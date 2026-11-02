"""One detached, sequential, resumable 25-model final ensemble experiment."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import optuna
import torch

from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.storage import study_name
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    fit_preprocessing,
    stable_json_bytes,
)
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
)

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "artifacts/training/phase8"
MODULE = "wildfire_risk_uncertainty.training.phase8"
SELECTED = {1: 0, 2: 1, 3: 0, 4: 1, 5: 2}
EPOCHS = 100
MEMBERS = 5
MAX_ATTEMPTS = 3
STOP = False


def utc() -> str:
    return datetime.now(UTC).isoformat()


def pid_start(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def child_parent_death(expected_parent: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), signal.SIGKILL)


def gpu_snapshot() -> tuple[float, float, float] | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        return tuple(float(item.strip()) for item in value.splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def available_ram() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def swap_used() -> int:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in {"SwapTotal", "SwapFree"}:
            values[key] = int(line.split()[1]) * 1024
    return values["SwapTotal"] - values["SwapFree"]


def proc_stats(pid: int) -> tuple[int, int, int]:
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
        io_lines = Path(f"/proc/{pid}/io").read_text().splitlines()
        read_bytes = next(int(line.split()[1]) for line in io_lines if line.startswith("read_bytes:"))
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return rss, read_bytes, int(fields[9])
    except (OSError, StopIteration, IndexError):
        return 0, 0, 0


def digest(value: object) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def chosen_trials() -> dict:
    result = {}
    for outer, number in SELECTED.items():
        database = ROOT / f"artifacts/hpo/outer_{outer}_gradnorm_v3.sqlite3"
        study = optuna.load_study(study_name=study_name(outer, gradnorm=True),
                                  storage=f"sqlite:///{database.resolve()}")
        complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        first_five = complete[:5]
        if len(first_five) != 5 or max(first_five, key=lambda trial: (trial.value, -trial.number)).number != number:
            raise ValueError(f"Outer {outer} selection differs from first-five-success HPO record")
        trial = study.trials[number]
        result[str(outer)] = {"trial_number": number, "value": trial.value,
                              "configuration": trial.params, "study_name": study.study_name,
                              "database_sha256": sha256(database),
                              "first_five_successful_trial_numbers": [item.number for item in first_five]}
    pilot = optuna.load_study(study_name=study_name(3, gradnorm=True),
                              storage=f"sqlite:///{(ROOT / 'artifacts/hpo/outer_3_gradnorm_v3.sqlite3').resolve()}")
    pilot_complete = [trial for trial in pilot.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if (len(pilot_complete) != 20 or
            max(pilot_complete, key=lambda trial: (trial.value, -trial.number)).number != 19):
        raise ValueError("Outer 3 full-pilot sensitivity trial is missing")
    result["3"]["full_pilot_sensitivity_trial"] = 19
    result["3"]["full_pilot_sensitivity_value"] = pilot.trials[19].value
    return result


def create_manifest() -> dict:
    cache = read_json(ROOT / "artifacts/training/hpo_cache_v1/manifest.json")
    files = ["training/phase8.py", "training/trainer.py", "training/cache.py", "training/data.py",
             "training/checkpoint.py", "training/cache_checkpoint.py", "training/gradnorm.py",
             "training/losses.py", "training/model.py", "training/preprocessing.py"]
    source = ROOT / "src/wildfire_risk_uncertainty"
    return {"protocol": "phase8-final-ensemble-v1", "created_at": utc(),
            "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_file_sha256": {name: sha256(source / name) for name in files},
            "detached_helper_sha256": sha256(ROOT / "scripts/benchmark_07a_detached.py"),
            "project_sha256": sha256(ROOT / "pyproject.toml"),
            "dependency_lock_sha256": sha256(ROOT / "uv.lock"),
            "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "cache_manifest_sha256": sha256(ROOT / "artifacts/training/hpo_cache_v1/manifest.json"),
            "cache_source_hashes": cache["source_hashes"],
            "outer_fold_manifest_sha256": sha256(ROOT / "artifacts/spatial_validation/05b/manifest.json"),
            "feature_schema_sha256": sha256(ROOT / "reports/dataset/canonical/schema.json"),
            "selected_trials": chosen_trials(), "outer_studies": list(SELECTED),
            "members_per_study": MEMBERS, "epochs": EPOCHS, "optimizer": "AdamW",
            "scheduler": "none", "precision": "BF16 autocast", "loss_balancing": "GradNorm",
            "gradnorm_alpha": 1.5, "gradnorm_weight_learning_rate": 0.025,
            "seed_formula": "42000 + 1000 * outer_study + member_index (member_index=0..4)",
            "data_order_seed": 42, "outer_test_policy": "never accessed during final training",
            "execution": "one sequential GPU worker; maximum three attempts per model"}


def verify_manifest(path: Path) -> dict:
    manifest = read_json(path / "manifest.json")
    current = create_manifest()
    current["created_at"] = manifest["created_at"]
    if manifest != current:
        raise ValueError("Frozen Phase 8 source, data, or HPO provenance differs")
    return manifest


def model_key(outer: int, member: int) -> str:
    return f"outer_{outer}_member_{member}"


def worker_command(path: Path, outer: int, member: int, parent_pid: int) -> list[str]:
    """Use the importable module name even when invoked through a console script."""
    return [sys.executable, "-m", MODULE, "_worker", str(path), str(outer),
            str(member), str(parent_pid)]


def retryable_worker_exit(returncode: int) -> bool:
    """Signals can interrupt a worker; Python exceptions require inspection."""
    return returncode < 0


def model_dir(path: Path, key: str) -> Path:
    result = path / "models" / key
    result.mkdir(parents=True, exist_ok=True)
    return result


def checkpoint_metadata(manifest: dict, key: str, seed: int, state_sha: str) -> dict:
    return {"manifest_sha256": digest(manifest), "model_key": key,
            "initialization_seed": seed, "outer_train_preprocessing_sha256": state_sha,
            "partition_role": "complete_outer_train"}


def valid_checkpoint(path: Path, model, optimizer, cache, state, metadata, gradnorm):
    try:
        return load_cache_checkpoint(path, model, optimizer, cache=cache, state=state,
                                     expected_metadata=metadata, gradnorm=gradnorm)
    except ValueError as exc:
        if "provenance" in str(exc) or "mode mismatch" in str(exc):
            raise
        return None
    except (OSError, RuntimeError, EOFError, KeyError, pickle.UnpicklingError,
            json.JSONDecodeError):
        return None


def quarantine_incomplete(target: Path) -> None:
    """Clear only artifacts for the epoch being replayed; retain forensic copies."""
    for candidate in (target, target.with_suffix(".tmp"),
                      target.with_suffix(target.suffix + ".sha256.json"),
                      target.with_name(target.name + ".sha256.json.tmp")):
        if candidate.exists():
            quarantine = candidate.with_name(candidate.name + ".corrupt")
            if quarantine.exists():
                raise FileExistsError(quarantine)
            candidate.rename(quarantine)


def outer_preprocessing(path: Path, outer: int, cache: HpoCache,
                        schema_sha256: str) -> PreprocessingState:
    """Fit once on outer TRAIN, then restore the exact frozen state."""
    inputs = tuple(cache.manifest["inputs"])
    state_path = path / "preprocessing" / f"outer_{outer}.json"
    if state_path.exists():
        record = read_json(state_path)
        state = PreprocessingState(inputs, tuple(record["means_after_fixed_transform"]),
                                   tuple(record["standard_deviations_after_imputation"]),
                                   tuple(record["missing_counts"][name] for name in inputs),
                                   record["rows"], record["flp_valid_rows"], record["schema_sha256"])
        if record["sha256"] != state.digest() or state.schema_sha256 != schema_sha256:
            raise ValueError("Outer TRAIN preprocessing hash mismatch")
        return state
    state = fit_preprocessing(cache.outer_train_chunks(outer_study=outer, epoch=None),
                              inputs, schema_sha256)
    if state.rows != cache.manifest["outer_train_counts"][outer - 1]:
        raise ValueError("Outer TRAIN support differs from frozen fold summary")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(state_path, {**state.serializable(), "sha256": state.digest()})
    return state


def worker(path: Path, outer: int, member: int, parent_pid: int) -> None:
    child_parent_death(parent_pid)
    manifest = verify_manifest(path)
    key = model_key(outer, member)
    directory = model_dir(path, key)
    configuration = manifest["selected_trials"][str(outer)]["configuration"]
    seed = 42000 + 1000 * outer + member
    seed_everything(seed)
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    inputs = tuple(cache.manifest["inputs"])
    state = outer_preprocessing(path, outer, cache, manifest["feature_schema_sha256"])
    model, optimizer = make_model(inputs=len(inputs), depth=configuration["depth"],
                                  width=configuration["width"],
                                  learning_rate=configuration["learning_rate"],
                                  weight_decay=configuration["weight_decay"])
    gradnorm = GradNorm("cuda")
    metadata = checkpoint_metadata(manifest, key, seed, state.digest())
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    epoch = global_step = 0
    for candidate in sorted(checkpoints.glob("epoch_*.pt"), reverse=True):
        restored = valid_checkpoint(candidate, model, optimizer, cache, state, metadata, gradnorm)
        if restored is not None:
            epoch, global_step = restored
            if candidate.name != f"epoch_{epoch:03d}.pt":
                raise ValueError("Checkpoint epoch and filename disagree")
            break
    if epoch == EPOCHS:
        final = checkpoints / f"epoch_{EPOCHS:03d}.pt"
        write_json(directory / "complete.json", {"model_key": key, "completed_at": utc(),
                                                   "checkpoint_sha256": sha256(final),
                                                   "preprocessing_sha256": state.digest()})
        return
    for next_epoch in range(epoch + 1, EPOCHS + 1):
        if (path / "pause.request.json").exists():
            return
        metrics = train_epoch(ROOT, state, model, optimizer, next_epoch,
                              outer_study=outer, inner_fold=0,
                              batch_size=configuration["batch_size"], cache=cache,
                              gradnorm=gradnorm, global_step_base=global_step)
        global_step += metrics["optimizer_steps"]
        target = checkpoints / f"epoch_{next_epoch:03d}.pt"
        if target.exists() and valid_checkpoint(target, model, optimizer, cache, state, metadata, gradnorm) is not None:
            raise ValueError("Existing later checkpoint blocks recovery; inspect trajectory")
        quarantine_incomplete(target)
        save_cache_checkpoint(target, model, optimizer, completed_epoch=next_epoch,
                              global_step=global_step, cache=cache, state=state,
                              metadata=metadata, gradnorm=gradnorm)
        if valid_checkpoint(target, model, optimizer, cache, state, metadata, gradnorm) != (next_epoch, global_step):
            raise ValueError("New checkpoint failed integrity and state restoration")
        write_json(directory / "progress.json", {"model_key": key, "completed_epoch": next_epoch,
                                                  "global_step": global_step, "updated_at": utc(),
                                                  "training": metrics, "checkpoint_sha256": sha256(target)})
    final = checkpoints / f"epoch_{EPOCHS:03d}.pt"
    write_json(directory / "complete.json", {"model_key": key, "completed_at": utc(),
                                               "checkpoint_sha256": sha256(final),
                                               "preprocessing_sha256": state.digest()})


def complete(path: Path, key: str) -> bool:
    directory = model_dir(path, key)
    marker = directory / "complete.json"
    final = directory / "checkpoints" / f"epoch_{EPOCHS:03d}.pt"
    if not marker.exists():
        return False
    sidecar = final.with_suffix(final.suffix + ".sha256.json")
    try:
        record = read_json(marker)
        return (record["model_key"] == key and final.exists() and sidecar.exists() and
                sha256(final) == record["checkpoint_sha256"] and
                read_json(sidecar)["sha256"] == record["checkpoint_sha256"])
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def request_pause(_signum, _frame) -> None:
    global STOP
    STOP = True


def supervise(path: Path) -> None:
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    lock = (path / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify_manifest(path)
    ledger_path = path / "ledger.json"
    ledger = read_json(ledger_path)
    ledger.update(state="running", controller_pid=os.getpid(),
                  controller_pid_start=pid_start(os.getpid()), started_at=utc(), error=None)
    write_json(ledger_path, ledger)
    try:
        for outer in manifest["outer_studies"]:
            for member in range(MEMBERS):
                key = model_key(outer, member)
                if complete(path, key):
                    ledger["models"][key]["state"] = "complete"
                    continue
                if STOP or (path / "pause.request.json").exists():
                    ledger["state"] = "paused"
                    write_json(ledger_path, ledger)
                    return
                entry = ledger["models"][key]
                while entry["failures"] < MAX_ATTEMPTS:
                    entry["attempts"] += 1
                    entry["state"] = "running"
                    ledger["current_model"] = key
                    write_json(ledger_path, ledger)
                    with (path / "worker.stdout.log").open("a") as out, (path / "worker.stderr.log").open("a") as err:
                        proc = subprocess.Popen(worker_command(path, outer, member, os.getpid()), cwd=ROOT,
                                                stdin=subprocess.DEVNULL, stdout=out, stderr=err)
                        initial_swap = swap_used()
                        while proc.poll() is None:
                            if STOP:
                                write_json(path / "pause.request.json", {"requested_at": utc()})
                            progress = model_dir(path, key) / "progress.json"
                            if progress.exists():
                                entry["completed_epoch"] = read_json(progress)["completed_epoch"]
                            entry["worker_rss_bytes"] = proc_stats(proc.pid)[0]
                            entry["available_ram_bytes"] = available_ram()
                            entry["swap_growth_bytes"] = max(0, swap_used() - initial_swap)
                            gpu = gpu_snapshot()
                            entry["gpu_vram_used_mib"] = gpu[0] if gpu else None
                            if (entry["available_ram_bytes"] < 2 * 1024**3 or
                                    entry["swap_growth_bytes"] > 128 * 1024**2 or
                                    (gpu and gpu[0] > 15500)):
                                ledger["resource_safety_event"] = {"model_key": key, "at": utc(),
                                                                    "available_ram_bytes": entry["available_ram_bytes"],
                                                                    "swap_growth_bytes": entry["swap_growth_bytes"],
                                                                    "gpu_vram_used_mib": entry["gpu_vram_used_mib"]}
                                write_json(path / "pause.request.json", ledger["resource_safety_event"])
                            write_json(ledger_path, ledger)
                            time.sleep(5)
                        result = proc.wait()
                    if result == 0 and complete(path, key):
                        entry.update(state="complete", completed_epoch=EPOCHS, completed_at=utc())
                        write_json(ledger_path, ledger)
                        break
                    if result == 0 and (path / "pause.request.json").exists():
                        entry["state"] = "paused"
                        ledger["state"] = "paused"
                        write_json(ledger_path, ledger)
                        return
                    entry.update(state="retrying" if retryable_worker_exit(result) else "failed",
                                 last_exit_code=result, last_failure_at=utc(),
                                 failures=entry["failures"] + 1)
                    write_json(ledger_path, ledger)
                    if not retryable_worker_exit(result):
                        raise RuntimeError(f"{key} exited {result}; deterministic worker error, inspect worker.stderr.log")
                    time.sleep(min(30, 5 * entry["failures"]))
                else:
                    raise RuntimeError(f"{key} failed after {MAX_ATTEMPTS} attempts; inspect worker.stderr.log")
        ledger.update(state="complete", current_model=None, completed_at=utc())
        write_json(ledger_path, ledger)
    except BaseException:
        ledger.update(state="failed", error=traceback.format_exc(), ended_at=utc())
        write_json(ledger_path, ledger)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)


def launch(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    manifest = create_manifest()
    write_json(path / "manifest.json", manifest)
    keys = [model_key(outer, member) for outer in SELECTED for member in range(MEMBERS)]
    write_json(path / "ledger.json", {"state": "starting", "created_at": utc(),
                                       "models": {key: {"state": "pending", "attempts": 0, "failures": 0,
                                                        "completed_epoch": 0} for key in keys}})
    start_controller(path)


def start_controller(path: Path) -> None:
    verify_manifest(path)
    existing = read_json(path / "ledger.json")
    if (existing.get("controller_pid") and existing.get("controller_pid_start") and
            pid_start(existing["controller_pid"]) == existing["controller_pid_start"]):
        raise RuntimeError("Phase 8 controller is already running")
    # Hash the large immutable cache once per controller launch, not once per model.
    HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=True)
    with (path / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Phase 8 controller is already running") from exc
        pause = path / "pause.request.json"
        if pause.exists():
            pause.unlink()
        with (path / "supervisor.stdout.log").open("a") as out, (path / "supervisor.stderr.log").open("a") as err:
            proc = subprocess.Popen([sys.executable, "-m", MODULE, "_supervise", str(path)],
                                    cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err)
        ledger = read_json(path / "ledger.json")
        ledger.update(state="starting", controller_pid=proc.pid,
                      controller_pid_start=pid_start(proc.pid))
        write_json(path / "ledger.json", ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "resume", "pause", "status", "_supervise", "_worker"))
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("outer", nargs="?", type=int)
    parser.add_argument("member", nargs="?", type=int)
    parser.add_argument("parent_pid", nargs="?", type=int)
    args = parser.parse_args()
    if args.action == "launch":
        name = args.experiment or datetime.now(UTC).strftime("phase8-%Y%m%dT%H%M%SZ")
        if not name.startswith("phase8-") or "/" in name or ".." in name:
            raise ValueError("Invalid experiment name")
        launch(RUNS / name)
        print(RUNS / name)
        return
    if not args.experiment:
        parser.error("Experiment path or name required")
    path = Path(args.experiment) if args.action.startswith("_") else RUNS / args.experiment
    if args.action == "_worker":
        worker(path, args.outer, args.member, args.parent_pid)
    elif args.action == "_supervise":
        supervise(path)
    elif args.action == "resume":
        start_controller(path)
    elif args.action == "pause":
        write_json(path / "pause.request.json", {"requested_at": utc()})
    else:
        ledger = read_json(path / "ledger.json")
        if (ledger.get("state") in {"starting", "running"} and
                ledger.get("controller_pid_start") and
                pid_start(ledger["controller_pid"]) != ledger["controller_pid_start"]):
            ledger["state"] = "interrupted"
            ledger["error"] = "Supervisor process disappeared; run resume to recover checkpoints"
        for key, entry in ledger["models"].items():
            progress = path / "models" / key / "progress.json"
            if progress.exists():
                entry["completed_epoch"] = read_json(progress)["completed_epoch"]
        print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
