"""Detached, supervised full-epoch Phase 7A training calibration."""

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.hpo.records import task_metrics
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/benchmark/07a_detached"
CONFIG = {"kind": "full_epoch", "outer_study": 3, "folds": [1, 2, 3, 4, 5],
          "processes": 2, "threads_each": 4, "data_loader_workers": 0,
          "depth": 3, "width": 128, "batch_size": 65536,
          "learning_rate": 0.001, "weight_decay": 0.000316227766,
          "precision": "BF16", "seed": 42, "epochs": 3,
          "reference_run": "07a-20261020T183149Z-b5b9a54d",
          "gpu_vram_stop_mib": 15500, "min_available_ram_bytes": 2 * 1024**3,
          "max_swap_growth_bytes": 128 * 1024**2}


def utc() -> str:
    return datetime.now(UTC).isoformat()


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(stable_json_bytes(value))
    tmp.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def pid_start(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (OSError, IndexError):
        return None


def run_dir(experiment_id: str) -> Path:
    if not experiment_id.startswith("07a-") or "/" in experiment_id or ".." in experiment_id:
        raise ValueError("Invalid experiment ID")
    path = RUNS / experiment_id
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def status(path: Path) -> dict:
    record = read(path / "status.json")
    if record["state"] in {"starting", "running", "stopping"}:
        pid = record.get("supervisor_pid")
        start = record.get("supervisor_pid_start")
        if pid and start and pid_start(pid) != start:
            record.update(state="failed", ended_at=utc(),
                          error="Supervisor disappeared; worker PDEATHSIG terminates children")
            for worker_record in record.get("workers", {}).values():
                if worker_record["state"] == "running":
                    worker_record.update(state="parent_lost", ended_at=utc(), exit_code=None)
            write(path / "status.json", record)
    record["progress"] = {key: read(progress_path) for key in record.get("workers", {})
                          if (progress_path := path / f"fold_{key}.progress.json").exists()}
    return record


def request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


STOP_REQUESTED = False


def child_parent_death(expected_parent: int) -> None:
    """Kernel kills a worker if its supervisor exits before reaping it."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), signal.SIGTERM)


def gpu_snapshot() -> tuple[float, float, float] | None:
    try:
        value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory",
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
    """Current RSS, bytes fetched from storage, and major faults."""
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
        io_lines = Path(f"/proc/{pid}/io").read_text().splitlines()
        read_bytes = next(int(line.split()[1]) for line in io_lines if line.startswith("read_bytes:"))
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return rss, read_bytes, int(stat[9])
    except (OSError, StopIteration, IndexError):
        return 0, 0, 0


def state_digest(value: object) -> str:
    digest = hashlib.sha256()

    def add(item: object) -> None:
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous().numpy()
            digest.update(stable_json_bytes({"dtype": str(data.dtype), "shape": data.shape}))
            digest.update(data.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                add(child)
        else:
            digest.update(stable_json_bytes(item))

    add(value)
    return digest.hexdigest()


def reference_check(path: Path, config: dict, fold: int, training: dict, state_sha: str) -> None:
    reference_path = RUNS / config["reference_run"] / f"fold_{fold}.json"
    reference = read(reference_path)
    keys = ("bp_loss", "flp_loss", "combined_loss", "bp_loss_sum", "flp_loss_sum",
            "bp_rows", "flp_valid_rows", "optimizer_steps")
    differences = {key: [reference["training"][key], training[key]] for key in keys
                   if reference["training"][key] != training[key]}
    if reference["state_sha256"] != state_sha:
        differences["state_sha256"] = [reference["state_sha256"], state_sha]
    write(path / f"fold_{fold}.reference_check.json", {"equal": not differences,
                                                         "differences": differences,
                                                         "reference_run": config["reference_run"]})
    if differences:
        raise ValueError(f"Epoch-1 reference mismatch for fold {fold}: {differences}")


def progress(path: Path, fold: int, epoch: int, phase: str) -> None:
    write(path / f"fold_{fold}.progress.json", {"fold": fold, "epoch": epoch,
                                                 "phase": phase, "updated_at": utc()})


def worker(path: Path, fold: int, parent_pid: int) -> None:
    child_parent_death(parent_pid)
    config = read(path / "config.json")
    cpus = sorted(os.sched_getaffinity(0))
    if config["kind"] == "full_epoch":
        setup_start = time.perf_counter()
        os.sched_setaffinity(0, cpus[fold % 2::2])
        torch.set_num_threads(config["threads_each"])
        torch.set_num_interop_threads(1)
        cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
        state_start = time.perf_counter()
        state = load_states(ROOT, cache, config["outer_study"])[fold]
        state_load_seconds = time.perf_counter() - state_start
        counts = support_counts(ROOT, config["outer_study"])[fold]
        if (state.rows != counts["train"]["rows"] or
                state.flp_valid_rows != counts["train"]["flp_valid"]):
            raise ValueError("Frozen inner TRAIN support mismatch")
        seed_everything(config["seed"])
        model, optimizer = make_model(33, config["depth"], config["width"],
                                      config["learning_rate"], config["weight_decay"])
        setup_seconds = time.perf_counter() - setup_start
        write(path / f"fold_{fold}.ready.json", {"fold": fold, "ready_at": utc(),
                                                 "pid": os.getpid(), "state_sha256": state.digest()})
        epochs = []
        global_step = 0
        metadata = {"benchmark": "phase_07a_3_epoch", "outer_study": config["outer_study"],
                    "inner_fold": fold, "configuration": {key: config[key] for key in (
                        "depth", "width", "batch_size", "learning_rate", "weight_decay", "seed")}}
        for epoch in range(1, config["epochs"] + 1):
            if (path / "stop.request.json").exists():
                break
            progress(path, fold, epoch, "train")
            training = train_epoch(ROOT, state, model, optimizer, epoch,
                                   outer_study=config["outer_study"], inner_fold=fold,
                                   batch_size=config["batch_size"], cache=cache)
            task_metrics(training)
            if epoch == 1:
                reference_check(path, config, fold, training, state.digest())
            global_step += training["optimizer_steps"]
            progress(path, fold, epoch, "validation")
            validation = validate(ROOT, state, model,
                                  counts["validation"]["rows"],
                                  counts["validation"]["flp_valid"],
                                  outer_study=config["outer_study"], inner_fold=fold,
                                  batch_size=config["batch_size"], cache=cache)
            task_metrics(validation)
            progress(path, fold, epoch, "checkpoint_save")
            checkpoint = path / f"fold_{fold}_epoch_{epoch:03d}.pt"
            before_model = state_digest(model.state_dict())
            before_optimizer = state_digest(optimizer.state_dict())
            save_start = time.perf_counter()
            save_cache_checkpoint(checkpoint, model, optimizer, completed_epoch=epoch,
                                  global_step=global_step, cache=cache, state=state,
                                  metadata=metadata)
            save_seconds = time.perf_counter() - save_start
            progress(path, fold, epoch, "checkpoint_load")
            load_start = time.perf_counter()
            restored_epoch, restored_step = load_cache_checkpoint(
                checkpoint, model, optimizer, cache=cache, state=state,
                expected_metadata=metadata)
            load_seconds = time.perf_counter() - load_start
            if (restored_epoch != epoch or restored_step != global_step or
                    state_digest(model.state_dict()) != before_model or
                    state_digest(optimizer.state_dict()) != before_optimizer):
                raise ValueError(f"Checkpoint resume mismatch for fold {fold}, epoch {epoch}")
            rss, read_bytes, major_faults = proc_stats(os.getpid())
            epoch_record = {"fold": fold, "epoch": epoch, "finished_at": utc(),
                            "training": training, "validation": validation,
                            "checkpoint_save_seconds": save_seconds,
                            "checkpoint_load_seconds": load_seconds,
                            "checkpoint_bytes": checkpoint.stat().st_size,
                            "checkpoint_sha256": read(checkpoint.with_suffix(".pt.sha256.json"))["sha256"],
                            "model_sha256": before_model, "optimizer_sha256": before_optimizer,
                            "global_step": global_step, "current_rss_bytes": rss,
                            "read_bytes": read_bytes, "major_faults": major_faults,
                            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                            "all_finite": all(math.isfinite(v) for v in (
                                training["bp_loss"], training["flp_loss"],
                                validation["bp_loss"], validation["flp_loss"]))}
            write(path / f"fold_{fold}_epoch_{epoch:03d}.json", epoch_record)
            epochs.append(epoch_record)
            if (path / "stop.request.json").exists():
                break
        progress(path, fold, len(epochs), "stopped" if len(epochs) < config["epochs"] else "complete")
        artifact = {"fold": fold, "finished_at": utc(), "epochs": epochs,
                    "stopped_early": len(epochs) < config["epochs"],
                    "setup_seconds": setup_seconds,
                    "preprocessing_state_load_seconds": state_load_seconds,
                    "state_sha256": state.digest(), "cache_source_hashes": cache.manifest["source_hashes"]}
    else:
        write(path / f"fold_{fold}.ready.json", {"fold": fold, "ready_at": utc(),
                                                 "pid": os.getpid()})
        time.sleep(config["smoke_seconds"])
        artifact = {"fold": fold, "finished_at": utc(), "smoke": True}
    write(path / f"fold_{fold}.json", artifact)


def terminate_children(active: dict[int, subprocess.Popen], record: dict) -> None:
    for proc in active.values():
        if proc.poll() is None:
            proc.terminate()
    for fold, proc in active.items():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        record["workers"][str(fold)].update(state="terminated", exit_code=proc.returncode,
                                             ended_at=utc())


def supervise(path: Path) -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = read(path / "config.json")
    record = read(path / "status.json")
    record.update(state="running", started_at=utc(), supervisor_pid=os.getpid(),
                  supervisor_pid_start=pid_start(os.getpid()))
    write(path / "status.json", record)
    active: dict[int, subprocess.Popen] = {}
    pending = list(config["folds"])
    completed: list[int] = []
    failure = None
    max_vram = 0.0
    samples = (path / "resource_samples.csv").open("w", newline="")
    columns = ["sampled_at", "elapsed_seconds", "active_folds", "worker_rss_bytes_sum",
               "worker_read_bytes_sum", "worker_major_faults_sum", "available_ram_bytes",
               "swap_used_bytes", "gpu_vram_used_mib", "gpu_utilization_percent",
               "gpu_memory_utilization_percent"]
    sample_writer = csv.DictWriter(samples, fieldnames=columns, lineterminator="\n")
    sample_writer.writeheader()
    start_time = time.monotonic()
    initial_swap = swap_used()
    try:
        while pending or active:
            if STOP_REQUESTED and record["state"] != "stopping":
                record.update(state="stopping", stop_requested_at=utc())
                write(path / "stop.request.json", {"requested_at": record["stop_requested_at"]})
                write(path / "status.json", record)
            while pending and len(active) < config["processes"] and not STOP_REQUESTED:
                fold = pending.pop(0)
                with (path / f"fold_{fold}.stdout.log").open("a") as out, (path / f"fold_{fold}.stderr.log").open("a") as err:
                    parent = os.getpid()
                    proc = subprocess.Popen([sys.executable, __file__, "_worker", path.name, str(fold),
                                             str(parent)],
                                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                            close_fds=True)
                active[fold] = proc
                record.setdefault("workers", {})[str(fold)] = {"pid": proc.pid, "pid_start": pid_start(proc.pid),
                                                              "started_at": utc(), "state": "running"}
                record["pending_folds"] = pending
                write(path / "status.json", record)
            for fold, proc in list(active.items()):
                result = proc.poll()
                if result is None:
                    continue
                proc.wait()
                del active[fold]
                item = record["workers"][str(fold)]
                artifact_path = path / f"fold_{fold}.json"
                early_stop = artifact_path.exists() and read(artifact_path).get("stopped_early", False)
                item.update(ended_at=utc(), exit_code=result,
                            state=("stopped" if early_stop else "complete")
                            if result == 0 and artifact_path.exists() else "failed")
                if item["state"] == "complete":
                    completed.append(fold)
                elif item["state"] == "failed":
                    failure = f"fold {fold} exited {result}; inspect its stderr log"
                record.update(completed_folds=completed, pending_folds=pending)
                write(path / "status.json", record)
            if failure:
                terminate_children(active, record)
                active.clear()
                break
            if config["kind"] == "full_epoch" and active:
                gpu = gpu_snapshot()
                used = gpu[0] if gpu is not None else None
                ram = available_ram()
                swap = swap_used()
                worker_stats = [proc_stats(proc.pid) for proc in active.values()]
                sample_writer.writerow({"sampled_at": utc(),
                                        "elapsed_seconds": time.monotonic() - start_time,
                                        "active_folds": ",".join(map(str, sorted(active))),
                                        "worker_rss_bytes_sum": sum(x[0] for x in worker_stats),
                                        "worker_read_bytes_sum": sum(x[1] for x in worker_stats),
                                        "worker_major_faults_sum": sum(x[2] for x in worker_stats),
                                        "available_ram_bytes": ram,
                                        "swap_used_bytes": swap,
                                        "gpu_vram_used_mib": used,
                                        "gpu_utilization_percent": gpu[1] if gpu else None,
                                        "gpu_memory_utilization_percent": gpu[2] if gpu else None})
                samples.flush()
                if used is not None:
                    max_vram = max(max_vram, used)
                if (used is not None and used > config["gpu_vram_stop_mib"] or
                        ram < config["min_available_ram_bytes"] or
                        swap - initial_swap > config["max_swap_growth_bytes"]):
                    failure = "Resource safety gate: VRAM, available RAM, or swap growth"
                    terminate_children(active, record)
                    active.clear()
                    break
            if active:
                time.sleep(1)
        final_state = "failed" if failure else ("stopped" if STOP_REQUESTED else "complete")
        record.update(state=final_state, ended_at=utc(), error=failure,
                      completed_folds=completed, pending_folds=pending,
                      peak_sampled_device_vram_mib=max_vram)
        write(path / "status.json", record)
    except BaseException:
        terminate_children(active, record)
        record.update(state="failed", ended_at=utc(), error=traceback.format_exc())
        write(path / "status.json", record)
        raise
    finally:
        samples.close()


def launch(args) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.kind == "full_epoch":
        if len(os.sched_getaffinity(0)) < 8 or not torch.cuda.is_available():
            raise RuntimeError("Two workers require at least eight accessible logical CPUs and CUDA")
        free, total = torch.cuda.mem_get_info()
        if total < 8 * 1024**3 or free < 8 * 1024**3 or available_ram() < 4 * 1024**3:
            raise RuntimeError("Insufficient GPU or guest RAM for validated two-worker calibration")
        config = dict(CONFIG)
        reference_path = RUNS / config["reference_run"]
        if read(reference_path / "status.json")["state"] != "complete" or any(
            not (reference_path / f"fold_{fold}.json").exists() for fold in config["folds"]
        ):
            raise RuntimeError("Complete five-fold epoch-1 reference is required")
        reference_config = read(reference_path / "config.json")
        for key in ("outer_study", "folds", "processes", "threads_each", "depth", "width",
                    "batch_size", "learning_rate", "weight_decay", "precision", "seed"):
            if config[key] != reference_config[key]:
                raise ValueError(f"Epoch-1 reference configuration differs: {key}")
        for previous in RUNS.glob("07a-*/config.json"):
            if previous.parent == reference_path:
                continue
            if read(previous).get("kind") == "full_epoch" and status(previous.parent)["state"] in {
                "starting", "running", "stopping"
            }:
                raise RuntimeError(f"Another full-epoch calibration is active: {previous.parent.name}")
    else:
        config = {"kind": "smoke", "folds": [1], "processes": 1,
                  "smoke_seconds": args.smoke_seconds}
    config.update(created_at=utc(), source_git_sha=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    experiment_id = "07a-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path = RUNS / experiment_id
    path.mkdir(mode=0o755)
    write(path / "config.json", config)
    write(path / "status.json", {"experiment_id": experiment_id, "state": "starting",
                                  "created_at": config["created_at"], "pending_folds": config["folds"],
                                  "completed_folds": [], "workers": {}})
    with (path / "supervisor.stdout.log").open("a") as out, (path / "supervisor.stderr.log").open("a") as err:
        proc = subprocess.Popen([sys.executable, __file__, "_supervise", experiment_id], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True, close_fds=True)
    # Readiness check only; never wait for calibration completion.
    for _ in range(100):
        current = status(path)
        if current["state"] == "running" and any(path.glob("fold_*.ready.json")):
            print(json.dumps({"experiment_id": experiment_id, "status": current["state"],
                              "supervisor_pid": proc.pid, "run_dir": str(path),
                              "status_command": f"{sys.executable} {__file__} status {experiment_id}"}))
            return
        if current["state"] == "failed" or proc.poll() is not None:
            raise RuntimeError(f"Detached supervisor failed: {path / 'supervisor.stderr.log'}")
        time.sleep(0.1)
    raise TimeoutError(f"Supervisor did not become ready: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("launch")
    start.add_argument("--kind", choices=["full_epoch", "smoke"], default="full_epoch")
    start.add_argument("--smoke-seconds", type=int, default=30)
    for name in ("status", "results", "stop", "_supervise"):
        sub.add_parser(name).add_argument("experiment_id")
    internal = sub.add_parser("_worker")
    internal.add_argument("experiment_id")
    internal.add_argument("fold", type=int)
    internal.add_argument("parent_pid", type=int)
    args = parser.parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "_supervise":
        supervise(run_dir(args.experiment_id))
    elif args.command == "_worker":
        worker(run_dir(args.experiment_id), args.fold, args.parent_pid)
    elif args.command == "status":
        print(json.dumps(status(run_dir(args.experiment_id)), indent=2))
    elif args.command == "results":
        path = run_dir(args.experiment_id)
        print(json.dumps({"status": status(path), "artifacts": {
            p.stem: read(p) for p in path.glob("fold_[0-9].json")}}, indent=2))
    else:
        path = run_dir(args.experiment_id)
        current = status(path)
        if current["state"] not in {"starting", "running", "stopping"}:
            print(json.dumps(current, indent=2))
            return
        pid = current.get("supervisor_pid")
        if not pid or pid_start(pid) != current.get("supervisor_pid_start"):
            raise RuntimeError("Supervisor is not alive; inspect status")
        os.kill(pid, signal.SIGTERM)
        print(f"Graceful stop requested for {args.experiment_id}; active folds finish first")


if __name__ == "__main__":
    main()
