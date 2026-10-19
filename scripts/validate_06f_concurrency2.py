"""Final isolated validation of two-fold concurrency for Checkpoint 6F."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from wildfire_risk_uncertainty.hpo.config import Configuration
from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/training/06f_resource_optimization/final_validation"
CACHE_PATH = ROOT / "artifacts/training/hpo_cache_v1"
CONFIG = Configuration(3, 64, 0.00020511104188433984, 1.493656855461762e-05, 32768)
VRAM_STOP_MIB = 15_500


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(stable_json_bytes(value))
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def state_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(
                stable_json_bytes({"dtype": str(array.dtype), "shape": array.shape})
            )
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            digest.update(stable_json_bytes(item))

    update(value)
    return digest.hexdigest()


def worker(suite: str, run: int, concurrency: int, fold: int, epochs: int) -> None:
    cache = HpoCache(ROOT, CACHE_PATH, check_files=False)
    state = load_states(ROOT, cache, 1)[fold]
    counts = support_counts(ROOT, 1)[fold]
    seed_everything(42)
    model, optimizer = make_model(
        len(state.inputs),
        CONFIG.depth,
        CONFIG.width,
        CONFIG.learning_rate,
        CONFIG.weight_decay,
    )
    records = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        training = train_epoch(
            ROOT,
            state,
            model,
            optimizer,
            epoch,
            outer_study=1,
            inner_fold=fold,
            batch_size=CONFIG.batch_size,
            cache=cache,
        )
        validation = validate(
            ROOT,
            state,
            model,
            counts["validation"]["rows"],
            counts["validation"]["flp_valid"],
            outer_study=1,
            inner_fold=fold,
            batch_size=CONFIG.batch_size,
            cache=cache,
        )
        records.append(
            {
                "epoch": epoch,
                "training": training,
                "validation": validation,
                "model_sha256": state_digest(model.state_dict()),
                "optimizer_sha256": state_digest(optimizer.state_dict()),
                "epoch_wall_seconds": time.perf_counter() - epoch_started,
            }
        )
    result = {
        "suite": suite,
        "run": run,
        "concurrency": concurrency,
        "fold": fold,
        "configuration": CONFIG.serializable(),
        "epochs": records,
        "preprocessing_sha256": state.digest(),
        "worker_wall_seconds": time.perf_counter() - started,
    }
    write_json(OUT / f"{suite}_run{run}_c{concurrency}_fold{fold}.json", result)


def memory_status() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(rest.split()[0]) * 1024
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


def gpu_status() -> tuple[float, float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=8,
    )
    return tuple(
        float(value.strip()) for value in result.stdout.splitlines()[0].split(",")
    )


def warm_cache() -> float:
    """Read all mmap inputs before a measured run; time is excluded."""
    started = time.perf_counter()
    names = (
        "features.npy",
        "bp.npy",
        "flp.npy",
        "flp_valid.npy",
        "inner_fold.npy",
        "partition_offsets.npy",
    )
    total = 0
    for name in names:
        with (CACHE_PATH / name).open("rb", buffering=16 * 1024 * 1024) as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                total += len(block)
    if total <= 0:
        raise RuntimeError("Cache warming read no bytes")
    return time.perf_counter() - started


METRIC_KEYS = (
    "bp_rows",
    "flp_valid_rows",
    "optimizer_steps",
    "bp_loss_sum",
    "flp_loss_sum",
    "bp_loss",
    "flp_loss",
    "combined_loss",
)


def compare_fold(candidate: dict, reference: dict) -> list[str]:
    mismatch = []
    if candidate["preprocessing_sha256"] != reference["preprocessing_sha256"]:
        mismatch.append("preprocessing_sha256")
    if len(candidate["epochs"]) != len(reference["epochs"]):
        mismatch.append("epochs")
        return mismatch
    for left, right in zip(candidate["epochs"], reference["epochs"], strict=True):
        epoch = right["epoch"]
        if left["epoch"] != epoch:
            mismatch.append(f"epoch_{epoch}.identity")
        for role in ("training", "validation"):
            for key in METRIC_KEYS:
                if left[role][key] != right[role][key]:
                    mismatch.append(f"epoch_{epoch}.{role}.{key}")
        for key in ("model_sha256", "optimizer_sha256"):
            if left[key] != right[key]:
                mismatch.append(f"epoch_{epoch}.{key}")
    return mismatch


def validate_support(result: dict, counts: dict) -> None:
    fold = result["fold"]
    for record in result["epochs"]:
        for role, support_role in (("training", "train"), ("validation", "validation")):
            metrics = record[role]
            if metrics["bp_rows"] != counts[fold][support_role]["rows"]:
                raise RuntimeError(
                    f"Fold {fold} epoch {record['epoch']} {role} rows mismatch"
                )
            if metrics["flp_valid_rows"] != counts[fold][support_role]["flp_valid"]:
                raise RuntimeError(
                    f"Fold {fold} epoch {record['epoch']} {role} FLP support mismatch"
                )
            expected_batches = math.ceil(metrics["bp_rows"] / CONFIG.batch_size)
            if metrics["optimizer_steps"] != expected_batches:
                raise RuntimeError(
                    f"Fold {fold} epoch {record['epoch']} {role} batch count mismatch"
                )


def execute(suite: str, run: int, concurrency: int, epochs: int) -> dict:
    expected = [
        OUT / f"{suite}_run{run}_c{concurrency}_fold{fold}.json" for fold in range(1, 6)
    ]
    if any(path.exists() for path in expected):
        if not all(path.exists() for path in expected):
            raise RuntimeError(
                f"Partial prior output requires HITL: {suite} run {run} c{concurrency}"
            )
        raise RuntimeError(
            f"Refusing to rerun completed output: {suite} run {run} c{concurrency}"
        )
    warm_seconds = warm_cache()
    available_start, swap_start = memory_status()
    pending = list(range(1, 6))
    active: dict[int, tuple[subprocess.Popen, object]] = {}
    all_processes = []
    samples = []
    failure = None
    started = time.monotonic()
    while pending or active:
        while pending and len(active) < concurrency:
            fold = pending.pop(0)
            stderr_path = OUT / f"{suite}_run{run}_c{concurrency}_fold{fold}.stderr"
            stderr = stderr_path.open("w")
            process = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    suite,
                    str(run),
                    str(concurrency),
                    str(fold),
                    str(epochs),
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
            active[fold] = (process, stderr)
            all_processes.append(process)
        time.sleep(2)
        elapsed = time.monotonic() - started
        available, swap = memory_status()
        try:
            utilization, power, vram = gpu_status()
        except (subprocess.SubprocessError, ValueError):
            utilization = power = vram = float("nan")
        samples.append(
            {
                "elapsed_seconds": elapsed,
                "available_ram_bytes": available,
                "swap_used_bytes": swap,
                "gpu_util_percent": utilization,
                "gpu_power_w": power,
                "gpu_vram_mib": vram,
            }
        )
        if swap - swap_start > 128 * 1024**2 or available < 2 * 1024**3:
            failure = "swap or RAM pressure"
        if vram > VRAM_STOP_MIB:
            failure = "VRAM pressure"
        if elapsed > 1_800:
            failure = "possible deadlock"
        for fold, (process, stderr) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stderr.close()
            del active[fold]
            if code != 0:
                failure = f"fold {fold} exited {code}"
        if failure:
            for process in all_processes:
                if process.poll() is None:
                    process.terminate()
            for process in all_processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            for _, stderr in active.values():
                stderr.close()
            write_json(
                OUT / f"{suite}_run{run}_c{concurrency}_STOP.json",
                {"reason": failure, "samples": samples},
            )
            raise RuntimeError(failure)
    wall_seconds = time.monotonic() - started
    results = [read_json(path) for path in expected]
    counts = support_counts(ROOT, 1)
    for result in results:
        validate_support(result, counts)
    record = {
        "suite": suite,
        "run": run,
        "concurrency": concurrency,
        "epochs": epochs,
        "wall_seconds": wall_seconds,
        "warm_cache_seconds_excluded": warm_seconds,
        "rows": sum(
            sum(
                epoch[role]["bp_rows"]
                for epoch in result["epochs"]
                for role in ("training", "validation")
            )
            for result in results
        ),
        "peak_gpu_vram_mib": max(sample["gpu_vram_mib"] for sample in samples),
        "mean_gpu_util_percent": sum(sample["gpu_util_percent"] for sample in samples)
        / len(samples),
        "min_available_ram_bytes": min(
            sample["available_ram_bytes"] for sample in samples
        ),
        "peak_swap_used_bytes": max(sample["swap_used_bytes"] for sample in samples),
        "available_ram_at_start_bytes": available_start,
        "samples": len(samples),
    }
    write_json(OUT / f"{suite}_run{run}_c{concurrency}_summary.json", record)
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = HpoCache(ROOT, CACHE_PATH, check_files=False)
    metadata = {
        "source_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "outer_study": 1,
        "trial": 0,
        "configuration": CONFIG.serializable(),
        "cache_manifest_sha256": hashlib.sha256(
            stable_json_bytes(cache.manifest)
        ).hexdigest(),
        "timing_repetitions": 3,
        "regression_epochs": 4,
        "concurrency_candidates": [1, 2],
        "vram_stop_mib": VRAM_STOP_MIB,
    }
    metadata_path = OUT / "metadata.json"
    if metadata_path.exists() and read_json(metadata_path) != metadata:
        raise RuntimeError("Existing final-validation metadata differs")
    write_json(metadata_path, metadata)
    timing = []
    schedules = ((1, 2), (2, 1), (1, 2))
    original_reference = {
        fold: read_json(OUT.parent / f"c1_fold{fold}.json") for fold in range(1, 6)
    }
    for run, schedule in enumerate(schedules, start=1):
        run_records = {}
        for concurrency in schedule:
            record = execute("timing", run, concurrency, 1)
            run_records[concurrency] = record
            for fold in range(1, 6):
                result = read_json(
                    OUT / f"timing_run{run}_c{concurrency}_fold{fold}.json"
                )
                mismatch = compare_fold(
                    result,
                    {
                        **original_reference[fold],
                        "epochs": [
                            {
                                "epoch": 1,
                                "training": original_reference[fold]["training"],
                                "validation": original_reference[fold]["validation"],
                                "model_sha256": original_reference[fold][
                                    "model_sha256"
                                ],
                                "optimizer_sha256": original_reference[fold][
                                    "optimizer_sha256"
                                ],
                            }
                        ],
                        "preprocessing_sha256": original_reference[fold][
                            "state_sha256"
                        ],
                    },
                )
                if mismatch:
                    raise RuntimeError(
                        f"Timing numerical mismatch run {run} c{concurrency} fold {fold}: {mismatch}"
                    )
            print(
                f"timing run {run} c{concurrency}: {record['wall_seconds']:.3f}s",
                flush=True,
            )
        timing.append(
            {
                "run": run,
                "c1_wall_seconds": run_records[1]["wall_seconds"],
                "c2_wall_seconds": run_records[2]["wall_seconds"],
                "speedup": run_records[1]["wall_seconds"]
                / run_records[2]["wall_seconds"],
            }
        )
        write_json(OUT / "timing_results.json", timing)
    regression = {}
    for concurrency in (1, 2):
        regression[concurrency] = execute("regression", 1, concurrency, 4)
        print(
            f"regression c{concurrency}: {regression[concurrency]['wall_seconds']:.3f}s",
            flush=True,
        )
        if concurrency == 2:
            for fold in range(1, 6):
                reference = read_json(OUT / f"regression_run1_c1_fold{fold}.json")
                candidate = read_json(OUT / f"regression_run1_c2_fold{fold}.json")
                mismatch = compare_fold(candidate, reference)
                if mismatch:
                    raise RuntimeError(f"Regression mismatch fold {fold}: {mismatch}")
    result = {
        "status": "PASS",
        "exact_equivalence": True,
        "timing": timing,
        "regression": {
            "c1_wall_seconds": regression[1]["wall_seconds"],
            "c2_wall_seconds": regression[2]["wall_seconds"],
            "speedup": regression[1]["wall_seconds"] / regression[2]["wall_seconds"],
        },
    }
    write_json(OUT / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=5)
    arguments = parser.parse_args()
    if arguments.worker:
        suite_arg, run_arg, concurrency_arg, fold_arg, epochs_arg = arguments.worker
        worker(
            suite_arg,
            int(run_arg),
            int(concurrency_arg),
            int(fold_arg),
            int(epochs_arg),
        )
    else:
        main()
