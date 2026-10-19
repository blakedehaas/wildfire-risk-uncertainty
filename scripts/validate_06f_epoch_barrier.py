"""Four-epoch c1/c2 regression with observed common-epoch barriers."""

import hashlib
import json
import math
import multiprocessing as mp
import queue
import subprocess
import time

from scripts.validate_06f_concurrency2 import (
    CACHE_PATH,
    METRIC_KEYS,
    OUT,
    ROOT,
    VRAM_STOP_MIB,
    gpu_status,
    memory_status,
    state_digest,
    warm_cache,
    write_json,
)
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

CONFIG = Configuration(3, 64, 0.00020511104188433984, 1.493656855461762e-05, 32768)


def pool_worker(folds: tuple[int, ...], commands: mp.Queue, results: mp.Queue) -> None:
    try:
        cache = HpoCache(ROOT, CACHE_PATH, check_files=False)
        states = load_states(ROOT, cache, 1)
        counts = support_counts(ROOT, 1)
        models = {}
        for fold in folds:
            seed_everything(42)
            models[fold] = make_model(
                len(states[fold].inputs),
                CONFIG.depth,
                CONFIG.width,
                CONFIG.learning_rate,
                CONFIG.weight_decay,
            )
        results.put({"status": "ready", "folds": folds})
        while True:
            epoch = commands.get()
            if epoch is None:
                return
            records = []
            for fold in folds:
                state = states[fold]
                model, optimizer = models[fold]
                started = time.perf_counter()
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
                    counts[fold]["validation"]["rows"],
                    counts[fold]["validation"]["flp_valid"],
                    outer_study=1,
                    inner_fold=fold,
                    batch_size=CONFIG.batch_size,
                    cache=cache,
                )
                records.append(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "training": training,
                        "validation": validation,
                        "model_sha256": state_digest(model.state_dict()),
                        "optimizer_sha256": state_digest(optimizer.state_dict()),
                        "preprocessing_sha256": state.digest(),
                        "fold_wall_seconds": time.perf_counter() - started,
                    }
                )
            results.put({"status": "complete", "epoch": epoch, "records": records})
    except BaseException as error:
        results.put(
            {"status": "error", "type": type(error).__name__, "message": str(error)}
        )
        raise


def compare(candidate: dict, reference: dict) -> list[str]:
    mismatches = []
    for role in ("training", "validation"):
        for key in METRIC_KEYS:
            if candidate[role][key] != reference[role][key]:
                mismatches.append(f"{role}.{key}")
    for key in ("model_sha256", "optimizer_sha256", "preprocessing_sha256"):
        if candidate[key] != reference[key]:
            mismatches.append(key)
    return mismatches


def assignments(concurrency: int) -> tuple[tuple[int, ...], ...]:
    return ((1, 2, 3, 4, 5),) if concurrency == 1 else ((1, 3, 5), (2, 4))


def run(concurrency: int) -> dict:
    path = OUT / f"barrier_c{concurrency}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if (
            existing.get("concurrency") != concurrency
            or existing.get("configuration") != CONFIG.serializable()
            or len(existing.get("epochs", [])) != 4
        ):
            raise RuntimeError(f"Existing result is incomplete or incompatible: {path}")
        return existing
    warm_seconds = warm_cache()
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    command_queues = []
    processes = []
    total_started = time.monotonic()
    for folds in assignments(concurrency):
        commands = context.Queue()
        process = context.Process(
            target=pool_worker, args=(folds, commands, result_queue)
        )
        process.start()
        command_queues.append(commands)
        processes.append(process)
    ready = 0
    while ready < concurrency:
        message = result_queue.get(timeout=180)
        if message["status"] != "ready":
            raise RuntimeError(f"Worker initialization failed: {message}")
        ready += 1
    swap_start = memory_status()[1]
    epoch_rows = []
    all_records = {}
    samples = []
    for epoch in range(1, 5):
        epoch_started = time.monotonic()
        for commands in command_queues:
            commands.put(epoch)
        completed = 0
        records = []
        while completed < concurrency:
            try:
                message = result_queue.get(timeout=2)
            except queue.Empty:
                available, swap = memory_status()
                utilization, power, vram = gpu_status()
                samples.append(
                    {
                        "epoch": epoch,
                        "available_ram_bytes": available,
                        "swap_used_bytes": swap,
                        "gpu_util_percent": utilization,
                        "gpu_power_w": power,
                        "gpu_vram_mib": vram,
                    }
                )
                if swap - swap_start > 128 * 1024**2 or available < 2 * 1024**3:
                    raise RuntimeError("swap or RAM pressure")
                if vram > VRAM_STOP_MIB:
                    raise RuntimeError("VRAM pressure")
                if time.monotonic() - epoch_started > 600:
                    raise RuntimeError("possible deadlock")
                continue
            if message["status"] == "error":
                raise RuntimeError(f"Worker failed: {message}")
            if message["status"] != "complete" or message["epoch"] != epoch:
                raise RuntimeError(f"Unexpected worker message: {message}")
            records.extend(message["records"])
            completed += 1
        epoch_wall = time.monotonic() - epoch_started
        records.sort(key=lambda record: record["fold"])
        if [record["fold"] for record in records] != [1, 2, 3, 4, 5]:
            raise RuntimeError("Epoch does not contain all five folds")
        for record in records:
            fold = record["fold"]
            expected = support_counts(ROOT, 1)[fold]
            for role, support_role in (
                ("training", "train"),
                ("validation", "validation"),
            ):
                metrics = record[role]
                if metrics["bp_rows"] != expected[support_role]["rows"]:
                    raise RuntimeError("row mismatch")
                if metrics["flp_valid_rows"] != expected[support_role]["flp_valid"]:
                    raise RuntimeError("support mismatch")
                if metrics["optimizer_steps"] != math.ceil(
                    metrics["bp_rows"] / CONFIG.batch_size
                ):
                    raise RuntimeError("batch-count mismatch")
        all_records[epoch] = records
        epoch_rows.append({"epoch": epoch, "wall_seconds": epoch_wall})
        print(f"barrier c{concurrency} epoch {epoch}: {epoch_wall:.3f}s", flush=True)
    for commands in command_queues:
        commands.put(None)
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"Worker exited {process.exitcode}")
    result = {
        "concurrency": concurrency,
        "configuration": CONFIG.serializable(),
        "epochs": epoch_rows,
        "records_by_epoch": all_records,
        "total_wall_seconds": time.monotonic() - total_started,
        "warm_cache_seconds_excluded": warm_seconds,
        "peak_gpu_vram_mib": max(sample["gpu_vram_mib"] for sample in samples),
        "mean_gpu_util_percent": sum(sample["gpu_util_percent"] for sample in samples)
        / len(samples),
        "min_available_ram_bytes": min(
            sample["available_ram_bytes"] for sample in samples
        ),
        "peak_swap_used_bytes": max(sample["swap_used_bytes"] for sample in samples),
        "resource_samples": len(samples),
    }
    write_json(path, result)
    return result


def main() -> None:
    metadata = {
        "source_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "outer_study": 1,
        "trial": 0,
        "configuration": CONFIG.serializable(),
        "cache_manifest_sha256": hashlib.sha256(
            stable_json_bytes(HpoCache(ROOT, CACHE_PATH, check_files=False).manifest)
        ).hexdigest(),
        "epoch_barriers": True,
        "persistent_worker_processes": True,
        "c2_fold_assignment": [[1, 3, 5], [2, 4]],
    }
    write_json(OUT / "barrier_metadata.json", metadata)
    reference = run(1)
    candidate = run(2)
    for epoch in range(1, 5):
        for left, right in zip(
            candidate["records_by_epoch"][str(epoch)],
            reference["records_by_epoch"][str(epoch)],
            strict=True,
        ):
            mismatch = compare(left, right)
            if mismatch:
                write_json(
                    OUT / "barrier_STOP.json",
                    {"epoch": epoch, "fold": right["fold"], "mismatches": mismatch},
                )
                raise RuntimeError(
                    f"Epoch {epoch} fold {right['fold']} mismatch: {mismatch}"
                )
    result = {
        "status": "PASS",
        "exact_equivalence": True,
        "per_epoch": [
            {
                "epoch": epoch,
                "c1_wall_seconds": reference["epochs"][epoch - 1]["wall_seconds"],
                "c2_wall_seconds": candidate["epochs"][epoch - 1]["wall_seconds"],
                "speedup": reference["epochs"][epoch - 1]["wall_seconds"]
                / candidate["epochs"][epoch - 1]["wall_seconds"],
            }
            for epoch in range(1, 5)
        ],
        "total": {
            "c1_wall_seconds": reference["total_wall_seconds"],
            "c2_wall_seconds": candidate["total_wall_seconds"],
            "speedup": reference["total_wall_seconds"]
            / candidate["total_wall_seconds"],
        },
    }
    write_json(OUT / "barrier_result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
