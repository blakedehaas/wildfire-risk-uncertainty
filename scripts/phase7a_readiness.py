"""Reproduce the Phase 7A calibration readiness audit from persisted artifacts."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/benchmark/07a_detached/07a-20261020T193344Z-f965d452"
OUTPUT = ROOT / "reports/benchmark/07a/readiness.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit(run: Path) -> dict:
    config, status = read(run / "config.json"), read(run / "status.json")
    checks = {}
    checks["complete_lifecycle"] = (status["state"] == "complete" and
                                    status["error"] is None and
                                    set(status["completed_folds"]) == set(config["folds"]) and
                                    all(w["state"] == "complete" and w["exit_code"] == 0
                                        for w in status["workers"].values()))
    checks["frozen_configuration"] = all(config[key] == value for key, value in {
        "kind": "full_epoch", "outer_study": 3, "folds": [1, 2, 3, 4, 5],
        "epochs": 3, "processes": 2, "threads_each": 4,
        "depth": 3, "width": 128, "batch_size": 65536,
        "learning_rate": 0.001, "weight_decay": 0.000316227766,
        "precision": "BF16", "seed": 42}.items())
    epochs = []
    reference_ok = arithmetic_ok = checkpoints_ok = finite_ok = supports_ok = True
    memory_growth = []
    for fold in config["folds"]:
        fold_record = read(run / f"fold_{fold}.json")
        reference_ok &= read(run / f"fold_{fold}.reference_check.json")["equal"]
        reference = read(run.parent / config["reference_run"] / f"fold_{fold}.json")
        reference_keys = ("bp_loss", "flp_loss", "combined_loss", "bp_loss_sum",
                          "flp_loss_sum", "bp_rows", "flp_valid_rows", "optimizer_steps")
        rss = []
        for epoch in range(1, config["epochs"] + 1):
            record = read(run / f"fold_{fold}_epoch_{epoch:03d}.json")
            epochs.append(record)
            if epoch == 1:
                reference_ok &= (fold_record["state_sha256"] == reference["state_sha256"] and
                                 all(record["training"][key] == reference["training"][key]
                                     for key in reference_keys))
            for task in ("training", "validation"):
                metric = record[task]
                arithmetic_ok &= (math.isclose(metric["bp_loss"], metric["bp_loss_sum"] /
                                               metric["bp_rows"], abs_tol=1e-12) and
                                  math.isclose(metric["flp_loss"], metric["flp_loss_sum"] /
                                               metric["flp_valid_rows"], abs_tol=1e-12) and
                                  math.isclose(metric["combined_loss"],
                                               (metric["bp_loss"] + metric["flp_loss"]) / 2,
                                               abs_tol=1e-12))
                supports_ok &= (metric["bp_rows"] == fold_record["epochs"][0][task]["bp_rows"] and
                                metric["flp_valid_rows"] ==
                                fold_record["epochs"][0][task]["flp_valid_rows"])
                finite_ok &= all(math.isfinite(metric[key]) for key in
                                 ("bp_loss", "flp_loss", "combined_loss", "wall_seconds"))
            checkpoint = run / f"fold_{fold}_epoch_{epoch:03d}.pt"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoints_ok &= (digest == record["checkpoint_sha256"] ==
                               read(checkpoint.with_suffix(".pt.sha256.json"))["sha256"] and
                               checkpoint.stat().st_size == record["checkpoint_bytes"] and
                               record["global_step"] == epoch * record["training"]["optimizer_steps"] and
                               len(record["model_sha256"]) == 64 and
                               len(record["optimizer_sha256"]) == 64 and
                               record["checkpoint_load_seconds"] > 0)
            finite_ok &= record["all_finite"]
            rss.append(record["current_rss_bytes"])
        memory_growth.append(rss[-1] - rss[0])
        supports_ok &= len(fold_record["epochs"]) == config["epochs"] and not fold_record["stopped_early"]
    checks.update(reference_equivalence=reference_ok, task_arithmetic=arithmetic_ok,
                  constant_validation_and_training_support=supports_ok,
                  checkpoint_sha256_and_restore=checkpoints_ok,
                  finite_losses_and_gradients=finite_ok,
                  stable_worker_rss=max(memory_growth) < 256 * 1024**2)
    with (run / "resource_samples.csv").open(newline="") as handle:
        samples = list(csv.DictReader(handle))
    vram = [float(row["gpu_vram_used_mib"]) for row in samples if row["gpu_vram_used_mib"]]
    util = [float(row["gpu_utilization_percent"]) for row in samples
            if row["gpu_utilization_percent"]]
    swap = [int(row["swap_used_bytes"]) for row in samples]
    available = [int(row["available_ram_bytes"]) for row in samples]
    checks["stable_resources"] = (bool(samples) and max(vram) <= config["gpu_vram_stop_mib"] and
                                  min(available) >= config["min_available_ram_bytes"] and
                                  max(swap) - min(swap) <= config["max_swap_growth_bytes"])
    start = datetime.fromisoformat(status["started_at"])
    end = datetime.fromisoformat(status["ended_at"])
    wall = (end - start).total_seconds()
    training = [sum(r["training"]["wall_seconds"] for r in epochs if r["epoch"] == epoch)
                for epoch in range(1, 4)]
    validation = [sum(r["validation"]["wall_seconds"] for r in epochs if r["epoch"] == epoch)
                  for epoch in range(1, 4)]
    warm = sum(training[1:] + validation[1:]) / 2
    # Five fold workers are scheduled two at a time. The observed run wall time,
    # divided by three, captures that schedule and startup/cache costs.
    cv_100_seconds = wall / 3 * 100
    return {"calibration_experiment_id": run.name, "source_git_sha": config["source_git_sha"],
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checks": checks, "ready": all(checks.values()), "fold_epochs": len(epochs),
            "resource_samples": len(samples), "wall_seconds": wall,
            "five_fold_epoch_equivalent_seconds": wall / 3,
            "training_seconds_by_epoch_sum": training,
            "validation_seconds_by_epoch_sum": validation,
            "warm_train_plus_validation_worker_seconds_per_five_folds": warm,
            "checkpoint_save_seconds_total": sum(r["checkpoint_save_seconds"] for r in epochs),
            "checkpoint_load_seconds_total": sum(r["checkpoint_load_seconds"] for r in epochs),
            "checkpoint_bytes_total": sum(r["checkpoint_bytes"] for r in epochs),
            "peak_sampled_vram_mib": max(vram),
            "median_sampled_gpu_utilization_percent": statistics.median(util),
            "minimum_available_ram_bytes": min(available),
            "swap_growth_bytes": max(swap) - min(swap),
            "maximum_worker_rss_growth_bytes": max(memory_growth),
            "projected_cv_100_epochs_hours": cv_100_seconds / 3600,
            "projected_one_model_100_epochs_minutes_range": [
                min(sum(r[task]["wall_seconds"] for task in ("training", "validation"))
                    for r in epochs if r["epoch"] > 1) * 100 / 60,
                max(sum(r[task]["wall_seconds"] for task in ("training", "validation"))
                    for r in epochs if r["epoch"] > 1) * 100 / 60],
            "projected_six_weight_five_outer_hours": cv_100_seconds * 30 / 3600,
            "projection_assumption": "Outer 3 two-worker, three-epoch mean wall time repeats across 100 epochs and all outer TRAIN studies; architecture, cache, and hardware remain fixed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
