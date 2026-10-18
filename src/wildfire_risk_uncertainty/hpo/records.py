"""Atomic authoritative local records and raw-support objective arithmetic."""

import json
import os
import statistics
from pathlib import Path

from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(stable_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def task_metrics(raw: dict) -> dict:
    bp_count = int(raw["bp_rows"])
    flp_count = int(raw["flp_valid_rows"])
    if bp_count <= 0 or flp_count <= 0:
        raise ValueError("Both task supports must be positive")
    bp_mean = float(raw["bp_loss_sum"]) / bp_count
    flp_mean = float(raw["flp_loss_sum"]) / flp_count
    objective = 0.5 * (bp_mean + flp_mean)
    if abs(bp_mean - raw["bp_loss"]) > 1e-12 or abs(flp_mean - raw["flp_loss"]) > 1e-12:
        raise ValueError("Raw sums/counts disagree with task means")
    if abs(objective - raw["combined_loss"]) > 1e-12:
        raise ValueError("Raw task means disagree with objective")
    return {"bp_loss_sum": raw["bp_loss_sum"], "bp_count": bp_count,
            "flp_loss_sum": raw["flp_loss_sum"], "flp_valid_count": flp_count,
            "bp_loss_mean": bp_mean, "flp_loss_mean": flp_mean,
            "equal_task_objective": objective}


def fold_epoch_record(*, outer_study: int, trial_number: int, inner_fold: int,
                      epoch: int, configuration: dict, training: dict, validation: dict,
                      global_step: int, milestone: bool) -> dict:
    train = task_metrics(training)
    valid = task_metrics(validation)
    return {"outer_study": outer_study, "trial_number": trial_number,
            "inner_fold": inner_fold, "epoch": epoch, "resource_milestone": milestone,
            "global_optimizer_step": global_step, "train_rows": training["bp_rows"],
            "validation_rows": validation["bp_rows"], "training": train,
            "validation": valid, "configuration": configuration,
            "precision_mode": "BF16 autocast", "activation": "ReLU", "optimizer": "AdamW",
            "dropout": 0, "training_wall_seconds": training["wall_seconds"],
            "validation_wall_seconds": validation["wall_seconds"],
            "training_rows_per_second": training["rows_per_second"],
            "validation_rows_per_second": validation["rows_per_second"],
            "optimizer_steps": training["optimizer_steps"], "all_finite": True,
            "peak_allocated_vram_bytes": max(training["peak_vram_bytes"], validation["peak_vram_bytes"]),
            "peak_reserved_vram_bytes": max(training["peak_reserved_vram_bytes"],
                                            validation["peak_reserved_vram_bytes"]),
            "peak_process_rss_bytes": max(training["peak_rss_bytes"], validation["peak_rss_bytes"])}


def aggregate(epoch: int, records: list[dict]) -> dict:
    if len(records) != 5 or sorted(r["inner_fold"] for r in records) != [1, 2, 3, 4, 5]:
        raise ValueError("Exactly five distinct inner folds must reach a common epoch")
    if any(r["epoch"] != epoch for r in records):
        raise ValueError("Spatial aggregation requires a common resource level")
    values = [r["validation"]["equal_task_objective"] for r in records]
    return {"resource_epoch": epoch,
            "fold_validation_bp_losses": [r["validation"]["bp_loss_mean"] for r in records],
            "fold_validation_flp_losses": [r["validation"]["flp_loss_mean"] for r in records],
            "fold_validation_objectives": values,
            "validation_bp_mean": statistics.fmean(r["validation"]["bp_loss_mean"] for r in records),
            "validation_flp_mean": statistics.fmean(r["validation"]["flp_loss_mean"] for r in records),
            "J_e": statistics.fmean(values), "fold_objective_std_population": statistics.pstdev(values),
            "fold_objective_min": min(values), "fold_objective_max": max(values),
            "training_objective_mean": statistics.fmean(
                r["training"]["equal_task_objective"] for r in records),
            "training_bp_mean": statistics.fmean(r["training"]["bp_loss_mean"] for r in records),
            "training_flp_mean": statistics.fmean(r["training"]["flp_loss_mean"] for r in records)}


class TrialRecord:
    def __init__(self, root: Path, study_name: str, trial_number: int) -> None:
        self.path = root / study_name / f"trial_{trial_number:03d}"
        self.path.mkdir(parents=True, exist_ok=True)

    def manifest(self, value: dict) -> None:
        path = self.path / "manifest.json"
        if path.exists() and read_json(path) != value:
            raise ValueError("Existing trial manifest differs; refusing overwrite")
        write_json(path, value)

    def fold(self, record: dict) -> None:
        path = self.path / f"fold_{record['inner_fold']}_epoch_{record['epoch']:03d}.json"
        if path.exists():
            raise ValueError("Fold/epoch record already exists")
        write_json(path, record)

    def spatial(self, record: dict) -> None:
        path = self.path / f"aggregate_epoch_{record['resource_epoch']:03d}.json"
        if path.exists():
            raise ValueError("Aggregate resource record already exists")
        write_json(path, record)

    def state(self, value: dict) -> None:
        write_json(self.path / "state.json", value)
