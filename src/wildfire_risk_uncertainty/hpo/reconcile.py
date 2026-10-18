"""Independent raw-metric, Optuna, and online W&B history reconciliation."""

import math
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

import optuna
import wandb

from wildfire_risk_uncertainty.hpo.records import read_json


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def _online_run(link: dict):
    parts = urlparse(link["run_url"]).path.strip("/").split("/")
    if len(parts) < 4 or parts[-2] != "runs" or parts[-1] != link["run_id"]:
        raise ValueError("Invalid W&B run URL/ID linkage")
    return wandb.Api(timeout=20).run(f"{parts[-4]}/{parts[-3]}/{parts[-1]}")


def _history(run, keys: list[str]) -> list[dict]:
    for attempt in range(3):
        rows = list(run.scan_history(keys=keys))
        if rows:
            return rows
        if attempt < 2:
            time.sleep(2)
    raise ValueError(f"Online W&B run lacks required history keys: {keys}")


def reconcile_trial(trial: optuna.trial.FrozenTrial, trial_dir: Path,
                    *, check_fold_history: bool = False) -> dict:
    link = read_json(trial_dir / "wandb_link.json")
    aggregate_files = sorted(trial_dir.glob("aggregate_epoch_*.json"))
    fold_checks = aggregate_checks = remote_checks = 0
    for path in sorted(trial_dir.glob("fold_*_epoch_*.json")):
        record = read_json(path)
        for role in ("training", "validation"):
            values = record[role]
            bp = values["bp_loss_sum"] / values["bp_count"]
            flp = values["flp_loss_sum"] / values["flp_valid_count"]
            if not (close(bp, values["bp_loss_mean"]) and
                    close(flp, values["flp_loss_mean"]) and
                    close(0.5 * (bp + flp), values["equal_task_objective"])):
                raise ValueError(f"Raw support reconciliation failed: {path}")
            fold_checks += 1
    for path in aggregate_files:
        value = read_json(path)
        epoch = value["resource_epoch"]
        folds = [read_json(trial_dir / f"fold_{fold}_epoch_{epoch:03d}.json")
                 for fold in range(1, 6)]
        recomputed = statistics.fmean(f["validation"]["equal_task_objective"] for f in folds)
        dispersion = statistics.pstdev(f["validation"]["equal_task_objective"] for f in folds)
        if not close(value["J_e"], recomputed) or not close(
                value["fold_objective_std_population"], dispersion):
            raise ValueError("Five-fold local J_e/dispersion reconciliation failed")
        if epoch in trial.intermediate_values and not close(trial.intermediate_values[epoch], recomputed):
            raise ValueError("Optuna intermediate differs from raw five-fold calculation")
        mirror = read_json(trial_dir / "wandb_mirror" / f"aggregate_epoch_{epoch:03d}.json")
        if not close(mirror["spatial_cv/validation_objective_mean"], recomputed):
            raise ValueError("W&B local mirror differs from raw five-fold calculation")
        aggregate_checks += 1
    run = _online_run(link)
    if run.state != "finished":
        raise ValueError("W&B online run was not finalized")
    rows = (_history(run, ["spatial_cv/resource_epoch", "spatial_cv/validation_objective_mean"])
            if aggregate_files else [])
    by_epoch = {int(r["spatial_cv/resource_epoch"]): r["spatial_cv/validation_objective_mean"]
                for r in rows}
    for path in aggregate_files:
        value = read_json(path)
        if value["resource_epoch"] not in by_epoch or not close(
                by_epoch[value["resource_epoch"]], value["J_e"]):
            raise ValueError("Online W&B spatial history differs from local J_e")
        remote_checks += 1
    if check_fold_history:
        for fold in range(1, 6):
            keys = [f"fold_{fold}/epoch", f"fold_{fold}/training/equal_task_objective",
                    f"fold_{fold}/validation/equal_task_objective"]
            rows = _history(run, keys)
            by_epoch = {int(r[keys[0]]): r for r in rows}
            for path in sorted(trial_dir.glob(f"fold_{fold}_epoch_*.json")):
                record = read_json(path)
                remote = by_epoch.get(record["epoch"])
                if remote is None or not all(close(remote[keys[i]], record[role]["equal_task_objective"])
                                             for i, role in ((1, "training"), (2, "validation"))):
                    raise ValueError("Online W&B fold epoch curve differs from local record")
                remote_checks += 1
    return {"trial_number": trial.number, "trial_state": trial.state.name,
            "wandb_run_id": link["run_id"], "wandb_run_url": link["run_url"],
            "raw_task_mean_checks": fold_checks, "five_fold_aggregate_checks": aggregate_checks,
            "online_wandb_history_checks": remote_checks,
            "serialization_tolerance": {"relative": 1e-12, "absolute": 1e-12}}
