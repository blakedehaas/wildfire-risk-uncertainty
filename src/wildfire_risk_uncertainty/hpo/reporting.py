"""Reconstruct compact 6E reports from authoritative local trial records."""

import csv
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.records import read_json, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def real_smoke_tables(root: Path) -> dict:
    report = root / "reports/hpo/06e"
    local = root / "artifacts/hpo/06e/records/checkpoint_6e_outer_3_fixed_real_smoke_v1/trial_000"
    rows = []
    for fold in range(1, 6):
        record = read_json(local / f"fold_{fold}_epoch_001.json")
        row = {"outer_study": 3, "trial_number": 0, "inner_fold": fold, "epoch": 1,
               "global_optimizer_step": record["global_optimizer_step"]}
        for role in ("training", "validation"):
            metrics = record[role]
            prefix = "train" if role == "training" else "validation"
            row.update({f"{prefix}_{key}": metrics[key] for key in (
                "bp_loss_sum", "bp_count", "flp_loss_sum", "flp_valid_count",
                "bp_loss_mean", "flp_loss_mean", "equal_task_objective")})
            row[f"{prefix}_wall_seconds"] = record[f"{role}_wall_seconds"]
            row[f"{prefix}_rows_per_second"] = record[f"{role}_rows_per_second"]
        row.update({key: record[key] for key in (
            "optimizer_steps", "peak_allocated_vram_bytes", "peak_reserved_vram_bytes",
            "peak_process_rss_bytes", "all_finite")})
        rows.append(row)
    write_csv(report / "real_smoke_fold_epoch1.csv", rows)
    summary = read_json(report / "real_smoke_trial_summary.json")
    summary["per_fold_epoch1"] = rows
    write_json(report / "real_smoke_trial_summary.json", summary)

    caches = []
    for fold in range(1, 6):
        state = read_json(root / "artifacts/hpo/preprocessing" / f"outer_3_inner_{fold}.json")
        caches.append({"inner_fold": fold, "state_sha256": state["state_sha256"],
                       "rows": state["state"]["rows"],
                       "flp_valid_rows": state["state"]["flp_valid_rows"],
                       "refit_identical": state["refit_identical"],
                       "scan_seconds": state["scan_seconds"]})
    write_json(report / "preprocessing_cache_validation.json", {"folds": caches,
        "all_repeated_inner_train_fits_identical": all(x["refit_identical"] for x in caches)})

    with (root / "reports/training/06d/epoch_metrics.csv").open() as handle:
        earlier = list(csv.DictReader(handle))
    checks = {}
    for kind in ("training", "validation"):
        prior = next(r for r in earlier if r["run"] == "reference" and
                     r["stage"] == "epoch_1" and r["kind"] == kind)
        for component, prior_key in (("bp_loss_mean", "bp_loss"),
                                     ("flp_loss_mean", "flp_loss"),
                                     ("equal_task_objective", "combined_loss")):
            current = rows[0][f"{'train' if kind == 'training' else 'validation'}_{component}"]
            checks[f"{kind}_{component}"] = {"6d_value": float(prior[prior_key]),
                                              "6e_value": current,
                                              "absolute_difference": abs(current - float(prior[prior_key]))}
    write_json(report / "six_d_fold1_replay.json", {
        "same_fixed_configuration_and_fold": True, "component_checks": checks,
        "all_component_losses_bitwise_equal": all(x["absolute_difference"] == 0 for x in checks.values())})

    study = optuna.load_study(study_name="checkpoint_6e_outer_3_fixed_real_smoke_v1",
                              storage=f"sqlite:///{(root / 'artifacts/hpo/06e/fixed_real_smoke.sqlite3').resolve()}")
    trial = study.trials[0]
    trial_seconds = (trial.datetime_complete - trial.datetime_start).total_seconds()
    train_valid_seconds = sum(r["train_wall_seconds"] + r["validation_wall_seconds"] for r in rows)
    result = {"optuna_trial_wall_seconds": trial_seconds,
              "summed_training_validation_wall_seconds": train_valid_seconds,
              "other_trial_wall_seconds": trial_seconds - train_valid_seconds,
              "interpretation": "Combined Optuna, local/W&B tracking, model setup, and other orchestration time; W&B-only overhead was not isolated.",
              "preprocessing_scan_seconds_before_trial": sum(sum(x["scan_seconds"]) for x in caches)}
    write_json(report / "tracking_overhead.json", result)
    return result
