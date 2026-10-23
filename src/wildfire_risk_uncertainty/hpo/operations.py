"""Read-only study accounting and explicit lifetime-budget extension."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import sampler
from wildfire_risk_uncertainty.hpo.records import read_json
from wildfire_risk_uncertainty.hpo.status import status
from wildfire_risk_uncertainty.hpo.storage import (
    BP_CE_NUMERICAL_REVISION,
    _registered_revision_matches,
    extend_trial_limit,
    metadata,
    restore_sampler_state,
    study_controller_lock,
    study_name,
)
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes


def _database(root: Path, outer_study: int) -> Path:
    return root / "artifacts/hpo" / f"outer_{outer_study}_gradnorm_v3.sqlite3"


def _study(root: Path, outer_study: int) -> optuna.Study:
    database = _database(root, outer_study)
    if not database.exists():
        raise ValueError("GradNorm HPO study has not started")
    return optuna.load_study(study_name=study_name(outer_study, gradnorm=True),
                             storage=f"sqlite:///{database.resolve()}", sampler=sampler(),
                             pruner=optuna.pruners.NopPruner())


def _active_detached_controller(root: Path, outer_study: int) -> bool:
    for path in (root / "artifacts/benchmark/07a_detached").glob("07a-*/config.json"):
        config = read_json(path)
        if config.get("kind") != "gradnorm_hpo" or config.get("outer_study") != outer_study:
            continue
        state_path = path.parent / "status.json"
        if not state_path.exists():
            continue
        record = read_json(state_path)
        if record.get("state") not in {"starting", "running", "stopping"}:
            continue
        pid = record.get("supervisor_pid")
        if not pid:
            return True  # Unresolved startup must be inspected before extension.
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        except OSError:
            continue
        if fields[0] != "Z" and fields[19] == record.get("supervisor_pid_start"):
            return True
    return False


def _resource_summary(root: Path, outer_study: int) -> dict:
    runs = []
    for path in (root / "artifacts/benchmark/07a_detached").glob("07a-*/config.json"):
        config = read_json(path)
        if config.get("kind") == "gradnorm_hpo" and config.get("outer_study") == outer_study:
            status_path = path.parent / "status.json"
            if status_path.exists():
                runs.append(read_json(status_path))
    available = [run["minimum_sampled_available_ram_bytes"] for run in runs
                 if run.get("minimum_sampled_available_ram_bytes") is not None]
    return {"controller_attempts": len(runs),
            "peak_device_vram_mib": max((run.get("peak_sampled_device_vram_mib", 0)
                                         for run in runs), default=0),
            "peak_process_tree_rss_bytes": max((run.get("peak_sampled_worker_rss_bytes", 0)
                                                for run in runs), default=0),
            "minimum_available_ram_bytes": min(available) if available else None,
            "maximum_swap_growth_bytes": max((run.get("maximum_sampled_swap_growth_bytes", 0)
                                              for run in runs), default=0),
            "resource_safety_events": [run["resource_safety_event"] for run in runs
                                       if "resource_safety_event" in run]}


def extend(root: Path, outer_study: int, new_limit: int) -> dict:
    if not 11 <= new_limit <= 150:
        raise ValueError("Lifetime trial cap must be 11..150")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise RuntimeError("Extension requires a clean committed worktree")
    database = _database(root, outer_study)
    with study_controller_lock(database):
        if _active_detached_controller(root, outer_study):
            raise RuntimeError("A detached controller is active for this study")
        study = _study(root, outer_study)
        frozen = study.user_attrs.get("frozen_metadata")
        current = metadata(root, outer_study, kind="gradnorm_hpo")
        if (not isinstance(frozen, dict) or
                (frozen != current and not _registered_revision_matches(study, current))):
            raise ValueError("Current source/environment differs from frozen study methodology")
        cache = HpoCache(root, root / "artifacts/training/hpo_cache_v1")
        if frozen["cache_manifest_sha256"] != hashlib.sha256(
            stable_json_bytes(cache.manifest)).hexdigest():
            raise ValueError("Frozen HPO cache identity differs")
        states = load_states(root, cache, outer_study)
        if frozen["preprocessing_hashes_by_fold"] != {str(k): value.digest()
                                                       for k, value in states.items()}:
            raise ValueError("Frozen fold preprocessing identity differs")
        restore_sampler_state(study)
        unreviewed = [trial.number for trial in study.trials
                      if trial.state == optuna.trial.TrialState.FAIL and
                      not trial.user_attrs.get("failure", {}).get("numerical_failure") and
                      not (trial.number == 1 and study.user_attrs.get("numerical_revision"))]
        if unreviewed:
            raise ValueError(f"Failed trials require inspection before extension: {unreviewed}")
        return extend_trial_limit(study, new_limit)


def adopt_bp_ce_fix(root: Path, outer_study: int) -> dict:
    """Register a reviewed source revision; never change finished trial state."""
    if outer_study != 3:
        raise ValueError("This numerical recovery is specific to outer study 3")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise RuntimeError("Revision adoption requires a clean committed worktree")
    database = _database(root, outer_study)
    with study_controller_lock(database):
        if _active_detached_controller(root, outer_study):
            raise RuntimeError("A detached controller is active for this study")
        study = _study(root, outer_study)
        frozen = study.user_attrs["frozen_metadata"]
        current = metadata(root, outer_study, kind="gradnorm_hpo")
        if current["bp_ce_numerical_revision"] != BP_CE_NUMERICAL_REVISION:
            raise ValueError("Unexpected numerical implementation revision")
        ignored = {"source_git_sha", "bp_ce_numerical_revision"}
        if any(frozen.get(key) != value for key, value in current.items() if key not in ignored):
            raise ValueError("Frozen scientific/environment provenance differs")
        if [trial.state.name for trial in study.trials[:2]] != ["COMPLETE", "FAIL"]:
            raise ValueError("Expected completed trial 0 and failed trial 1")
        original = root / "reports/hpo/07b/trial1_replay_fold1.json"
        corrected = root / "reports/hpo/07b/trial1_replay_fold1_stable_ce.json"
        before, after = read_json(original), read_json(corrected)
        checkpoint = root / "artifacts/hpo/records" / study.study_name / "trial_001" / \
            "checkpoints/fold_1_epoch_075.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (before["original_source_git_sha"] != frozen["source_git_sha"] or
                before["checkpoint_sha256"] != digest or after["checkpoint_sha256"] != digest or
                before["outcome"] != "numerical_exception" or
                "bp_rate_ce=0.0" not in before["exception"] or
                after["outcome"] != "epoch_completed"):
            raise ValueError("Reviewed replay evidence does not match persisted trial 1")
        existing = study.user_attrs.get("numerical_revision")
        revision = {"source_git_sha": current["source_git_sha"],
                    "original_source_git_sha": frozen["source_git_sha"],
                    "marker": BP_CE_NUMERICAL_REVISION,
                    "failure_replay_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                    "corrected_replay_sha256": hashlib.sha256(corrected.read_bytes()).hexdigest(),
                    "trial_1_state": "FAIL", "trial_1_checkpoint_epoch": 75}
        if existing is not None and existing != revision:
            raise ValueError("Different numerical revision is already registered")
        if existing is None:
            study.set_user_attr("numerical_revision", revision)
        return {"study": study.study_name, "registered_revision": revision,
                "trial_0_state": "COMPLETE", "trial_1_state": "FAIL",
                "trial_limit": study.user_attrs["trial_budget"]["current"]}


def trial_rows(root: Path, outer_study: int) -> list[dict]:
    study = _study(root, outer_study)
    records_root = root / "artifacts/hpo/records" / study.study_name
    rows = []
    for trial in study.trials:
        local = records_root / f"trial_{trial.number:03d}"
        epochs = sorted(local.glob("fold_*_epoch_*.json"))
        fold_rows = [read_json(path) for path in epochs]
        final = local / "aggregate_epoch_100.json"
        aggregate = read_json(final) if final.exists() else {}
        rows.append({"study": study.study_name, "trial_number": trial.number,
                     "state": trial.state.name, "value": trial.value,
                     "started_at": trial.datetime_start.isoformat() if trial.datetime_start else "",
                     "ended_at": trial.datetime_complete.isoformat() if trial.datetime_complete else "",
                     "wall_seconds": ((trial.datetime_complete - trial.datetime_start).total_seconds()
                                      if trial.datetime_start and trial.datetime_complete else ""),
                     **{key: trial.params.get(key, "") for key in
                        ("depth", "width", "learning_rate", "weight_decay", "batch_size")},
                     "completed_fold_epochs": len(fold_rows),
                     "completed_fits": sum(row["epoch"] == 100 for row in fold_rows),
                     "train_seconds": sum(row["training_wall_seconds"] for row in fold_rows),
                     "validation_seconds": sum(row["validation_wall_seconds"] for row in fold_rows),
                     "checkpoint_seconds": sum(row.get("checkpoint_wall_seconds", 0)
                                               for row in fold_rows),
                     "peak_process_rss_bytes": max((row["peak_process_rss_bytes"]
                                                    for row in fold_rows), default=0),
                     "peak_allocated_vram_bytes": max((row["peak_allocated_vram_bytes"]
                                                      for row in fold_rows), default=0),
                     "failure": json.dumps(trial.user_attrs.get("failure", {})),
                     "fold_objectives": json.dumps(aggregate.get("fold_validation_objectives", [])),
                     "fold_objective_std": aggregate.get("fold_objective_std_population", ""),
                     "local_record": str(local)})
    return rows


def summary(root: Path, outer_study: int) -> dict:
    progress = status(root, outer_study, gradnorm=True)
    if progress["state"] == "not_started":
        return progress
    rows = trial_rows(root, outer_study)
    completed = [row for row in rows if row["state"] == "COMPLETE"]
    train = sum(row["train_seconds"] for row in rows)
    validation = sum(row["validation_seconds"] for row in rows)
    checkpoints = sum(row["checkpoint_seconds"] for row in rows)
    fits = sum(row["completed_fits"] for row in rows)
    finalized_fits = sum(row["completed_fits"] for row in completed)
    controller_active = _active_detached_controller(root, outer_study)
    complete_fit_seconds = sum(row["train_seconds"] + row["validation_seconds"]
                               for row in completed)
    return {**progress, "best_objective": max((row["value"] for row in completed), default=None),
            "completed_fits": fits, "training_seconds": train,
            "validation_seconds": validation, "checkpoint_seconds": checkpoints,
            "measured_compute_seconds": train + validation + checkpoints,
            "mean_measured_seconds_per_completed_fit": (complete_fit_seconds / finalized_fits
                                                        if finalized_fits else None),
            "unfinalized_trial_compute_seconds": train + validation - complete_fit_seconds,
            "completed_trial_wall_seconds": sum(row["wall_seconds"] for row in completed),
            "mean_wall_seconds_per_completed_trial": (sum(row["wall_seconds"] for row in completed)
                                                       / len(completed) if completed else None),
            "remaining_trial_slots": max(0, progress["trial_limit"] - len(rows)),
            "failed_trials": [row["trial_number"] for row in rows if row["state"] == "FAIL"],
            "resource": _resource_summary(root, outer_study),
            "controller_active": controller_active,
            "interrupted_trial_numbers": [row["trial_number"] for row in rows
                                          if row["state"] == "RUNNING" and not controller_active]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "results", "export", "extend", "adopt-bp-ce-fix"):
        item = sub.add_parser(command)
        item.add_argument("--outer-study", type=int, choices=range(1, 6), required=True)
        if command == "export":
            item.add_argument("--output", type=Path, required=True)
        if command == "extend":
            item.add_argument("--new-limit", type=int, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "status":
        result = summary(root, args.outer_study)
    elif args.command == "extend":
        result = extend(root, args.outer_study, args.new_limit)
    elif args.command == "adopt-bp-ce-fix":
        result = adopt_bp_ce_fix(root, args.outer_study)
    elif args.command == "results":
        result = trial_rows(root, args.outer_study)
    else:
        rows = trial_rows(root, args.outer_study)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                    ["study", "trial_number", "state", "value"])
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(args.output)
        result = {"output": str(args.output), "trials": len(rows)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
