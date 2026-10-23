"""Epoch-boundary continuation of one RUNNING Optuna trial without replacement."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import optuna

from wildfire_risk_uncertainty.hpo.config import MILESTONES, Configuration
from wildfire_risk_uncertainty.hpo.records import (
    TrialRecord,
    aggregate,
    fold_epoch_record,
    read_json,
    write_json,
)
from wildfire_risk_uncertainty.hpo.wandb_tracking import DisabledTracker


class ResumableProvider(Protocol):
    def __call__(self, fold: int, epoch: int) -> tuple[dict, dict]: ...

    def restore(self, fold: int, completed_epoch: int, global_step: int) -> None: ...

    def save(self, fold: int, epoch: int, global_step: int) -> None: ...

    def cleanup(self, fold: int, epoch: int) -> None: ...


def _existing_folds(local: TrialRecord, max_epochs: int) -> dict[int, int]:
    completed = {}
    for fold in range(1, 6):
        epochs = sorted(int(path.stem.split("_")[-1]) for path in
                        local.path.glob(f"fold_{fold}_epoch_*.json"))
        if epochs != list(range(1, len(epochs) + 1)) or len(epochs) > max_epochs:
            raise ValueError("Persisted fold epochs are not a contiguous valid prefix")
        completed[fold] = len(epochs)
    if max(completed.values()) - min(completed.values()) > 1:
        raise ValueError("Persisted folds differ by more than one unfinished common epoch")
    return completed


def _summary(state: str, local: TrialRecord, trial: optuna.Trial, started: float,
             last_epoch: int, tracker: DisabledTracker) -> dict:
    records = [read_json(path) for path in sorted(local.path.glob("fold_*_epoch_*.json"))]
    common = [read_json(path) for path in sorted(local.path.glob("aggregate_epoch_*.json"))]
    ranked = common and common[0].get("objective_direction") == "maximize"
    return {"trial_state": state, "final_resource_epoch": last_epoch,
            "objective_at_last_common_resource": common[-1]["J_e"] if common else None,
            "best_observed_common_resource_objective": (
                max((x["J_e"] for x in common), default=None) if ranked else
                min((x["J_e"] for x in common), default=None)),
            "total_runtime_seconds": time.perf_counter() - started,
            "total_training_rows_processed": sum(r["train_rows"] for r in records),
            "total_validation_rows_processed": sum(r["validation_rows"] for r in records),
            "prune_epoch": last_epoch if state == "PRUNED" else None,
            "local_record_path": str(local.path), "wandb_run_id": tracker.run_id,
            "wandb_sync_state": tracker.sync_state, "trial_number": trial.number}


def run_resumable_trial(*, trial: optuna.Trial, outer_study: int,
                        configuration: Configuration, provider: ResumableProvider,
                        record_root: Path, study_name: str, provenance: dict,
                        tracker: DisabledTracker, max_epochs: int = 100,
                        pause_after_epoch: int | None = None,
                        pause_requested: Callable[[], bool] | None = None,
                        references: dict[int, dict] | None = None) -> float | None:
    """Return None only at a complete common-epoch pause; Optuna stays RUNNING."""
    local = TrialRecord(record_root, study_name, trial.number)
    config = configuration.serializable()
    saved_config = trial.user_attrs.get("suggested_configuration")
    if saved_config is not None and saved_config != config:
        raise ValueError("Persisted Optuna configuration differs from resumed trial")
    trial.set_user_attr("suggested_configuration", config)
    local.manifest({"study_name": study_name, "outer_study": outer_study,
                    "trial_number": trial.number, "configuration": config,
                    "provenance": provenance, "max_epochs": max_epochs})
    link = local.path / "wandb_link.json"
    if link.exists():
        if read_json(link)["run_id"] != tracker.run_id:
            raise ValueError("Resumed W&B run identity differs from local trial record")
    else:
        write_json(link, {"run_id": tracker.run_id, "run_url": tracker.run_url})
    completed = _existing_folds(local, max_epochs)
    steps = {}
    for fold in range(1, 6):
        last = completed[fold]
        step = read_json(local.path / f"fold_{fold}_epoch_{last:03d}.json")["global_optimizer_step"] if last else 0
        steps[fold] = step
    concurrent = all(hasattr(provider, name) for name in
                     ("restore_many", "run_epoch", "save_many", "cleanup_many"))
    if concurrent:
        provider.restore_many(completed, steps)
    else:
        for fold in range(1, 6):
            provider.restore(fold, completed[fold], steps[fold])
    started = time.perf_counter()
    last_common = 0
    for epoch in range(1, max_epochs + 1):
        folds = []
        if concurrent:
            missing = [fold for fold in range(1, 6) if epoch > completed[fold]]
            results = provider.run_epoch(epoch, missing) if missing else {}
            new_records = {}
            for fold in range(1, 6):
                if epoch <= completed[fold]:
                    record = read_json(local.path / f"fold_{fold}_epoch_{epoch:03d}.json")
                else:
                    training, validation = results[fold]
                    steps[fold] += training["optimizer_steps"]
                    record = fold_epoch_record(
                        outer_study=outer_study, trial_number=trial.number, inner_fold=fold,
                        epoch=epoch, configuration=config, training=training,
                        validation=validation, global_step=steps[fold],
                        milestone=epoch in MILESTONES,
                        reference=references[fold] if references is not None else None)
                    new_records[fold] = record
                folds.append(record)
            if new_records:
                provider.save_many({fold: (epoch, steps[fold]) for fold in new_records})
                for fold, record in new_records.items():
                    local.fold(record)
                    tracker.fold(record)
                    completed[fold] = epoch
                provider.cleanup_many({fold: epoch for fold in new_records})
        else:
            for fold in range(1, 6):
                if epoch <= completed[fold]:
                    record = read_json(local.path / f"fold_{fold}_epoch_{epoch:03d}.json")
                else:
                    training, validation = provider(fold, epoch)
                    steps[fold] += training["optimizer_steps"]
                    record = fold_epoch_record(
                        outer_study=outer_study, trial_number=trial.number, inner_fold=fold,
                        epoch=epoch, configuration=config, training=training,
                        validation=validation, global_step=steps[fold],
                        milestone=epoch in MILESTONES,
                        reference=references[fold] if references is not None else None)
                    provider.save(fold, epoch, steps[fold])
                    local.fold(record)
                    provider.cleanup(fold, epoch)
                    tracker.fold(record)
                    completed[fold] = epoch
                folds.append(record)
        calculated = aggregate(epoch, folds)
        aggregate_path = local.path / f"aggregate_epoch_{epoch:03d}.json"
        if aggregate_path.exists():
            spatial = read_json(aggregate_path)
            if spatial["J_e"] != calculated["J_e"] or spatial["fold_validation_objectives"] != calculated["fold_validation_objectives"]:
                raise ValueError("Persisted common-resource objective differs from fold records")
            if epoch in MILESTONES and trial._get_latest_trial().intermediate_values.get(epoch) != spatial["J_e"]:
                raise ValueError("Optuna intermediate objective differs from local aggregate")
        else:
            spatial = {**calculated, "outer_study": outer_study,
                       "trial_number": trial.number, "resource_milestone": epoch in MILESTONES,
                       "pruning_decision": "not_evaluated", "pruned": False,
                       "trial_state_after_decision": "RUNNING"}
            if epoch in MILESTONES:
                existing = trial._get_latest_trial().intermediate_values.get(epoch)
                if existing is None:
                    trial.report(spatial["J_e"], step=epoch)
                elif existing != spatial["J_e"]:
                    raise ValueError("Optuna intermediate objective differs after interruption")
                prune = trial.should_prune()
                spatial["pruning_decision"] = "PRUNE" if prune else "CONTINUE"
                spatial["pruned"] = prune
                spatial["trial_state_after_decision"] = "PRUNED" if prune else "RUNNING"
            local.spatial(spatial)
            tracker.spatial(spatial)
        last_common = epoch
        if spatial["pruned"]:
            summary = _summary("PRUNED", local, trial, started, epoch, tracker)
            local.state(summary)
            tracker.finish(summary)
            raise optuna.TrialPruned(f"Five-fold J_{epoch} pruned trial")
        if pause_after_epoch == epoch or (pause_requested is not None and pause_requested()):
            summary = _summary("PAUSED", local, trial, started, epoch, tracker)
            local.state(summary)
            tracker.finish(summary)
            return None
    summary = _summary("COMPLETE", local, trial, started, last_common, tracker)
    local.state(summary)
    tracker.finish(summary)
    return read_json(local.path / f"aggregate_epoch_{max_epochs:03d}.json")["J_e"]
