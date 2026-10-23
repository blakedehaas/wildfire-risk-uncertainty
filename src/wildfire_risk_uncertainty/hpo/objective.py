"""Five-fold epoch-synchronous Optuna objective and pruning boundary."""

import time
from collections.abc import Callable
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import MILESTONES, Configuration
from wildfire_risk_uncertainty.hpo.records import (
    TrialRecord,
    aggregate,
    fold_epoch_record,
    write_json,
)
from wildfire_risk_uncertainty.hpo.wandb_tracking import DisabledTracker

Provider = Callable[[int, int], tuple[dict, dict]]


def run_trial(*, trial: optuna.Trial, outer_study: int, configuration: Configuration,
              provider: Provider, record_root: Path, study_name: str, provenance: dict,
              tracker: DisabledTracker, max_epochs: int = 100,
              references: dict[int, dict] | None = None) -> float:
    """Run all five folds to each epoch before a spatial score/pruning decision."""
    trial.set_user_attr("suggested_configuration", configuration.serializable())
    local = TrialRecord(record_root, study_name, trial.number)
    local.manifest({"study_name": study_name, "outer_study": outer_study,
                    "trial_number": trial.number, "configuration": configuration.serializable(),
                    "provenance": provenance, "max_epochs": max_epochs})
    write_json(local.path / "wandb_link.json", {"run_id": tracker.run_id,
                                               "run_url": tracker.run_url})
    started = time.perf_counter()
    steps = {fold: 0 for fold in range(1, 6)}
    train_rows = valid_rows = 0
    common: list[dict] = []
    last_epoch = 0
    state = "RUNNING"
    failure: dict = {}
    try:
        for epoch in range(1, max_epochs + 1):
            folds = []
            for fold in range(1, 6):
                training, validation = provider(fold, epoch)
                steps[fold] += training["optimizer_steps"]
                record = fold_epoch_record(
                    outer_study=outer_study, trial_number=trial.number, inner_fold=fold,
                    epoch=epoch, configuration=configuration.serializable(),
                    training=training, validation=validation, global_step=steps[fold],
                    milestone=epoch in MILESTONES,
                    reference=references[fold] if references is not None else None,
                )
                local.fold(record)  # Local authoritative record precedes online logging.
                tracker.fold(record)
                folds.append(record)
                train_rows += record["train_rows"]
                valid_rows += record["validation_rows"]
            spatial = aggregate(epoch, folds)
            spatial.update({"outer_study": outer_study, "trial_number": trial.number,
                            "resource_milestone": epoch in MILESTONES,
                            "pruning_decision": "not_evaluated", "pruned": False,
                            "trial_state_after_decision": "RUNNING"})
            if epoch in MILESTONES:
                trial.report(spatial["J_e"], step=epoch)
                prune = trial.should_prune()
                spatial["pruning_decision"] = "PRUNE" if prune else "CONTINUE"
                spatial["pruned"] = prune
                spatial["trial_state_after_decision"] = "PRUNED" if prune else "RUNNING"
            local.spatial(spatial)
            tracker.spatial(spatial)
            common.append(spatial)
            last_epoch = epoch
            if spatial["pruned"]:
                state = "PRUNED"
                raise optuna.TrialPruned(f"Five-fold J_{epoch} pruned trial")
        state = "COMPLETE"
        return common[-1]["J_e"]
    except optuna.TrialPruned:
        state = "PRUNED"
        raise
    except Exception as exc:
        state = "FAILED"
        failure = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
        raise
    finally:
        summary = {"trial_state": state, "final_resource_epoch": last_epoch,
                   "objective_at_last_common_resource": common[-1]["J_e"] if common else None,
                   "best_observed_common_resource_objective": (
                       max((x["J_e"] for x in common), default=None) if references is not None else
                       min((x["J_e"] for x in common), default=None)),
                   "total_runtime_seconds": time.perf_counter() - started,
                   "total_training_rows_processed": train_rows,
                   "total_validation_rows_processed": valid_rows,
                   "prune_epoch": last_epoch if state == "PRUNED" else None,
                   "local_record_path": str(local.path), "wandb_run_id": tracker.run_id,
                   "wandb_sync_state": tracker.sync_state}
        local.state({**summary, **failure})
        tracker.finish(summary)
        if tracker.sync_state != summary["wandb_sync_state"]:
            local.state({**summary, **failure, "wandb_sync_state": tracker.sync_state})
