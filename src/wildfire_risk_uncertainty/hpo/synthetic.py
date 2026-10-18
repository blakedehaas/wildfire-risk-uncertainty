"""Fast deterministic five-fold fixture for Optuna, pruning, and persistence."""

from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import suggest
from wildfire_risk_uncertainty.hpo.objective import run_trial
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.storage import metadata, open_study
from wildfire_risk_uncertainty.hpo.wandb_tracking import OnlineTracker

STUDY_NAME = "checkpoint_6e_synthetic_spatial_hpo_v1"


class SyntheticFailure(RuntimeError):
    """Intentional fixture failure to distinguish FAILED from PRUNED."""


def raw_metrics(bp_mean: float, flp_mean: float, *, rows: int = 1000, flp_rows: int = 700) -> dict:
    return {"bp_loss_sum": bp_mean * rows, "flp_loss_sum": flp_mean * flp_rows,
            "bp_rows": rows, "flp_valid_rows": flp_rows, "bp_loss": bp_mean,
            "flp_loss": flp_mean, "combined_loss": 0.5 * (bp_mean + flp_mean),
            "optimizer_steps": 1, "wall_seconds": 0.001, "rows_per_second": rows / 0.001,
            "peak_vram_bytes": 0, "peak_reserved_vram_bytes": 0, "peak_rss_bytes": 0}


def run(root: Path, *, phase: str) -> dict:
    artifact = root / "artifacts/hpo/06e"
    report = root / "reports/hpo/06e"
    artifact.mkdir(parents=True, exist_ok=True)
    expected = metadata(root, 1, kind="synthetic_validation")
    storage_file = artifact / "synthetic.sqlite3"

    def open_again():
        return open_study(storage_file, STUDY_NAME, expected)

    def objective(trial: optuna.Trial) -> float:
        configuration = suggest(trial)
        local_dir = artifact / "records" / STUDY_NAME / f"trial_{trial.number:03d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        tracker = OnlineTracker(outer_study=1, trial_number=trial.number,
                                configuration=configuration.serializable(),
                                provenance=expected, local_dir=local_dir,
                                checkpoint="checkpoint-6e")

        def provider(fold: int, epoch: int) -> tuple[dict, dict]:
            if trial.number == 5 and fold == 3 and epoch == 1:
                raise SyntheticFailure("Intentional synthetic failed-trial fixture")
            penalty = 0.25 * trial.number
            decay = 0.1 / (epoch + 1)
            training = raw_metrics(0.2 + penalty + 0.01 * fold + decay,
                                   0.8 + penalty + 0.02 * fold + decay)
            validation = raw_metrics(0.25 + penalty + 0.015 * fold + decay,
                                     0.85 + penalty + 0.025 * fold + decay)
            return training, validation

        return run_trial(trial=trial, outer_study=1, configuration=configuration,
                         provider=provider, record_root=artifact / "records",
                         study_name=STUDY_NAME, provenance=expected,
                         tracker=tracker, max_epochs=100)

    if phase == "first":
        study = open_again()
        if len(study.trials) > 3:
            raise ValueError("Synthetic first phase cannot run after later trials")
        to_first = max(0, 3 - len(study.trials))
        if to_first:
            study.optimize(objective, n_trials=to_first, n_jobs=1, catch=(SyntheticFailure,))
        history = [{"number": t.number, "state": t.state.name, "params": t.params,
                    "intermediate_values": {str(k): v for k, v in t.intermediate_values.items()}}
                   for t in study.get_trials(deepcopy=True)]
        write_json(artifact / "synthetic_first_process_history.json", {"trials": history})
        return {"first_process_trials": len(history)}
    if phase != "second":
        raise ValueError(phase)
    before_restart = read_json(artifact / "synthetic_first_process_history.json")["trials"]
    study = open_again()
    reloaded = [{"number": t.number, "state": t.state.name, "params": t.params,
                 "intermediate_values": {str(k): v for k, v in t.intermediate_values.items()}}
                for t in study.get_trials(deepcopy=True)]
    if before_restart != reloaded:
        raise ValueError("Optuna history changed on process-level study reload")
    remaining = max(0, 7 - len(reloaded))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, catch=(SyntheticFailure,))
    trials = study.get_trials(deepcopy=True)
    numbers = [t.number for t in trials]
    states = {name: sum(t.state.name == name for t in trials)
              for name in ("COMPLETE", "PRUNED", "FAIL")}
    if len(trials) != 7 or len(set(numbers)) != 7 or not states["COMPLETE"] or not states["PRUNED"] or not states["FAIL"]:
        raise ValueError(f"Synthetic state coverage failed: {states}")
    checks = []
    for trial in trials:
        local_dir = artifact / "records" / STUDY_NAME / f"trial_{trial.number:03d}"
        state = read_json(local_dir / "state.json")
        if state["trial_state"] != ("FAILED" if trial.state.name == "FAIL" else trial.state.name):
            raise ValueError("Optuna/local trial state disagreement")
        for epoch, value in trial.intermediate_values.items():
            local = read_json(local_dir / f"aggregate_epoch_{epoch:03d}.json")
            if local["J_e"] != value or len(local["fold_validation_objectives"]) != 5:
                raise ValueError("Optuna/local five-fold intermediate disagreement")
            checks.append({"trial_number": trial.number, "resource_epoch": epoch,
                           "J_e": value, "state": trial.state.name,
                           "wandb_run_id": read_json(local_dir / "wandb_link.json")["run_id"]})
    storage_result = {"study_name": STUDY_NAME, "storage": str(storage_file.relative_to(root)),
                      "trials_before_restart": len(before_restart),
                      "history_preserved_after_reload": before_restart == reloaded,
                      "no_duplicate_trial_numbers": len(numbers) == len(set(numbers)),
                      "completed_trials_not_rerun": all(t["number"] in numbers for t in before_restart),
                      "trial_numbers": numbers, "states": states,
                      "sampler_restart_note": "Realized persisted history is authoritative; future TPE suggestions are not claimed bitwise identical across restart."}
    result = {"synthetic_only": True, "states": states, "trial_numbers": numbers,
              "intermediate_local_optuna_checks": checks,
              "resource_milestones": [1, 4, 16, 64],
              "study_metadata_verified": study.user_attrs["frozen_metadata"] == expected}
    write_json(report / "study_storage_validation.json", storage_result)
    write_json(report / "synthetic_orchestration_validation.json", result)
    return result
