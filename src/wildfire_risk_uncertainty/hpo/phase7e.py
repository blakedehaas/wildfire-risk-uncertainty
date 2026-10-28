"""Frozen Phase 7E five-success geographic HPO protocol."""

import hashlib
import json
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import (
    GRADNORM_TRAINING_VERSION,
    MAX_EPOCHS,
    SEARCH_SPACE,
    SEED,
    TPE_STARTUP_TRIALS,
    search_space_hash,
)
from wildfire_risk_uncertainty.hpo.storage import study_name
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes

OUTERS = (1, 2, 4, 5)
SUCCESS_TARGET = 5
ATTEMPT_CAP = 25
MANIFEST = Path("reports/hpo/07e_protocol/protocol_manifest.json")


def completed(trials: list[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]


def select_best(trials: list[optuna.trial.FrozenTrial], count: int = SUCCESS_TARGET) -> optuna.trial.FrozenTrial:
    """Frozen objective, first successful completions, deterministic earliest tie."""
    choices = completed(sorted(trials, key=lambda trial: trial.number))[:count]
    if len(choices) != count or any(trial.value is None for trial in choices):
        raise ValueError(f"Exactly {count} completed valued trials required")
    return max(choices, key=lambda trial: (trial.value, -trial.number))


def next_action(trials: list[optuna.trial.FrozenTrial], *, target: int = SUCCESS_TARGET,
                attempt_cap: int = ATTEMPT_CAP) -> tuple[str, int | None]:
    """Return resume, suggest, complete, or exhausted without mutating a study."""
    if len(trials) > attempt_cap:
        raise ValueError("Study exceeded the attempt safety cap")
    running = [trial for trial in trials if trial.state == optuna.trial.TrialState.RUNNING]
    if len(running) > 1:
        raise ValueError("Multiple RUNNING trials in a single-worker study")
    if any(trial.state == optuna.trial.TrialState.PRUNED for trial in trials):
        raise ValueError("Unexpected pruning in NopPruner study")
    successes = len(completed(trials))
    if successes > target:
        raise ValueError("Study exceeded success target")
    if running:
        return "resume", running[0].number
    if successes == target:
        return "complete", None
    if len(trials) == attempt_cap:
        return "exhausted", None
    return "suggest", None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_manifest(root: Path) -> dict:
    """Read-only source and data fingerprints; no outer TEST targets are opened."""
    cache = json.loads((root / "artifacts/training/hpo_cache_v1/manifest.json").read_text())
    studies = {}
    for outer in OUTERS:
        states = {}
        for fold in range(1, 6):
            record = json.loads((root / f"artifacts/hpo/preprocessing_cache_v1/outer_{outer}_inner_{fold}.json").read_text())
            states[str(fold)] = record["state_sha256"]
        studies[str(outer)] = {
            "optuna_study": study_name(outer, gradnorm=True),
            "sqlite": f"artifacts/hpo/outer_{outer}_gradnorm_v3.sqlite3",
            "record_root": f"artifacts/hpo/records/{study_name(outer, gradnorm=True)}",
            "outer_partition": outer, "inner_folds": [1, 2, 3, 4, 5],
            "preprocessing_hashes_by_fold": states,
        }
    return {
        "protocol": "phase-7e-five-success-random-search-v1",
        "approved_pilot_commit": "79bae026721fed316a40bda62a0010e192f5be95",
        "outer_studies": list(OUTERS), "pilot_outer_study": 3,
        "pilot_primary_selection": "original_best_of_20_successes_trial_19",
        "pilot_comparable_selection": "best_of_first_5_successes_frozen_objective_earliest_trial_tie",
        "pilot_study_database_sha256": _sha(root / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"),
        "success_target": SUCCESS_TARGET, "attempt_cap": ATTEMPT_CAP,
        "failure_policy": "FAIL retained; replace with next Optuna suggestion until 5 COMPLETE or 25 attempts; RUNNING resumes",
        "sampler": {"implementation": "Optuna TPESampler", "seed": SEED,
                    "n_startup_trials": TPE_STARTUP_TRIALS,
                    "effective_phase": "random startup only for 5 completed successes"},
        "search_space": SEARCH_SPACE, "search_space_sha256": search_space_hash(),
        "pruner": "NopPruner", "epochs": MAX_EPOCHS, "fold_workers": 2,
        "training": {"implementation_version": GRADNORM_TRAINING_VERSION,
                     "initialization_seed": SEED, "data_order_seed": SEED,
                     "optimizer": "AdamW", "scheduler": "none", "precision": "BF16 autocast"},
        "objective": "mean_five_inner_folds_of_equal_BP_FLP_validation_log_skill_v1",
        "loss_balancing": {"method": "GradNorm", "alpha": 1.5,
                           "weight_optimizer": "Adam", "weight_learning_rate": 0.025},
        "model_selection": "maximum_frozen_objective_among_first_5_COMPLETE_earliest_trial_tie",
        "outer_test_policy": "never used for HPO, protocol, or readiness checks",
        "reporting": "revised_protocol_geographic_sensitivity; pilot_outer_3_result_separate",
        "implementation_files_sha256": {name: _sha(root / name) for name in (
            "src/wildfire_risk_uncertainty/hpo/config.py",
            "src/wildfire_risk_uncertainty/hpo/objective.py",
            "src/wildfire_risk_uncertainty/hpo/records.py",
            "src/wildfire_risk_uncertainty/hpo/real.py",
            "src/wildfire_risk_uncertainty/hpo/storage.py",
            "src/wildfire_risk_uncertainty/hpo/phase7e.py",
            "src/wildfire_risk_uncertainty/training/trainer.py",
            "scripts/benchmark_07a_detached.py",
        )},
        "data_fingerprints": {
            "dataset_manifest_sha256": _sha(root / "reports/dataset/canonical/manifest.json"),
            "feature_schema_sha256": _sha(root / "reports/dataset/canonical/schema.json"),
            "feature_order_sha256": hashlib.sha256(stable_json_bytes(cache["inputs"])).hexdigest(),
            "outer_fold_manifest_sha256": _sha(root / "artifacts/spatial_validation/05b/manifest.json"),
            "inner_fold_manifest_sha256": _sha(root / "reports/spatial_validation/06c/manifest.json"),
            "cache_manifest_sha256": hashlib.sha256(stable_json_bytes(cache)).hexdigest(),
            "partition_offsets_sha256": cache["files_sha256"]["partition_offsets.npy"],
        },
        "studies": studies,
    }


def validate_manifest(root: Path, outer_study: int) -> dict:
    if outer_study not in OUTERS:
        raise ValueError("Phase 7E excludes outer study 3")
    path = root / MANIFEST
    recorded = json.loads(path.read_text())
    if recorded != expected_manifest(root):
        raise ValueError("Phase 7E frozen protocol/data fingerprints differ")
    return recorded
