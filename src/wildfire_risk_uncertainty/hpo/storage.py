"""Persistent single-worker Optuna studies with strict frozen provenance."""

import hashlib
import subprocess
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import (
    GRADNORM_TRAINING_VERSION,
    MAX_EPOCHS,
    SCHEMA_VERSION,
    SEARCH_SPACE,
    SEED,
    TRAINING_VERSION,
    pruner,
    sampler,
    search_space_hash,
)
from wildfire_risk_uncertainty.training.data import model_inputs
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes


def study_name(outer_study: int, *, gradnorm: bool = False) -> str:
    if outer_study not in range(1, 6):
        raise ValueError("Outer study must be 1..5")
    return (f"wildfire_outer_{outer_study}_gradnorm_hpo_v3" if gradnorm else
            f"wildfire_outer_{outer_study}_hpo_v1")


def metadata(root: Path, outer_study: int, *, kind: str = "real_hpo") -> dict:
    inputs = model_inputs(root / "reports/dataset/canonical/schema.json")
    return {"outer_study": outer_study, "kind": kind,
            "source_git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            ).strip(),
            "hpo_schema_version": SCHEMA_VERSION,
            "training_implementation_version": (GRADNORM_TRAINING_VERSION
                                                if kind == "gradnorm_hpo" else TRAINING_VERSION),
            **({"loss_balancing": {"method": "GradNorm", "alpha": 1.5,
                                    "weight_optimizer": "Adam", "weight_learning_rate": 0.025,
                                    "rate_loss": "masked_cross_entropy",
                                    "initial_reference": "first_epoch_separate_support_mean_CE"},
                "ranking_objective": "mean_five_inner_folds_of_equal_BP_FLP_validation_log_skill_v1"}
               if kind == "gradnorm_hpo" else {}),
            "seed": SEED, "sampler": {"name": "TPESampler", "seed": SEED},
            "pruner": ({"name": "NopPruner"} if kind == "gradnorm_hpo" else
                       {"name": "SuccessiveHalvingPruner", "min_resource": 1,
                        "reduction_factor": 4, "min_early_stopping_rate": 0}),
            "search_space": SEARCH_SPACE, "search_space_sha256": search_space_hash(),
            "maximum_resource_epochs": MAX_EPOCHS,
            "feature_order_sha256": hashlib.sha256(stable_json_bytes(list(inputs))).hexdigest(),
            "inner_fold_manifest_sha256": hashlib.sha256(
                (root / "reports/spatial_validation/06c/manifest.json").read_bytes()).hexdigest(),
            "outer_fold_manifest_sha256": hashlib.sha256(
                (root / "artifacts/spatial_validation/05b/manifest.json").read_bytes()).hexdigest()}


def open_study(storage_file: Path, name: str, expected_metadata: dict) -> optuna.Study:
    storage_file.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_file.resolve()}"
    existing = name in optuna.study.get_all_study_names(storage=storage)
    selected_pruner = (optuna.pruners.NopPruner() if expected_metadata.get("kind") == "gradnorm_hpo"
                       else pruner())
    direction = "maximize" if expected_metadata.get("kind") == "gradnorm_hpo" else "minimize"
    study = optuna.create_study(study_name=name, storage=storage, direction=direction,
                                sampler=sampler(), pruner=selected_pruner, load_if_exists=True)
    if existing:
        if study.user_attrs.get("frozen_metadata") != expected_metadata:
            raise ValueError("Persisted study metadata/search-space is incompatible")
    else:
        study.set_user_attr("frozen_metadata", expected_metadata)
    return study
