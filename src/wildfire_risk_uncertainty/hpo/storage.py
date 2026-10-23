"""Persistent single-worker Optuna studies with strict frozen provenance."""

import hashlib
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb

from wildfire_risk_uncertainty.hpo.config import (
    GRADNORM_TRAINING_VERSION,
    MAX_EPOCHS,
    SCHEMA_VERSION,
    SEARCH_SPACE,
    SEED,
    TPE_STARTUP_TRIALS,
    TRAINING_VERSION,
    pruner,
    sampler,
    search_space_hash,
)
from wildfire_risk_uncertainty.training.data import model_inputs
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import DATA_ORDER_VERSION

BP_CE_NUMERICAL_REVISION = "bp-ce-zero-tail-softplus-fallback-v1"
EXTENSION_REPAIR_MARKER = "gradnorm-hpo-extension-provenance-v1"


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
            "seed": SEED, "sampler": {"name": "TPESampler", "seed": SEED,
                                      **({"n_startup_trials": TPE_STARTUP_TRIALS}
                                         if kind == "gradnorm_hpo" else {})},
            **({"worktree_clean_at_launch": not bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root)),
                "bp_ce_numerical_revision": BP_CE_NUMERICAL_REVISION,
                "training_configuration": {"optimizer": "AdamW", "scheduler": "none",
                                           "epochs": MAX_EPOCHS, "precision": "BF16 autocast",
                                           "initialization_seed": SEED,
                                           "data_order_seed": SEED,
                                           "data_order_version": DATA_ORDER_VERSION,
                                           "fold_workers": 2, "optuna_n_jobs": 1},
                "runtime_environment": {
                "python": platform.python_version(), "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "optuna": optuna.__version__,
                "numpy": np.__version__, "wandb": wandb.__version__,
                "platform": platform.platform(), "cpu_model": platform.processor(),
                "cpu_count": os.cpu_count(),
                "accessible_logical_cpus": len(os.sched_getaffinity(0)),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_memory_bytes": (torch.cuda.get_device_properties(0).total_memory
                                           if torch.cuda.is_available() else None),
            }} if kind == "gradnorm_hpo" else {}),
            "pruner": ({"name": "NopPruner"} if kind == "gradnorm_hpo" else
                       {"name": "SuccessiveHalvingPruner", "min_resource": 1,
                        "reduction_factor": 4, "min_early_stopping_rate": 0}),
            "search_space": SEARCH_SPACE, "search_space_sha256": search_space_hash(),
            "maximum_resource_epochs": MAX_EPOCHS,
            "feature_order_sha256": hashlib.sha256(stable_json_bytes(list(inputs))).hexdigest(),
            **({"dataset_manifest_sha256": hashlib.sha256(
                (root / "reports/dataset/canonical/manifest.json").read_bytes()).hexdigest(),
            "feature_schema_sha256": hashlib.sha256(
                (root / "reports/dataset/canonical/schema.json").read_bytes()).hexdigest()}
               if kind == "gradnorm_hpo" else {}),
            "inner_fold_manifest_sha256": hashlib.sha256(
                (root / "reports/spatial_validation/06c/manifest.json").read_bytes()).hexdigest(),
            "outer_fold_manifest_sha256": hashlib.sha256(
                (root / "artifacts/spatial_validation/05b/manifest.json").read_bytes()).hexdigest()}


def _registered_revision_matches(study: optuna.Study, expected: dict) -> bool:
    frozen = study.user_attrs.get("frozen_metadata")
    revision = study.user_attrs.get("numerical_revision")
    if not isinstance(frozen, dict) or not isinstance(revision, dict):
        return False
    approved_source = revision.get("source_git_sha")
    extension = study.user_attrs.get("extension_repair_revision")
    if extension is not None:
        if (not isinstance(extension, dict) or
                extension.get("approved_bp_ce_source_git_sha") != approved_source or
                extension.get("marker") != EXTENSION_REPAIR_MARKER):
            return False
        approved_source = extension.get("source_git_sha")
    if (approved_source != expected.get("source_git_sha") or
            revision.get("marker") != expected.get("bp_ce_numerical_revision") or
            revision.get("original_source_git_sha") != frozen.get("source_git_sha")):
        return False
    ignored = {"source_git_sha", "bp_ce_numerical_revision"}
    return {k: v for k, v in frozen.items() if k not in ignored} == {
        k: v for k, v in expected.items() if k not in ignored}


def open_study(storage_file: Path, name: str, expected_metadata: dict,
               *, allow_registered_revision: bool = False) -> optuna.Study:
    storage_file.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_file.resolve()}"
    existing = name in optuna.study.get_all_study_names(storage=storage)
    selected_pruner = (optuna.pruners.NopPruner() if expected_metadata.get("kind") == "gradnorm_hpo"
                       else pruner())
    direction = "maximize" if expected_metadata.get("kind") == "gradnorm_hpo" else "minimize"
    study = optuna.create_study(study_name=name, storage=storage, direction=direction,
                                sampler=sampler(), pruner=selected_pruner, load_if_exists=True)
    if existing:
        if (study.user_attrs.get("frozen_metadata") != expected_metadata and
                not (allow_registered_revision and
                     _registered_revision_matches(study, expected_metadata))):
            raise ValueError("Persisted study metadata/search-space is incompatible")
    else:
        study.set_user_attr("frozen_metadata", expected_metadata)
    return study


@contextmanager
def study_controller_lock(storage_file: Path):
    """One process may suggest, train, or change the budget for a study."""
    import fcntl

    path = storage_file.with_suffix(".controller.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another controller owns this HPO study") from exc
        yield


def _rng_json(rng: np.random.RandomState) -> dict:
    name, keys, position, gaussian, cached = rng.get_state()
    return {"name": name, "keys": keys.tolist(), "position": position,
            "gaussian": gaussian, "cached": cached}


def _set_rng_json(rng: np.random.RandomState, state: dict) -> None:
    rng.set_state((state["name"], np.asarray(state["keys"], dtype=np.uint32),
                   state["position"], state["gaussian"], state["cached"]))


def restore_sampler_state(study: optuna.Study) -> None:
    """Restore Optuna 5 TPE and startup samplers after reopening SQLite."""
    saved = study.user_attrs.get("sampler_state")
    suggested = [trial.number for trial in study.trials if trial.params]
    if saved is None:
        if suggested:
            raise ValueError("Persisted study has suggestions but no sampler RNG state")
        return
    if saved["optuna_version"] != optuna.__version__:
        raise ValueError("Optuna version differs from persisted sampler RNG state")
    if saved["trial_number"] != (max(suggested) if suggested else -1):
        raise ValueError("Sampler RNG state does not cover the last suggested trial")
    _set_rng_json(study.sampler._rng.rng, saved["tpe"])
    _set_rng_json(study.sampler._random_sampler._rng.rng, saved["startup"])


def save_sampler_state(study: optuna.Study, trial_number: int) -> None:
    study.set_user_attr("sampler_state", {
        "trial_number": trial_number,
        "optuna_version": optuna.__version__,
        "tpe": _rng_json(study.sampler._rng.rng),
        "startup": _rng_json(study.sampler._random_sampler._rng.rng),
    })


def initialize_trial_limit(study: optuna.Study, limit: int) -> None:
    budget = study.user_attrs.get("trial_budget")
    if budget is None:
        if study.trials:
            raise ValueError("Existing study lacks a lifetime trial limit")
        study.set_user_attr("trial_budget", {"initial": limit, "current": limit,
                                             "history": [{"from": None, "to": limit}]})
    elif budget["current"] != limit:
        raise ValueError("Requested trial limit differs from persisted lifetime cap")


def extend_trial_limit(study: optuna.Study, new_limit: int) -> dict:
    budget = study.user_attrs.get("trial_budget")
    old = budget["current"] if budget is not None else None
    if old is None or new_limit <= old or new_limit < len(study.trials):
        raise ValueError("Extension must raise the persisted lifetime cap")
    history = list(budget["history"])
    history.append({"from": old, "to": new_limit})
    study.set_user_attr("trial_budget", {"initial": budget["initial"], "current": new_limit,
                                         "history": history})
    return {"study": study.study_name, "previous_limit": old,
            "trial_limit": new_limit, "existing_trials": len(study.trials),
            "maximum_new_suggestions": new_limit - len(study.trials)}
