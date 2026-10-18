"""Fixed one-epoch five-fold real-data tracking smoke via the frozen 6D trainer."""

import csv
import hashlib
import time
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import FIXED_SMOKE, suggest
from wildfire_risk_uncertainty.hpo.objective import run_trial
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.storage import metadata, open_study, study_name
from wildfire_risk_uncertainty.hpo.wandb_tracking import OnlineTracker
from wildfire_risk_uncertainty.training.data import inner_chunks, model_inputs
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    fit_preprocessing,
)
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

STUDY_NAME = "checkpoint_6e_outer_3_fixed_real_smoke_v1"


def support_counts(root: Path, outer_study: int) -> dict[int, dict]:
    with (root / "reports/spatial_validation/06c/tables/target_support_by_inner_fold.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["outer_study"]) == outer_study]
    by_fold = {}
    for fold in range(1, 6):
        selected = {r["role"]: r for r in rows if int(r["inner_fold"]) == fold}
        if set(selected) != {"train", "validation"}:
            raise ValueError("Missing frozen 6C fold support")
        by_fold[fold] = {role: {"rows": int(row["rows"]),
                                "flp_valid": int(row["flp_target_valid"]),
                                "bp_zero": int(row["bp_zero"]),
                                "bp_positive": int(row["bp_positive"])}
                         for role, row in selected.items()}
    return by_fold


def state_from_record(d: dict) -> PreprocessingState:
    s = d["state"]
    return PreprocessingState(tuple(s["inputs"]), tuple(s["means_after_fixed_transform"]),
                              tuple(s["standard_deviations_after_imputation"]),
                              tuple(s["missing_counts"][name] for name in s["inputs"]),
                              s["rows"], s["flp_valid_rows"], s["schema_sha256"])


def preprocessing_state(root: Path, outer_study: int, fold: int,
                        provenance: dict, counts: dict) -> tuple[PreprocessingState, dict]:
    path = root / "artifacts/hpo/preprocessing" / f"outer_{outer_study}_inner_{fold}.json"
    inputs = model_inputs(root / "reports/dataset/canonical/schema.json")
    schema_hash = hashlib.sha256((root / "reports/dataset/canonical/schema.json").read_bytes()).hexdigest()
    expected = {"outer_study": outer_study, "inner_fold": fold,
                "feature_order_sha256": provenance["feature_order_sha256"],
                "inner_fold_manifest_sha256": provenance["inner_fold_manifest_sha256"],
                "outer_fold_manifest_sha256": provenance["outer_fold_manifest_sha256"],
                "schema_sha256": schema_hash}
    if path.exists():
        existing = read_json(path)
        if existing["provenance"] != expected:
            raise ValueError("Cached fold preprocessing provenance mismatch")
        state = state_from_record(existing)
        if state.digest() != existing["state_sha256"]:
            raise ValueError("Cached fold preprocessing hash mismatch")
        if state.rows != counts["train"]["rows"] or state.flp_valid_rows != counts["train"]["flp_valid"]:
            raise ValueError("Cached fold preprocessing support mismatch")
        return state, existing

    canonical = root / "artifacts/dataset/canonical/partitions"
    outer = root / "artifacts/spatial_validation/05b/partitions"
    inner = root / "artifacts/spatial_validation/06c/partitions"
    fits = []
    for _ in range(2):
        start = time.perf_counter()
        state = fit_preprocessing(inner_chunks(
            canonical, outer, inner, inputs, role="train", outer_study=outer_study,
            inner_fold=fold,
        ), inputs, schema_hash)
        fits.append((state, time.perf_counter() - start))
    if fits[0][0].digest() != fits[1][0].digest():
        raise ValueError("Repeated INNER TRAIN preprocessing refit differs")
    if state.rows != counts["train"]["rows"] or state.flp_valid_rows != counts["train"]["flp_valid"]:
        raise ValueError("INNER TRAIN preprocessing support differs from frozen 6C")
    record = {"provenance": expected, "state": state.serializable(),
              "state_sha256": state.digest(), "refit_identical": True,
              "scan_seconds": [seconds for _, seconds in fits]}
    write_json(path, record)
    return state, record


class RealProvider:
    def __init__(self, root: Path, outer_study: int, configuration,
                 states: dict[int, PreprocessingState], counts: dict[int, dict]) -> None:
        self.root = root
        self.outer_study = outer_study
        self.configuration = configuration
        self.states = states
        self.counts = counts
        self.models = {}

    def __call__(self, fold: int, epoch: int) -> tuple[dict, dict]:
        if fold not in self.models:
            seed_everything(42)
            self.models[fold] = make_model(
                inputs=len(self.states[fold].inputs), depth=self.configuration.depth,
                width=self.configuration.width, learning_rate=self.configuration.learning_rate,
                weight_decay=self.configuration.weight_decay,
            )
        model, optimizer = self.models[fold]
        state = self.states[fold]
        training = train_epoch(self.root, state, model, optimizer, epoch,
                               outer_study=self.outer_study, inner_fold=fold,
                               batch_size=self.configuration.batch_size)
        validation = validate(self.root, state, model,
                              self.counts[fold]["validation"]["rows"],
                              self.counts[fold]["validation"]["flp_valid"],
                              outer_study=self.outer_study, inner_fold=fold,
                              batch_size=self.configuration.batch_size)
        return training, validation


def run_smoke(root: Path) -> dict:
    outer_study = 3
    artifact = root / "artifacts/hpo/06e"
    report = root / "reports/hpo/06e"
    provenance = metadata(root, outer_study, kind="fixed_real_smoke")
    counts = support_counts(root, outer_study)
    states = {}
    preprocessing = {}
    for fold in range(1, 6):
        states[fold], preprocessing[fold] = preprocessing_state(
            root, outer_study, fold, provenance, counts[fold],
        )
    study = open_study(artifact / "fixed_real_smoke.sqlite3", STUDY_NAME, provenance)
    if study.trials:
        raise ValueError("Fixed real smoke study already has a trial; refusing a second real-data run")
    study.enqueue_trial(FIXED_SMOKE.serializable())

    def objective(trial: optuna.Trial) -> float:
        configuration = suggest(trial)
        if configuration != FIXED_SMOKE:
            raise ValueError("Fixed real smoke configuration differs from prospectively frozen values")
        local_dir = artifact / "records" / STUDY_NAME / f"trial_{trial.number:03d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        provider = RealProvider(root, outer_study, configuration, states, counts)
        model_parameter_count = sum(p.numel() for p in RiskMLP(
            len(states[1].inputs), depth=configuration.depth, width=configuration.width,
        ).parameters())
        tracker = OnlineTracker(outer_study=outer_study, trial_number=trial.number,
                                configuration=configuration.serializable(),
                                provenance={**provenance,
                                    "preprocessing_hashes_by_fold": {str(k): states[k].digest() for k in states},
                                    "model_parameter_count": model_parameter_count},
                                local_dir=local_dir, checkpoint="checkpoint-6e")
        return run_trial(trial=trial, outer_study=outer_study, configuration=configuration,
                         provider=provider, record_root=artifact / "records",
                         study_name=STUDY_NAME, provenance=provenance, tracker=tracker,
                         max_epochs=1)

    study.optimize(objective, n_trials=1, n_jobs=1)
    trial = study.trials[0]
    if trial.state != optuna.trial.TrialState.COMPLETE or set(trial.intermediate_values) != {1}:
        raise ValueError("Fixed real smoke did not complete exactly one common resource epoch")
    local_dir = artifact / "records" / STUDY_NAME / "trial_000"
    aggregate_record = read_json(local_dir / "aggregate_epoch_001.json")
    result = {"scientific_hpo": False, "outer_study": outer_study,
              "configuration": FIXED_SMOKE.serializable(), "inner_fold_support": counts,
              "preprocessing_hashes_by_fold": {str(k): states[k].digest() for k in states},
              "preprocessing_scans_by_fold": {str(k): preprocessing[k]["scan_seconds"] for k in preprocessing},
              "J_1": aggregate_record["J_e"], "fold_dispersion_std_population":
                  aggregate_record["fold_objective_std_population"],
              "fold_min": aggregate_record["fold_objective_min"],
              "fold_max": aggregate_record["fold_objective_max"],
              "optuna_intermediate_J_1": trial.intermediate_values[1],
              "optuna_trial_state": trial.state.name,
              "wandb_link": read_json(local_dir / "wandb_link.json")}
    write_json(report / "real_smoke_trial_summary.json", result)
    return result


def run_real_hpo(root: Path, outer_study: int) -> None:
    """Future 6F entry point; the 6E command never calls this function."""
    provenance = metadata(root, outer_study, kind="real_hpo")
    counts = support_counts(root, outer_study)
    states = {fold: preprocessing_state(root, outer_study, fold, provenance, counts[fold])[0]
              for fold in range(1, 6)}
    name = study_name(outer_study)
    artifact = root / "artifacts/hpo"
    study = open_study(artifact / f"outer_{outer_study}.sqlite3", name, provenance)
    if len(study.trials) > 30:
        raise ValueError("Persisted study exceeds the frozen 30-trial budget")

    def objective(trial: optuna.Trial) -> float:
        configuration = suggest(trial)
        local_dir = artifact / "records" / name / f"trial_{trial.number:03d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        provider = RealProvider(root, outer_study, configuration, states, counts)
        model_parameter_count = sum(p.numel() for p in RiskMLP(
            len(states[1].inputs), depth=configuration.depth, width=configuration.width,
        ).parameters())
        tracker = OnlineTracker(outer_study=outer_study, trial_number=trial.number,
                                configuration=configuration.serializable(),
                                provenance={**provenance,
                                    "preprocessing_hashes_by_fold": {str(k): states[k].digest() for k in states},
                                    "model_parameter_count": model_parameter_count},
                                local_dir=local_dir, checkpoint="checkpoint-6f")
        return run_trial(trial=trial, outer_study=outer_study, configuration=configuration,
                         provider=provider, record_root=artifact / "records", study_name=name,
                         provenance=provenance, tracker=tracker, max_epochs=100)

    remaining = 30 - len(study.trials)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
