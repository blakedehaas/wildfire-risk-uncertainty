"""Fixed one-epoch five-fold real-data tracking smoke via the frozen 6D trainer."""

import csv
import hashlib
import signal
import time
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import (
    FIXED_SMOKE,
    GRADNORM_TRAINING_VERSION,
    Configuration,
    suggest,
)
from wildfire_risk_uncertainty.hpo.objective import run_trial
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.resume import run_resumable_trial
from wildfire_risk_uncertainty.hpo.skill import fold_references
from wildfire_risk_uncertainty.hpo.storage import metadata, open_study, study_name
from wildfire_risk_uncertainty.hpo.wandb_tracking import OnlineTracker
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.data import inner_chunks, model_inputs
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    fit_preprocessing,
    stable_json_bytes,
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
                 states: dict[int, PreprocessingState], counts: dict[int, dict],
                 cache: HpoCache | None = None, trial_number: int | None = None,
                 checkpoint_root: Path | None = None,
                 retain_checkpoints: bool = False,
                 loss_balancing: str = "equal") -> None:
        self.root = root
        self.outer_study = outer_study
        self.configuration = configuration
        self.states = states
        self.counts = counts
        self.cache = cache
        self.trial_number = trial_number
        self.checkpoint_root = checkpoint_root
        self.retain_checkpoints = retain_checkpoints
        if loss_balancing not in {"equal", "gradnorm"}:
            raise ValueError("Unknown loss-balancing method")
        self.loss_balancing = loss_balancing
        self.models = {}

    def _model(self, fold: int):
        if fold not in self.models:
            seed_everything(42)
            model, optimizer = make_model(
                inputs=len(self.states[fold].inputs), depth=self.configuration.depth,
                width=self.configuration.width, learning_rate=self.configuration.learning_rate,
                weight_decay=self.configuration.weight_decay,
            )
            gradnorm = GradNorm("cuda") if self.loss_balancing == "gradnorm" else None
            self.models[fold] = (model, optimizer, gradnorm)
        return self.models[fold]

    def _checkpoint_path(self, fold: int, epoch: int) -> Path:
        if self.checkpoint_root is None:
            raise ValueError("Resumable HPO requires an explicit checkpoint directory")
        return self.checkpoint_root / f"fold_{fold}_epoch_{epoch:03d}.pt"

    def _checkpoint_metadata(self, fold: int) -> dict:
        return {"outer_study": self.outer_study, "trial_number": self.trial_number,
                "inner_fold": fold, "configuration": self.configuration.serializable(),
                "seed": 42,
                **({"loss_balancing": "gradnorm", "gradnorm_alpha": 1.5,
                    "gradnorm_training_version": GRADNORM_TRAINING_VERSION}
                   if self.loss_balancing == "gradnorm" else {})}

    def restore(self, fold: int, completed_epoch: int, global_step: int) -> None:
        if completed_epoch == 0:
            self._model(fold)
            return
        if self.cache is None:
            raise ValueError("Resumable HPO requires a validated cache")
        model, optimizer, gradnorm = self._model(fold)
        epoch, step = load_cache_checkpoint(
            self._checkpoint_path(fold, completed_epoch), model, optimizer,
            cache=self.cache, state=self.states[fold],
            expected_metadata=self._checkpoint_metadata(fold), gradnorm=gradnorm)
        if epoch != completed_epoch or step != global_step:
            raise ValueError("Checkpoint epoch/optimizer step differs from local fold record")

    def save(self, fold: int, epoch: int, global_step: int) -> None:
        if self.cache is None:
            raise ValueError("Resumable HPO requires a validated cache")
        model, optimizer, gradnorm = self._model(fold)
        save_cache_checkpoint(self._checkpoint_path(fold, epoch), model, optimizer,
                              completed_epoch=epoch, global_step=global_step,
                              cache=self.cache, state=self.states[fold],
                              metadata=self._checkpoint_metadata(fold), gradnorm=gradnorm)

    def cleanup(self, fold: int, epoch: int) -> None:
        if self.retain_checkpoints:
            return
        previous = self._checkpoint_path(fold, epoch - 2) if epoch > 2 else None
        if previous is not None:
            previous.unlink(missing_ok=True)
            previous.with_suffix(previous.suffix + ".sha256.json").unlink(missing_ok=True)

    def __call__(self, fold: int, epoch: int) -> tuple[dict, dict]:
        model, optimizer, gradnorm = self._model(fold)
        state = self.states[fold]
        training = train_epoch(self.root, state, model, optimizer, epoch,
                               outer_study=self.outer_study, inner_fold=fold,
                               batch_size=self.configuration.batch_size, cache=self.cache,
                               gradnorm=gradnorm)
        validation = validate(self.root, state, model,
                              self.counts[fold]["validation"]["rows"],
                              self.counts[fold]["validation"]["flp_valid"],
                              outer_study=self.outer_study, inner_fold=fold,
                              batch_size=self.configuration.batch_size, cache=self.cache,
                              report_excess=gradnorm is not None)
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


def run_real_hpo(root: Path, outer_study: int, *, cache_dir: Path,
                 pause_after_epoch: int | None = None,
                 gradnorm_objective_reviewed: bool = False,
                 external_stop_file: Path | None = None) -> None:
    """Prepared GradNorm HPO entry point; objective review is a launch gate."""
    if not gradnorm_objective_reviewed:
        raise RuntimeError("Explicit reviewed-objective launch flag is required for GradNorm HPO")
    from wildfire_risk_uncertainty.hpo.fold_workers import PersistentFoldProvider
    from wildfire_risk_uncertainty.training.cache_states import load_states

    cache = HpoCache(root, cache_dir)
    plan = read_json(root / "reports/training/06f_prelaunch/projected_hpo_runtime.json")
    cache_sha = hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest()
    if not plan["accepted_mmap_for_6f"] or plan["cache_manifest_sha256"] != cache_sha:
        raise ValueError("Validated HPO cache acceptance/provenance mismatch")
    provenance = {**metadata(root, outer_study, kind="gradnorm_hpo"),
                  "cache_version": cache.manifest["version"],
                  "cache_manifest_sha256": cache_sha,
                  "partition_offsets_sha256": cache.manifest["files_sha256"]["partition_offsets.npy"]}
    references = fold_references(cache, root, outer_study)
    provenance["inner_fold_constant_references"] = {str(k): v for k, v in references.items()}
    states = load_states(root, cache, outer_study)
    name = study_name(outer_study, gradnorm=True)
    artifact = root / "artifacts/hpo"
    study = open_study(artifact / f"outer_{outer_study}_gradnorm_v3.sqlite3", name, provenance)
    requested_pause = False

    def request_pause(_signal: int, _frame: object) -> None:
        nonlocal requested_pause
        requested_pause = True

    signal.signal(signal.SIGINT, request_pause)
    signal.signal(signal.SIGTERM, request_pause)
    while True:
        if requested_pause or (external_stop_file is not None and external_stop_file.exists()):
            return
        trials = study.get_trials(deepcopy=True)
        if len(trials) > 30 or any(t.state == optuna.trial.TrialState.FAIL for t in trials):
            raise ValueError("Real study exceeded 30 trials or contains FAILED trial; stop for HITL")
        running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
        if len(running) > 1:
            raise ValueError("More than one RUNNING trial in frozen single-worker study")
        if not running and len(trials) == 30:
            return
        if running:
            trial = optuna.Trial(study, running[0]._trial_id)
            saved = trial.user_attrs.get("suggested_configuration")
            if saved is None:
                if (artifact / "records" / name / f"trial_{trial.number:03d}" / "manifest.json").exists():
                    raise ValueError("RUNNING trial has local work but no persisted configuration")
                configuration = suggest(trial)
                trial.set_user_attr("suggested_configuration", configuration.serializable())
            else:
                configuration = Configuration(**saved)
                if configuration.serializable() != trial.params:
                    raise ValueError("RUNNING Optuna trial configuration mismatch")
        else:
            trial = study.ask()
            configuration = suggest(trial)
            trial.set_user_attr("suggested_configuration", configuration.serializable())
        execution = {"inner_fold_workers": 2, "persistent_workers": True,
                     "common_epoch_barrier": True, "optuna_n_jobs": 1}
        saved_execution = trial.user_attrs.get("fold_execution")
        if saved_execution is not None and saved_execution != execution:
            raise ValueError("RUNNING Optuna trial fold-execution topology mismatch")
        trial.set_user_attr("fold_execution", execution)
        local_dir = artifact / "records" / name / f"trial_{trial.number:03d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        provider = PersistentFoldProvider(
            root, outer_study, configuration, cache_path=cache_dir,
            trial_number=trial.number, checkpoint_root=local_dir / "checkpoints",
            loss_balancing="gradnorm")
        model_parameter_count = sum(p.numel() for p in RiskMLP(
            len(states[1].inputs), depth=configuration.depth, width=configuration.width,
        ).parameters())
        link = local_dir / "wandb_link.json"
        run_id = read_json(link)["run_id"] if link.exists() else None
        tracker = OnlineTracker(outer_study=outer_study, trial_number=trial.number,
                                configuration=configuration.serializable(),
                                provenance={**provenance,
                                    "preprocessing_hashes_by_fold": {str(k): states[k].digest() for k in states},
                                    "model_parameter_count": model_parameter_count,
                                    "fold_execution": execution},
                                local_dir=local_dir, checkpoint="checkpoint-07b", run_id=run_id)
        if not link.exists():
            write_json(link, {"run_id": tracker.run_id, "run_url": tracker.run_url})
        try:
            try:
                result = run_resumable_trial(
                    trial=trial, outer_study=outer_study, configuration=configuration,
                    provider=provider, record_root=artifact / "records", study_name=name,
                    provenance=provenance, tracker=tracker, max_epochs=100,
                    references=references,
                    pause_after_epoch=pause_after_epoch,
                    pause_requested=lambda: requested_pause or
                    (external_stop_file is not None and external_stop_file.exists()))
            finally:
                provider.close()
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if requested_pause:
                return
        except Exception:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        else:
            if result is None:
                return  # Leave this exact trial RUNNING for a later invocation.
            study.tell(trial, result)
            if requested_pause:
                return
