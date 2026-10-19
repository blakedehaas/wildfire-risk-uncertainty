"""Deterministic HPO orchestration checks without real-data training."""

from pathlib import Path

import optuna
import pytest

from wildfire_risk_uncertainty.hpo.cli import write_figure_manifest
from wildfire_risk_uncertainty.hpo.config import (
    MILESTONES,
    SEARCH_SPACE,
    pruner,
    sampler,
    search_space_hash,
    suggest,
)
from wildfire_risk_uncertainty.hpo.objective import run_trial
from wildfire_risk_uncertainty.hpo.real import RealProvider, preprocessing_state
from wildfire_risk_uncertainty.hpo.records import (
    TrialRecord,
    aggregate,
    fold_epoch_record,
    read_json,
    task_metrics,
    write_json,
)
from wildfire_risk_uncertainty.hpo.resume import run_resumable_trial
from wildfire_risk_uncertainty.hpo.storage import metadata, open_study
from wildfire_risk_uncertainty.hpo.synthetic import raw_metrics
from wildfire_risk_uncertainty.hpo.visualization import (
    curated_real_figures,
    trial_figures,
)
from wildfire_risk_uncertainty.hpo.wandb_tracking import (
    DisabledTracker,
    MockTracker,
    OnlineTracker,
    fold_payload,
    require_online_authentication,
    spatial_payload,
)
from wildfire_risk_uncertainty.training.data import model_inputs
from wildfire_risk_uncertainty.training.preprocessing import PreprocessingState


class BatchToyProvider:
    def __init__(self, penalty: float, checkpoints: dict | None = None) -> None:
        self.penalty = penalty
        self.checkpoints = checkpoints if checkpoints is not None else {}
        self.states = {}

    def restore_many(self, completed, steps):
        self.states = {fold: self.checkpoints.get((fold, completed[fold]), 0)
                       for fold in range(1, 6)}

    def run_epoch(self, epoch, folds):
        result = {}
        for fold in folds:
            self.states[fold] += 1
            value = 0.2 + self.penalty + fold / 100 + self.states[fold] / 1000
            result[fold] = (raw_metrics(value, 0.8 + value),
                            raw_metrics(value + 0.1, 0.9 + value))
        return result

    def save_many(self, folds):
        for fold, (epoch, _step) in folds.items():
            self.checkpoints[(fold, epoch)] = self.states[fold]

    def cleanup_many(self, folds):
        pass


class SequentialToyProvider:
    def __init__(self, penalty: float) -> None:
        self.penalty = penalty
        self.states = {}

    def restore(self, fold, completed_epoch, global_step):
        self.states[fold] = completed_epoch

    def __call__(self, fold, epoch):
        self.states[fold] += 1
        value = 0.2 + self.penalty + fold / 100 + self.states[fold] / 1000
        return raw_metrics(value, 0.8 + value), raw_metrics(value + 0.1, 0.9 + value)

    def save(self, fold, epoch, global_step):
        pass

    def cleanup(self, fold, epoch):
        pass


def test_frozen_sampler_pruner_space_and_seed() -> None:
    study = optuna.create_study(sampler=sampler(), pruner=pruner())
    trial = study.ask()
    config = suggest(trial)
    assert SEARCH_SPACE["depth"]["choices"] == [2, 3]
    assert SEARCH_SPACE["width"]["choices"] == [64, 128, 256]
    assert SEARCH_SPACE["batch_size"]["choices"] == [32768, 65536, 131072]
    assert trial.distributions["learning_rate"] == optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True)
    assert trial.distributions["weight_decay"] == optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True)
    assert config.serializable() == trial.params
    assert len(search_space_hash()) == 64 and MILESTONES == (1, 4, 16, 64)
    assert study.pruner._min_resource == 1
    assert study.pruner._reduction_factor == 4
    assert study.pruner._min_early_stopping_rate == 0
    a = optuna.create_study(sampler=sampler()).ask()
    b = optuna.create_study(sampler=sampler()).ask()
    assert suggest(a) == suggest(b)


def test_raw_objective_and_exact_five_fold_aggregate() -> None:
    raw = raw_metrics(0.2, 0.8)
    values = task_metrics(raw)
    assert values["bp_loss_sum"] == pytest.approx(200)
    assert values["flp_loss_sum"] == pytest.approx(560)
    assert values["equal_task_objective"] == pytest.approx(0.5)
    records = [fold_epoch_record(outer_study=3, trial_number=0, inner_fold=f, epoch=1,
                                 configuration={}, training=raw,
                                 validation=raw_metrics(0.2 + f / 100, 0.8 + f / 100),
                                 global_step=1, milestone=True) for f in range(1, 6)]
    result = aggregate(1, records)
    assert result["J_e"] == pytest.approx(0.53)
    assert result["fold_objective_min"] == pytest.approx(0.51)
    assert result["fold_objective_max"] == pytest.approx(0.55)
    assert result["fold_objective_std_population"] > 0
    with pytest.raises(ValueError, match="Exactly five"):
        aggregate(1, records[:1])
    with pytest.raises(ValueError, match="common resource"):
        aggregate(4, records)


def test_storage_resume_metadata_and_no_duplicate_trials(tmp_path) -> None:
    root = Path.cwd()
    meta = metadata(root, 1, kind="synthetic_validation")
    db = tmp_path / "study.sqlite3"
    study = open_study(db, "test_study", meta)
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=2)
    snapshot = [(t.number, t.params, t.value) for t in study.trials]
    reloaded = open_study(db, "test_study", meta)
    assert [(t.number, t.params, t.value) for t in reloaded.trials] == snapshot
    reloaded.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
    assert [t.number for t in reloaded.trials] == [0, 1, 2]
    with pytest.raises(ValueError, match="incompatible"):
        open_study(db, "test_study", {**meta, "seed": 99})


def test_five_folds_precede_pruning_and_local_state_survives(tmp_path) -> None:
    study = optuna.create_study(sampler=sampler(), pruner=pruner())
    fold_calls = []

    def objective(trial):
        config = suggest(trial)
        tracker = MockTracker()

        def provider(fold, epoch):
            fold_calls.append((trial.number, fold, epoch))
            penalty = trial.number * 0.5
            return raw_metrics(0.2 + penalty, 0.8 + penalty), raw_metrics(
                0.3 + penalty, 0.9 + penalty)

        return run_trial(trial=trial, outer_study=1, configuration=config, provider=provider,
                         record_root=tmp_path, study_name="trial_test", provenance={"fixture": True},
                         tracker=tracker, max_epochs=4)

    study.optimize(objective, n_trials=4)
    assert study.trials[0].state.name == "COMPLETE"
    assert set(study.trials[0].intermediate_values) == {1, 4}
    assert any(t.state.name == "PRUNED" for t in study.trials)
    for trial in study.trials:
        assert [(f, e) for _, f, e in fold_calls if _ == trial.number and e == 1] == [
            (f, 1) for f in range(1, 6)]
        local = tmp_path / "trial_test" / f"trial_{trial.number:03d}"
        state = read_json(local / "state.json")
        assert state["trial_state"] == trial.state.name
        for epoch, value in trial.intermediate_values.items():
            record = read_json(local / f"aggregate_epoch_{epoch:03d}.json")
            assert record["J_e"] == value
            assert len(record["fold_validation_objectives"]) == 5
            assert record["pruning_decision"] in {"PRUNE", "CONTINUE"}
        if trial.state.name == "PRUNED":
            assert state["prune_epoch"] in MILESTONES
            assert state["total_training_rows_processed"] == 5000 * state["prune_epoch"]
            assert state["total_validation_rows_processed"] == 5000 * state["prune_epoch"]
            assert not (local / f"aggregate_epoch_{state['prune_epoch'] + 1:03d}.json").exists()


def test_failed_trial_keeps_completed_fold_records(tmp_path) -> None:
    study = optuna.create_study(sampler=sampler(), pruner=pruner())

    def objective(trial):
        configuration = suggest(trial)

        def provider(fold, epoch):
            if fold == 3:
                raise RuntimeError("synthetic fold failure")
            return raw_metrics(0.2, 0.8), raw_metrics(0.3, 0.9)

        return run_trial(trial=trial, outer_study=1, configuration=configuration,
                         provider=provider, record_root=tmp_path, study_name="failed_fixture",
                         provenance={}, tracker=MockTracker(), max_epochs=1)

    study.optimize(objective, n_trials=1, catch=(RuntimeError,))
    assert study.trials[0].state.name == "FAIL"
    local = tmp_path / "failed_fixture/trial_000"
    state = read_json(local / "state.json")
    assert state["trial_state"] == "FAILED" and state["exception_type"] == "RuntimeError"
    assert state["total_training_rows_processed"] == 2000
    assert (local / "fold_1_epoch_001.json").exists() and (local / "fold_2_epoch_001.json").exists()
    assert not (local / "aggregate_epoch_001.json").exists()


def test_tracker_payloads_disabled_mock_and_local_before_failure(tmp_path) -> None:
    raw = raw_metrics(0.2, 0.8)
    record = fold_epoch_record(outer_study=3, trial_number=0, inner_fold=2, epoch=4,
                               configuration={}, training=raw, validation=raw,
                               global_step=4, milestone=True)
    payload = fold_payload(record)
    assert payload["fold_2/epoch"] == 4
    assert payload["fold_2/training/bp_loss_sum"] == raw["bp_loss_sum"]
    assert payload["fold_2/validation/equal_task_objective"] == 0.5
    spatial = aggregate(4, [{**record, "inner_fold": f} for f in range(1, 6)])
    assert spatial_payload(spatial)["spatial_cv/validation_objective_mean"] == spatial["J_e"]
    disabled = DisabledTracker(); disabled.fold(record); disabled.spatial(spatial); disabled.finish({})
    mock = MockTracker(); mock.fold(record); mock.spatial(spatial); mock.finish({"trial_state": "COMPLETE"})
    assert len(mock.events) == 2 and mock.summary["trial_state"] == "COMPLETE"
    local = TrialRecord(tmp_path, "study", 0)
    local.manifest({"config": 1})
    local.fold(record)
    with pytest.raises(ValueError, match="already exists"):
        local.fold(record)
    assert read_json(local.path / "fold_2_epoch_004.json")["training"]["bp_count"] == 1000


def test_pruned_visualization_uses_only_recorded_epochs(tmp_path) -> None:
    local = TrialRecord(tmp_path, "study", 0)
    for fold in range(1, 6):
        raw = raw_metrics(0.2 + fold / 100, 0.8)
        local.fold(fold_epoch_record(outer_study=1, trial_number=0, inner_fold=fold,
                                     epoch=1, configuration={}, training=raw,
                                     validation=raw, global_step=1, milestone=True))
    records = [read_json(local.path / f"fold_{fold}_epoch_001.json") for fold in range(1, 6)]
    value = aggregate(1, records)
    value.update({"resource_milestone": True, "pruning_decision": "PRUNE", "pruned": True})
    local.spatial(value)
    files = trial_figures(local.path, tmp_path / "figures", "pruned")
    assert len(files) == 19 and all(Path(f).is_file() for f in files)
    assert not list(local.path.glob("*epoch_004.json"))


def test_curated_figures_and_tiered_manifest_from_saved_metrics(tmp_path) -> None:
    local = TrialRecord(tmp_path, "study", 0)
    for fold in range(1, 6):
        raw = raw_metrics(0.2 + fold / 100, 0.8)
        local.fold(fold_epoch_record(outer_study=3, trial_number=0, inner_fold=fold,
                                     epoch=1, configuration={}, training=raw,
                                     validation=raw, global_step=1, milestone=True))
    records = [read_json(local.path / f"fold_{fold}_epoch_001.json") for fold in range(1, 6)]
    spatial = aggregate(1, records)
    spatial.update({"resource_milestone": True, "pruning_decision": "CONTINUE"})
    local.spatial(spatial)
    report_dir = tmp_path / "reports/hpo/06e/figures"
    files = curated_real_figures(local.path, report_dir)
    assert len(files) == 6 and all(Path(p).is_file() for p in files)
    artifact_dir = tmp_path / "artifacts/hpo/06e/figures"
    detailed = trial_figures(local.path, artifact_dir, "detail")
    assert len(detailed) == 19
    write_figure_manifest(tmp_path)
    manifest = read_json(tmp_path / "reports/hpo/06e/figures_manifest.json")
    assert len(manifest["committed_report_figures"]) == 6
    assert manifest["detailed_artifact_figures"]["count"] == 19
    assert all("artifacts/hpo/" not in path for path in manifest["committed_report_figures"])


def test_real_hpo_auth_preflight_does_not_reveal_credentials(monkeypatch) -> None:
    class MissingApi:
        def __init__(self, **_kwargs):
            raise RuntimeError("secret credential value")

    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.wandb_tracking.wandb.Api", MissingApi)
    with pytest.raises(RuntimeError, match="stop for HITL") as error:
        require_online_authentication()
    assert "secret credential value" not in str(error.value)


def test_online_adapter_names_axes_summary_and_degraded_local_record(tmp_path, monkeypatch) -> None:
    captured = {}

    class FakeRun:
        id = "fixture-id"
        url = "https://wandb.ai/entity/project/runs/fixture-id"

        def __init__(self):
            self.axes = []
            self.events = []
            self.summary = {}
            self.fail_logging = False
            self.finished = False

        def define_metric(self, *args, **kwargs):
            self.axes.append((args, kwargs))

        def log(self, value):
            if self.fail_logging:
                raise RuntimeError("synthetic W&B outage")
            self.events.append(value)

        def finish(self):
            self.finished = True

    fake = FakeRun()

    def fake_init(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.wandb_tracking.wandb.init", fake_init)
    tracker = OnlineTracker(outer_study=3, trial_number=7,
                            configuration={"depth": 2}, provenance={"git_sha": "fixture"},
                            local_dir=tmp_path, checkpoint="checkpoint-6e")
    assert captured["name"] == "outer-3-trial-007" and captured["group"] == "outer-3"
    assert captured["job_type"] == "hpo-trial" and captured["mode"] == "online"
    resumed_tracker = OnlineTracker(outer_study=3, trial_number=7,
                                    configuration={"depth": 2}, provenance={"git_sha": "fixture"},
                                    local_dir=tmp_path, checkpoint="checkpoint-6f",
                                    run_id="fixture-id")
    assert resumed_tracker.run_id == tracker.run_id
    assert captured["id"] == "fixture-id" and captured["resume"] == "must"
    assert "WANDB_API_KEY" not in captured["config"]
    assert any(args == ("fold_3/*",) and kwargs.get("step_metric") == "fold_3/epoch"
               for args, kwargs in fake.axes)
    assert any(args == ("spatial_cv/*",) and kwargs.get("step_metric") == "spatial_cv/resource_epoch"
               for args, kwargs in fake.axes)
    raw = raw_metrics(0.2, 0.8)
    fold = fold_epoch_record(outer_study=3, trial_number=7, inner_fold=3, epoch=1,
                             configuration={}, training=raw, validation=raw,
                             global_step=1, milestone=True)
    tracker.fold(fold)
    spatial = aggregate(1, [{**fold, "inner_fold": k} for k in range(1, 6)])
    tracker.spatial(spatial)
    assert len(fake.events) == 2
    assert read_json(tmp_path / "wandb_mirror/aggregate_epoch_001.json")[
        "spatial_cv/validation_objective_mean"] == spatial["J_e"]
    fake.fail_logging = True
    tracker.fold({**fold, "epoch": 2})
    assert tracker.sync_state == "degraded_pending"
    assert read_json(tmp_path / "wandb_sync_state.json")["state"] == "degraded_pending"
    tracker.finish({"trial_state": "COMPLETE"})
    assert fake.finished and fake.summary["trial_state"] == "COMPLETE"


def test_real_provider_reuses_frozen_trainer_and_33_input_order(tmp_path, monkeypatch) -> None:
    inputs = model_inputs(Path.cwd() / "reports/dataset/canonical/schema.json")
    assert len(inputs) == 33 and "x_5070" not in inputs and "BP" not in inputs
    state = PreprocessingState(inputs, (0.0,) * 33, (1.0,) * 33, (0,) * 33, 10, 8, "schema")
    calls = []
    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.real.seed_everything", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.real.make_model", lambda **kw: ("model", "optimizer"))

    def fake_train(root, state, model, optimizer, epoch, **kwargs):
        calls.append(("training", kwargs["outer_study"], kwargs["inner_fold"], epoch))
        return raw_metrics(0.2, 0.8)

    def fake_validation(root, state, model, expected_rows, expected_flp, **kwargs):
        calls.append(("validation", kwargs["outer_study"], kwargs["inner_fold"], epoch))
        return raw_metrics(0.3, 0.9)

    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.real.train_epoch", fake_train)
    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.real.validate", fake_validation)
    config = suggest(optuna.create_study(sampler=sampler()).ask())
    provider = RealProvider(tmp_path, 3, config, {1: state}, {1: {"validation": {
        "rows": 1000, "flp_valid": 700}}})
    epoch = 1
    training, validation = provider(1, epoch)
    assert training["bp_loss"] == 0.2 and validation["flp_loss"] == 0.9
    assert ("training", 3, 1, 1) in calls and ("validation", 3, 1, 1) in calls


def test_cached_preprocessing_rejects_wrong_fold_provenance(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "reports/dataset/canonical/schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_text("fixture")
    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.real.model_inputs", lambda path: ("feature",))
    path = tmp_path / "artifacts/hpo/preprocessing/outer_3_inner_1.json"
    write_json(path, {"provenance": {"outer_study": 2}, "state": {}})
    provenance = {"feature_order_sha256": "feature", "inner_fold_manifest_sha256": "inner",
                  "outer_fold_manifest_sha256": "outer"}
    with pytest.raises(ValueError, match="provenance mismatch"):
        preprocessing_state(tmp_path, 3, 1, provenance, {"train": {"rows": 10, "flp_valid": 8}})


def test_concurrent_batch_path_matches_sequential_pruning(tmp_path) -> None:
    snapshots = []
    for mode in ("sequential", "concurrent"):
        study = optuna.create_study(sampler=sampler(), pruner=pruner())
        record_root = tmp_path / mode

        def objective(trial, mode=mode, record_root=record_root):
            configuration = suggest(trial)
            provider = (SequentialToyProvider(trial.number * 0.5)
                        if mode == "sequential" else BatchToyProvider(trial.number * 0.5))
            return run_resumable_trial(
                trial=trial, outer_study=1, configuration=configuration,
                provider=provider, record_root=record_root, study_name="pruning_fixture",
                provenance={"fixture": True}, tracker=MockTracker(), max_epochs=4)

        study.optimize(objective, n_trials=4, n_jobs=1)
        snapshots.append([{"number": trial.number, "state": trial.state.name,
                           "intermediate": dict(trial.intermediate_values), "value": trial.value}
                          for trial in study.trials])
    assert snapshots[0] == snapshots[1]
    assert any(row["state"] == "PRUNED" for row in snapshots[0])
    for trial in range(4):
        left = tmp_path / "sequential/pruning_fixture" / f"trial_{trial:03d}"
        right = tmp_path / "concurrent/pruning_fixture" / f"trial_{trial:03d}"
        assert [read_json(path)["J_e"] for path in sorted(left.glob("aggregate_*.json"))] == [
            read_json(path)["J_e"] for path in sorted(right.glob("aggregate_*.json"))]


def test_concurrent_batch_pause_resume_has_no_duplicate_records(tmp_path) -> None:
    study = optuna.create_study(sampler=sampler(), pruner=optuna.pruners.NopPruner())
    trial = study.ask()
    configuration = suggest(trial)
    checkpoints = {}
    tracker = MockTracker()
    common = {"trial": trial, "outer_study": 1, "configuration": configuration,
              "record_root": tmp_path, "study_name": "resume_fixture",
              "provenance": {"fixture": True}, "max_epochs": 4}
    result = run_resumable_trial(
        **common, provider=BatchToyProvider(0.0, checkpoints), tracker=tracker,
        pause_after_epoch=2)
    assert result is None and study.trials[0].state.name == "RUNNING"
    run_id = read_json(tmp_path / "resume_fixture/trial_000/wandb_link.json")["run_id"]
    resumed_tracker = MockTracker()
    result = run_resumable_trial(
        **common, provider=BatchToyProvider(0.0, checkpoints), tracker=resumed_tracker)
    assert result is not None
    study.tell(trial, result)
    local = tmp_path / "resume_fixture/trial_000"
    assert len(list(local.glob("fold_*_epoch_*.json"))) == 20
    assert len(list(local.glob("aggregate_epoch_*.json"))) == 4
    assert study.trials[0].number == 0 and study.trials[0].state.name == "COMPLETE"
    assert run_id == tracker.run_id == resumed_tracker.run_id
    assert len(resumed_tracker.events) == 12
