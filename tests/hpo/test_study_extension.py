"""Cheap persistent-study controls; no model training."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pytest

from wildfire_risk_uncertainty.hpo import operations
from wildfire_risk_uncertainty.hpo.config import suggest
from wildfire_risk_uncertainty.hpo.status import status
from wildfire_risk_uncertainty.hpo.storage import (
    extend_trial_limit,
    initialize_trial_limit,
    open_study,
    restore_sampler_state,
    save_sampler_state,
    study_controller_lock,
)
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes


def test_reopen_extend_preserves_history_and_rng(tmp_path: Path) -> None:
    database = tmp_path / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
    name = "wildfire_outer_3_gradnorm_hpo_v3"
    frozen = {"kind": "gradnorm_hpo", "search_space": "frozen-fixture"}
    study = open_study(database, name, frozen)
    initialize_trial_limit(study, 20)
    for _ in range(20):
        trial = study.ask()
        config = suggest(trial)
        trial.set_user_attr("suggested_configuration", config.serializable())
        save_sampler_state(study, trial.number)
        trial.report(float(trial.number), step=4)
        study.tell(trial, float(trial.number))
    before = [(trial.number, trial.params, trial.value, trial.intermediate_values)
              for trial in study.trials]
    rng = study.user_attrs["sampler_state"]

    reopened = open_study(database, name, frozen)
    restore_sampler_state(reopened)
    assert reopened.user_attrs["sampler_state"] == rng
    assert reopened.sampler._rng.rng.get_state()[2] == rng["tpe"]["position"]
    assert np.array_equal(reopened.sampler._rng.rng.get_state()[1], rng["tpe"]["keys"])
    assert np.array_equal(reopened.sampler._random_sampler._rng.rng.get_state()[1],
                          rng["startup"]["keys"])
    change = extend_trial_limit(reopened, 30)
    assert change["maximum_new_suggestions"] == 10
    assert status(tmp_path, 3, gradnorm=True)["trial_limit"] == 30
    assert reopened.user_attrs["trial_budget"]["initial"] == 20
    assert reopened.user_attrs["trial_budget"]["history"] == [
        {"from": None, "to": 20}, {"from": 20, "to": 30}]
    with pytest.raises(ValueError, match="persisted lifetime cap"):
        initialize_trial_limit(reopened, 20)
    with pytest.raises(ValueError, match="incompatible"):
        open_study(database, name, {"kind": "gradnorm_hpo", "search_space": "changed"})
    with pytest.raises(ValueError, match="raise"):
        extend_trial_limit(reopened, 30)
    for _ in range(change["maximum_new_suggestions"]):
        trial = reopened.ask()
        suggest(trial)
        save_sampler_state(reopened, trial.number)
        reopened.tell(trial, float(trial.number))
    assert [trial.number for trial in reopened.trials] == list(range(30))
    assert [(trial.number, trial.params, trial.value, trial.intermediate_values)
            for trial in reopened.trials[:20]] == before
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in reopened.trials)
    progress = operations.summary(tmp_path, 3)
    assert progress["trial_states"]["COMPLETE"] == 30
    assert progress["best_objective"] == 29.0
    assert progress["remaining_trial_slots"] == 0
    second = extend_trial_limit(reopened, 40)
    assert second["maximum_new_suggestions"] == 10
    assert status(tmp_path, 3, gradnorm=True)["pending_trial_slots"] == 10


def test_study_controller_lock_rejects_second_owner(tmp_path: Path) -> None:
    database = tmp_path / "study.sqlite3"
    with study_controller_lock(database), pytest.raises(
        RuntimeError, match="Another controller"
    ), study_controller_lock(database):
        pass


def test_extension_checks_identity_and_active_controller(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    database = tmp_path / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
    manifest = {"fixture": True}
    digest = hashlib.sha256(stable_json_bytes(manifest)).hexdigest()
    frozen = {"kind": "gradnorm_hpo", "source_git_sha": "fixture",
              "cache_manifest_sha256": digest,
              "preprocessing_hashes_by_fold": {"1": "state"}}
    study = open_study(database, "wildfire_outer_3_gradnorm_hpo_v3", frozen)
    initialize_trial_limit(study, 20)
    monkeypatch.setattr(operations.subprocess, "check_output", lambda *args, **kwargs: b"")
    monkeypatch.setattr(operations, "metadata", lambda *args, **kwargs: {
        "kind": "gradnorm_hpo", "source_git_sha": "fixture"})
    monkeypatch.setattr(operations, "HpoCache", lambda *args, **kwargs:
                        SimpleNamespace(manifest=manifest))
    monkeypatch.setattr(operations, "load_states", lambda *args, **kwargs: {
        1: SimpleNamespace(digest=lambda: "state")})
    monkeypatch.setattr(operations, "_active_detached_controller", lambda *args: True)
    with pytest.raises(RuntimeError, match="active"):
        operations.extend(tmp_path, 3, 30)
    assert status(tmp_path, 3, gradnorm=True)["trial_limit"] == 20
    monkeypatch.setattr(operations, "_active_detached_controller", lambda *args: False)
    result = operations.extend(tmp_path, 3, 30)
    assert result["study"] == study.study_name
    assert result["trial_limit"] == 30
    assert status(tmp_path, 3, gradnorm=True)["trial_limit"] == 30
