"""Cheap persistent-study controls; no model training."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pytest

from wildfire_risk_uncertainty.hpo import operations
from wildfire_risk_uncertainty.hpo.config import suggest
from wildfire_risk_uncertainty.hpo.status import status
from wildfire_risk_uncertainty.hpo.storage import (
    BP_CE_NUMERICAL_REVISION,
    EXTENSION_REPAIR_MARKER,
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
    database = tmp_path / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
    manifest = {"fixture": True}
    digest = hashlib.sha256(stable_json_bytes(manifest)).hexdigest()
    frozen = {"kind": "gradnorm_hpo", "source_git_sha": "fixture",
              "cache_manifest_sha256": digest,
              "preprocessing_hashes_by_fold": {"1": "state"}}
    study = open_study(database, "wildfire_outer_3_gradnorm_hpo_v3", frozen)
    initialize_trial_limit(study, 20)
    monkeypatch.setattr(operations.subprocess, "check_output", lambda *args, **kwargs: b"")
    monkeypatch.setattr(operations, "_current_provenance", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(operations, "_active_detached_controller", lambda *args: True)
    with pytest.raises(RuntimeError, match="active"):
        operations.extend(tmp_path, 3, 30)
    assert status(tmp_path, 3, gradnorm=True)["trial_limit"] == 20
    monkeypatch.setattr(operations, "_active_detached_controller", lambda *args: False)
    result = operations.extend(tmp_path, 3, 30)
    assert result["study"] == study.study_name
    assert result["trial_limit"] == 30
    assert status(tmp_path, 3, gradnorm=True)["trial_limit"] == 30


def test_extension_after_registered_numerical_and_infrastructure_revisions(
    tmp_path: Path, monkeypatch,
) -> None:
    database = tmp_path / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
    frozen = {"kind": "gradnorm_hpo", "source_git_sha": "original",
              "cache_manifest_sha256": "cache", "inner_fold_constant_references": {"1": "reference"},
              "preprocessing_hashes_by_fold": {"1": "state"}, "runtime_environment": {"gpu": "same"}}
    study = open_study(database, "wildfire_outer_3_gradnorm_hpo_v3", frozen)
    initialize_trial_limit(study, 20)
    first = study.ask()
    suggest(first)
    save_sampler_state(study, first.number)
    first.report(0.05, step=4)
    study.tell(first, 0.1)
    failed = study.ask()
    suggest(failed)
    save_sampler_state(study, failed.number)
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    old_trials = [(trial.number, trial.state, trial.params, trial.value,
                   trial.intermediate_values) for trial in study.trials]
    old_sampler = study.user_attrs["sampler_state"]
    study.set_user_attr("numerical_revision", {
        "source_git_sha": "approved", "original_source_git_sha": "original",
        "marker": BP_CE_NUMERICAL_REVISION})
    current = {**frozen, "source_git_sha": "infrastructure",
               "bp_ce_numerical_revision": BP_CE_NUMERICAL_REVISION}
    monkeypatch.setattr(operations.subprocess, "check_output", lambda *args, **kwargs: b"")
    monkeypatch.setattr(operations, "_active_detached_controller", lambda *args: False)
    monkeypatch.setattr(operations, "_current_provenance", lambda *args: current)
    with pytest.raises(ValueError, match="source/environment"):
        operations.extend(tmp_path, 3, 21)
    study.set_user_attr("extension_repair_revision", {
        "source_git_sha": "infrastructure", "approved_bp_ce_source_git_sha": "approved",
        "marker": EXTENSION_REPAIR_MARKER})
    result = operations.extend(tmp_path, 3, 21)
    assert result["previous_limit"] == 20
    assert result["trial_limit"] == 21
    assert result["existing_trials"] == 2
    assert result["maximum_new_suggestions"] == 19
    reopened = operations._study(tmp_path, 3)
    assert reopened.user_attrs["trial_budget"] == {
        "initial": 20, "current": 21,
        "history": [{"from": None, "to": 20}, {"from": 20, "to": 21}]}
    assert status(tmp_path, 3, gradnorm=True)["trial_limit"] == 21
    assert reopened.user_attrs["sampler_state"] == old_sampler
    assert [(trial.number, trial.state, trial.params, trial.value,
             trial.intermediate_values) for trial in reopened.trials] == old_trials
    assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in reopened.trials) + 19 == 20
    current["runtime_environment"] = {"gpu": "changed"}
    with pytest.raises(ValueError, match="source/environment"):
        operations.extend(tmp_path, 3, 22)
    assert operations._study(tmp_path, 3).user_attrs["trial_budget"]["current"] == 21


def test_extension_repair_registration_checks_lineage_scope_and_history(
    tmp_path: Path, monkeypatch,
) -> None:
    database = tmp_path / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
    frozen = {"kind": "gradnorm_hpo", "source_git_sha": "original", "cache": "frozen"}
    study = open_study(database, "wildfire_outer_3_gradnorm_hpo_v3", frozen)
    initialize_trial_limit(study, 20)
    complete = study.ask()
    suggest(complete)
    save_sampler_state(study, complete.number)
    study.tell(complete, 0.2)
    failed = study.ask()
    suggest(failed)
    save_sampler_state(study, failed.number)
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    study.set_user_attr("numerical_revision", {
        "source_git_sha": operations.APPROVED_BP_CE_SOURCE_SHA,
        "original_source_git_sha": "original", "marker": BP_CE_NUMERICAL_REVISION})
    current = {**frozen, "source_git_sha": "new-sha",
               "bp_ce_numerical_revision": BP_CE_NUMERICAL_REVISION}
    changed = "src/wildfire_risk_uncertainty/hpo/operations.py\n"

    def git_output(args, **kwargs):
        if args[1] == "status":
            return b""
        if args[1] == "diff":
            return changed
        raise AssertionError(args)

    monkeypatch.setattr(operations.subprocess, "check_output", git_output)
    monkeypatch.setattr(operations.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(operations, "_active_detached_controller", lambda *args: False)
    monkeypatch.setattr(operations, "_current_provenance", lambda *args: current)
    before = [(t.number, t.state, t.value) for t in study.trials]
    rng = study.user_attrs["sampler_state"]
    monkeypatch.setattr(operations.subprocess, "check_output", lambda args, **kwargs:
                        b"" if args[1] == "status" else "src/wildfire_risk_uncertainty/training/trainer.py\n")
    with pytest.raises(ValueError, match="unauthorized paths"):
        operations.adopt_extension_repair(tmp_path, 3)
    monkeypatch.setattr(operations.subprocess, "check_output", git_output)
    result = operations.adopt_extension_repair(tmp_path, 3)
    assert result["registered_revision"]["source_git_sha"] == "new-sha"
    reopened = operations._study(tmp_path, 3)
    assert [(t.number, t.state, t.value) for t in reopened.trials] == before
    assert reopened.user_attrs["sampler_state"] == rng
    assert reopened.user_attrs["trial_budget"]["current"] == 20


def test_current_provenance_includes_runner_cache_reference_and_state_fields(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest = {"version": "cache-v1", "files_sha256": {"partition_offsets.npy": "offsets"}}
    monkeypatch.setattr(operations, "metadata", lambda *args, **kwargs: {"source_git_sha": "new"})
    monkeypatch.setattr(operations, "HpoCache", lambda *args, **kwargs:
                        SimpleNamespace(manifest=manifest))
    monkeypatch.setattr(operations, "load_states", lambda *args, **kwargs: {
        1: SimpleNamespace(digest=lambda: "state")})
    monkeypatch.setattr(operations, "fold_references", lambda *args, **kwargs: {
        1: {"bp_train_probability": 0.1}})
    value = operations._current_provenance(tmp_path, 3)
    assert value == {"source_git_sha": "new", "cache_version": "cache-v1",
                     "cache_manifest_sha256": hashlib.sha256(
                         stable_json_bytes(manifest)).hexdigest(),
                     "partition_offsets_sha256": "offsets",
                     "inner_fold_constant_references": {"1": {"bp_train_probability": 0.1}},
                     "preprocessing_hashes_by_fold": {"1": "state"}}
