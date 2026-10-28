"""Five-success Phase 7E protocol checks without training or target access."""

import json
from pathlib import Path

import optuna
import pytest

from wildfire_risk_uncertainty.hpo import operations
from wildfire_risk_uncertainty.hpo.config import sampler, suggest
from wildfire_risk_uncertainty.hpo.phase7e import (
    ATTEMPT_CAP,
    MANIFEST,
    OUTERS,
    SUCCESS_TARGET,
    next_action,
    select_best,
    validate_manifest,
)
from wildfire_risk_uncertainty.hpo.real import run_real_hpo
from wildfire_risk_uncertainty.hpo.status import status
from wildfire_risk_uncertainty.hpo.storage import (
    open_study,
    restore_sampler_state,
    save_sampler_state,
    study_name,
)
from wildfire_risk_uncertainty.hpo.wandb_tracking import OnlineTracker


def test_success_budget_replaces_failures_and_stops_exactly_at_five(tmp_path: Path) -> None:
    assert OUTERS == (1, 2, 4, 5)
    assert SUCCESS_TARGET == 5 and ATTEMPT_CAP == 25
    assert sampler()._n_startup_trials == 10
    db = tmp_path / "artifacts/hpo/outer_1_gradnorm_v3.sqlite3"
    study = open_study(db, study_name(1, gradnorm=True), {"kind": "gradnorm_hpo"})
    study.set_user_attr("success_budget", {"target": 5, "attempt_cap": 25, "protocol_sha256": "fixture"})
    for number in range(7):
        assert next_action(study.trials)[0] == "suggest"
        trial = study.ask()
        suggest(trial)
        save_sampler_state(study, trial.number)
        assert next_action(study.trials) == ("resume", number)
        study.tell(trial, state=optuna.trial.TrialState.FAIL) if number in (1, 4) else study.tell(trial, .1 + number / 100)
    assert next_action(study.trials) == ("complete", None)
    assert len(study.trials) == 7
    progress = status(tmp_path, 1, gradnorm=True)
    assert progress["state"] == "budget_reached"
    assert progress["trial_states"]["COMPLETE"] == 5
    assert progress["trial_states"]["FAIL"] == 2
    assert progress["remaining_successes"] == 0
    assert select_best(study.trials).number == 6
    with pytest.raises(ValueError, match="Exactly 5"):
        select_best(study.trials[:6])


def test_failure_replacement_sampling_survives_resume(tmp_path: Path) -> None:
    metadata = {"kind": "gradnorm_hpo", "outer_study": 2}
    db = tmp_path / "outer_2.sqlite3"
    name = study_name(2, gradnorm=True)
    study = open_study(db, name, metadata)
    first = study.ask()
    suggest(first)
    save_sampler_state(study, first.number)
    study.tell(first, state=optuna.trial.TrialState.FAIL)
    second = study.ask()
    expected = suggest(second)
    save_sampler_state(study, second.number)
    study.tell(second, .2)

    replay_db = tmp_path / "replay.sqlite3"
    replay = open_study(replay_db, name, metadata)
    same_first = replay.ask()
    suggest(same_first)
    save_sampler_state(replay, same_first.number)
    replay.tell(same_first, state=optuna.trial.TrialState.FAIL)
    reopened = open_study(replay_db, name, metadata)
    restore_sampler_state(reopened)
    replacement = reopened.ask()
    assert suggest(replacement) == expected
    assert next_action(reopened.trials) == ("resume", replacement.number)


def test_attempt_cap_and_selection_tie() -> None:
    study = optuna.create_study(direction="maximize")
    for number in range(25):
        trial = study.ask()
        study.tell(trial, state=optuna.trial.TrialState.FAIL if number >= 2 else optuna.trial.TrialState.COMPLETE,
                   values=None if number >= 2 else .3)
    assert next_action(study.trials) == ("exhausted", None)
    assert select_best(study.trials, count=2).number == 0


def test_manifest_isolation_and_provenance(tmp_path: Path, monkeypatch) -> None:
    from wildfire_risk_uncertainty.hpo import phase7e

    root = Path.cwd()
    frozen = phase7e.expected_manifest(root)
    assert len({v["sqlite"] for v in frozen["studies"].values()}) == 4
    assert len({v["record_root"] for v in frozen["studies"].values()}) == 4
    assert all(v["outer_partition"] == int(k) for k, v in frozen["studies"].items())
    assert all(v["inner_folds"] == [1, 2, 3, 4, 5] for v in frozen["studies"].values())
    assert "3" not in frozen["studies"]
    (tmp_path / MANIFEST).parent.mkdir(parents=True)
    (tmp_path / MANIFEST).write_text(json.dumps(frozen))
    monkeypatch.setattr(phase7e, "expected_manifest", lambda _root: frozen)
    assert validate_manifest(tmp_path, 1) == frozen
    with pytest.raises(ValueError, match="excludes"):
        validate_manifest(tmp_path, 3)
    broken = {**frozen, "success_target": 3}
    (tmp_path / MANIFEST).write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="fingerprints"):
        validate_manifest(tmp_path, 1)


def test_outer_3_and_wrong_budget_block_before_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protected"):
        run_real_hpo(tmp_path, 3, cache_dir=tmp_path / "none",
                     gradnorm_objective_reviewed=True, trial_limit=5)
    with pytest.raises(ValueError, match="five successful"):
        run_real_hpo(tmp_path, 1, cache_dir=tmp_path / "none",
                     gradnorm_objective_reviewed=True, trial_limit=20)
    assert not (tmp_path / "artifacts").exists()
    (tmp_path / MANIFEST).parent.mkdir(parents=True)
    (tmp_path / MANIFEST).write_text("{}")
    with pytest.raises(RuntimeError, match="frozen"):
        operations.extend(tmp_path, 3, 30)


def test_wandb_identity_is_study_specific_and_random_search(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeRun:
        id = "fixture"
        url = "https://example.invalid/fixture"

        def define_metric(self, *args, **kwargs):
            pass

    def fake_init(**kwargs):
        captured.update(kwargs)
        return FakeRun()

    monkeypatch.setattr("wildfire_risk_uncertainty.hpo.wandb_tracking.wandb.init", fake_init)
    identity = study_name(1, gradnorm=True)
    OnlineTracker(outer_study=1, trial_number=0, configuration={"depth": 2},
                  provenance={"study_name": identity, "phase7e_protocol_sha256": "fixture",
                              "pruner": {"name": "NopPruner"}}, local_dir=tmp_path,
                  checkpoint="checkpoint-07b")
    assert captured["group"] == identity
    assert captured["name"] == f"{identity}-trial-000"
    assert "random-search" in captured["tags"]
    assert "tpe" not in captured["tags"]
