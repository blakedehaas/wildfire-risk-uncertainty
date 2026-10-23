"""Numerical failure and continued-study controls without real training."""

import queue
from pathlib import Path

import optuna
import pytest

from wildfire_risk_uncertainty.hpo.fold_workers import (
    FoldWorkerError,
    PersistentFoldProvider,
)
from wildfire_risk_uncertainty.hpo.real import pending_trial
from wildfire_risk_uncertainty.hpo.storage import (
    BP_CE_NUMERICAL_REVISION,
    open_study,
)


def test_primary_worker_error_and_abort_do_not_wait_for_closed_response() -> None:
    class DeadWorker:
        exitcode = 1

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    results = queue.Queue()
    results.put({"status": "error", "exception_type": "FloatingPointError",
                 "exception_message": "Invalid BP CE rate loss; bp_rate_ce=0.0",
                 "traceback": "original traceback"})
    worker = DeadWorker()
    with pytest.raises(FoldWorkerError, match="bp_rate_ce=0.0") as caught:
        PersistentFoldProvider._expect(results, "epoch", worker)
    assert caught.value.exception_type == "FloatingPointError"
    provider = object.__new__(PersistentFoldProvider)
    provider._closed = False
    provider._workers = [(worker, queue.Queue(), results)]
    provider.close(abort=True)
    assert provider._closed


def test_failed_trial_consumes_slot_but_does_not_block_new_suggestion(tmp_path: Path) -> None:
    study = open_study(tmp_path / "study.sqlite3", "continue_fixture",
                       {"kind": "gradnorm_hpo"})
    first = study.ask()
    study.tell(first, 0.1)
    failed = study.ask()
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    assert pending_trial(study.trials, 20) is None
    assert study.ask().number == 2
    with pytest.raises(ValueError, match="lifetime trial limit"):
        pending_trial(study.trials, 2)


def test_finished_trial_is_immutable_and_registered_source_revision_is_explicit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "study.sqlite3"
    frozen = {"kind": "gradnorm_hpo", "source_git_sha": "original", "cache": "frozen"}
    study = open_study(database, "revision_fixture", frozen)
    trial = study.ask()
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    with pytest.raises(optuna.exceptions.UpdateFinishedTrialError):
        study._storage.set_trial_state_values(trial._trial_id, optuna.trial.TrialState.RUNNING)
    revised = {**frozen, "source_git_sha": "corrected",
               "bp_ce_numerical_revision": BP_CE_NUMERICAL_REVISION}
    with pytest.raises(ValueError, match="incompatible"):
        open_study(database, "revision_fixture", revised, allow_registered_revision=True)
    study.set_user_attr("numerical_revision", {
        "source_git_sha": "corrected", "original_source_git_sha": "original",
        "marker": BP_CE_NUMERICAL_REVISION})
    reopened = open_study(database, "revision_fixture", revised,
                          allow_registered_revision=True)
    assert reopened.trials[0].state == optuna.trial.TrialState.FAIL
    assert reopened.user_attrs["frozen_metadata"] == frozen
    with pytest.raises(ValueError, match="incompatible"):
        open_study(database, "revision_fixture", {**revised, "cache": "changed"},
                   allow_registered_revision=True)
