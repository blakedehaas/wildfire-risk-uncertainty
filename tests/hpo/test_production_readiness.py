"""Bounded GradNorm study controls without executing training."""

from pathlib import Path

import optuna
import pytest

from wildfire_risk_uncertainty.hpo.config import TPE_STARTUP_TRIALS, sampler
from wildfire_risk_uncertainty.hpo.real import run_real_hpo
from wildfire_risk_uncertainty.hpo.status import status
from wildfire_risk_uncertainty.hpo.storage import open_study, study_name


def test_tpe_startup_and_no_pruning_study_status(tmp_path: Path) -> None:
    assert TPE_STARTUP_TRIALS == 10
    assert sampler()._n_startup_trials == 10
    assert status(tmp_path, 3, gradnorm=True)["state"] == "not_started"
    name = study_name(3, gradnorm=True)
    database = tmp_path / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
    study = open_study(database, name, {"kind": "gradnorm_hpo", "trial_limit": 20})
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    trial = study.ask()
    current = status(tmp_path, 3, gradnorm=True)
    assert current["trials"] == 1
    assert current["trial_limit"] == 20
    assert current["active_trial"]["trial_number"] == trial.number
    assert current["active_trial"]["last_common_epoch"] == 0
    study.tell(trial, 0.1)
    assert status(tmp_path, 3, gradnorm=True)["trial_states"]["COMPLETE"] == 1
    for _ in range(19):
        study.tell(study.ask(), 0.1)
    assert status(tmp_path, 3, gradnorm=True)["state"] == "budget_reached"


@pytest.mark.parametrize("limit", [None, 0, 10, 31])
def test_invalid_trial_limit_fails_before_data_access(tmp_path: Path, limit: int | None) -> None:
    with pytest.raises(ValueError, match="11..30-trial"):
        run_real_hpo(tmp_path, 1, cache_dir=tmp_path / "missing-cache",
                     gradnorm_objective_reviewed=True, trial_limit=limit)
    assert not (tmp_path / "artifacts").exists()
