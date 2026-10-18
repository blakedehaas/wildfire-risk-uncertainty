"""Process-level synthetic active-trial pause, interruption, and continuation."""

import json
import subprocess
import sys


def test_same_trial_pause_mid_epoch_replay_and_trajectory(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    interrupted = tmp_path / "interrupted"

    def phase(action, root, *, returncode=0):
        result = subprocess.run(
            [sys.executable, "-m", "wildfire_risk_uncertainty.hpo.resume_fixture",
             action, str(root)], capture_output=True, text=True, check=False)
        assert result.returncode == returncode, result.stderr

    phase("baseline", baseline)
    phase("pause", interrupted)
    phase("interrupt", interrupted, returncode=37)
    phase("resume", interrupted)
    expected = json.loads((baseline / "result.json").read_text())
    actual = json.loads((interrupted / "result.json").read_text())
    assert actual == expected
    assert actual["trial_number"] == 0 and actual["state"] == "COMPLETE"
    assert actual["wandb_run_id"] == "mock-run"
    assert actual["fold_epoch_record_count"] == 20
    assert set(actual["intermediate_values"]) == {"1", "4"}
