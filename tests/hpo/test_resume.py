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


def test_three_completed_folds_survive_interruption(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    interrupted = tmp_path / "interrupted"
    for action, root, code in (("baseline", baseline, 0), ("pause", interrupted, 0),
                               ("interrupt_after_three", interrupted, 37)):
        result = subprocess.run([sys.executable, "-m", "wildfire_risk_uncertainty.hpo.resume_fixture",
                                 action, str(root)], capture_output=True, text=True, check=False)
        assert result.returncode == code, result.stderr
    local = interrupted / "records/restart_fixture/trial_000"
    saved = [local / f"fold_{fold}_epoch_002.json" for fold in (1, 2, 3)]
    original = [path.read_bytes() for path in saved]
    result = subprocess.run([sys.executable, "-m", "wildfire_risk_uncertainty.hpo.resume_fixture",
                             "resume", str(interrupted)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert [path.read_bytes() for path in saved] == original
    assert json.loads((baseline / "result.json").read_text()) == json.loads(
        (interrupted / "result.json").read_text())


def test_corrupt_checkpoint_stops_resume(tmp_path) -> None:
    root = tmp_path / "interrupted"
    result = subprocess.run([sys.executable, "-m", "wildfire_risk_uncertainty.hpo.resume_fixture",
                             "pause", str(root)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    checkpoint = root / "records/restart_fixture/trial_000/checkpoints/fold_1_epoch_001.json"
    checkpoint.write_text("{}")
    result = subprocess.run([sys.executable, "-m", "wildfire_risk_uncertainty.hpo.resume_fixture",
                             "resume", str(root)], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "checkpoint hash mismatch" in result.stderr.lower()
