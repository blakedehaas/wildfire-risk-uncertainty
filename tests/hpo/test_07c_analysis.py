"""Focused checks for the read-only pilot analysis arithmetic."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_07c_outer3_hpo.py"
spec = importlib.util.spec_from_file_location("analysis_07c", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fold(k, bp_ce, flp_ce):
    return {"inner_fold": k, "validation": {"bp_loss_mean": bp_ce, "flp_loss_mean": flp_ce},
            "validation_log_skill": {"bp": 1 - bp_ce / 2, "flp": 1 - flp_ce / 4,
                                     "bp_reference_ce": 2, "flp_reference_ce": 4},
            "training_wall_seconds": 10, "validation_wall_seconds": 2}


def test_frozen_equal_task_equal_fold_arithmetic():
    folds = [fold(k, 1, k) for k in range(1, 6)]
    result = analysis.summarize_folds(folds)
    assert result["objective"] == pytest.approx(.375)
    assert result["bp_skill"] == pytest.approx(.5)
    assert result["flp_skill"] == pytest.approx(.25)
    assert result["worst_fold"] == pytest.approx(.125)
    assert result["train_seconds"] == 50


def test_rejects_incomplete_and_inconsistent_records():
    folds = [fold(k, 1, k) for k in range(1, 6)]
    with pytest.raises(ValueError, match="five distinct"):
        analysis.summarize_folds(folds[:-1])
    folds[0]["validation_log_skill"]["bp"] = 0.2
    with pytest.raises(ValueError, match="disagrees"):
        analysis.summarize_folds(folds)
