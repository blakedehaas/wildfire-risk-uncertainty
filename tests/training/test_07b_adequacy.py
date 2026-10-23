"""Check the frozen diagnostic's probabilistic-score arithmetic."""

import math
import runpy
from pathlib import Path

import pytest

_module = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                             "scripts/diagnose_07b_bp_adequacy.py"))
empty = _module["empty"]
summarize = _module["summarize"]


def test_soft_target_scores_and_climatological_skill() -> None:
    targets = (0.0, 0.2, 0.8)
    predictions = (0.1, 0.3, 0.7)
    climate = 0.25
    sums = empty()
    for y, p in zip(targets, predictions, strict=True):
        sums["n"] += 1
        sums["target"] += y
        sums["prediction"] += p
        sums["target2"] += y * y
        sums["prediction2"] += p * p
        sums["cross"] += y * p
        sums["absolute"] += abs(p - y)
        sums["square"] += (p - y) ** 2
        sums["ce"] += -y * math.log(p) - (1 - y) * math.log1p(-p)
        sums["entropy"] += -sum(q * math.log(q) for q in (y, 1 - y) if q)
        sums["brier_base"] += (climate - y) ** 2
    result = summarize(sums, climate)
    assert result["rows"] == 3
    assert result["bias"] == pytest.approx(sum(predictions) / 3 - sum(targets) / 3)
    assert result["brier"] == pytest.approx(sum((p - y) ** 2 for y, p in zip(
        targets, predictions, strict=True)) / 3)
    assert result["excess_log_loss"] >= 0
    assert result["log_skill"] == pytest.approx(
        result["climatology_excess_log_loss"] - result["excess_log_loss"])
