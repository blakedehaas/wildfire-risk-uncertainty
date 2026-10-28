"""Arithmetic checks for the retrospective budget and selection analysis."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_07d_compute_reduction.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("analysis_07d", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_budget_uses_successful_completion_order_and_preserves_phase():
    rows = [
        {"number": 0, "successful_order": 1, "sampler_phase": "startup", "objective": .4},
        {"number": 2, "successful_order": 2, "sampler_phase": "startup", "objective": .3},
        {"number": 3, "successful_order": 3, "sampler_phase": "TPE", "objective": .5},
    ]
    result = analysis.budget_trajectory(rows)
    assert [r["best_trial"] for r in result] == [0, 0, 3]
    assert [r["gap_to_final_best"] for r in result] == pytest.approx([.1, .1, 0])
    assert result[-1]["last_phase"] == "TPE"
    with pytest.raises(ValueError, match="Nonconsecutive"):
        analysis.budget_trajectory(rows[:1] + rows[2:])


def test_subset_budget_exact_combinations_and_regret():
    rows = [{"objective": value} for value in (.1, .2, .3)]
    result = analysis.subset_budget(rows, (1, 2, 3))
    assert [r["subset_count"] for r in result] == [3, 3, 1]
    assert [r["median_regret"] for r in result] == pytest.approx([.1, 0, 0])
    assert [r["winner_included_fraction"] for r in result] == pytest.approx([1 / 3, 2 / 3, 1])


def test_rank_regret_is_evaluated_on_full_objective():
    final = {1: .1, 2: .3, 3: .2}
    screen = {1: .4, 2: .2, 3: .3}
    result = analysis.rank_comparison(screen, final)
    assert result["selected_trial"] == 1
    assert result["final_winner"] == 2
    assert result["regret"] == pytest.approx(.2)
    assert result["top_agreement"] == 0
    assert result["spearman"] == pytest.approx(-1)
    with pytest.raises(ValueError, match="same candidates"):
        analysis.rank_comparison({1: .1}, final)
