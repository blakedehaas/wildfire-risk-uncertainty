"""CPU-only smoke tests for the detached Pareto result path."""

import pytest

from wildfire_risk_uncertainty.hpo.records import write_json
from wildfire_risk_uncertainty.training.phase7a_run import (
    _check_metrics,
    analyze,
    tasks_from_proposals,
)


def test_task_plan_and_weighted_metric_arithmetic():
    proposals = {str(outer): {"candidates": [0.1, 0.3, 0.4, 0.5, 0.7, 0.9]}
                 for outer in range(1, 6)}
    tasks = tasks_from_proposals(proposals)
    assert len(tasks) == 150
    assert tasks[0] == {"id": 1, "outer_study": 1, "alpha": 0.1, "inner_fold": 1}
    assert tasks[-1] == {"id": 150, "outer_study": 5, "alpha": 0.9, "inner_fold": 5}
    _check_metrics({"bp_loss_sum": 4, "bp_rows": 2, "bp_loss": 2,
                    "flp_loss_sum": 9, "flp_valid_rows": 3, "flp_loss": 3,
                    "combined_loss": 2.7}, 0.3)
    with pytest.raises(ValueError):
        _check_metrics({"bp_loss_sum": 4, "bp_rows": 2, "bp_loss": 2,
                        "flp_loss_sum": 9, "flp_valid_rows": 3, "flp_loss": 3,
                        "combined_loss": 2.5}, 0.3)


def test_complete_synthetic_results_produce_reproducible_pareto_outputs(tmp_path):
    proposals = {str(outer): {"candidates": [0.1, 0.3, 0.4, 0.5, 0.7, 0.9]}
                 for outer in range(1, 6)}
    tasks = tasks_from_proposals(proposals)
    write_json(tmp_path / "config.json", {"kind": "pareto", "max_epochs": 100,
                                          "tasks": tasks, "proposals": proposals})
    with pytest.raises(ValueError, match="150"):
        analyze(tmp_path)
    for task in tasks:
        alpha = task["alpha"]
        write_json(tmp_path / f"fold_{task['id']}_epoch_100.json",
                   {"validation": {"bp_rows": 100 + task["inner_fold"],
                                    "flp_valid_rows": 60 + task["inner_fold"],
                                    "bp_excess_log_loss": alpha**2,
                                    "flp_excess_log_loss": (1 - alpha)**2,
                                    "bp_loss": 0.5 + alpha**2,
                                    "flp_loss": 0.8 + (1 - alpha)**2}})
    result = analyze(tmp_path)
    assert len(result["selections"]) == 5
    assert (tmp_path / "pareto_table.csv").exists()
    assert len(list((tmp_path / "figures").glob("outer_*_pareto.png"))) == 5
    assert result == analyze(tmp_path)
