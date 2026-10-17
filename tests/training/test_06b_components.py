"""Focused checks of frozen training math and benchmark isolation."""

import csv
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from wildfire_risk_uncertainty.benchmark.training_throughput import (
    cost_tables,
    packed_batches,
    write_json,
)
from wildfire_risk_uncertainty.training.data import (
    median_train_fold,
    model_inputs,
    training_batches,
)
from wildfire_risk_uncertainty.training.losses import (
    equal_task_loss,
    loss_sums,
    streaming_batch_loss,
)
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.preprocessing import (
    DENSITIES,
    fit_preprocessing,
    fixed_transform,
    stable_json_bytes,
    transform,
)


def test_soft_losses_mask_and_equal_task_denominators():
    bp = torch.tensor([0.2, 0.8, 0.0])
    bp_logits = torch.zeros(3)
    flp_logits = torch.zeros(3, 6)
    q = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1], [float("nan")] * 6,
                      [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]])
    valid = torch.tensor([True, False, True])
    bp_sum, flp_sum, bp_n, flp_n = loss_sums(bp_logits, flp_logits, bp, q, valid)
    assert bp_n == 3 and flp_n == 2
    assert torch.isfinite(flp_sum)
    assert bp_sum.item() == pytest.approx(3 * np.log(2))
    assert flp_sum.item() == pytest.approx(2 * np.log(6))
    assert equal_task_loss(bp_sum, flp_sum, bp_n, flp_n).item() == pytest.approx(
        0.5 * (np.log(2) + np.log(6))
    )
    # The first valid FLP is low-BP; it retains the same conditional weight.
    changed_bp = torch.tensor([0.99, 0.8, 0.0])
    _, flp_sum2, _, _ = loss_sums(bp_logits, flp_logits, changed_bp, q, valid)
    assert flp_sum2.item() == pytest.approx(flp_sum.item())


def test_streaming_batch_objective_reconstructs_global_separate_means():
    # Unequal FLP support across two minibatches must not alter the task weight.
    a = streaming_batch_loss(torch.tensor(2.0), torch.tensor(1.0), 2, 4, 1)
    b = streaming_batch_loss(torch.tensor(4.0), torch.tensor(0.0), 2, 4, 1)
    assert ((a + b) / 2).item() == pytest.approx(0.5 * (6 / 4 + 1 / 1))


def test_model_shapes_and_normalized_probabilities():
    model = RiskMLP(33, depth=2, width=128)
    bp, flp = model(torch.zeros(4, 33))
    p, q = model.probabilities(torch.zeros(4, 33))
    assert bp.shape == (4,) and flp.shape == (4, 6)
    assert p.shape == (4,) and q.shape == (4, 6)
    assert torch.allclose(q.sum(dim=1), torch.ones(4))


def test_only_four_densities_log1p_and_training_only_statistics():
    inputs = (*DENSITIES, "VEG2019PC", "wui_failure")
    train = np.array([[0, 1, 3, 7, 0.5, 0], [3, np.nan, 8, 15, 0.7, 1]], dtype=np.float32)
    heldout = np.array([[100, 100, 100, 100, 0.9, 1]], dtype=np.float32)
    fixed = fixed_transform(train, inputs)
    assert fixed[0, 0] == 0
    assert fixed[0, 1] == pytest.approx(np.log1p(1))
    assert fixed[0, 4] == pytest.approx(0.5)
    batches = [(train, np.zeros(2), np.zeros((2, 6)), np.ones(2, dtype=bool))]
    state = fit_preprocessing(iter(batches), inputs, "schema-hash")
    assert state.missing_counts[1] == 1
    assert state.means[1] == pytest.approx(np.log1p(1))
    before = state.digest()
    transformed = transform(heldout, state)
    assert transformed.shape == (1, 6)
    assert transformed[0, -1] == 1.0  # binary flag is not standardized
    assert transformed[0, 4] == pytest.approx((0.9 - 0.6) / 0.1)
    imputed = transform(train, state)
    assert imputed[1, 1] == pytest.approx(0.0)  # missing density uses TRAIN mean
    assert state.digest() == before
    assert fit_preprocessing(iter(batches), inputs, "schema-hash").digest() == before
    assert stable_json_bytes(state.serializable()) == stable_json_bytes(state.serializable())


def test_median_fold_selection_uses_counts(tmp_path):
    path = tmp_path / "folds.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["fold", "train_rows"])
        writer.writeheader()
        writer.writerows([{"fold": i, "train_rows": n} for i, n in
                          ((1, 90), (2, 70), (3, 80), (4, 60), (5, 50))])
    assert median_train_fold(path) == (2, 70)


def test_training_reader_returns_no_outer_test_rows(tmp_path):
    canonical = tmp_path / "canonical"; canonical.mkdir()
    folds = tmp_path / "folds"; folds.mkdir()
    name = "tile.parquet"
    pq.write_table(pa.table({"x": [1.0, 999.0, 3.0], "BP": [0.1, 999.0, 0.3],
                             **{f"FLP{k}": [1 / 6, 999.0, 1 / 6] for k in range(1, 7)},
                             "flp_target_valid": [True, False, True]}), canonical / name)
    pq.write_table(pa.table({"train_eligible_fold_3": [True, False, True]}), folds / name)
    parts = list(training_batches(canonical, folds, 3, ("x",)))
    assert len(parts) == 1
    features, bp, _, _ = parts[0]
    assert features[:, 0].tolist() == [1.0, 3.0]
    assert bp.tolist() == pytest.approx([0.1, 0.3])
    packed = list(packed_batches(iter(parts), 2))
    assert len(packed) == 1 and packed[0][0].shape == (2, 1)


def test_schema_inputs_are_exact_33():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    inputs = model_inputs(root / "reports/dataset/canonical/schema.json")
    assert len(inputs) == 33 and len(set(inputs)) == 33
    schema = json.loads((root / "reports/dataset/canonical/schema.json").read_text())
    roles = {column["name"]: column["role"] for column in schema["columns"]}
    assert sum(roles[name] == "predictor" for name in inputs) == 25
    assert sum(roles[name] == "diagnostic" for name in inputs) == 8
    assert "x_5070" not in inputs and "flp_target_valid" not in inputs


def test_benchmark_configuration_serializes_stably(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(first, {"seed": 42, "batch_sizes": [1024, 2048], "precision": "BF16"})
    write_json(second, {"precision": "BF16", "batch_sizes": [1024, 2048], "seed": 42})
    assert first.read_bytes() == second.read_bytes()


def test_nested_hpo_costs_include_five_independent_outer_studies():
    budgets, scaling = cost_tables(80_000_000, training_rate=4_000_000, validation_rate=5_000_000)
    for scenario in ("no_pruning", "idealized_ASHA_rf4"):
        one = next(r for r in budgets if r["scenario"] == scenario
                   and r.get("scope") == "per_outer_study" and r["trials_per_outer_study"] == 30)
        five = next(r for r in budgets if r["scenario"] == scenario
                    and r.get("scope") == "complete_nested_hpo" and r["trials_per_outer_study"] == 30)
        assert five["gpu_seconds"] == pytest.approx(5 * one["gpu_seconds"])
        assert five["total_planned_trials"] == 150
        eight_gpu = next(r for r in scaling if r["scenario"] == scenario
                         and r["scope"] == "complete_nested_hpo"
                         and r["trials_per_outer_study"] == 30 and r["gpus"] == 8)
        assert eight_gpu["whole_study_schedule_wall_hours"] == pytest.approx(one["gpu_hours"])
