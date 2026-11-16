"""Numerical and evaluation-integrity checks for the paired uncertainty comparison."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wildfire_risk_uncertainty.analysis import phase11a as p
from wildfire_risk_uncertainty.training.cache import sha256


def test_jensen_shannon_matches_mean_kl_and_handles_endpoints():
    members = np.array([
        [0, 1, 0.001, 0.2, 0.5, 0, 1e-15],
        [0, 1, 0.002, 0.3, 0.5, 0, 2e-15],
        [0, 1, 0.003, 0.4, 0.5, 1, 3e-15],
        [0, 1, 0.004, 0.5, 0.5, 1, 4e-15],
        [0, 1, 0.005, 0.6, 0.5, 0, 5e-15],
    ], dtype=np.float64)
    js = p.jensen_shannon(members)
    np.testing.assert_allclose(js, p.mean_kl(members), atol=1e-14)
    assert np.all((js >= 0) & (js <= np.log(2)))
    np.testing.assert_allclose(js[[0, 1, 4]], 0, atol=1e-14)
    assert js[5] > 0
    assert p.bernoulli_entropy(np.array([.5]))[0] > js[4]


def test_jensen_shannon_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="finite"):
        p.jensen_shannon(np.full((5, 3), np.nan))
    with pytest.raises(ValueError, match="five"):
        p.jensen_shannon(np.full((4, 3), .5))


def test_selection_is_deterministic_unique_and_seeded_by_study():
    a = p.selected_rows(10_000, 1, n=1000)
    b = p.selected_rows(10_000, 1, n=1000)
    c = p.selected_rows(10_000, 2, n=1000)
    np.testing.assert_array_equal(a, b)
    assert np.unique(a).size == 1000
    assert not np.array_equal(a, c)
    assert (np.bincount(a % 100, minlength=100) > 0).all()


def test_train_exclusion_and_id_target_alignment():
    cache = SimpleNamespace(arrays={
        "cell_id": np.array([10, 20, 30]),
        "bp": np.array([.1, .2, .3], dtype=np.float32),
        "inner_fold": np.array([[0, 1], [0, 0], [2, 0]], dtype=np.uint8),
    })
    plan = pa.table({"global_row": [0, 1], "cell_id": [10, 20],
                     "bp_target": np.array([.1, .2], dtype=np.float32)})
    np.testing.assert_array_equal(p._inputs(cache, plan, 1), [0, 1])
    bad_id = pa.table({"global_row": [0, 1], "cell_id": [10, 99],
                       "bp_target": np.array([.1, .2], dtype=np.float32)})
    with pytest.raises(ValueError, match="Cell IDs"):
        p._inputs(cache, bad_id, 1)
    with pytest.raises(ValueError, match="TRAIN"):
        p._inputs(cache, plan.take(pa.array([1, 0])), 2)


def test_rejection_uses_same_high_error_group_and_cell_id_ties():
    ids = np.array([4, 3, 2, 1])
    error = np.array([1., 2., 3., 4.])
    score = np.array([.5, .5, .5, .5])
    high = np.array([False, False, False, True])
    mae, recall, mass = p.rejection(error, score, ids, high, .25)
    assert (mae, recall, mass) == (2., 1., .4)
    random = p.rejection(error, None, ids, high, .25)
    assert random == (2.5, .25, .25)
    assert p._top_indices(score, ids, .25).tolist() == [3]


def test_conditional_summary_weights_deciles_by_cell_count():
    rows = [
        {"study": 1, "rejected_fraction": .1, "method": "variance", "n": 10,
         "baseline_mae": 1., "remaining_mae": .5, "high_error_recall": .2},
        {"study": 1, "rejected_fraction": .1, "method": "variance", "n": 30,
         "baseline_mae": 3., "remaining_mae": 2., "high_error_recall": .4},
    ]
    # Complete the other study/fraction/method combinations with identical rows.
    expanded = []
    for study in range(1, 6):
        for fraction in p.CONDITIONAL_FRACTIONS:
            for method in ("variance", "jensen_shannon", "random"):
                expanded.extend({**r, "study": study, "rejected_fraction": fraction,
                                 "method": method} for r in rows)
    output = p._conditional_summary(expanded)
    assert len(output) == 30
    assert output[0]["baseline_mae"] == 2.5
    assert output[0]["remaining_mae"] == 1.625
    assert output[0]["mean_within_decile_high_error_recall"] == pytest.approx(.3)


@pytest.mark.skipif(not (p.OUT / "predictions/complete.json").exists(),
                    reason="Phase 11A generated artifacts are not available")
def test_completed_artifacts_reproduce_moments_ids_and_provenance():
    marker = json.loads((p.OUT / "sample/complete.json").read_text())
    assert marker["outcome_independent"] is True
    assert marker["phase9_manifest_sha256"] == sha256(p.RUN / "evaluation_manifest.json")
    cache = p.cache_for()
    for study in range(1, 6):
        plan = p.read_plan(study)
        source, _ = p.source_table(study)
        choice = p.selected_rows(len(source), study)
        np.testing.assert_array_equal(plan["cell_id"].to_numpy(), source["cell_id"].to_numpy()[choice])
        assert np.unique(source["column"].to_numpy()[choice] % 256).size == 256
        rows = p._inputs(cache, plan, study)
        path = p.prediction_path(study)
        record = json.loads(path.with_suffix(".json").read_text())
        assert record["sha256"] == sha256(path)
        assert record["sample_sha256"] == sha256(p.plan_path(study))
        table = pq.read_table(path)
        np.testing.assert_array_equal(table["cell_id"].to_numpy(), plan["cell_id"].to_numpy())
        np.testing.assert_array_equal(table["global_row"].to_numpy(), rows)
        members = np.stack([table[f"member_{member}_bp"].to_numpy() for member in range(5)])
        np.testing.assert_allclose(members.mean(axis=0), table["phase9_bp_mean"].to_numpy(), atol=1e-8)
        np.testing.assert_allclose(members.var(axis=0), table["phase9_bp_between_variance"].to_numpy(), atol=1e-11)
        np.testing.assert_allclose(p.jensen_shannon(members), table["bp_jensen_shannon"].to_numpy(), atol=1e-10)
        assert record["max_phase9_mean_difference"] < 1e-8
        assert record["max_phase9_variance_difference"] < 1e-11
