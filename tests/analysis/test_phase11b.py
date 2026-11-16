"""Small synthetic checks for Phase 11B conformal arithmetic and provenance."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wildfire_risk_uncertainty.analysis import phase11b as p
from wildfire_risk_uncertainty.training.cache import sha256


def _data(ids, rows, target, prediction):
    n = len(ids)
    return {"cell_id": np.array(ids, dtype=np.int64),
            "global_row": np.array(rows, dtype=np.int64),
            "outer_study": np.ones(n, dtype=np.uint8),
            "bp_target": np.array(target, dtype=np.float32),
            "prediction": np.array(prediction, dtype=np.float32)}


def test_conformal_quantile_exact_rank_ties_and_infinity():
    scores = np.array([.4, .1, .1, .3, .2, .5, .7, .6, .8])
    # n=9: rank ceil(10 * .8) = 8, and ceil(10 * .9) = 9.
    assert p.conformal_quantile(scores, .8) == .7
    assert p.conformal_quantile(scores, .9) == .8
    assert p.conformal_quantile(np.array([.1]), .95) == np.inf
    assert p.conformal_quantile(np.array([.1, .1, .2, .2]), .6) == .2
    with pytest.raises(ValueError, match="nonempty"):
        p.conformal_quantile(np.array([]), .9)
    with pytest.raises(ValueError, match="nonnegative"):
        p.conformal_quantile(np.array([-.1]), .9)


def test_clipping_coverage_and_interval_score():
    lower, upper = p.intervals(np.array([.05, .5, .95]), .1)
    np.testing.assert_allclose(lower, [0, .4, .85])
    np.testing.assert_allclose(upper, [.15, .6, 1])
    metrics = p.interval_metrics(np.array([0, .4, .8]), lower, upper, .9)
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["mean_width"] == pytest.approx((.15 + .2 + .15) / 3)
    assert metrics["interval_score"] == pytest.approx((.15 + .2 + .15 + 20 * .05) / 3)
    lo, hi = p.intervals(np.array([.1, .9]), np.inf)
    np.testing.assert_array_equal(lo, [0, 0])
    np.testing.assert_array_equal(hi, [1, 1])
    assert p.interval_metrics(np.array([0, 1]), lo, hi, .95) == {
        "coverage": 1.0, "mean_width": 1.0, "interval_score": 1.0}


def test_split_is_deterministic_disjoint_and_outcome_independent():
    a, b = p.split_indices(101, 1)
    aa, bb = p.split_indices(101, 1)
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(b, bb)
    assert len(a) == 50 and len(b) == 51
    np.testing.assert_array_equal(np.sort(np.r_[a, b]), np.arange(101))
    assert not np.array_equal(a, p.split_indices(101, 2)[0])


def test_study_evaluation_uses_one_evaluation_half_and_calibration_only_scores():
    test = _data(range(10), range(10), np.linspace(0, .9, 10), np.full(10, .1))
    dead = _data(range(10, 20), range(10, 20), np.full(10, .8), np.full(10, .1))
    rows, strata, meta = p.evaluate_study(test, dead, 1, seed=7)
    assert len(rows) == 6
    assert meta["n_test_calibration"] == meta["n_test_evaluation"] == 5
    q_test = next(r["conformal_quantile"] for r in rows if r["calibration"] == "within_test"
                  and r["nominal_coverage"] == .8)
    q_dead = next(r["conformal_quantile"] for r in rows if r["calibration"] == "dead_zone_transfer"
                  and r["nominal_coverage"] == .8)
    assert q_test != q_dead
    assert {r["n_evaluation"] for r in rows} == {5}
    assert {r["bp_stratum"] for r in strata} == {"[0.01, 1]"}
    _, eval_idx = p.split_indices(10, 1, seed=7)
    changed = {**test, "bp_target": test["bp_target"].copy()}
    changed["bp_target"][eval_idx] = 0.99
    altered, _, _ = p.evaluate_study(changed, dead, 1, seed=7)
    assert [r["conformal_quantile"] for r in rows] == [r["conformal_quantile"] for r in altered]
    assert any(a["coverage"] != b["coverage"] for a, b in zip(rows, altered))


def test_study_membership_alignment_and_train_exclusion():
    test = _data([11, 12], [0, 1], [.1, .2], [.1, .2])
    dead = _data([13, 14], [2, 3], [.3, .4], [.3, .4])
    cache = SimpleNamespace(arrays={
        "cell_id": np.array([11, 12, 13, 14]),
        "bp": np.array([.1, .2, .3, .4], dtype=np.float32),
        "inner_fold": np.zeros((4, 5), dtype=np.uint8),
    })
    p.validate_pair(test, dead, 1, cache)
    with pytest.raises(ValueError, match="cell IDs"):
        p.validate_pair(test, {**dead, "cell_id": np.array([13, 99])}, 1, cache)
    cache.arrays["inner_fold"][2, 0] = 1
    with pytest.raises(ValueError, match="TRAIN"):
        p.validate_pair(test, dead, 1, cache)
    cache.arrays["inner_fold"][2, 0] = 0
    with pytest.raises(ValueError, match="overlap"):
        p.validate_pair(test, {**dead, "cell_id": np.array([11, 14]), "global_row": np.array([0, 3]),
                                    "bp_target": np.array([.1, .4], dtype=np.float32)}, 1, cache)


def test_equal_study_summary_uses_study_not_pixel_weights():
    rows = [{"row_type": "study", "study": 1, "calibration": "within_test", "nominal_coverage": .9,
             "n_evaluation": 10, "coverage": .8, "mean_width": .1, "interval_score": .2},
            {"row_type": "study", "study": 2, "calibration": "within_test", "nominal_coverage": .9,
             "n_evaluation": 100, "coverage": 1., "mean_width": .3, "interval_score": .4}]
    output = p.equal_study_rows(rows)
    assert len(output) == 1
    assert output[0]["coverage"] == pytest.approx(.9)
    assert output[0]["mean_width"] == pytest.approx(.2)
    assert output[0]["n_studies"] == 2


def test_paired_transfer_differences():
    rows = [{"study": 1, "nominal_coverage": .9, "calibration": "within_test",
             "n_evaluation": 5, "coverage": .8, "mean_width": .1, "interval_score": .2},
            {"study": 1, "nominal_coverage": .9, "calibration": "dead_zone_transfer",
             "n_evaluation": 5, "coverage": .6, "mean_width": .3, "interval_score": .4}]
    p.add_transfer_deltas(rows)
    assert rows[1]["delta_coverage"] == pytest.approx(-.2)
    assert rows[1]["delta_mean_width"] == pytest.approx(.2)
    assert rows[1]["delta_interval_score"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="counts differ"):
        p.add_transfer_deltas([rows[0], {**rows[1], "n_evaluation": 4}])


def test_source_hash_and_row_count_validation(tmp_path):
    path = tmp_path / "predictions.parquet"
    pq.write_table(pa.table({"cell_id": [10, 20], "bp_target": [.1, .2]}), path)
    sidecar = {"study": 1, "n": 2, "sha256": sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(sidecar))
    data, record = p._read_source(path, ("cell_id", "bp_target"), {"study": 1})
    np.testing.assert_array_equal(data["cell_id"], [10, 20])
    assert record == sidecar
    with pytest.raises(ValueError, match="provenance"):
        p._read_source(path, ("cell_id",), {"study": 2})
    path.with_suffix(".json").write_text(json.dumps({**sidecar, "n": 3}))
    with pytest.raises(ValueError, match="row count"):
        p._read_source(path, ("cell_id",), {"study": 1})
