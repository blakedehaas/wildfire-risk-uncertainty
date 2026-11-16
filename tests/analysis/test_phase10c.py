import numpy as np
import pyarrow as pa

from wildfire_risk_uncertainty.analysis.phase10c import (
    sample_mask,
    standardize_bp_strata,
    summarize,
    table_for,
)


def test_hash_sample_is_partition_invariant_and_study_specific():
    ids = np.arange(100_000, dtype=np.int64)
    whole = sample_mask(ids, 3)
    split = np.concatenate([sample_mask(ids[:31_217], 3), sample_mask(ids[31_217:], 3)])
    assert np.array_equal(whole, split)
    assert .008 < whole.mean() < .012
    assert not np.array_equal(whole, sample_mask(ids, 4))


def test_inference_table_preserves_mask_and_coordinates():
    plan = pa.table({"cell_id": [10, 11], "global_row": [100, 101], "row": [2, 3],
        "column": [4, 5], "x_5070": [100., 200.], "y_5070": [300., 400.]})
    flp = np.array([[1., 0, 0, 0, 0, 0], [np.nan] * 6], dtype=np.float32)
    predicted = np.full((2, 6), 1 / 6, dtype=np.float32)
    table = table_for(plan, 2, np.array([.01, 0.], dtype=np.float32), flp,
        np.array([True, False]), np.array([.02, .001]), np.array([.1, .2]),
        predicted, np.array([270., 540.]))
    assert table.num_rows == 2
    assert table["outer_study"].to_pylist() == [2, 2]
    assert table["cell_id"].to_pylist() == [10, 11]
    assert table["flp_valid"].to_pylist() == [True, False]
    assert np.isnan(table["flp1_target"].to_numpy()[1])
    assert table["nearest_train_distance_m"].to_pylist() == [270., 540.]


def test_common_bp_mix_uses_same_weights_for_dead_and_test():
    rows = []
    for zone, counts, errors in (("DEAD", (3, 1), (.1, .2)), ("TEST", (1, 3), (.3, .4))):
        for n, error in zip(counts, errors, strict=True):
            rows.append({"study": 1, "zone": zone, "n": n, "bp_mae": error, "mean_disagreement": error / 10})
    # The function expects all five studies; replicate the two strata for each.
    all_rows = [{**row, "study": study} for study in range(1, 6) for row in rows]
    summary = standardize_bp_strata(all_rows)
    assert np.isclose(summary[0]["common_mix_bp_mae"], .15)
    assert np.isclose(summary[1]["common_mix_bp_mae"], .35)


def test_distance_and_bp_strata_preserve_all_sampled_rows():
    n = 200
    plan = pa.table({"cell_id": np.arange(n), "global_row": np.arange(n),
        "row": np.arange(n), "column": np.zeros(n, dtype=int),
        "x_5070": np.arange(n, dtype=float), "y_5070": np.zeros(n)})
    target = np.r_[np.linspace(.001, .0015, n // 2), np.linspace(.01, .0105, n // 2)]
    pred = np.r_[np.full(n // 2, .002), np.full(n // 2, .004)]
    variance = np.linspace(0, .001, n)
    flp = np.tile([1., 0, 0, 0, 0, 0], (n, 1))
    distance = np.r_[np.full(n // 2, 10_000.), np.full(n // 2, 175_000.)]
    table = table_for(plan, 1, target, flp, np.ones(n, dtype=bool), pred,
        variance, flp, distance)
    summary, bins, bp_strata, _coverage, distance_bp_strata = summarize(1, "dead", table)
    assert summary["n"] == n
    assert sum(row["n"] for row in bins) == n
    assert sum(row["n"] for row in bp_strata) == n
    assert sum(row["n"] for row in distance_bp_strata) == n
    assert np.isclose(summary["bp_mae"], .0035)
    assert [row["n"] for row in bins if row["n"]] == [100, 100]
