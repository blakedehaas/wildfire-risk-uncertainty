import numpy as np
import pytest

from wildfire_risk_uncertainty.analysis.phase12 import (
    aggregate,
    categorical_entropy,
    disagreement,
    high_error_mask,
    quintile_rows,
    rejection_rows,
    stable_descending,
    total_variation,
    train_constant,
    validate_distribution,
)


def test_tv_disagreement_and_entropy():
    target = np.array([[1., 0, 0, 0, 0, 0], [.5, .5, 0, 0, 0, 0]])
    pred = np.array([[.5, .5, 0, 0, 0, 0], [.5, .5, 0, 0, 0, 0]])
    assert np.allclose(total_variation(pred, target), [.5, 0])
    assert np.allclose(disagreement(np.full((2, 6), .01)), [.06, .06])
    assert np.allclose(categorical_entropy(pred), np.log(2))
    with pytest.raises(ValueError):
        validate_distribution(np.array([[.2] * 6]))
    with pytest.raises(ValueError):
        disagreement(np.array([[-.1] * 6]))


def test_train_baseline_uses_only_valid_eligible_targets():
    target = np.eye(6)[:4]
    total, n = train_constant(target, np.array([True, True, False, True]),
                              np.array([True, False, True, True]))
    assert n == 2
    assert np.array_equal(total / n, [.5, 0, 0, .5, 0, 0])


def test_exact_ties_and_rejection():
    ids = np.array([30, 10, 20, 40, 50, 60, 70, 80, 90, 100])
    score = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    assert np.array_equal(stable_descending(score, ids)[:3], [1, 2, 0])
    error = np.array([.1, .9, .2, .1, .1, .1, .1, .1, .1, .1])
    high = high_error_mask(error, ids)
    assert np.flatnonzero(high).tolist() == [1]
    rows = rejection_rows(1, "disagreement", score, error, high, ids)
    assert rows[2]["rejected_n"] == 1
    assert rows[2]["high_error_recall"] == 1
    random = rejection_rows(1, "random", None, error, high)
    assert random[3]["high_error_recall"] == .2


def test_quintile_and_equal_vs_cell_weighted_aggregation():
    score = np.arange(10, dtype=float)
    error = score / 10
    groups = quintile_rows(1, score, error, np.arange(10))
    assert [g["n"] for g in groups] == [2] * 5
    assert [g["mean_tv"] for g in groups] == pytest.approx([.05, .25, .45, .65, .85])
    rows = []
    for study, n, value in [(1, 10, .2), (2, 20, .4), (3, 10, .2), (4, 10, .2), (5, 10, .2)]:
        for method in ("disagreement", "entropy", "random"):
            for fraction in (0, .05, .10, .20, .50):
                rows.append({"study": study, "method": method, "rejected_fraction": fraction,
                             "study_n": n, "rejected_n": round(n * fraction),
                             "high_error_n": round(n * .1),
                             "remaining_mean_tv": value, "high_error_recall": value})
    result = aggregate(rows)
    equal = next(r for r in result if r["aggregation"] == "equal_study" and r["method"] == "random" and r["rejected_fraction"] == .2)
    pooled = next(r for r in result if r["aggregation"] == "pooled_cell_weighted" and r["method"] == "random" and r["rejected_fraction"] == .2)
    assert equal["remaining_mean_tv"] == pytest.approx(.24)
    assert pooled["remaining_mean_tv"] == pytest.approx(4 / 15)
