import numpy as np

from wildfire_risk_uncertainty.analysis.phase10a import (
    correlation_sums,
    entropy,
    exact_tail_mask,
    pearson_from_sums,
    rejection_metrics,
)


def test_rejection_metrics_finds_known_largest_errors():
    error = np.array([.01, .02, .03, .50, .90], dtype=np.float32)
    high = np.array([False, False, False, True, True])
    score = np.array([0., 1., 2., 3., 4.])
    mae, recall, mass = rejection_metrics(score, error, high, .4)
    assert np.isclose(mae, .02)
    assert recall == 1
    assert np.isclose(mass, 1.4 / 1.46)


def test_entropy_turns_at_half_and_correlation_sums_merge():
    p = np.array([.1, .2, .5, .8], dtype=np.float32)
    h = entropy(p)
    assert h[0] < h[1] < h[2]
    assert np.isclose(h[1], h[3])
    x = np.array([1., 2., 3., 4.])
    y = np.array([2., 4., 1., 3.])
    merged = correlation_sums(x[:2], y[:2]) + correlation_sums(x[2:], y[2:])
    assert np.isclose(pearson_from_sums(merged), np.corrcoef(x, y)[0, 1])


def test_exact_tail_masks_keep_requested_support_under_ties():
    values = np.array([0., 0., 0., 0., 1., 1., 1., 1., 1., 1.])
    low = exact_tail_mask(values, .2, largest=False)
    high = exact_tail_mask(values, .1, largest=True)
    assert low.sum() == 2
    assert high.sum() == 1
    assert (values[low] == 0).all()
    assert (values[high] == 1).all()
