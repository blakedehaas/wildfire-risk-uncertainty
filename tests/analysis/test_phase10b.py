import numpy as np

from wildfire_risk_uncertainty.analysis.phase10b import (
    geographic_concentration,
    quantile_labels,
    summarize_group,
)


def test_distance_quantiles_have_exact_support_with_ties():
    values = np.array([300., 300., 300., 300., 500., 500., 500., 700., 700., 700.])
    labels = quantile_labels(values)
    assert np.array_equal(np.bincount(labels), np.array([2, 2, 2, 2, 2]))
    assert all(values[labels == k].max() <= values[labels == k+1].min() for k in range(4))


def test_group_summary_uses_selected_cells_only():
    data = {"distance": np.array([220000., 300000., 400000.]),
        "variance": np.array([.1, .2, .3]), "pred": np.array([.01, .02, .03]),
        "bp": np.array([.02, .04, .06])}
    row = summarize_group(2, "Q1", data, np.array([True, False, True]), np.array([.01, .02, .03]))
    assert row["n"] == 2
    assert np.isclose(row["distance_median_km"], 310)
    assert np.isclose(row["mean_disagreement"], .2)
    assert np.isclose(row["mean_absolute_error"], .02)


def test_geographic_concentration_keeps_failure_and_cell_denominators_separate():
    bins = [{"study": study, "all_n": 1000, "failure_n": 10 if k == 0 else 1,
        "failure_rate": .01 if k == 0 else .001} for study in range(1, 6) for k in range(10)]
    result = geographic_concentration(bins)
    assert len(result) == 5
    assert result[0]["top_bin_n"] == 1
    assert np.isclose(result[0]["fraction_of_failures_in_top_bins"], 10 / 19)
    assert np.isclose(result[0]["fraction_of_cells_in_top_bins"], .1)
