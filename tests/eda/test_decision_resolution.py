"""Synthetic decision oracles; no source scans."""

import numpy as np

from wildfire_risk_uncertainty.eda.decision_resolution import (
    classify_target,
    missingness_summary,
    pattern,
    target_rows,
    write_csv,
)


def test_target_partition_and_tolerance():
    bp = np.array([0, 1, 1, 1, 1, np.nan, np.nan], dtype=float)
    good = np.array([True] * 5 + [False, False])
    flp = np.zeros((6, 7), dtype=float)
    flp[0, 1] = 0.5
    flp[1, 1] = 0.5 + 0.9e-6
    flp[0, 2] = 0.5
    flp[1, 2] = 0.5 + 1.1e-6
    flp[0, 3] = 1
    flp[0, 4] = 1
    flp[0, 5] = 1
    valid = np.ones((6, 7), dtype=bool)
    valid[0, 3] = False
    valid[0, 6] = False
    states = classify_target(bp, good, flp, valid)
    assert sum(int(v.sum()) for v in states.values()) == 7
    assert states['retained_bp_zero_flp_present'][0]
    assert states['retained_bp_positive_flp_valid'][1]
    assert states['retained_bp_positive_flp_present_non_normalized'][2]
    assert states['retained_bp_positive_flp_missing_invalid'][3]
    assert states['excluded_bp_invalid_flp_present'][5]
    assert states['excluded_bp_invalid_flp_invalid'][6]
    counts = {key: int(value.sum()) for key, value in states.items()}
    counts['total_raster_cells'] = 7
    rows = {row['category']: row for row in target_rows(counts)}
    assert rows['bp_valid_canonical']['count'] == 5
    assert rows['bp_valid_masked_from_flp']['count'] == 3
    assert rows['strict_flp_valid']['count'] == 2
    assert target_rows(counts) == target_rows(counts)


def test_repeated_block_patterns():
    assert pattern([[2, 3, 1, 1], [2, 3, 1, 1]], [0, 1]) == 'repeated_unchanged'
    assert pattern([[0, 0, 0, 0], [2, 3, 1, 1]], [1, 0]) == 'zeroed_public_private_positive'
    assert pattern([[1, 1, 1, 0], [2, 2, 2, 0]], [0, 0]) == 'varied_nonzero'


def test_missingness_summary():
    values = [0, np.nan, 4, np.nan]
    groups = [0, 0, 1, 1]
    assert missingness_summary(values, groups) == {"0": (1, 1), "1": (1, 1)}


def test_deterministic_csv(tmp_path):
    rows = [{"category": "a", "count": 1}, {"category": "b", "count": 2}]
    path = tmp_path / "out.csv"
    write_csv(path, rows)
    first = path.read_bytes()
    write_csv(path, rows)
    assert path.read_bytes() == first
