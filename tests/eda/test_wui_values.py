"""Small WUI value oracles; no full FileGDB scans."""

import numpy as np
import pytest

from wildfire_risk_uncertainty.eda.wui_values import (
    NumericSummary,
    batch_offsets,
    flag_codes,
    missing,
    transition_counts,
)


def test_flag_counts_and_transitions():
    left = np.array([0, 0, 1, 2, None, 9], dtype=object)
    right = np.array([0, 1, 2, 1, 0, None], dtype=object)
    assert flag_codes(left).tolist() == [0, 0, 1, 2, 3, 4]
    matrix = transition_counts(left, right)
    assert matrix.sum() == 6
    assert matrix[0, 0] == matrix[0, 1] == matrix[1, 2] == 1
    assert matrix[2, 1] == matrix[3, 0] == matrix[4, 3] == 1


def test_numeric_missing_domain_and_quantiles():
    summary = NumericSummary((0, 100))
    summary.update([0, 50, np.nan], 0, stride=2)
    summary.update([100, 110, -1], 3, stride=2)
    row = summary.row('percent')
    assert row['rows'] == 6 and row['missing'] == 1
    assert row['min'] == -1 and row['max'] == 110
    assert row['zero_count'] == 1 and row['out_of_domain_count'] == 2
    assert row['sample_valid'] == 2
    assert row['q50'] == pytest.approx(55)
    assert missing(np.array([None, 'x'], dtype=object)).tolist() == [True, False]


def test_deterministic_batches():
    assert list(batch_offsets(7, 3)) == [(0, 3), (3, 3), (6, 1)]
    assert list(batch_offsets(7, 3)) == list(batch_offsets(7, 3))
    with pytest.raises(ValueError):
        list(batch_offsets(5, 0))


def test_duplicate_id_accounting():
    import sqlite3

    from wildfire_risk_uncertainty.eda.wui_values import summarize_ids

    connection = sqlite3.connect(':memory:')
    connection.execute('CREATE TABLE ids (id TEXT NOT NULL)')
    connection.executemany('INSERT INTO ids VALUES (?)', [('a',), ('b',), ('b',),
                                                           ('c',), ('c',), ('c',)])
    result, frequencies = summarize_ids(connection, 7, 1)
    assert result == {'total_rows': 7, 'missing_BLK20': 1,
                      'unique_nonmissing_BLK20': 3, 'IDs_once': 1,
                      'IDs_multiple': 2, 'rows_in_duplicated_IDs': 5,
                      'maximum_multiplicity': 3}
    assert dict(frequencies) == {1: 1, 2: 1, 3: 1}


def test_numeric_category_keys_across_null_upcast():
    from wildfire_risk_uncertainty.eda.wui_values import category_key

    assert [category_key('PUBFLAG', x) for x in (0, 0.0, 1, 1.0, np.nan)] == [
        '0', '0', '1', '1', None,
    ]


def test_wkb_bounds_and_deterministic_fid_sampling(monkeypatch, tmp_path):
    import struct

    from wildfire_risk_uncertainty.eda import wui_values

    ring = [(0., 0.), (2., 0.), (2., 4.), (0., 4.), (0., 0.)]
    wkb = b'\x01' + struct.pack('<II', 6, 1)
    wkb += b'\x01' + struct.pack('<III', 3, 1, len(ring))
    wkb += b''.join(struct.pack('<dd', *point) for point in ring)
    assert wui_values.wkb_bounds(wkb) == (0, 0, 2, 4)
    seen = []

    def fake_read(source, *, fids, **kwargs):
        seen.extend(fids.tolist())
        return ({'fields': ['WUIFLAG2020']}, fids, [wkb] * len(fids),
                [np.zeros(len(fids), dtype=np.int32)])

    monkeypatch.setattr(wui_values, 'read', fake_read)
    first = wui_values.spatial_sample(tmp_path, total=10, requested=4, batch=2)
    second = wui_values.spatial_sample(tmp_path, total=10, requested=4, batch=2)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 3)
    assert seen == [1, 4, 7, 10] * 2
