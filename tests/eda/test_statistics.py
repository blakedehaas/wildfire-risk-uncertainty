"""Small synthetic numerical oracles; never scan full real rasters here."""

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from wildfire_risk_uncertainty.eda import dillon
from wildfire_risk_uncertainty.eda.statistics import Accumulator


def make_sources(root, different=False):
    folder = root / dillon.DILLON_DIR
    folder.mkdir(parents=True)
    data = np.tile(np.array([-9999, 0, .1, .25, .5, .9, 1, 1.1, -.1, np.nan], dtype='float32'), (35, 7))
    arrays = []
    for i, name in enumerate(dillon.FILENAMES):
        values = data.copy()
        if different and i == 1:
            values[0, 1] = -9999
            values[0, 0] = .4
        with rasterio.open(folder / name, 'w', driver='GTiff', height=values.shape[0],
                           width=values.shape[1], count=1, dtype='float32', nodata=-9999,
                           crs='EPSG:5070', transform=from_origin(0, 9450, 270, 270)) as src:
            src.write(values, 1)
        arrays.append(values)
    return arrays


def test_streaming_moments_are_stable():
    values = 1e9 + np.arange(100, dtype=np.float64) / 100
    acc = Accumulator()
    for block in np.array_split(values, 13):
        acc.update(block)
    row = acc.summary(len(values), values)
    assert row['mean'] == pytest.approx(values.mean(), abs=1e-6)
    assert row['std'] == pytest.approx(values.std(ddof=0), abs=1e-6)


@pytest.mark.parametrize('different', [False, True])
def test_exact_statistics_and_masks(tmp_path, different):
    arrays = make_sources(tmp_path, different)
    result = dillon.analyze(tmp_path, requested=10000, window_size=32)
    masks = [(x != -9999) & np.isfinite(x) for x in arrays]
    for i, (x, valid, row) in enumerate(zip(arrays, masks, result['summaries'], strict=True)):
        values = x[valid].astype('float64')
        assert row['valid_pixels'] == valid.sum()
        assert row['invalid_pixels'] == (~valid).sum()
        assert row['mean'] == pytest.approx(values.mean(), abs=1e-12)
        assert row['std'] == pytest.approx(values.std(), abs=1e-12)
        assert row['min'] == values.min()
        assert row['max'] == values.max()
        assert row['below_0'] == (values < 0).sum()
        assert row['above_1'] == (values > 1).sum()
        assert row['outside_fraction'] == pytest.approx(((values < 0) | (values > 1)).mean())
        assert row['q50'] == np.quantile(values, .5)
        assert row['unmasked_nonfinite'] == np.isnan(x).sum()
        assert result['accumulators'][i].linear.sum() == ((values >= 0) & (values <= 1)).sum()
    all_valid = np.logical_and.reduce(masks)
    any_valid = np.logical_or.reduce(masks)
    assert result['patterns'][127] == all_valid.sum()
    assert result['patterns'][1:].sum() == any_valid.sum()
    assert result['patterns'][1:127].sum() == (2 if different else 0)
    assert result['coverage_all'].sum() == all_valid.sum()
    assert result['coverage_any'].sum() == any_valid.sum()
    assert result['coverage_total'].sum() == arrays[0].size


def test_sampling_is_deterministic_across_window_sizes(tmp_path):
    arrays = make_sources(tmp_path)
    first = dillon.analyze(tmp_path, requested=300, window_size=32)
    second = dillon.analyze(tmp_path, requested=300, window_size=64)
    rows, cols, stride = dillon.sample_grid(*arrays[0].shape, requested=300)
    assert stride == 3
    expected = arrays[0][np.ix_(rows, cols)]
    expected = np.where((expected != -9999) & np.isfinite(expected), expected, np.nan)
    np.testing.assert_equal(first['samples'][0], expected)
    np.testing.assert_equal(first['samples'], second['samples'])
    assert [s['q95'] for s in first['summaries']] == [s['q95'] for s in second['summaries']]
    assert first['provenance']['seed'] is None
    assert first['provenance']['actual_sample_locations'] <= 300


def test_products_deterministic_and_sources_unchanged(tmp_path):
    make_sources(tmp_path, different=True)
    source_files = list((tmp_path / dillon.DILLON_DIR).iterdir())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result = dillon.analyze(tmp_path, requested=300, window_size=32)
    dillon.write_products(result, tmp_path)
    base = tmp_path / 'reports/eda/raw/dillon'
    outputs = {p: p.read_bytes() for p in base.rglob('*') if p.is_file()}
    assert (base / 'tables/distribution_summary.csv') in outputs
    assert (base / 'figures/bp_map.png') in outputs
    assert len(list((base / 'figures').glob('*.png'))) == 19
    assert all(p.stat().st_size > 1000 for p in (base / 'figures').glob('*.png'))
    result2 = dillon.analyze(tmp_path, requested=300, window_size=32)
    dillon.write_products(result2, tmp_path)
    assert outputs == {p: p.read_bytes() for p in outputs}
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    assert not plt.get_fignums()
    assert plt.get_backend().lower() == 'agg'
