"""Unit tests plus metadata-only integration tests requiring local source data."""

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from wildfire_risk_uncertainty.eda import cli, dillon, wui

ROOT = Path(__file__).resolve().parents[2]


def test_expected_sources_exist():
    for name in dillon.FILENAMES:
        assert (ROOT / dillon.DILLON_DIR / name).is_file(), name
    assert (ROOT / wui.GDB).is_dir()
    assert (ROOT / wui.METADATA).is_file()


def test_actual_raster_alignment():
    rows = dillon.inventory(ROOT)
    assert len(rows) == 7
    assert all(dillon.alignment(rows).values())


@pytest.mark.parametrize('key,value,failed', [
    ('width', 2, {'shape', 'grid'}),
    ('transform', [270, 0, 1, 0, -270, 0], {'transform', 'grid'}),
    ('crs_wkt', 'different', {'crs', 'grid'}),
    ('bounds', [0, 0, 1, 1], {'bounds'}),
    ('nodata', [None], {'nodata_convention'}),
])
def test_alignment_detects_mismatch(key, value, failed):
    row = {'width': 1, 'height': 1, 'transform': [1, 0, 0, 0, -1, 1],
           'crs_wkt': 'test', 'bounds': [0, 0, 1, 2], 'nodata': [-9999]}
    other = deepcopy(row)
    other[key] = value
    assert {k for k, v in dillon.alignment([row, other]).items() if not v} == failed


def test_empty_alignment_rejected():
    with pytest.raises(ValueError):
        dillon.alignment([])


def test_raster_reader_preserves_bytes(tmp_path):
    directory = tmp_path / dillon.DILLON_DIR
    directory.mkdir(parents=True)
    for name in dillon.FILENAMES:
        with rasterio.open(directory / name, 'w', driver='GTiff', width=2, height=3,
                           count=1, dtype='float32', crs='EPSG:5070', nodata=-9999,
                           transform=from_origin(0, 810, 270, 270)) as src:
            src.write(np.ones((1, 3, 2), dtype='float32'))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    rows = dillon.inventory(tmp_path)
    assert rows[0]['width'] == 2 and rows[0]['height'] == 3
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_schema_metadata_matches_and_dates():
    metadata = wui.metadata(ROOT)
    layers, fields = wui.inventory(ROOT, metadata['attributes'])
    assert layers and all(layer['capabilities']['fast_feature_count'] for layer in layers)
    assert all(field['metadata_label'] for field in fields)
    assert next(f for f in fields if f['field'] == 'fips')['metadata_label'] == 'FIPS'
    assert {y for f in fields for y in f['schema_years'].split(';') if y} == {
        '1990', '1992', '2000', '2001', '2010', '2011', '2019', '2020',
    }
    assert '1911' not in {y for m in metadata['year_mentions'] for y in m['years']}


def test_output_paths(tmp_path):
    assert cli.output_paths(tmp_path) == {
        name: tmp_path / 'reports/eda/raw' / name / 'tables' for name in ('dillon', 'wui')
    }


def test_actual_sources_unchanged_and_outputs_deterministic():
    before = cli.source_state(ROOT)
    cli.run_inventory(ROOT)
    report = ROOT / 'reports/eda/raw'
    files = [*report.rglob('*.json'), *report.rglob('*.csv')]
    first = {p: p.read_bytes() for p in files}
    cli.run_inventory(ROOT)
    assert first == {p: p.read_bytes() for p in files}
    assert cli.source_state(ROOT) == before


def test_output_symlink_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'source_state', lambda root: {})
    monkeypatch.setattr(dillon, 'inventory', lambda root: [])
    monkeypatch.setattr(wui, 'metadata', lambda root: {'attributes': {}})
    monkeypatch.setattr(wui, 'inventory', lambda root, definitions: ([], []))
    (tmp_path / 'reports').symlink_to(ROOT / 'data', target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        cli.run_inventory(tmp_path)


@pytest.mark.parametrize('suffix', ['.tif', '.tfw', '.tif.xml', '.tif.aux.xml', '.tif.ovr'])
def test_source_guard_covers_sidecars(tmp_path, suffix):
    directory = tmp_path / dillon.DILLON_DIR
    directory.mkdir(parents=True)
    metadata = tmp_path / wui.METADATA
    metadata.parent.mkdir(parents=True)
    metadata.write_text('<metadata/>')
    sidecar = directory / ('CONUS_BP' + suffix)
    before = cli.source_state(tmp_path)
    sidecar.write_bytes(b'original')
    added = cli.source_state(tmp_path)
    assert added != before
    assert sidecar.relative_to(tmp_path).as_posix() in added
    sidecar.write_bytes(b'modified content')
    assert cli.source_state(tmp_path) != added
    sidecar.unlink()
    assert cli.source_state(tmp_path) == before
