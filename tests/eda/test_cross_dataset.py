"""Synthetic geometry and report oracles; no complete WUI source scan."""

import json
import struct

import numpy as np
import pytest
from rasterio.warp import transform, transform_bounds

from wildfire_risk_uncertainty.eda import cross_dataset as cross


def multipolygon(rings):
    blob = b'\x01' + struct.pack('<II', 6, 1)
    blob += b'\x01' + struct.pack('<II', 3, len(rings))
    for ring in rings:
        blob += struct.pack('<I', len(ring))
        blob += b''.join(struct.pack('<dd', *xy) for xy in ring)
    return blob


def test_crs_transform_and_overlap():
    x, y = transform('EPSG:4269', 'EPSG:5070', [-96], [23])
    assert x[0] == pytest.approx(0, abs=1e-6)
    assert y[0] == pytest.approx(0, abs=1e-6)
    bounds = transform_bounds('EPSG:4269', 'EPSG:5070', -97, 22, -95, 24, densify_pts=128)
    assert bounds[0] < 0 < bounds[2] and bounds[1] < 0 < bounds[3]
    assert cross.extent_overlap((0, 0, 2, 2), (1, 1, 3, 3)) == (1, 1, 2, 2)
    assert cross.extent_overlap((0, 0, 1, 1), (1, 0, 2, 1)) is None


def test_fid_selection_and_projected_polygon_area(monkeypatch):
    assert cross.selected_fids(10, 4).tolist() == [1, 4, 7, 10]
    exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]
    monkeypatch.setattr(cross, 'transform', lambda _a, _b, xs, ys: (xs, ys))
    assert cross.projected_polygon_area(multipolygon([exterior, hole])) == 96
    assert cross.projected_polygon_area(multipolygon([exterior])) == 100


def test_real_projection_area_positive():
    ring = [(-96, 35), (-95.99, 35), (-95.99, 35.01), (-96, 35.01), (-96, 35)]
    area = cross.projected_polygon_area(multipolygon([ring]))
    assert 900_000 < area < 1_100_000


def test_scale_and_variable_types():
    summary = cross.scale_summary([0, cross.CELL_AREA_M2, 4 * cross.CELL_AREA_M2])
    assert (summary['smaller_than_cell'], summary['equal_to_cell'], summary['larger_than_cell']) == (1, 1, 1)
    assert summary['ratio_q50'] == 1
    assert summary['equivalent_side_m_q50'] == 270
    assert summary['equivalent_radius_m_q50'] == pytest.approx((72900 / np.pi) ** .5)
    for kind, fields in cross.VARIABLE_TYPES.items():
        assert all(cross.variable_type(field) == kind for field in fields)
    with pytest.raises(ValueError):
        cross.variable_type('Shape_Area')


def test_sample_geometry_batches_are_deterministic(monkeypatch, tmp_path):
    ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    seen = []

    def fake_read(_source, *, fids, **_kwargs):
        seen.extend(fids.tolist())
        return {}, fids, [multipolygon([ring])] * len(fids), []

    monkeypatch.setattr(cross, 'read', fake_read)
    monkeypatch.setattr(cross, 'transform', lambda _a, _b, xs, ys: (xs, ys))
    one = cross.sample_areas(tmp_path, 10, requested=4, batch=2)
    two = cross.sample_areas(tmp_path, 10, requested=4, batch=2)
    assert seen == [1, 4, 7, 10] * 2
    np.testing.assert_array_equal(one[0], two[0])
    assert one[1:] == (4, 0)


def test_deterministic_product_generation(tmp_path):
    raster = {'crs': 'EPSG:5070', 'width': 2, 'height': 2,
              'transform': [270, 0, 0, 0, -270, 540], 'bounds': [0, 0, 540, 540]}
    layer = {'crs': 'EPSG:4269', 'geometry_type': 'MultiPolygon',
             'features': 3, 'total_bounds': [-97, 22, -95, 24]}
    result = {'raster': raster, 'layer': layer, 'transformed': [-100, -100, 600, 600],
              'overlap': (0, 0, 540, 540), 'areas': np.array([100., 72900., 100000.]),
              'requested': 3, 'null': 0, 'scale': cross.scale_summary([100, 72900, 100000]),
              'bp_valid_cells': 2, 'common_valid_cells': 2}
    cross.write_products(result, tmp_path)
    base = tmp_path / 'reports/eda/raw/cross_dataset/tables'
    output = tmp_path / 'reports/eda/raw/cross_dataset'
    original = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    cross.write_products(result, tmp_path)
    assert original == {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert json.loads((base / 'spatial_compatibility.json').read_text())['bbox_overlap_area_m2'] == 540 ** 2
