"""Small synthetic checks for native targets, area aggregation and reproducibility."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import Polygon, box

from wildfire_risk_uncertainty.dataset import create_dataset as cd


class IdentityProjector:
    def transform(self, x, y):
        return x, y


class IdentityTransformer:
    @staticmethod
    def from_crs(*_args, **_kwargs):
        return IdentityProjector()


@pytest.fixture
def identity(monkeypatch):
    monkeypatch.setattr(cd, 'Transformer', IdentityTransformer)


def polygon(fid, geometry, **overrides):
    attrs = dict.fromkeys(cd.FIELDS)
    attrs.update(WUIFLAG2020=1, WUICLASS2020='Low_Dens_Intermix', WATER20=0,
                 HUDEN2020=10.0, POPDEN2020=20.0, OCCHUDEN2020=3.0,
                 VACHUDEN2020=4.0, VEG2019PC=50.0, AWATER20PC=0.0)
    attrs.update(overrides)
    return {'fid': fid, 'wkb': geometry.wkb, **attrs}


def single_cell():
    return [{'cell_id': 0, 'row': 0, 'column': 0}]


def test_cell_id_endpoints_and_order():
    tiles = cd.full_tiles(10000)
    assert tiles[0].row == tiles[0].column == 0
    assert tiles[-1].row + tiles[-1].height == cd.GRID_HEIGHT
    assert tiles[-1].column + tiles[-1].width == cd.GRID_WIDTH
    assert cd.GRID_WIDTH * cd.GRID_HEIGHT - 1 == (cd.GRID_HEIGHT-1) * cd.GRID_WIDTH + cd.GRID_WIDTH-1
    ids = [r * cd.GRID_WIDTH + c for r in (0, 1, cd.GRID_HEIGHT-1)
           for c in (0, 1, cd.GRID_WIDTH-1)]
    assert len(ids) == len(set(ids))


def test_target_masks_and_bp_exclusion():
    transform = Affine(270, 0, 0, 0, -270, 270)
    arrays = [np.array([[0, 0.4, -9999, 0.3, 0.2, 0.5]], dtype=np.float32)]
    flps = [np.array([[1/6, 1/6, 1/6, 1/6, 0.2, 1/6]], dtype=np.float32) for _ in range(6)]
    flps[0][0, 3] = -9999
    flps[0][0, 5] = -0.1
    arrays += flps
    with MemoryFile():
        paths = []
        for i, array in enumerate(arrays):
            path = f'/vsimem/test_{i}.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=6, height=1, count=1,
                               dtype='float32', nodata=-9999, transform=transform) as ds:
                ds.write(array, 1)
            paths.append(path)
        with __import__('contextlib').ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(p)) for p in paths]
            rows = cd.target_rows(sources, cd.Tile(0, 0, 1, 6), transform)
        assert [r['column'] for r in rows] == [0, 1, 3, 4, 5]
        assert [r['cell_id'] for r in rows] == [0, 1, 3, 4, 5]
        assert rows[0]['BP'] == 0 and rows[0]['flp_not_applicable_bp_zero']
        assert not rows[0]['flp_target_valid']
        assert rows[1]['flp_target_valid']
        assert rows[2]['FLP1'] is None and rows[2]['flp_sum'] is None
        assert rows[2]['flp_missing_invalid'] and not rows[2]['flp_target_valid']
        assert rows[3]['flp_present_non_normalized'] and not rows[3]['flp_target_valid']
        assert rows[4]['flp_present_non_normalized'] and not rows[4]['flp_in_range']


def test_mixed_categories_intensive_nulls_and_boundary(identity):
    rows = single_cell()
    left = polygon(2, box(0, 0, 135, 270), HUDEN2020=10.0,
                   WUICLASS2020='Water', POPDEN2020=None)
    right = polygon(1, box(135, 0, 270, 270), HUDEN2020=30.0,
                    WUICLASS2020='Low_Dens_Intermix', WUIFLAG2020=None)
    diagnostics = cd.aggregate_polygons(rows, [left, right], Affine(270, 0, 0, 0, -270, 270),
                                        cd.Tile(0, 0, 1, 1))
    row = rows[0]
    assert not diagnostics[0]['failure']
    assert row['wui_covered_fraction'] == 1
    assert row[cd.category_column('WUICLASS2020', 'Water')] == 0.5
    assert row[cd.category_column('WUICLASS2020', 'Low_Dens_Intermix')] == 0.5
    assert row['WUIFLAG2020__source_null_share'] == 0.5
    assert row['HUDEN2020'] == 20
    assert row['POPDEN2020'] == 20
    assert row['POPDEN2020__source_null_area_fraction'] == 0.5
    assert row['POPDEN2020__valid_area_fraction'] == 0.5


def test_partial_and_uncovered(identity):
    transform = Affine(270, 0, 0, 0, -270, 270)
    row = single_cell()
    cd.aggregate_polygons(row, [polygon(1, box(0, 0, 135, 270), POPDEN2020=None)],
                          transform, cd.Tile(0, 0, 1, 1))
    assert row[0]['wui_covered_fraction'] == 0.5
    assert row[0]['WATER20__uncovered_share'] == 0.5
    assert row[0]['POPDEN2020'] is None
    assert row[0]['POPDEN2020__source_null_area_fraction'] == 0.5
    blank = single_cell()
    cd.aggregate_polygons(blank, [], transform, cd.Tile(0, 0, 1, 1))
    assert blank[0]['wui_covered_fraction'] == 0
    assert blank[0]['WATER20__uncovered_share'] == 1
    assert blank[0]['HUDEN2020'] is None


def test_overlap_and_invalid_geometry(identity):
    transform = Affine(270, 0, 0, 0, -270, 270)
    rows = single_cell()
    bad = Polygon([(0, 0), (270, 270), (0, 270), (270, 0), (0, 0)])
    diagnostics = cd.aggregate_polygons(rows, [polygon(4, bad)], transform, cd.Tile(0, 0, 1, 1))
    assert diagnostics[0]['failure'] and diagnostics[0]['source_ids'] == [4]
    assert rows[0]['HUDEN2020'] is None
    rows = single_cell()
    diagnostics = cd.aggregate_polygons(rows, [polygon(2, box(0, 0, 200, 270)),
                                               polygon(1, box(100, 0, 270, 270))],
                                        transform, cd.Tile(0, 0, 1, 1))
    assert 'material_polygon_overlap' in diagnostics[0]['reasons']
    assert diagnostics[0]['source_ids'] == [1, 2]
    assert rows[0]['WUICLASS2020__uncovered_share'] is None


def test_deterministic_schema_parquet_and_seed(tmp_path):
    specs = cd.column_specs()
    assert [s['name'] for s in specs] == [s['name'] for s in cd.column_specs()]
    schema = cd.arrow_schema(specs)
    import pyarrow as pa
    rows = [{name: None for name in schema.names}]
    rows[0].update(cell_id=0, row=0, column=0, x_5070=0., y_5070=0.,
                   grid_id='x', wui_source_id='y', bp_valid=True,
                   wui_failure=False, wui_failure_source_ids='[]')
    table = pa.Table.from_pylist(rows, schema=schema)
    hashes = []
    for name in ('a', 'b'):
        path = tmp_path / f'{name}.parquet'
        pq.write_table(table, path, compression='zstd', compression_level=3,
                       row_group_size=cd.ROW_GROUP_SIZE, use_dictionary=False,
                       write_statistics=False, data_page_version='1.0', store_schema=True)
        hashes.append(cd.sha256(path))
    assert hashes[0] == hashes[1]
    assert cd.REPRODUCIBILITY_SEED == 42
    assert json.loads(json.dumps(specs)) == specs


def test_progress_mode_does_not_change_parquet(tmp_path, monkeypatch):
    class FakeSource:
        width = cd.GRID_WIDTH
        height = cd.GRID_HEIGHT
        crs = rasterio.crs.CRS.from_epsg(5070)
        transform = Affine(270, 0, -2362635, 0, -270, 3267675)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    seen = []

    class FakeBar:
        def __init__(self, **kwargs):
            seen.append({'disabled': kwargs['disable'], 'total': kwargs['total'], 'updates': 0})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, amount):
            seen[-1]['updates'] += amount

    monkeypatch.setattr(cd.rasterio, 'open', lambda *_args: FakeSource())
    monkeypatch.setattr(cd, 'target_rows', lambda *_args: [])
    monkeypatch.setattr(cd, 'tqdm', FakeBar)
    tile = cd.Tile(0, 0, 1, 1)
    results = []
    for progress in (True, False):
        base = tmp_path / str(progress)
        cd.build(Path('.'), [tile], base / 'parquet', base / 'report', progress=progress)
        results.append(cd.sha256(base / 'parquet' / f'{tile.name}.parquet'))
    assert seen == [{'disabled': False, 'total': 1, 'updates': 1},
                    {'disabled': True, 'total': 1, 'updates': 1}]
    assert results[0] == results[1]


def test_intersection_failure_is_reported(identity, monkeypatch):
    class BrokenProjected:
        is_empty = False
        is_valid = True
        bounds = (0, 0, 270, 270)

        def intersection(self, _cell):
            raise ValueError('synthetic overlay failure')

    monkeypatch.setattr(cd.shapely, 'transform', lambda *_args, **_kwargs: BrokenProjected())
    rows = single_cell()
    diagnostics = cd.aggregate_polygons(rows, [polygon(9, box(0, 0, 270, 270))],
                                        Affine(270, 0, 0, 0, -270, 270), cd.Tile(0, 0, 1, 1))
    assert diagnostics[0]['reasons'] == ['intersection:ValueError']
    assert diagnostics[0]['source_ids'] == [9]
    assert rows[0]['HUDEN2020'] is None


def test_cli_no_progress_and_seed(monkeypatch, tmp_path):
    import sys

    class Source:
        transform = Affine(270, 0, 0, 0, -270, 270)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    captured = {}
    monkeypatch.setattr(cd.rasterio, 'open', lambda *_args: Source())
    monkeypatch.setattr(cd, 'build', lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, 'argv', ['wildfire-create-dataset', '--pilot', '--no-progress',
                                      '--seed', '17', '--root', str(tmp_path)])
    cd.main()
    assert captured['seed'] == 17
    assert captured['progress'] is False


def test_numerical_overlap_tolerated_without_renormalization(identity):
    transform = Affine(270, 0, 0, 0, -270, 270)
    rows = single_cell()
    diagnostics = cd.aggregate_polygons(
        rows, [polygon(2, box(135 - .0002, 0, 270, 270)),
               polygon(1, box(0, 0, 135, 270))], transform, cd.Tile(0, 0, 1, 1))
    assert cd.AREA_FRACTION_TOLERANCE == 1e-6
    assert cd.AREA_TOLERANCE_M2 == pytest.approx(.0729)
    assert diagnostics[0]['overlap_pair_count'] == 1
    assert diagnostics[0]['max_pair_overlap_m2'] == pytest.approx(.054)
    assert not rows[0]['wui_failure']
    assert rows[0]['wui_tolerated_overlap_source_ids'] == '[1,2]'
    assert rows[0]['wui_covered_fraction'] > 1
    assert rows[0]['WATER20__uncovered_share'] < 0


def test_material_overlap_is_masked(identity):
    transform = Affine(270, 0, 0, 0, -270, 270)
    rows = single_cell()
    diagnostics = cd.aggregate_polygons(
        rows, [polygon(1, box(0, 0, 135, 270)),
               polygon(2, box(135 - .0004, 0, 270, 270))],
        transform, cd.Tile(0, 0, 1, 1))
    assert diagnostics[0]['max_pair_overlap_m2'] == pytest.approx(.108)
    assert diagnostics[0]['failure']
    assert 'material_polygon_overlap' in diagnostics[0]['reasons']
    assert rows[0]['HUDEN2020'] is None
    assert rows[0]['wui_failure_source_ids'] == '[1,2]'


def test_accepted_make_valid_repair_and_provenance(identity):
    transform = Affine(270, 0, 0, 0, -270, 270)
    invalid = Polygon([(0, 0), (270, 0), (270, 270), (0, 270),
                       (0, 0), (135, 0), (0, 0)])
    assert not invalid.is_valid
    results = []
    for _ in range(2):
        rows = single_cell()
        repairs = []
        diagnostics = cd.aggregate_polygons(rows, [polygon(7, invalid)], transform,
                                            cd.Tile(0, 0, 1, 1), repair_records=repairs)
        assert not diagnostics[0]['failure']
        assert rows[0]['wui_repaired_source_ids'] == '[7]'
        assert rows[0]['HUDEN2020'] == 10
        assert repairs[0]['accepted']
        assert repairs[0]['relative_area_change'] <= 1e-6
        results.append((rows, repairs, diagnostics))
    assert results[0] == results[1]


def test_repair_with_material_area_change_rejected(identity, monkeypatch):
    transform = Affine(270, 0, 0, 0, -270, 270)
    invalid = Polygon([(0, 0), (270, 0), (270, 270), (0, 270),
                       (0, 0), (135, 0), (0, 0)])
    monkeypatch.setattr(cd.shapely, 'make_valid', lambda _geometry: box(0, 0, 100, 270))
    rows, repairs = single_cell(), []
    diagnostics = cd.aggregate_polygons(rows, [polygon(7, invalid)], transform,
                                        cd.Tile(0, 0, 1, 1), repair_records=repairs)
    assert diagnostics[0]['failure']
    assert repairs[0]['accepted'] is False
    assert repairs[0]['relative_area_change'] > 1e-6
    assert rows[0]['wui_repaired_source_ids'] == '[]'
    assert rows[0]['HUDEN2020'] is None
