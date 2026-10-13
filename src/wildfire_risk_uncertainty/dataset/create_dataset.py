"""Deterministic, windowed construction of the canonical 270 m dataset.

Run with ``PYTHONHASHSEED=42 uv run wildfire-create-dataset --pilot``.
Each requested native raster window is an independent, sorted Parquet partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import tomllib
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
import rasterio
import shapely
from pyproj import Transformer, __proj_version__
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm

from wildfire_risk_uncertainty.eda.dillon import DILLON_DIR, FILENAMES
from wildfire_risk_uncertainty.eda.wui import GDB

REPRODUCIBILITY_SEED = 42
GRID_WIDTH = 17_372
GRID_HEIGHT = 11_283
CELL_AREA_M2 = 72_900.0
FLP_TOLERANCE = 1e-6
AREA_FRACTION_TOLERANCE = 1e-6
AREA_TOLERANCE_M2 = CELL_AREA_M2 * AREA_FRACTION_TOLERANCE
SHARE_TOLERANCE = AREA_FRACTION_TOLERANCE
ROW_GROUP_SIZE = 65_536
SCHEMA_VERSION = '1.1.0'
CATEGORICAL = {
    'WUIFLAG2020': ('0', '1', '2'),
    'WUICLASS2020': tuple(sorted((
        'High_Dens_Interface', 'High_Dens_Intermix', 'High_Dens_NoVeg',
        'Low_Dens_Interface', 'Low_Dens_Intermix', 'Low_Dens_NoVeg',
        'Med_Dens_Interface', 'Med_Dens_Intermix', 'Med_Dens_NoVeg',
        'Uninhabited_NoVeg', 'Uninhabited_Veg', 'Very_Low_Dens_NoVeg',
        'Very_Low_Dens_Veg', 'Water',
    ))),
    'WATER20': ('0', '1'),
}
INTENSIVE = ('HUDEN2020', 'POPDEN2020', 'OCCHUDEN2020', 'VACHUDEN2020',
             'VEG2019PC', 'AWATER20PC')
FIELDS = (*CATEGORICAL, *INTENSIVE)
# Fixed examples selected from geographic locations, not a random source search.
PILOT_LOCATIONS = (
    ('denver_urban', -104.9903, 39.7392, 64),
    ('chicago_urban', -87.6298, 41.8781, 8),
    ('lake_michigan', -87.5600, 41.8800, 8),
    ('rural_kansas', -99.0000, 38.5000, 8),
    ('seattle_edge', -122.1700, 47.5200, 8),
    ('pacific_coast', -124.2000, 41.0000, 8),
)


@dataclass(frozen=True, order=True)
class Tile:
    row: int
    column: int
    height: int
    width: int

    @property
    def name(self) -> str:
        return f'row_{self.row:05d}_col_{self.column:05d}_h{self.height:03d}_w{self.width:03d}'


def pilot_tiles(transform) -> list[Tile]:
    transformer = Transformer.from_crs('EPSG:4269', 'EPSG:5070', always_xy=True)
    tiles = []
    for _, lon, lat, size in PILOT_LOCATIONS:
        x, y = transformer.transform(lon, lat)
        row, col = rasterio.transform.rowcol(transform, x, y)
        tiles.append(Tile(int(max(0, row - size // 2)), int(max(0, col - size // 2)), size, size))
    return sorted(set(tiles))


def full_tiles(tile_size: int) -> list[Tile]:
    if tile_size < 1:
        raise ValueError('tile size must be positive')
    return [Tile(r, c, min(tile_size, GRID_HEIGHT-r), min(tile_size, GRID_WIDTH-c))
            for r in range(0, GRID_HEIGHT, tile_size)
            for c in range(0, GRID_WIDTH, tile_size)]


def category_column(field: str, level: str) -> str:
    return f'{field}__share__{level}'


def column_specs() -> list[dict]:
    specs = []

    def add(name, dtype, units, source, aggregation, role, null_semantics):
        specs.append({'name': name, 'dtype': dtype, 'units': units, 'source': source,
                      'aggregation_rule': aggregation, 'role': role,
                      'null_semantics': null_semantics})

    for name, dtype, units, rule in (
        ('cell_id', 'int64', None, 'row * 17372 + column'),
        ('row', 'int32', 'native grid index', 'raster row'),
        ('column', 'int32', 'native grid index', 'raster column'),
        ('x_5070', 'float64', 'm', 'native pixel center'),
        ('y_5070', 'float64', 'm', 'native pixel center'),
    ):
        add(name, dtype, units, 'Dillon grid', rule, 'provenance', 'never null')
    add('grid_id', 'string', None, 'Dillon LF2020', 'constant EPSG:5070/270m',
        'provenance', 'never null')
    add('wui_source_id', 'string', None, 'SILVIS V4', 'constant source GDB identity',
        'provenance', 'never null')
    for name in ('BP', *(f'FLP{i}' for i in range(1, 7))):
        add(name, 'float32', 'probability', f'Dillon {name}', 'native raster value, no resampling',
            'target', 'null iff GDAL invalid or nonfinite; BP-invalid row excluded')
    for name, dtype in (
        ('bp_valid', 'bool'), ('flps_present', 'bool'), ('flp_sum', 'float64'),
        ('flp_in_range', 'bool'), ('flp_target_valid', 'bool'),
        ('flp_missing_invalid', 'bool'), ('flp_present_non_normalized', 'bool'),
        ('flp_not_applicable_bp_zero', 'bool'),
    ):
        add(name, dtype, 'probability' if name == 'flp_sum' else None,
            'Dillon BP/FLP1-6', 'strict target-mask rule in manifest', 'diagnostic',
            'flp_sum null when any FLP absent; other diagnostics never null')
    add('wui_covered_fraction', 'float64', 'cell area fraction', 'SILVIS V4',
        'sum polygon/cell intersection area / 72900', 'diagnostic',
        'null on WUI geometry/topology failure')
    for field, levels in CATEGORICAL.items():
        for level in levels:
            add(category_column(field, level), 'float64', 'cell area fraction',
                f'SILVIS V4 {field}', f'intersection area with category {level} / 72900',
                'predictor', 'null on WUI geometry/topology failure')
        for suffix, rule in (('source_null_share', 'source-null intersection area / 72900'),
                             ('uncovered_share', '1 - WUI covered fraction')):
            add(f'{field}__{suffix}', 'float64', 'cell area fraction', f'SILVIS V4 {field}',
                rule, 'diagnostic', 'null on WUI geometry/topology failure')
    for field in INTENSIVE:
        units = 'percent' if field.endswith('PC') else 'source density units'
        add(field, 'float64', units, f'SILVIS V4 {field}',
            'sum(intersection area * finite value) / finite-value intersection area',
            'predictor', 'null when valid area zero or WUI failure')
        for suffix, rule in (('valid_area_fraction', 'finite-value area / 72900'),
                             ('source_null_area_fraction', 'source-null area / 72900')):
            add(f'{field}__{suffix}', 'float64', 'cell area fraction', f'SILVIS V4 {field}',
                rule, 'diagnostic', 'null on WUI geometry/topology failure')
    add('wui_failure', 'bool', None, 'overlay diagnostics', 'any geometry/topology failure',
        'diagnostic', 'never null')
    add('wui_failure_source_ids', 'string', None, 'SILVIS OBJECTID',
        'sorted JSON source-ID list', 'diagnostic', 'never null; [] if no failure')
    add('wui_repaired_source_ids', 'string', None, 'SILVIS OBJECTID',
        'sorted JSON IDs of accepted make_valid geometries contributing to cell',
        'diagnostic', 'never null; [] if none')
    add('wui_tolerated_overlap_source_ids', 'string', None, 'SILVIS OBJECTID',
        'sorted JSON IDs in positive overlaps within 1e-6 cell-area tolerance',
        'diagnostic', 'never null; [] if none')
    return specs


def arrow_schema(specs: list[dict]) -> pa.Schema:
    types = {'int64': pa.int64(), 'int32': pa.int32(), 'float32': pa.float32(),
             'float64': pa.float64(), 'bool': pa.bool_(), 'string': pa.string()}
    return pa.schema([pa.field(s['name'], types[s['dtype']], nullable=s['null_semantics'] != 'never null')
                      for s in specs])


def target_rows(sources, tile: Tile, transform) -> list[dict]:
    window = Window(tile.column, tile.row, tile.width, tile.height)
    blocks = [src.read(1, window=window, masked=True) for src in sources]
    valid = [~np.ma.getmaskarray(b) & np.isfinite(b.data) for b in blocks]
    result = []
    for dr in range(tile.height):
        for dc in range(tile.width):
            if not valid[0][dr, dc]:
                continue
            row, col = tile.row + dr, tile.column + dc
            x, y = rasterio.transform.xy(transform, row, col, offset='center')
            bp = float(blocks[0].data[dr, dc])
            flps = [float(blocks[i].data[dr, dc]) if valid[i][dr, dc] else None
                    for i in range(1, 7)]
            present = all(v is not None for v in flps)
            flp_sum = sum(flps) if present else None  # Python float is float64; fixed FLP order.
            in_range = present and all(0 <= v <= 1 for v in flps)
            target_valid = bp > 0 and in_range and abs(flp_sum - 1) <= FLP_TOLERANCE
            result.append({
                'cell_id': row * GRID_WIDTH + col, 'row': row, 'column': col,
                'x_5070': float(x), 'y_5070': float(y),
                'grid_id': 'Dillon_LF2020_EPSG5070_270m',
                'wui_source_id': 'SILVIS_CONUS_WUI_V4_2020',
                'BP': bp, **{f'FLP{i}': flps[i-1] for i in range(1, 7)},
                'bp_valid': True, 'flps_present': present, 'flp_sum': flp_sum,
                'flp_in_range': in_range, 'flp_target_valid': target_valid,
                'flp_missing_invalid': bp > 0 and not present,
                'flp_present_non_normalized': bp > 0 and present and not target_valid,
                'flp_not_applicable_bp_zero': bp == 0,
            })
    return result


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _new_accumulator():
    return {'coverage': 0.0, 'pieces': [], 'categories': {
        f: {level: 0.0 for level in levels} for f, levels in CATEGORICAL.items()},
        'cat_null': {f: 0.0 for f in CATEGORICAL},
        'numerator': {f: 0.0 for f in INTENSIVE},
        'denominator': {f: 0.0 for f in INTENSIVE},
        'int_null': {f: 0.0 for f in INTENSIVE},
        'overlap_pairs': [], 'repaired': set(),
        'failures': set(), 'reasons': set()}


def approved_repair(projected):
    """Apply only the HITL-approved make_valid and projected-area gate."""
    original_area = float(projected.area)
    try:
        repaired = shapely.make_valid(projected)
        repaired_area = float(repaired.area)
    except (ValueError, TypeError, shapely.errors.GEOSException, OverflowError) as exc:
        return None, {
            'original_projected_area_m2': original_area,
            'repaired_projected_area_m2': None,
            'relative_area_change': None,
            'repair_result_type': None,
            'accepted': False,
            'reason': f'make_valid:{type(exc).__name__}',
        }
    relative_change = (abs(repaired_area - original_area) / original_area
                       if math.isfinite(original_area) and original_area > 0 else None)
    accepted = (not repaired.is_empty and repaired.geom_type in ('Polygon', 'MultiPolygon')
                and math.isfinite(original_area) and original_area > 0
                and math.isfinite(repaired_area) and repaired_area > 0
                and relative_change is not None and relative_change <= AREA_FRACTION_TOLERANCE)
    return repaired if accepted else None, {
        'original_projected_area_m2': original_area,
        'repaired_projected_area_m2': repaired_area,
        'relative_area_change': relative_change,
        'repair_result_type': repaired.geom_type,
        'accepted': accepted,
        'reason': 'accepted' if accepted else 'repair_criteria_failed',
    }


def aggregate_polygons(rows: list[dict], polygons: list[dict], transform,
                       tile: Tile, repair_records: list[dict] | None = None) -> list[dict]:
    """Project, intersect and reduce in ascending source OBJECTID order."""
    accum = {row['cell_id']: _new_accumulator() for row in rows}
    if not accum:
        return []
    projector = Transformer.from_crs('EPSG:4269', 'EPSG:5070', always_xy=True)
    cell_boxes = {row['cell_id']: box(*rasterio.windows.bounds(
        Window(row['column'], row['row'], 1, 1), transform)) for row in rows}
    for source in sorted(polygons, key=lambda p: p['fid']):
        fid = int(source['fid'])
        projected = None
        repair_recorded = False
        try:
            geometry = shapely.from_wkb(source['wkb']) if source['wkb'] is not None else None
            if geometry is None or geometry.is_empty:
                raise ValueError('invalid_source_geometry')
            projected = shapely.transform(geometry, projector.transform, interleaved=False)
            repaired = False
            if projected.is_empty or not geometry.is_valid or not projected.is_valid:
                replacement, record = approved_repair(projected)
                if repair_records is not None:
                    repair_records.append({'tile': tile.name, 'objectid': fid, **record})
                    repair_recorded = True
                if replacement is None:
                    raise ValueError('rejected_geometry_repair')
                projected = replacement
                repaired = True
            minx, miny, maxx, maxy = projected.bounds
            if not all(map(math.isfinite, (minx, miny, maxx, maxy))):
                raise ValueError('nonfinite_projected_bounds')
        except (ValueError, TypeError, shapely.errors.GEOSException, OverflowError) as exc:
            if repair_records is not None and not repair_recorded:
                repair_records.append({
                    'tile': tile.name, 'objectid': fid,
                    'original_projected_area_m2': float(projected.area) if projected is not None else None,
                    'repaired_projected_area_m2': None,
                    'relative_area_change': None, 'repair_result_type': None,
                    'accepted': False, 'reason': f'geometry:{type(exc).__name__}',
                })
            # The footprint of an invalid shape is uncertain. Its projected
            # envelope confines affected cells when that envelope is usable.
            bounds = projected.bounds if projected is not None else None
            for cid, acc in accum.items():
                if bounds and len(bounds) == 4 and all(map(math.isfinite, bounds)):
                    x0, y0, x1, y1 = cell_boxes[cid].bounds
                    if bounds[2] <= x0 or bounds[0] >= x1 or bounds[3] <= y0 or bounds[1] >= y1:
                        continue
                acc['failures'].add(fid)
                acc['reasons'].add(f'geometry:{type(exc).__name__}')
            continue
        # Native north-up affine grid: visit only cells in the projected envelope.
        col_start = max(tile.column, math.floor((minx - transform.c) / transform.a))
        col_stop = min(tile.column + tile.width, math.ceil((maxx - transform.c) / transform.a))
        row_start = max(tile.row, math.floor((transform.f - maxy) / -transform.e))
        row_stop = min(tile.row + tile.height, math.ceil((transform.f - miny) / -transform.e))
        for r in range(row_start, row_stop):
            for c in range(col_start, col_stop):
                cid = r * GRID_WIDTH + c
                if cid not in accum:
                    continue
                _intersect_one_cell(source, projected, fid, cell_boxes[cid], accum[cid], repaired)
    diagnostics = []
    for row in rows:
        cid = row['cell_id']
        acc = accum[cid]
        covered = acc['coverage'] / CELL_AREA_M2
        duplicate_area = 0.0
        if acc['overlap_pairs']:
            try:
                duplicate_area = max(0.0, acc['coverage'] - float(shapely.union_all(
                    [piece for _, piece in acc['pieces']]).area))
            except (ValueError, shapely.errors.GEOSException) as exc:
                acc['failures'].update(fid for fid, _ in acc['pieces'])
                acc['reasons'].add(f'union:{type(exc).__name__}')
        if duplicate_area > AREA_TOLERANCE_M2 or any(
                pair[2] > AREA_TOLERANCE_M2 for pair in acc['overlap_pairs']):
            acc['failures'].update(fid for pair in acc['overlap_pairs'] for fid in pair[:2])
            acc['reasons'].add('material_polygon_overlap')
        excess_area = acc['coverage'] - CELL_AREA_M2
        if excess_area > AREA_TOLERANCE_M2:
            acc['failures'].update(fid for fid, _ in acc['pieces'])
            acc['reasons'].add('material_coverage_excess')
        max_share_error = 0.0
        for field, levels in CATEGORICAL.items():
            accounted = (sum(acc['categories'][field][level] for level in levels)
                         + acc['cat_null'][field]) / CELL_AREA_M2
            share_error = abs(accounted + (1 - covered) - 1)
            max_share_error = max(max_share_error, share_error)
            if share_error > AREA_FRACTION_TOLERANCE:
                acc['failures'].update(fid for fid, _ in acc['pieces'])
                acc['reasons'].add(f'material_categorical_reconciliation:{field}')
        failure = bool(acc['reasons'])
        row['wui_failure'] = failure
        row['wui_failure_source_ids'] = json.dumps(sorted(acc['failures']), separators=(',', ':'))
        row['wui_repaired_source_ids'] = json.dumps(sorted(acc['repaired']), separators=(',', ':'))
        tolerated_ids = sorted({fid for pair in acc['overlap_pairs'] for fid in pair[:2]})
        row['wui_tolerated_overlap_source_ids'] = json.dumps(
            tolerated_ids if not failure else [], separators=(',', ':'))
        diagnostics.append({'tile': tile.name, 'cell_id': cid, 'coverage': covered,
                            'failure': failure, 'reasons': sorted(acc['reasons']),
                            'source_ids': sorted(acc['failures']),
                            'repaired_source_ids': sorted(acc['repaired']),
                            'overlap_source_ids': tolerated_ids,
                            'overlap_pair_count': len(acc['overlap_pairs']),
                            'max_pair_overlap_m2': max((pair[2] for pair in acc['overlap_pairs']), default=0.0),
                            'duplicate_overlap_m2': duplicate_area,
                            'coverage_excess_m2': max(0.0, excess_area),
                            'max_share_error': max_share_error})
        if failure:
            row['wui_covered_fraction'] = None
            for field, levels in CATEGORICAL.items():
                row.update({category_column(field, level): None for level in levels})
                row[f'{field}__source_null_share'] = None
                row[f'{field}__uncovered_share'] = None
            for field in INTENSIVE:
                row[field] = None
                row[f'{field}__valid_area_fraction'] = None
                row[f'{field}__source_null_area_fraction'] = None
            continue
        row['wui_covered_fraction'] = covered
        for field, levels in CATEGORICAL.items():
            for level in levels:
                row[category_column(field, level)] = acc['categories'][field][level] / CELL_AREA_M2
            row[f'{field}__source_null_share'] = acc['cat_null'][field] / CELL_AREA_M2
            row[f'{field}__uncovered_share'] = 1 - covered
        for field in INTENSIVE:
            denominator = acc['denominator'][field]
            row[field] = acc['numerator'][field] / denominator if denominator > 0 else None
            row[f'{field}__valid_area_fraction'] = denominator / CELL_AREA_M2
            row[f'{field}__source_null_area_fraction'] = acc['int_null'][field] / CELL_AREA_M2
    return diagnostics


def _intersect_one_cell(source, projected, fid, cell, acc, repaired=False):
    """Accumulate one source/cell pair; caller provides stable source order."""
    cx0, cy0, cx1, cy1 = cell.bounds
    minx, miny, maxx, maxy = projected.bounds
    if maxx <= cx0 or minx >= cx1 or maxy <= cy0 or miny >= cy1:
        return
    try:
        intersection = projected.intersection(cell)
        area = float(intersection.area)
        if not math.isfinite(area) or area < 0:
            raise ValueError('invalid_intersection_area')
        if area <= 0:
            return
        for prior_fid, prior in acc['pieces']:
            overlap = float(intersection.intersection(prior).area)
            if overlap > 0:
                acc['overlap_pairs'].append((prior_fid, fid, overlap))
        acc['pieces'].append((fid, intersection))
        acc['coverage'] += area
        if repaired:
            acc['repaired'].add(fid)
        for field, levels in CATEGORICAL.items():
            value = source.get(field)
            label = str(value) if value is not None else None
            if label is None:
                acc['cat_null'][field] += area
            elif label in levels:
                acc['categories'][field][label] += area
            else:
                acc['failures'].add(fid)
                acc['reasons'].add(f'unknown_category:{field}:{label}')
        for field in INTENSIVE:
            value = source.get(field)
            if _finite(value):
                acc['numerator'][field] += area * float(value)
                acc['denominator'][field] += area
            else:
                acc['int_null'][field] += area
    except (ValueError, TypeError, shapely.errors.GEOSException, OverflowError) as exc:
        acc['failures'].add(fid)
        acc['reasons'].add(f'intersection:{type(exc).__name__}')


def read_polygons(gdb: Path, tile: Tile, transform) -> list[dict]:
    bounds = rasterio.windows.bounds(Window(tile.column, tile.row, tile.width, tile.height), transform)
    native_bbox = transform_bounds('EPSG:5070', 'EPSG:4269', *bounds, densify_pts=21)
    metadata, table = pyogrio.read_arrow(gdb, bbox=native_bbox, columns=list(FIELDS), return_fids=True)
    if metadata['crs'] != 'EPSG:4269':
        raise ValueError(f'Unexpected SILVIS CRS: {metadata["crs"]}')
    columns = {name: table[name].to_pylist() for name in ('OBJECTID', *FIELDS, 'Shape')}
    return [dict(fid=columns['OBJECTID'][i], wkb=columns['Shape'][i],
                 **{field: columns[field][i] for field in FIELDS})
            for i in sorted(range(table.num_rows), key=lambda j: columns['OBJECTID'][j])]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def build(root: Path, tiles: list[Tile], output: Path, reports: Path,
          seed: int = REPRODUCIBILITY_SEED, progress: bool = True) -> None:
    random.seed(seed)
    np.random.default_rng(seed)  # No randomized operation is used.
    specs = column_specs()
    schema = arrow_schema(specs)
    tiles = sorted(set(tiles))
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    partition_paths = [output / f'{tile.name}.parquet' for tile in tiles]
    stale = sorted(set(output.glob('*.parquet')) - set(partition_paths))
    if stale:
        raise ValueError(f'Output contains partitions outside requested windows: {stale}')
    per_tile, coverage_rows = [], []
    mask_names = ('bp_valid', 'flps_present', 'flp_in_range', 'flp_target_valid',
                  'flp_missing_invalid', 'flp_present_non_normalized',
                  'flp_not_applicable_bp_zero')
    mask_counts = {name: 0 for name in mask_names}
    null_counts = {field: {'null_value_rows': 0, 'source_null_area_fraction_sum': 0.0,
                           'valid_area_fraction_sum': 0.0} for field in INTENSIVE}
    topology_fields = ['tile', 'cell_id', 'reasons', 'source_ids', 'pre_failure_coverage',
                       'max_pair_overlap_m2', 'duplicate_overlap_m2',
                       'coverage_excess_m2', 'max_share_error']
    tolerated_fields = ['tile', 'cell_id', 'source_ids', 'overlap_pair_count',
                        'max_pair_overlap_m2', 'duplicate_overlap_m2',
                        'coverage_excess_m2', 'raw_coverage_fraction', 'max_share_error']
    repair_fields = ['tile', 'objectid', 'original_projected_area_m2',
                     'repaired_projected_area_m2', 'relative_area_change',
                     'repair_result_type', 'accepted', 'reason']
    with ExitStack() as stack:
        topology_handle = stack.enter_context((reports / 'topology_summary.csv').open('w', newline=''))
        topology_writer = csv.DictWriter(topology_handle, fieldnames=topology_fields, lineterminator='\n')
        topology_writer.writeheader()
        tolerated_handle = stack.enter_context((reports / 'tolerated_topology_summary.csv').open('w', newline=''))
        tolerated_writer = csv.DictWriter(tolerated_handle, fieldnames=tolerated_fields, lineterminator='\n')
        tolerated_writer.writeheader()
        repair_handle = stack.enter_context((reports / 'geometry_repair_summary.csv').open('w', newline=''))
        repair_writer = csv.DictWriter(repair_handle, fieldnames=repair_fields, lineterminator='\n')
        repair_writer.writeheader()
        sources = [stack.enter_context(rasterio.open(root / DILLON_DIR / name))
                   for name in FILENAMES]
        reference = sources[0]
        if (reference.width, reference.height, reference.crs.to_string()) != (
                GRID_WIDTH, GRID_HEIGHT, 'EPSG:5070'):
            raise ValueError('Unexpected Dillon grid definition')
        if not all((s.width, s.height, s.crs, s.transform) == (
                reference.width, reference.height, reference.crs, reference.transform)
                for s in sources):
            raise ValueError('Dillon rasters do not share the native grid')
        bar = stack.enter_context(tqdm(total=len(tiles), desc='Building canonical dataset',
                                       unit='tile', disable=not progress, file=sys.stderr,
                                       dynamic_ncols=True, leave=True))
        for tile_index, tile in enumerate(tiles):
            rows = target_rows(sources, tile, reference.transform)
            polygons = read_polygons(root / GDB, tile, reference.transform) if rows else []
            repair_records = []
            diagnostics = aggregate_polygons(rows, polygons, reference.transform, tile,
                                             repair_records=repair_records)
            for record in sorted(repair_records, key=lambda record: record['objectid']):
                repair_writer.writerow(record)
            rows.sort(key=lambda row: row['cell_id'])
            table = pa.Table.from_pylist(rows, schema=schema) if rows else pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in schema], schema=schema)
            file = output / f'{tile.name}.parquet'
            temporary = file.with_suffix('.parquet.tmp')
            pq.write_table(table, temporary, compression='zstd', compression_level=3,
                           row_group_size=ROW_GROUP_SIZE, use_dictionary=False,
                           write_statistics=False, data_page_version='1.0',
                           store_schema=True)
            temporary.replace(file)
            per_tile.append({'tile': tile.name, 'rows': len(rows),
                             'bp_zero': sum(row['BP'] == 0 for row in rows),
                             'flp_target_valid': sum(row['flp_target_valid'] for row in rows),
                             'wui_failures': sum(row['wui_failure'] for row in rows),
                             'tolerated_overlap_cells': sum(
                                 bool(d['overlap_pair_count']) and not d['failure'] for d in diagnostics),
                             'tolerated_overlap_pairs': sum(
                                 d['overlap_pair_count'] for d in diagnostics if not d['failure']),
                             'accepted_repairs': sum(r['accepted'] for r in repair_records),
                             'rejected_repairs': sum(not r['accepted'] for r in repair_records)})
            for row in rows:
                for name in mask_names:
                    mask_counts[name] += bool(row[name])
                for field in INTENSIVE:
                    null_counts[field]['null_value_rows'] += row[field] is None
                    null_counts[field]['source_null_area_fraction_sum'] += (
                        row[f'{field}__source_null_area_fraction'] or 0.0)
                    null_counts[field]['valid_area_fraction_sum'] += (
                        row[f'{field}__valid_area_fraction'] or 0.0)
            good = [row for row in rows if not row['wui_failure']]
            coverage_rows.append({'tile': tile.name, 'rows': len(rows),
                                  'uncovered_rows': sum(row['wui_covered_fraction'] == 0 for row in good),
                                  'partial_rows': sum(0 < row['wui_covered_fraction'] < 1-SHARE_TOLERANCE for row in good),
                                  'mean_coverage': sum(row['wui_covered_fraction'] for row in good) / len(good) if good else None,
                                  'max_share_error': max((abs(sum(row[category_column(field, level)] for level in levels)
                                                              + row[f'{field}__source_null_share']
                                                              + row[f'{field}__uncovered_share'] - 1)
                                                          for row in good for field, levels in CATEGORICAL.items()), default=None)})
            for diagnostic in diagnostics:
                if diagnostic['failure']:
                    topology_writer.writerow({
                        'tile': diagnostic['tile'], 'cell_id': diagnostic['cell_id'],
                        'reasons': json.dumps(diagnostic['reasons'], separators=(',', ':')),
                        'source_ids': json.dumps(diagnostic['source_ids'], separators=(',', ':')),
                        'pre_failure_coverage': diagnostic['coverage'],
                        'max_pair_overlap_m2': diagnostic['max_pair_overlap_m2'],
                        'duplicate_overlap_m2': diagnostic['duplicate_overlap_m2'],
                        'coverage_excess_m2': diagnostic['coverage_excess_m2'],
                        'max_share_error': diagnostic['max_share_error']})
                elif diagnostic['overlap_pair_count'] or diagnostic['coverage_excess_m2'] > 0:
                    tolerated_writer.writerow({
                        'tile': diagnostic['tile'], 'cell_id': diagnostic['cell_id'],
                        'source_ids': json.dumps(diagnostic['overlap_source_ids'], separators=(',', ':')),
                        'overlap_pair_count': diagnostic['overlap_pair_count'],
                        'max_pair_overlap_m2': diagnostic['max_pair_overlap_m2'],
                        'duplicate_overlap_m2': diagnostic['duplicate_overlap_m2'],
                        'coverage_excess_m2': diagnostic['coverage_excess_m2'],
                        'raw_coverage_fraction': diagnostic['coverage'],
                        'max_share_error': diagnostic['max_share_error']})
            if tile_index < len(tiles) - 1:
                bar.update(1)
        write_csv(reports / 'pilot_summary.csv', per_tile,
                  ['tile', 'rows', 'bp_zero', 'flp_target_valid', 'wui_failures',
                   'tolerated_overlap_cells', 'tolerated_overlap_pairs',
                   'accepted_repairs', 'rejected_repairs'])
        write_csv(reports / 'target_mask_summary.csv',
                  [{'mask': name, 'count': mask_counts[name]} for name in mask_names], ['mask', 'count'])
        write_csv(reports / 'coverage_summary.csv', coverage_rows,
                  ['tile', 'rows', 'uncovered_rows', 'partial_rows', 'mean_coverage', 'max_share_error'])
        write_csv(reports / 'null_summary.csv',
                  [{'field': field, **null_counts[field]} for field in INTENSIVE],
                  ['field', 'null_value_rows', 'source_null_area_fraction_sum', 'valid_area_fraction_sum'])
        (reports / 'schema.json').write_text(json.dumps({'schema_version': SCHEMA_VERSION, 'columns': specs},
                                                        indent=2, sort_keys=True) + '\n')
        manifest = {
        'seed': seed, 'base_commit': 'd519589a43f342d29abf4c223cc9bc5e5f331bf9',
        'source_sha256': {
            'dillon_archive': 'b5313ad0ca2deac11b1284472731cc0aedea1bd3f0e92823e0d5356cb6dfff73',
            'silvis_archive': '0b3d1bfabd80fdc75a3e776c26f8708ac5a55ab43b7f87b081d69e413fe447d4',
            'silvis_metadata': '2edd4696ab151d267bd41a34e3a031d09b936466229617bbc16c6f55e2c0afd1'},
        'source_versions': {'dillon': 'RDS-2016-0034-3 LF2020', 'silvis': 'CONUS WUI V4 2020; vegetation 2019'},
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ('numpy', 'rasterio', 'pyogrio', 'shapely',
                                                       'pyproj', 'pyarrow', 'tqdm')},
        'lock_sha256': sha256(root / 'uv.lock'),
        'locked_packages': [
            {'name': package['name'], 'version': package['version']}
            for package in sorted(tomllib.loads((root / 'uv.lock').read_text())['package'],
                                  key=lambda package: package['name'])
        ],
        'gdal': rasterio.__gdal_version__, 'geos': shapely.geos_version_string,
        'proj': __proj_version__, 'crs': 'EPSG:5070',
        'grid': {'width': GRID_WIDTH, 'height': GRID_HEIGHT,
                 'transform': list(reference.transform)[:6], 'cell_area_m2': CELL_AREA_M2},
        'pilot_windows': [tile.__dict__ | {'name': tile.name} for tile in tiles],
        'tolerances': {'flp_sum': FLP_TOLERANCE,
                       'area_fraction': AREA_FRACTION_TOLERANCE,
                       'area_m2': AREA_TOLERANCE_M2, 'share': SHARE_TOLERANCE,
                       'geometry_repair_relative_area_change': AREA_FRACTION_TOLERANCE},
        'geometry_repair_policy': ('For invalid source geometry, apply shapely.make_valid in EPSG:5070; '
                                   'accept only non-empty Polygon/MultiPolygon with finite positive '
                                   'original and repaired projected areas and relative area change <=1e-6; '
                                   'otherwise mask affected WUI values.'),
        'topology_policy': ('Positive overlap, duplicate source coverage, coverage excess, and '
                            'categorical-share error <=1e-6 of cell area are tolerated and reported; '
                            'material discrepancies mask affected WUI values; no renormalization.'),
        'writer': {'format': 'Parquet', 'compression': 'zstd', 'compression_level': 3,
                   'row_group_size': ROW_GROUP_SIZE, 'use_dictionary': False,
                   'write_statistics': False, 'data_page_version': '1.0',
                   'store_schema': True},
        'schema_version': SCHEMA_VERSION,
        'target_mask_rule': 'BP>0 and all FLPs GDAL-valid, finite, in [0,1], and abs(float64 sum-1)<=1e-6',
        'aggregation': 'EPSG:5070 true cell/polygon intersection; categorical area / full cell; intensive finite-area weighted mean',
        'row_order': 'ascending cell_id within ascending tile order',
        'partitions': [{'name': p.name, 'sha256': sha256(p)} for p in partition_paths],
        }
        (reports / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        if tiles:
            bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/dataset/pilot'))
    parser.add_argument('--reports', type=Path, default=Path('reports/dataset/pilot'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pilot', action='store_true')
    group.add_argument('--full', action='store_true', help='Full grid; do not use before authorization')
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=REPRODUCIBILITY_SEED)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    with rasterio.open(args.root / DILLON_DIR / FILENAMES[0]) as source:
        tiles = pilot_tiles(source.transform) if args.pilot else full_tiles(args.tile_size)
    build(args.root, tiles, args.output, args.reports, seed=args.seed,
          progress=not args.no_progress)


if __name__ == '__main__':
    main()
