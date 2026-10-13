"""Bounded WUI geometry diagnostics against the reviewed Dillon 270 m grid."""

import csv
import json
import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
import rasterio
from pyogrio.raw import read
from rasterio.warp import transform, transform_bounds

from . import wui, wui_values

CELL_AREA_M2 = 270 * 270
SAMPLE_SIZE = 10_000
BATCH_SIZE = 250
QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)

VARIABLE_TYPES = {
    'categorical': ('WUIFLAG2020', 'WUICLASS2020', 'WATER20', 'PUBFLAG', 'BUFVEG'),
    'intensive': ('HUDEN2020', 'POPDEN2020', 'OCCHUDEN2020', 'VACHUDEN2020',
                  'VEG2019PC', 'AWATER20PC'),
    'extensive/count': ('HU2020', 'POP2020', 'OCCHU2020', 'VACHU2020'),
    'identifier/admin': ('BLK20', 'fips', 'STATE'),
}


def variable_type(field: str) -> str:
    for kind, fields in VARIABLE_TYPES.items():
        if field in fields:
            return kind
    raise ValueError(f'Unclassified WUI field: {field}')


def selected_fids(total: int, requested: int = SAMPLE_SIZE) -> np.ndarray:
    """Same evenly spaced FID selection as the reviewed WUI spatial view."""
    if total < 1 or requested < 1:
        raise ValueError('Positive feature count and sample size required')
    return np.unique(np.linspace(1, total, min(total, requested), dtype=np.int64))


def projected_polygon_area(wkb: bytes, source_crs='EPSG:4269', target_crs='EPSG:5070') -> float:
    """Project each MultiPolygon ring; subtract holes using planar shoelace areas."""
    if wkb is None:
        raise ValueError('Null geometry')
    view = memoryview(wkb)

    def header(offset, expected):
        if offset + 5 > len(view):
            raise ValueError('Truncated WKB')
        endian = '<' if view[offset] == 1 else '>'
        if struct.unpack_from(endian + 'I', view, offset + 1)[0] != expected:
            raise ValueError('Expected 2D WKB MultiPolygon/Polygon')
        return endian

    endian = header(0, 6)
    count = struct.unpack_from(endian + 'I', view, 5)[0]
    offset, total_area = 9, 0.0
    for _ in range(count):
        endian = header(offset, 3)
        rings = struct.unpack_from(endian + 'I', view, offset + 5)[0]
        offset += 9
        polygon_area = 0.0
        for ring in range(rings):
            points = struct.unpack_from(endian + 'I', view, offset)[0]
            offset += 4
            end = offset + points * 16
            if end > len(view):
                raise ValueError('Truncated WKB coordinates')
            coordinates = np.frombuffer(view, dtype=endian + 'f8', count=points * 2,
                                        offset=offset).reshape(-1, 2)
            offset = end
            if points < 4 or not np.isfinite(coordinates).all():
                raise ValueError('Invalid ring coordinates')
            x, y = transform(source_crs, target_crs, coordinates[:, 0].tolist(),
                             coordinates[:, 1].tolist())
            x, y = np.asarray(x), np.asarray(y)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite projected coordinates')
            x, y = x - x[0], y - y[0]
            area = abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2))
            polygon_area += area if ring == 0 else -area
        if polygon_area < 0:
            raise ValueError('Hole area exceeds exterior area')
        total_area += polygon_area
    if offset != len(view):
        raise ValueError('Unexpected trailing WKB bytes')
    return total_area


def extent_overlap(left, right):
    bounds = (max(left[0], right[0]), max(left[1], right[1]),
              min(left[2], right[2]), min(left[3], right[3]))
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        return None
    return bounds


def scale_summary(areas):
    values = np.asarray(areas, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Finite nonnegative projected areas required')
    ratio = values / CELL_AREA_M2
    result = {'sample_geometries': len(values), 'cell_area_m2': CELL_AREA_M2,
              'smaller_than_cell': int(np.count_nonzero(ratio < 1)),
              'equal_to_cell': int(np.count_nonzero(ratio == 1)),
              'larger_than_cell': int(np.count_nonzero(ratio > 1)),
              'smaller_fraction': float(np.mean(ratio < 1)),
              'larger_fraction': float(np.mean(ratio > 1))}
    for q in QUANTILES:
        area = float(np.quantile(values, q / 100))
        result[f'area_m2_q{q:02d}'] = area
        result[f'ratio_q{q:02d}'] = area / CELL_AREA_M2
        result[f'equivalent_side_m_q{q:02d}'] = math.sqrt(area)
        result[f'equivalent_radius_m_q{q:02d}'] = math.sqrt(area / math.pi)
    return result


def sample_areas(root: Path, total: int, requested=SAMPLE_SIZE, batch=BATCH_SIZE):
    fids = selected_fids(total, requested)
    areas, null_count = [], 0
    for start in range(0, len(fids), batch):
        chosen = fids[start:start + batch]
        _, returned, geometry, _ = read(root / wui.GDB, layer=wui_values.LAYER,
                                         columns=[], fids=chosen, read_geometry=True,
                                         return_fids=True)
        if not np.array_equal(np.asarray(returned), chosen):
            raise RuntimeError('Sparse geometry FIDs were missing or reordered')
        for blob in geometry:
            if blob is None:
                null_count += 1
            else:
                areas.append(projected_polygon_area(blob))
    return np.asarray(areas), len(fids), null_count


def _csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _plot_extents(path, dillon, wui_bounds):
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for bounds, label, color in ((dillon, 'Dillon raster envelope', '#bf5b17'),
                                 (wui_bounds, 'Projected WUI extent envelope', '#238b45')):
        x0, y0, x1, y1 = np.asarray(bounds) / 1000
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               linewidth=2, edgecolor=color, label=label))
    ax.autoscale_view()
    ax.set(xlabel='EPSG:5070 easting (km)', ylabel='EPSG:5070 northing (km)',
           title='Dataset bounding boxes; not valid-data footprints')
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_scale(path, areas):
    ratio = np.asarray(areas) / CELL_AREA_M2
    positive = ratio[ratio > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log10(positive), bins=65, color='#406b80')
    ax.axvline(0, color='#bf5b17', linewidth=2, label='One Dillon cell')
    ax.set(xlabel='log10(projected polygon area / 72,900 m²)',
           ylabel='Sampled polygon count', title=f'WUI geometry scale (n={len(ratio):,}; zero area={len(ratio)-len(positive):,})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def analyze(root: Path):
    raster = json.loads((root / 'reports/eda/raw/dillon/tables/rasters.json').read_text())[0]
    layer = json.loads((root / 'reports/eda/raw/wui/tables/layers.json').read_text())[0]
    if raster['crs'] != 'EPSG:5070' or layer['crs'] != 'EPSG:4269':
        raise ValueError('Reviewed CRS changed; rerun inventories')
    with (root / 'reports/eda/raw/dillon/tables/distribution_summary.csv').open(newline='') as handle:
        bp_row = next(row for row in csv.DictReader(handle) if row['raster'] == 'CONUS_BP')
    with (root / 'reports/eda/raw/dillon/tables/mask_overlap.csv').open(newline='') as handle:
        mask_row = next(csv.DictReader(handle))
    with rasterio.open(root / raster['path']) as source:
        if (source.width, source.height) != (raster['width'], raster['height']) or list(source.bounds) != raster['bounds']:
            raise ValueError('Dillon source differs from reviewed inventory')
    info = pyogrio.read_info(root / wui.GDB, layer=layer['layer_name'],
                                           force_feature_count=False, force_total_bounds=False)
    if info['features'] != layer['features'] or not np.allclose(info['total_bounds'], layer['total_bounds'], rtol=0, atol=1e-10):
        raise ValueError('WUI source differs from reviewed inventory')
    transformed = transform_bounds('EPSG:4269', 'EPSG:5070', *layer['total_bounds'], densify_pts=128)
    overlap = extent_overlap(raster['bounds'], transformed)
    areas, requested, null = sample_areas(root, layer['features'])
    return {'raster': raster, 'layer': layer, 'transformed': transformed,
            'overlap': overlap, 'areas': areas, 'requested': requested,
            'null': null, 'scale': scale_summary(areas),
            'bp_valid_cells': int(bp_row['valid_pixels']),
            'common_valid_cells': int(mask_row['valid_all_seven'])}


def write_products(result, root: Path):
    base = root / 'reports/eda/raw/cross_dataset'
    tables, figures = base / 'tables', base / 'figures'
    for directory in (tables, figures):
        if directory.resolve() != directory.absolute():
            raise ValueError(f'Output traverses symlink: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
    raster, layer = result['raster'], result['layer']
    spatial = {
        'dillon_crs': raster['crs'], 'dillon_width_cells': raster['width'],
        'dillon_height_cells': raster['height'], 'dillon_transform': raster['transform'],
        'dillon_bounds_m': raster['bounds'], 'dillon_cell_area_m2': CELL_AREA_M2,
        'dillon_valid_bp_cells': result['bp_valid_cells'],
        'dillon_common_valid_cells': result['common_valid_cells'],
        'wui_crs': layer['crs'], 'wui_geometry_type': layer['geometry_type'],
        'wui_feature_rows': layer['features'], 'wui_bounds_degrees': layer['total_bounds'],
        'wui_extent_bounds_5070_m': list(result['transformed']),
        'bbox_overlap_5070_m': list(result['overlap']) if result['overlap'] else None,
        'bbox_overlap_area_m2': ((result['overlap'][2] - result['overlap'][0]) *
                                  (result['overlap'][3] - result['overlap'][1])) if result['overlap'] else 0,
        'extent_transform': 'rasterio.warp.transform_bounds; 128 densification points per edge; transformed rectangular extent envelope, not feature union',
        'support_caveat': 'Bounding-box overlap does not imply identical valid-data support.',
    }
    (tables / 'spatial_compatibility.json').write_text(json.dumps(spatial, indent=2, sort_keys=True) + '\n')
    _csv(tables / 'sampled_polygon_scale_summary.csv', [
        {'metric': key, 'value': value} for key, value in {
            'requested_fids': result['requested'], 'null_sampled_geometries': result['null'],
            **result['scale']}.items()])
    _csv(tables / 'join_strategy_comparison.csv', [
        {'method': 'A cell-center point-in-polygon', 'determinism': 'Yes with explicit overlap tie rule',
         'cost': 'Moderate indexed point queries; many cells', 'boundary': 'Abrupt center assignment; edge overlap ignored',
         'small_polygons': 'Often missed', 'multiple_polygons_per_cell': 'Ambiguous; explicit precedence needed',
         'duplicated_BLK20': 'Treat geometry rows separately; ID is not unique',
         'information_loss': 'High for subcell polygons and fractions',
         'categorical': 'Possible dominant-at-center only', 'intensive': 'Point sample; not cell average',
         'extensive_count': 'Do not copy totals into cells', 'identifier_admin': 'Keep for lineage, not predictor'},
        {'method': 'B rasterization to Dillon grid', 'determinism': 'Yes with fixed order, all_touched and merge rules',
         'cost': 'Moderate to high, streaming/tiled possible', 'boundary': 'Center/default or all_touched changes edge results',
         'small_polygons': 'Default can miss; all_touched can inflate coverage',
         'multiple_polygons_per_cell': 'Last-wins unsafe; explicit reducer required',
         'duplicated_BLK20': 'Rows can be distinct parts; do not deduplicate blindly',
         'information_loss': 'Single-value burn loses composition and overlap',
         'categorical': 'Area/coverage shares if fractional rasterization; otherwise dominant code',
         'intensive': 'Weighted means require accumulation, not simple burn',
         'extensive_count': 'Conservative allocation requires area fractions',
         'identifier_admin': 'Potential trace layer only'},
        {'method': 'C polygon/raster area-overlap aggregation', 'determinism': 'Yes with stable geometry order and fixed precision/rules',
         'cost': 'Highest; spatial index and tiles needed', 'boundary': 'Explicit intersection area; slivers and invalid geometry need policy',
         'small_polygons': 'Retained as fractional coverage',
         'multiple_polygons_per_cell': 'Can aggregate, but overlapping source polygons need normalization',
         'duplicated_BLK20': 'Preserve parts; reconcile repeated count payloads before allocation',
         'information_loss': 'Least spatial loss at 270 m; still aggregates within cells',
         'categorical': 'Per-class area fractions; derive dominant class if desired',
         'intensive': 'Intersection-area weighted mean over valid covered area',
         'extensive_count': 'Allocate row totals by intersection/source-area fraction only after duplicate audit',
         'identifier_admin': 'Keep crosswalk/provenance, not ordinary predictor'},
    ])
    _csv(tables / 'temporal_compatibility.csv', [
        {'source': 'Dillon', 'snapshot': 'LF2020 circa-2020 landscape; hypothetical contemporary fire seasons', 'initial_pairing': 'Target context'},
        {'source': 'WUI/housing/population', 'snapshot': '1990, 2000, 2010, 2020', 'initial_pairing': '2020'},
        {'source': 'WUI vegetation', 'snapshot': '1992, 2001, 2011, 2019', 'initial_pairing': '2019; nearest available, not simultaneous'},
    ])
    _csv(tables / 'dataset_design_implications.csv', [
        {'field': field, 'variable_type': kind,
         'status': ('identifier/admin; retain for provenance, not ordinary predictor' if kind == 'identifier/admin' else
                    'semantic/null resolution before use' if field in {'PUBFLAG', 'BUFVEG'} else
                    'clean extreme/missing density values' if 'DEN' in field else
                    'plausible initial predictor; check redundancy and support'),
         'association': ('crosswalk only' if kind == 'identifier/admin' else
                         'area fractions / explicit category shares' if kind == 'categorical' else
                         'intersection-area weighted mean' if kind == 'intensive' else
                         'conservative fractional allocation after repeated-ID audit')}
        for kind, fields in VARIABLE_TYPES.items() for field in fields
    ])
    _plot_extents(figures / 'dillon_wui_extent_comparison.png', raster['bounds'], result['transformed'])
    _plot_scale(figures / 'sampled_wui_polygon_scale_distribution.png', result['areas'])
