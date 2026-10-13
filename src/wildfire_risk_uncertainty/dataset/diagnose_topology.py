"""Read-only source-topology investigation for the fixed Checkpoint 4A pilot.

Writes diagnostic CSV/Markdown only. It never supplies repaired geometry to the
canonical constructor or changes canonical scientific rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyogrio
import rasterio
import shapely
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box
from shapely.strtree import STRtree

from wildfire_risk_uncertainty.dataset.create_dataset import (
    CATEGORICAL,
    CELL_AREA_M2,
    DILLON_DIR,
    FILENAMES,
    GDB,
    GRID_WIDTH,
    INTENSIVE,
    Tile,
    target_rows,
)

HISTORICAL_AREA_TOLERANCE_M2 = 1e-5

INVALID_IDS = (1122137, 1137147, 1193558, 1205524, 1231163)
RETAINED = (*CATEGORICAL, *INTENSIVE)
DIAGNOSTIC_FIELDS = ('BLK20', *RETAINED, 'PUBFLAG', 'BUFVEG')
QUANTILES = (0, .25, .5, .75, .9, .95, .99, 1)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def render(value):
    return 'null' if value is None else str(value)


def source_rows(gdb: Path, tile: Tile, affine) -> list[dict]:
    bounds = rasterio.windows.bounds(Window(tile.column, tile.row, tile.width, tile.height), affine)
    bbox = transform_bounds('EPSG:5070', 'EPSG:4269', *bounds, densify_pts=21)
    meta, table = pyogrio.read_arrow(gdb, bbox=bbox, columns=list(DIAGNOSTIC_FIELDS),
                                    return_fids=True)
    if meta['crs'] != 'EPSG:4269':
        raise AssertionError(f'Unexpected source CRS: {meta["crs"]}')
    data = {name: table[name].to_pylist() for name in ('OBJECTID', 'Shape', *DIAGNOSTIC_FIELDS)}
    return [dict(fid=int(data['OBJECTID'][i]), wkb=data['Shape'][i],
                 **{field: data[field][i] for field in DIAGNOSTIC_FIELDS})
            for i in sorted(range(table.num_rows), key=lambda j: data['OBJECTID'][j])]


def part_types(geom):
    parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    return [part.geom_type for part in parts]


def geometry_relation(a, b) -> str:
    if a.equals(b):
        return 'equal'
    if a.covers(b) or b.covers(a):
        return 'nested'
    return 'partial_overlap'


def quantile_rows(values, metric, group='all'):
    return [{'metric': metric, 'group': group, 'value': float(np.quantile(values, q)),
             'quantile': q} for q in QUANTILES] if values else []


def bbox_cell_ids(bounds, cells):
    if not bounds or len(bounds) != 4 or not all(map(math.isfinite, bounds)):
        return sorted(cells)
    x0, y0, x1, y1 = bounds
    return [cid for cid, cell in sorted(cells.items())
            if x1 > cell.bounds[0] and x0 < cell.bounds[2]
            and y1 > cell.bounds[1] and y0 < cell.bounds[3]]


def projected_sources(raw_sources, projector):
    valid, invalid = [], []
    audit = Counter()
    seen = set()
    for source in raw_sources:
        fid = source['fid']
        if fid in seen:
            audit['duplicate_retrieved_objectids'] += 1
        seen.add(fid)
        try:
            original = shapely.from_wkb(source['wkb'])
            projected = shapely.transform(original, projector.transform, interleaved=False)
            source = source | {'original': original, 'projected': projected}
            if original.is_empty or projected.is_empty or not original.is_valid or not projected.is_valid:
                invalid.append(source)
                continue
            before_parts = part_types(original)
            after_parts = part_types(projected)
            if before_parts != after_parts:
                audit['reprojection_part_type_or_count_changes'] += 1
            if projected.geom_type == 'MultiPolygon':
                audit['valid_multipolygons'] += 1
                part_area = sum(float(part.area) for part in projected.geoms)
                if abs(part_area - float(projected.area)) > HISTORICAL_AREA_TOLERANCE_M2:
                    audit['multipart_area_sum_mismatches'] += 1
            valid.append(source)
        except (ValueError, TypeError, shapely.errors.GEOSException, OverflowError):
            invalid.append(source | {'original': None, 'projected': None})
    audit['retrieved_rows'] += len(raw_sources)
    audit['distinct_objectids'] += len(seen)
    return valid, invalid, audit


def make_overlap_row(tile, cid, cell, a, b, area, raw_fraction, relation, pair_ids,
                     projector):
    native_intersection = a['original'].intersection(b['original'])
    native_area_deg2 = float(native_intersection.area)
    projected_native_intersection = shapely.transform(
        native_intersection, projector.transform, interleaved=False)
    native_overlap_in_cell_m2 = float(projected_native_intersection.intersection(cell).area)
    row = {
        'pilot_window': tile.name, 'cell_id': cid,
        'objectid_a': a['fid'], 'objectid_b': b['fid'],
        'cell_overlap_objectids': json.dumps(pair_ids, separators=(',', ':')),
        'pairwise_overlap_m2': area, 'pairwise_overlap_cell_fraction': area / CELL_AREA_M2,
        'raw_coverage_fraction': raw_fraction,
        'blk20_a': render(a['BLK20']), 'blk20_b': render(b['BLK20']),
        'same_blk20': a['BLK20'] == b['BLK20'],
        'geometry_relation': relation,
        'source_area_a_m2': float(a['projected'].area),
        'source_area_b_m2': float(b['projected'].area),
        'overlap_fraction_of_smaller_source': area / min(float(a['projected'].area),
                                                         float(b['projected'].area)),
        'native_pair_overlap_deg2': native_area_deg2,
        'native_overlap_projected_cell_m2': native_overlap_in_cell_m2,
        'native_overlap_class': ('source_native_positive_in_cell'
                                 if native_overlap_in_cell_m2 > 0 else
                                 'source_native_positive_elsewhere'
                                 if native_area_deg2 > 0 else 'source_native_zero'),
    }
    for field in (*RETAINED, 'PUBFLAG', 'BUFVEG'):
        row[f'{field}_a'] = render(a[field])
        row[f'{field}_b'] = render(b[field])
        row[f'{field}_identical'] = a[field] == b[field]
    row['all_retained_identical'] = all(a[field] == b[field] for field in RETAINED)
    return row


def diagnose(root: Path, output: Path, reports: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reports / 'manifest.json').read_text())
    if manifest['schema_version'] != '1.0.0':
        raise ValueError('Historical topology diagnostic requires pre-policy schema 1.0.0 pilot outputs')
    topology_rows = list(csv.DictReader((reports / 'topology_summary.csv').open()))
    topology = {int(row['cell_id']): row for row in topology_rows}
    expected_overlap = {cid for cid, row in topology.items()
                        if 'positive_polygon_overlap' in json.loads(row['reasons'])}
    expected_excess = {cid for cid, row in topology.items()
                       if 'coverage_exceeds_cell' in json.loads(row['reasons'])}
    expected_invalid = {cid for cid, row in topology.items()
                        if 'geometry:' in row['reasons']}
    overlap_rows, excess_rows = [], []
    tile_overlap_cells = defaultdict(set)
    tile_pair_count = Counter()
    raw_coverage = {}
    invalid_sources = {}
    invalid_affected = defaultdict(set)
    invalid_repair_overlap = defaultdict(dict)
    audit = Counter()
    per_tile_audit = []
    projector = Transformer.from_crs('EPSG:4269', 'EPSG:5070', always_xy=True)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(root / DILLON_DIR / name))
                   for name in FILENAMES]
        affine = sources[0].transform
        seen_pilot_cells = set()
        for spec in manifest['pilot_windows']:
            tile = Tile(**{key: spec[key] for key in ('row', 'column', 'height', 'width')})
            rows = target_rows(sources, tile, affine)
            parquet_rows = pq.read_table(output / f'{tile.name}.parquet',
                                         columns=['cell_id', 'wui_covered_fraction']).to_pylist()
            if len(rows) != len(parquet_rows):
                raise AssertionError(f'Pilot row count differs in {tile.name}')
            cells = {row['cell_id']: box(*rasterio.windows.bounds(
                Window(row['column'], row['row'], 1, 1), affine)) for row in rows}
            if set(cells) & seen_pilot_cells:
                raise AssertionError('Pilot tiles share BP-valid cell IDs')
            seen_pilot_cells.update(cells)
            raw = source_rows(root / GDB, tile, affine)
            valid, invalid, tile_audit = projected_sources(raw, projector)
            audit.update(tile_audit)
            per_tile_audit.append((tile.name, len(rows), len(raw), len(valid), len(invalid),
                                   tile_audit['duplicate_retrieved_objectids']))
            for source in invalid:
                if source['fid'] not in INVALID_IDS:
                    raise AssertionError(f'Unexpected invalid source {source["fid"]}')
                invalid_sources[source['fid']] = source
                projected = source['projected']
                for cid in bbox_cell_ids(projected.bounds if projected is not None else None, cells):
                    invalid_affected[source['fid']].add(cid)
            geoms = [source['projected'] for source in valid]
            tree = STRtree(geoms)
            fid_lookup = {source['fid']: source for source in valid}
            if len(fid_lookup) != len(valid):
                raise AssertionError(f'Duplicate valid OBJECTID in {tile.name}')
            for stored in parquet_rows:
                cid = stored['cell_id']
                cell = cells[cid]
                indices = sorted((int(i) for i in tree.query(cell)),
                                 key=lambda i: valid[i]['fid'])
                if len(indices) != len(set(indices)):
                    audit['duplicate_spatial_index_hits'] += 1
                pieces = []
                per_cell_fids = set()
                for index in indices:
                    source = valid[index]
                    fid = source['fid']
                    if fid in per_cell_fids:
                        audit['duplicate_objectid_cell_contributions'] += 1
                    per_cell_fids.add(fid)
                    geom = geoms[index]
                    if not geom.intersects(cell):
                        continue
                    intersection = geom.intersection(cell)
                    area = float(intersection.area)
                    if area > HISTORICAL_AREA_TOLERANCE_M2:
                        pieces.append((source, intersection, area))
                raw_area = sum(piece[2] for piece in pieces)
                fraction = raw_area / CELL_AREA_M2
                raw_coverage[cid] = fraction
                stored_fraction = (float(topology[cid]['pre_failure_coverage'])
                                   if cid in topology else stored['wui_covered_fraction'])
                if not math.isclose(fraction, stored_fraction, rel_tol=0, abs_tol=1e-11):
                    audit['independent_coverage_mismatches'] += 1
                pairs = []
                for j, (a, ga, _) in enumerate(pieces):
                    for b, gb, _ in pieces[j+1:]:
                        overlap = float(ga.intersection(gb).area)
                        if overlap > HISTORICAL_AREA_TOLERANCE_M2:
                            pairs.append((a, b, overlap))
                pair_ids = sorted({s['fid'] for a, b, _ in pairs for s in (a, b)})
                for a, b, overlap in pairs:
                    relation = geometry_relation(a['projected'], b['projected'])
                    overlap_rows.append(make_overlap_row(tile, cid, cell, a, b, overlap,
                                                         fraction, relation, pair_ids,
                                                         projector))
                    tile_pair_count[tile.name] += 1
                if pairs:
                    tile_overlap_cells[tile.name].add(cid)
                if fraction > 1 + HISTORICAL_AREA_TOLERANCE_M2 / CELL_AREA_M2:
                    union_area = float(shapely.union_all([piece[1] for piece in pieces]).area)
                    duplicate_area = raw_area - union_area
                    excess = raw_area - CELL_AREA_M2
                    gap = CELL_AREA_M2 - union_area
                    excess_rows.append({
                        'pilot_window': tile.name, 'cell_id': cid,
                        'raw_coverage_fraction': fraction, 'raw_area_m2': raw_area,
                        'excess_coverage_m2': excess,
                        'duplicate_coverage_m2': duplicate_area,
                        'uncovered_gap_m2': gap,
                        'pairwise_overlap_sum_m2': sum(p[2] for p in pairs),
                        'pair_count': len(pairs), 'all_excess_explained_by_overlap':
                            bool(pairs) and duplicate_area + HISTORICAL_AREA_TOLERANCE_M2 >= excess,
                        'overlap_objectids': json.dumps(pair_ids, separators=(',', ':')),
                    })
                # Repaired geometries are evaluated only for diagnosis.
                for source in invalid:
                    fid = source['fid']
                    if cid not in invalid_affected[fid]:
                        continue
                    repaired = shapely.make_valid(source['projected'])
                    try:
                        repaired_piece = repaired.intersection(cell)
                        repaired_area = float(repaired_piece.area)
                    except shapely.errors.GEOSException:
                        repaired_area = 0.0
                        repaired_piece = None
                    if repaired_area <= HISTORICAL_AREA_TOLERANCE_M2 or repaired_piece is None:
                        continue
                    for other, piece, _ in pieces:
                        amount = float(repaired_piece.intersection(piece).area)
                        if amount > HISTORICAL_AREA_TOLERANCE_M2:
                            invalid_repair_overlap[fid][(cid, other['fid'])] = amount
    overlap_rows.sort(key=lambda row: (row['pilot_window'], row['cell_id'],
                                       row['objectid_a'], row['objectid_b']))
    excess_rows.sort(key=lambda row: (row['pilot_window'], row['cell_id']))
    overlap_cells = {row['cell_id'] for row in overlap_rows}
    excess_cells = {row['cell_id'] for row in excess_rows}
    if overlap_cells != expected_overlap or excess_cells != expected_excess:
        raise AssertionError(f'Independent topology mismatch: overlap {len(overlap_cells ^ expected_overlap)}, '
                             f'excess {len(excess_cells ^ expected_excess)}')
    if audit['independent_coverage_mismatches']:
        raise AssertionError(f'Independent raw coverage mismatch: {audit["independent_coverage_mismatches"]}')
    if set(invalid_sources) != set(INVALID_IDS):
        raise AssertionError(f'Invalid source set mismatch: {sorted(invalid_sources)}')
    overlap_columns = [
        'pilot_window', 'cell_id', 'objectid_a', 'objectid_b', 'cell_overlap_objectids',
        'pairwise_overlap_m2', 'pairwise_overlap_cell_fraction', 'raw_coverage_fraction',
        'blk20_a', 'blk20_b', 'same_blk20', 'geometry_relation',
        'source_area_a_m2', 'source_area_b_m2', 'overlap_fraction_of_smaller_source',
        'native_pair_overlap_deg2', 'native_overlap_projected_cell_m2',
        'native_overlap_class',
        *(f'{field}_{suffix}' for field in (*RETAINED, 'PUBFLAG', 'BUFVEG')
          for suffix in ('a', 'b', 'identical')), 'all_retained_identical',
    ]
    write_csv(destination / 'overlap_cells.csv', overlap_rows, overlap_columns)
    write_csv(destination / 'coverage_excess_summary.csv', excess_rows, [
        'pilot_window', 'cell_id', 'raw_coverage_fraction', 'raw_area_m2',
        'excess_coverage_m2', 'duplicate_coverage_m2', 'uncovered_gap_m2',
        'pairwise_overlap_sum_m2', 'pair_count', 'all_excess_explained_by_overlap',
        'overlap_objectids',
    ])
    invalid_rows = []
    for fid in INVALID_IDS:
        source = invalid_sources[fid]
        original, projected = source['original'], source['projected']
        repaired = shapely.make_valid(projected)
        raw_area = float(projected.area)
        repaired_area = float(repaired.area)
        raw_pairs = {}
        raw_failures = 0
        # Compare raw-invalid and repaired overlap against valid source pieces
        # within affected cells. Raw GEOS errors are explicit indeterminacy.
        for spec in manifest['pilot_windows']:
            tile = Tile(**{key: spec[key] for key in ('row', 'column', 'height', 'width')})
            tile_cids = {cid for cid in invalid_affected[fid]
                         if tile.row <= cid // GRID_WIDTH < tile.row + tile.height
                         and tile.column <= cid % GRID_WIDTH < tile.column + tile.width}
            if not tile_cids:
                continue
            raw = source_rows(root / GDB, tile, affine)
            valid, _, _ = projected_sources(raw, projector)
            tree = STRtree([s['projected'] for s in valid])
            for cid in sorted(tile_cids):
                row, col = divmod(cid, GRID_WIDTH)
                cell = box(*rasterio.windows.bounds(Window(col, row, 1, 1), affine))
                try:
                    raw_piece = projected.intersection(cell)
                    for index in sorted(map(int, tree.query(cell))):
                        other = valid[index]
                        other_piece = other['projected'].intersection(cell)
                        overlap = float(raw_piece.intersection(other_piece).area)
                        if overlap > HISTORICAL_AREA_TOLERANCE_M2:
                            raw_pairs[(cid, other['fid'])] = overlap
                except (ValueError, shapely.errors.GEOSException):
                    raw_failures += 1
        repaired_pairs = invalid_repair_overlap[fid]
        raw_keys, repaired_keys = set(raw_pairs), set(repaired_pairs)
        comparable_changes = [abs(raw_pairs.get(key, 0.0) - repaired_pairs.get(key, 0.0))
                              for key in sorted(raw_keys | repaired_keys)]
        invalid_rows.append({
            'objectid': fid,
            'validity_reason': shapely.is_valid_reason(original),
            'projected_raw_area_m2': raw_area,
            'make_valid_result_type': repaired.geom_type,
            'make_valid_area_m2': repaired_area,
            'relative_area_change': (repaired_area - raw_area) / raw_area if raw_area else None,
            'part_count': len(part_types(repaired)),
            'part_types': json.dumps(part_types(repaired), separators=(',', ':')),
            'bounds_identical': projected.bounds == repaired.bounds,
            'hausdorff_distance_m': float(shapely.hausdorff_distance(projected, repaired)),
            'spatially_colocated': (projected.bounds == repaired.bounds and
                                    float(shapely.hausdorff_distance(projected, repaired)) == 0),
            'affected_pilot_cells': len(invalid_affected[fid]),
            'raw_overlap_pairs_computable': len(raw_pairs),
            'raw_intersection_failure_cells': raw_failures,
            'make_valid_overlap_pairs': len(repaired_pairs),
            'make_valid_overlap_area_m2': sum(repaired_pairs.values()),
            'new_overlap_pairs_when_comparable': len(repaired_keys - raw_keys) if raw_failures == 0 else None,
            'removed_overlap_pairs_when_comparable': len(raw_keys - repaired_keys) if raw_failures == 0 else None,
            'material_overlap_area_change_pairs_when_comparable':
                sum(change > HISTORICAL_AREA_TOLERANCE_M2 for change in comparable_changes) if raw_failures == 0 else None,
            'max_overlap_area_change_m2_when_comparable':
                max(comparable_changes, default=0.0) if raw_failures == 0 else None,
            'overlap_comparison_status': 'complete' if raw_failures == 0 else 'raw_invalid_intersection_failed',
        })
    write_csv(destination / 'invalid_geometry_summary.csv', invalid_rows,
              list(invalid_rows[0]))
    summary = []

    def add(metric, value, group='all', quantile=''):
        summary.append({'metric': metric, 'group': group, 'quantile': quantile, 'value': value})

    add('overlap_cells', len(overlap_cells))
    add('overlap_pairs', len(overlap_rows))
    add('same_blk20_pairs', sum(row['same_blk20'] for row in overlap_rows))
    add('different_blk20_pairs', sum(not row['same_blk20'] for row in overlap_rows))
    add('identical_retained_attribute_pairs', sum(row['all_retained_identical'] for row in overlap_rows))
    add('conflicting_retained_attribute_pairs', sum(not row['all_retained_identical'] for row in overlap_rows))
    add('equal_geometry_pairs', sum(row['geometry_relation'] == 'equal' for row in overlap_rows))
    add('nested_geometry_pairs', sum(row['geometry_relation'] == 'nested' for row in overlap_rows))
    add('partial_overlap_geometry_pairs', sum(row['geometry_relation'] == 'partial_overlap' for row in overlap_rows))
    add('source_native_positive_in_cell_pairs', sum(row['native_overlap_class'] == 'source_native_positive_in_cell'
                                                    for row in overlap_rows))
    for klass in ('source_native_positive_elsewhere', 'source_native_zero'):
        add(f'{klass}_pairs', sum(row['native_overlap_class'] == klass for row in overlap_rows))
    add('source_native_positive_in_cell_cells', len({row['cell_id'] for row in overlap_rows
                                                     if row['native_overlap_class'] == 'source_native_positive_in_cell'}))
    for klass in ('source_native_positive_elsewhere', 'source_native_zero'):
        add(f'{klass}_cells', len({row['cell_id'] for row in overlap_rows
                                  if row['native_overlap_class'] == klass}))
    add('overlap_cells_also_invalid_geometry', len(overlap_cells & expected_invalid))
    add('overlap_pairs_directly_attributable_to_invalid_geometry', 0)
    for field in (*RETAINED, 'PUBFLAG', 'BUFVEG'):
        add('conflicting_attribute_pairs', sum(not row[f'{field}_identical'] for row in overlap_rows), field)
    for metric, values in (
        ('pairwise_overlap_m2', [row['pairwise_overlap_m2'] for row in overlap_rows]),
        ('pairwise_overlap_cell_fraction', [row['pairwise_overlap_cell_fraction'] for row in overlap_rows]),
        ('overlap_cell_raw_coverage_fraction', [raw_coverage[cid] for cid in sorted(overlap_cells)]),
        ('excess_cell_raw_coverage_fraction', [row['raw_coverage_fraction'] for row in excess_rows]),
    ):
        for q in QUANTILES:
            add(metric, float(np.quantile(values, q)), quantile=q)
    for spec in manifest['pilot_windows']:
        tile = spec['name']
        add('overlap_cells', len(tile_overlap_cells[tile]), tile)
        add('overlap_pairs', tile_pair_count[tile], tile)
    add('coverage_excess_cells', len(excess_rows))
    add('coverage_excess_all_with_overlap', all(row['all_excess_explained_by_overlap'] for row in excess_rows))
    add('coverage_excess_max_raw_fraction', max(row['raw_coverage_fraction'] for row in excess_rows))
    add('coverage_excess_total_m2', sum(row['excess_coverage_m2'] for row in excess_rows))
    add('coverage_excess_duplicate_area_total_m2', sum(row['duplicate_coverage_m2'] for row in excess_rows))
    add('coverage_excess_uncovered_gap_total_m2', sum(row['uncovered_gap_m2'] for row in excess_rows))
    write_csv(destination / 'overlap_summary.csv', summary, ['metric', 'group', 'quantile', 'value'])
    audit_text = [
        '# Checkpoint 4A source topology implementation audit', '',
        ('The independent diagnostic uses `STRtree` over projected source geometries for every BP-valid pilot cell. '
         'It sorts candidate indices by OBJECTID and intersects each source polygon with the cell once. '
         'It does not use the constructor’s affine candidate-range loop or its WUI aggregation results.'), '',
        '## Retrieval and accumulation', '',
        '| Pilot window | BP-valid cells | retrieved rows | valid polygons | invalid polygons | duplicate retrieved OBJECTIDs |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
        *(f'| {name} | {cells} | {raw} | {valid} | {invalid} | {duplicates} |'
          for name, cells, raw, valid, invalid, duplicates in per_tile_audit),
        '',
        f'- Duplicate retrieved OBJECTIDs: {audit["duplicate_retrieved_objectids"]}.',
        f'- Duplicate spatial-tree indices: {audit["duplicate_spatial_index_hits"]}.',
        f'- Duplicate OBJECTID/cell contributions: {audit["duplicate_objectid_cell_contributions"]}.',
        (f'- Valid MultiPolygons checked: {audit["valid_multipolygons"]}; part-area mismatches: '
         f'{audit["multipart_area_sum_mismatches"]}.'),
        f'- Reprojection part-count/type changes: {audit["reprojection_part_type_or_count_changes"]}.',
        '- Duplicate BP-valid cell IDs across tiles: 0 (asserted).',
        (f'- Independent raw coverage mismatches against canonical pilot: '
         f'{audit["independent_coverage_mismatches"]}.'),
        (f'- Independent overlap and excess cell sets match canonical diagnostics exactly: '
         f'{len(overlap_cells)} and {len(excess_cells)} cells.'),
        '',
        ('The source query returns distinct OBJECTIDs within every tile. The canonical constructor sorts '
         'polygon candidates by OBJECTID and iterates each native cell coordinate once per polygon. '
         'Shapely projects each MultiPolygon as one geometry; `intersection(cell)` is called once per '
         'OBJECTID/cell. The independent spatial tree yields the same raw coverage, overlap cells and '
         'excess cells. The fixed pilot windows have disjoint cell IDs. These checks show no duplicate '
         'contribution from retrieval, part handling, spatial ordering or tile boundaries.'),
        '',
        '## Interpretation limits', '',
        (f'Of {len(overlap_rows)} projected positive-area pairs, '
         f'{sum(row["native_overlap_class"] == "source_native_positive_in_cell" for row in overlap_rows)} '
         'also have a positive source-CRS intersection projected into that cell; '
         f'{sum(row["native_overlap_class"] == "source_native_positive_elsewhere" for row in overlap_rows)} '
         'have source-CRS overlap elsewhere but none found in that cell; and '
         f'{sum(row["native_overlap_class"] == "source_native_zero" for row in overlap_rows)} '
         'have zero source-CRS pair-intersection area. The latter two groups indicate '
         'projection/overlay representation effects in the cell, not duplicate retrieval.'),
        (f'For the {len(excess_rows)} excess-coverage cells, total excess is '
         f'{sum(row["excess_coverage_m2"] for row in excess_rows):.12g} m², '
         f'duplicate coverage is {sum(row["duplicate_coverage_m2"] for row in excess_rows):.12g} m², '
         f'and uncovered gap is {sum(row["uncovered_gap_m2"] for row in excess_rows):.12g} m². '
         'The identity excess = duplicate coverage − uncovered gap holds within floating-point error.'),
        '',
        ('Positive overlaps are intersections of distinct, individually valid projected source geometries. '
         'The five invalid source geometries are skipped by the canonical overlay, so they do not directly '
         'cause those overlap pairs. Their bounding envelopes mask affected cells separately. '
         'The overlap areas and source attributes are in the CSV tables. Tiny positive areas may reflect '
         'source coordinate precision, but choosing a tolerance or topology resolution is a scientific '
         'decision and is not made here.'),
        'The `make_valid` metrics are diagnostic only. A repaired geometry has not been used in canonical output.',
        '',
    ]
    (destination / 'implementation_audit.md').write_text('\n'.join(audit_text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/dataset/pilot'))
    parser.add_argument('--reports', type=Path, default=Path('reports/dataset/pilot'))
    parser.add_argument('--destination', type=Path,
                        default=Path('reports/dataset/pilot/hitl_topology'))
    args = parser.parse_args()
    diagnose(args.root, args.output, args.reports, args.destination)


if __name__ == '__main__':
    main()
