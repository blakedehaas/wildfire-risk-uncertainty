"""Independent, bounded-memory validation of the full canonical build."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

from wildfire_risk_uncertainty.dataset.create_dataset import CATEGORICAL, INTENSIVE
from wildfire_risk_uncertainty.eda.dillon import DILLON_DIR, FILENAMES

WIDTH = 17_372
HEIGHT = 11_283
TILE_SIZE = 256
EXPECTED = {
    'rows': 110_817_231,
    'bp_zero': 29_735_807,
    'bp_positive': 81_081_424,
    'flps_present': 110_809_720,
    'flp_target_valid': 81_068_773,
    'flp_missing_invalid': 18,
    'flp_present_non_normalized': 12_633,
    'flp_not_applicable_bp_zero': 29_735_807,
}
TILE_RE = re.compile(r'row_(\d{5})_col_(\d{5})_h(\d{3})_w(\d{3})\.parquet')


def _values(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _eq(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    if not np.array_equal(actual, expected, equal_nan=actual.dtype.kind == 'f'):
        raise AssertionError(f'{context} differs')


def _scan_csv(path: Path):
    with path.open(newline='') as stream:
        yield from csv.DictReader(stream)


def _pilot_comparison(full_output: Path, pilot_output: Path, pilot_manifest: Path,
                      schema: pa.Schema) -> dict:
    windows = json.loads(pilot_manifest.read_text())['pilot_windows']
    result = {}
    for window in windows:
        row0, col0 = window['row'], window['column']
        row1, col1 = row0 + window['height'], col0 + window['width']
        pieces = []
        for r in range(row0 // TILE_SIZE * TILE_SIZE, row1, TILE_SIZE):
            for c in range(col0 // TILE_SIZE * TILE_SIZE, col1, TILE_SIZE):
                name = f'row_{r:05d}_col_{c:05d}_h{min(TILE_SIZE, HEIGHT-r):03d}_w{min(TILE_SIZE, WIDTH-c):03d}.parquet'
                table = pq.read_table(full_output / name)
                mask = pc.and_(pc.and_(pc.greater_equal(table['row'], row0),
                                       pc.less(table['row'], row1)),
                               pc.and_(pc.greater_equal(table['column'], col0),
                                       pc.less(table['column'], col1)))
                pieces.append(table.filter(mask))
        extracted = pa.concat_tables(pieces).sort_by('cell_id')
        pilot = pq.read_table(pilot_output / f"{window['name']}.parquet")
        if extracted.schema != schema or pilot.schema != schema or extracted.num_rows != pilot.num_rows:
            raise AssertionError(f"Pilot schema/row mismatch: {window['name']}")
        for name in schema.names:
            if not extracted[name].combine_chunks().equals(pilot[name].combine_chunks()):
                raise AssertionError(f"Pilot value mismatch: {window['name']} / {name}")
        result[window['name']] = pilot.num_rows
    return {'windows_match': len(result), 'rows_match': sum(result.values()), 'window_rows': result}


def validate(root: Path, output: Path, run_reports: Path, pilot_output: Path,
             pilot_manifest: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_reports / 'manifest.json').read_text())
    pilot_schema = pq.read_schema(next(pilot_output.glob('*.parquet')))
    expected_names = [item['name'] for item in json.loads(
        (run_reports / 'schema.json').read_text())['columns']]
    if pilot_schema.names != expected_names:
        raise AssertionError('Canonical schema report differs from 4A')
    paths = sorted(output.glob('*.parquet'))
    if len(paths) != 3_060 or len(manifest['partitions']) != len(paths):
        raise AssertionError('Expected 3,060 complete partitions')
    manifest_hashes = {p['name']: p['sha256'] for p in manifest['partitions']}
    counts = Counter()
    nulls = {field: Counter() for field in INTENSIVE}
    category_nulls = {field: 0.0 for field in CATEGORICAL}
    maxima = {'categorical_share_error': 0.0, 'intensive_area_error': 0.0,
              'coverage_fraction': 0.0}
    hash_rows = []
    dataset_bytes = 0
    with ExitStack() as stack:
        rasters = [stack.enter_context(rasterio.open(root / DILLON_DIR / filename))
                   for filename in FILENAMES]
        for index, path in enumerate(paths):
            match = TILE_RE.fullmatch(path.name)
            if not match:
                raise AssertionError(f'Unexpected partition name: {path.name}')
            r, c, h, w = map(int, match.groups())
            expected_r, expected_c = divmod(index, math.ceil(WIDTH / TILE_SIZE))
            expected_r *= TILE_SIZE
            expected_c *= TILE_SIZE
            if (r, c, h, w) != (expected_r, expected_c, min(TILE_SIZE, HEIGHT-r),
                                 min(TILE_SIZE, WIDTH-c)):
                raise AssertionError(f'Partition grid gap/overlap: {path.name}')
            digest = _hash(path)
            if digest != manifest_hashes.get(path.name):
                raise AssertionError(f'Partition SHA256 mismatch: {path.name}')
            size = path.stat().st_size
            dataset_bytes += size
            hash_rows.append({'partition': path.name, 'sha256': digest, 'bytes': size})
            parquet = pq.ParquetFile(path)
            if parquet.schema_arrow != pilot_schema:
                raise AssertionError(f'Schema mismatch: {path.name}')
            table = parquet.read()
            rows = _values(table, 'row')
            cols = _values(table, 'column')
            ids = _values(table, 'cell_id')
            window = Window(c, r, w, h)
            blocks = [raster.read(1, window=window, masked=True) for raster in rasters]
            bp_support = ~np.ma.getmaskarray(blocks[0]) & np.isfinite(blocks[0].data)
            dr, dc = np.nonzero(bp_support)
            if len(ids) != len(dr):
                raise AssertionError(f'BP support count mismatch: {path.name}')
            expected_ids = (dr.astype(np.int64) + r) * WIDTH + dc + c
            _eq(ids, expected_ids, f'Cell IDs in {path.name}')
            if not np.all(rows == dr + r) or not np.all(cols == dc + c):
                raise AssertionError(f'Native coordinates differ in {path.name}')
            transform = rasters[0].transform
            x = transform.c + (cols.astype(np.float64) + 0.5) * transform.a
            y = transform.f + (rows.astype(np.float64) + 0.5) * transform.e
            if (not np.allclose(_values(table, 'x_5070'), x, atol=1e-8, rtol=0)
                    or not np.allclose(_values(table, 'y_5070'), y, atol=1e-8, rtol=0)):
                raise AssertionError(f'Pixel centers differ in {path.name}')
            if len(ids) and (pc.unique(table['grid_id']).to_pylist() != ['Dillon_LF2020_EPSG5070_270m']
                             or pc.unique(table['wui_source_id']).to_pylist() != ['SILVIS_CONUS_WUI_V4_2020']):
                raise AssertionError(f'Source identity differs in {path.name}')
            if not np.all(_values(table, 'bp_valid')):
                raise AssertionError(f'BP-invalid row in {path.name}')
            values = []
            for i, name in enumerate(('BP', *(f'FLP{j}' for j in range(1, 7)))):
                block = blocks[i]
                valid = ~np.ma.getmaskarray(block) & np.isfinite(block.data)
                expected = np.where(valid[dr, dc], block.data[dr, dc], np.nan)
                actual = _values(table, name)
                _eq(actual, expected, f'{name} native values in {path.name}')
                values.append(actual)
            bp = values[0]
            present = np.logical_and.reduce([np.isfinite(v) for v in values[1:]])
            in_range = present & np.logical_and.reduce(
                [(v >= 0) & (v <= 1) for v in values[1:]])
            flp_sum = np.full(len(ids), np.nan)
            if present.any():
                flp_sum[present] = sum(v[present].astype(np.float64) for v in values[1:])
            target_valid = (bp > 0) & in_range & (np.abs(flp_sum - 1) <= 1e-6)
            masks = {
                'flps_present': present,
                'flp_in_range': in_range,
                'flp_target_valid': target_valid,
                'flp_missing_invalid': (bp > 0) & ~present,
                'flp_present_non_normalized': (bp > 0) & present & ~target_valid,
                'flp_not_applicable_bp_zero': bp == 0,
            }
            _eq(_values(table, 'flp_sum'), flp_sum, f'FLP sum in {path.name}')
            for name, expected in masks.items():
                _eq(_values(table, name), expected, f'{name} in {path.name}')
                counts[name] += int(expected.sum())
            counts['rows'] += len(ids)
            counts['bp_zero'] += int((bp == 0).sum())
            counts['bp_positive'] += int((bp > 0).sum())
            failure = _values(table, 'wui_failure')
            counts['wui_failures'] += int(failure.sum())
            counts['tolerated_overlap_cells_from_rows'] += int(np.sum(
                _values(table, 'wui_tolerated_overlap_source_ids') != '[]'))
            counts['repaired_cells'] += int(np.sum(
                _values(table, 'wui_repaired_source_ids') != '[]'))
            coverage = _values(table, 'wui_covered_fraction')
            if np.any(np.isfinite(coverage[failure])) or np.any(~np.isfinite(coverage[~failure])):
                raise AssertionError(f'WUI failure null mismatch: {path.name}')
            if np.any((coverage[~failure] < -1e-6) | (coverage[~failure] > 1 + 1e-6)):
                raise AssertionError(f'Coverage outside tolerance: {path.name}')
            counts['uncovered'] += int(np.sum((coverage == 0) & ~failure))
            counts['partially_covered'] += int(np.sum((coverage > 0) & (coverage < 1 - 1e-6) & ~failure))
            maxima['coverage_fraction'] = max(maxima['coverage_fraction'],
                                              float(np.max(coverage[~failure], initial=0)))
            for field, levels in CATEGORICAL.items():
                shares = [_values(table, f'{field}__share__{level}') for level in levels]
                source_null = _values(table, f'{field}__source_null_share')
                uncovered = _values(table, f'{field}__uncovered_share')
                if (np.any(np.isfinite(source_null[failure]))
                        or np.any(np.isfinite(uncovered[failure]))
                        or any(np.any(np.isfinite(share[failure])) for share in shares)):
                    raise AssertionError(f'{field} failure shares present: {path.name}')
                error = np.abs(sum(share[~failure] for share in shares)
                               + source_null[~failure] + uncovered[~failure] - 1)
                maximum = float(np.max(error, initial=0))
                maxima['categorical_share_error'] = max(maxima['categorical_share_error'], maximum)
                if maximum > 1e-6:
                    raise AssertionError(f'{field} shares fail: {path.name}')
                category_nulls[field] += float(np.sum(source_null[~failure]))
            for field in INTENSIVE:
                value = _values(table, field)
                valid_area = _values(table, f'{field}__valid_area_fraction')
                null_area = _values(table, f'{field}__source_null_area_fraction')
                if (np.any(np.isfinite(value[failure]))
                        or np.any(np.isfinite(valid_area[failure]))
                        or np.any(np.isfinite(null_area[failure]))):
                    raise AssertionError(f'{field} failure area present: {path.name}')
                error = np.abs(valid_area[~failure] + null_area[~failure] - coverage[~failure])
                maximum = float(np.max(error, initial=0))
                maxima['intensive_area_error'] = max(maxima['intensive_area_error'], maximum)
                if maximum > 1e-6 or np.any(np.isnan(value[~failure]) != (valid_area[~failure] == 0)):
                    raise AssertionError(f'{field} area/null mismatch: {path.name}')
                nulls[field]['null_value_rows'] += int(np.isnan(value).sum())
                nulls[field]['source_null_area_cells'] += int(np.sum(null_area[~failure] > 0))
                nulls[field]['source_null_area_fraction_sum'] += float(np.sum(null_area[~failure]))
                nulls[field]['valid_area_fraction_sum'] += float(np.sum(valid_area[~failure]))
    for key, expected in EXPECTED.items():
        if counts[key] != expected:
            raise AssertionError(f'{key}: {counts[key]} != {expected}')
    checkpoint_3d = {record['category']: int(record['count']) for record in _scan_csv(
        root / 'reports/eda/raw/decision_resolution/target_support_accounting.csv')}
    source_mapping = {
        'rows': 'bp_valid_canonical', 'bp_zero': 'bp_valid_zero',
        'bp_positive': 'bp_valid_positive', 'flps_present': 'bp_valid_all_flps_valid',
        'flp_target_valid': 'strict_flp_valid',
        'flp_missing_invalid': 'bp_positive_missing_invalid_flps',
        'flp_present_non_normalized': 'bp_positive_present_non_normalized',
        'flp_not_applicable_bp_zero': 'bp_valid_zero',
    }
    for key, category in source_mapping.items():
        if counts[key] != checkpoint_3d[category]:
            raise AssertionError(f'Checkpoint 3D mismatch: {key}')
    if sum(int(row['rows']) for row in _scan_csv(run_reports / 'pilot_summary.csv')) != counts['rows']:
        raise AssertionError('Constructor per-tile row totals disagree')
    topology = Counter()
    reasons = Counter()
    for row in _scan_csv(run_reports / 'topology_summary.csv'):
        topology['failure_rows'] += 1
        reasons.update(json.loads(row['reasons']))
        for name in ('max_pair_overlap_m2', 'duplicate_overlap_m2', 'coverage_excess_m2',
                     'max_share_error'):
            target = name if name.startswith('max_') else f'max_{name}'
            topology[target] = max(topology[target], float(row[name]))
    if topology['failure_rows'] != counts['wui_failures']:
        raise AssertionError('WUI failure diagnostics disagree with Parquet')
    allowed = {'material_polygon_overlap', 'material_coverage_excess'}
    if any(reason not in allowed and not reason.startswith((
            'material_categorical_reconciliation:', 'geometry:', 'intersection:'))
           for reason in reasons):
        raise AssertionError(f'New topology failure type: {dict(reasons)}')
    for row in _scan_csv(run_reports / 'tolerated_topology_summary.csv'):
        topology['tolerated_rows'] += 1
        topology['tolerated_overlap_pairs'] += int(row['overlap_pair_count'])
        topology['tolerated_overlap_cells'] += int(row['overlap_pair_count'] > '0')
        for source, target in (('max_pair_overlap_m2', 'max_pair_overlap_m2'),
                               ('duplicate_overlap_m2', 'max_duplicate_overlap_m2'),
                               ('coverage_excess_m2', 'max_coverage_excess_m2'),
                               ('max_share_error', 'max_share_error')):
            topology[target] = max(topology[target], float(row[source]))
    if topology['tolerated_overlap_cells'] != counts['tolerated_overlap_cells_from_rows']:
        raise AssertionError('Tolerated overlap diagnostics disagree with Parquet')
    repairs = Counter()
    accepted_ids, rejected_ids = set(), set()
    for row in _scan_csv(run_reports / 'geometry_repair_summary.csv'):
        accepted = row['accepted'] == 'True'
        repairs['accepted' if accepted else 'rejected'] += 1
        repairs[f"reason:{row['reason']}"] += 1
        (accepted_ids if accepted else rejected_ids).add(int(row['objectid']))
        if accepted:
            original = float(row['original_projected_area_m2'])
            repaired = float(row['repaired_projected_area_m2'])
            change = float(row['relative_area_change'])
            if (row['repair_result_type'] not in ('Polygon', 'MultiPolygon')
                    or not all(math.isfinite(v) and v > 0 for v in (original, repaired))
                    or change > 1e-6):
                raise AssertionError(f"Repair policy violation at {row['objectid']}")
    repairs['unique_accepted_source_ids'] = len(accepted_ids)
    repairs['unique_rejected_source_ids'] = len(rejected_ids)
    repairs['accepted'] += 0
    repairs['rejected'] += 0
    if accepted_ids & rejected_ids:
        raise AssertionError('Geometry repair decision differs across tiles')
    pilot = _pilot_comparison(output, pilot_output, pilot_manifest, pilot_schema)
    summary = {
        'target_counts': dict(sorted(counts.items())),
        'dataset_bytes': dataset_bytes,
        'tile_count': len(paths), 'partition_count': len(paths),
        'hashes_match_manifest': True,
        'pilot_comparison': pilot,
        'topology': dict(sorted(topology.items())),
        'topology_reasons': dict(sorted(reasons.items())),
        'geometry_repairs': dict(sorted(repairs.items())),
        'maxima': maxima,
        'intensive_nulls': {field: dict(nulls[field]) for field in INTENSIVE},
        'categorical_source_null_area_fraction_sums': category_nulls,
    }
    return summary, hash_rows


def write_compact_reports(root: Path, run_reports: Path, reports: Path,
                          summary: dict, hashes: list[dict]) -> None:
    """Publish national aggregates and deterministic build metadata."""
    reports.mkdir(parents=True, exist_ok=True)

    def csv_report(name: str, fields: list[str], rows: list[dict]) -> None:
        with (reports / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    counts = summary['target_counts']
    rows = counts['rows']
    csv_report('target_mask_summary.csv', ['mask', 'count'],
               [{'mask': key, 'count': counts[key]} for key in (
                   'rows', 'bp_zero', 'bp_positive', 'flps_present', 'flp_target_valid',
                   'flp_missing_invalid', 'flp_present_non_normalized',
                   'flp_not_applicable_bp_zero')])
    csv_report('coverage_summary.csv', ['metric', 'count', 'fraction_of_rows'],
               [{'metric': key, 'count': counts[key], 'fraction_of_rows': counts[key] / rows}
                for key in ('wui_failures', 'uncovered', 'partially_covered')])
    topology = summary['topology']
    csv_report('topology_summary.csv', ['metric', 'value'],
               [{'metric': key, 'value': value} for key, value in sorted(topology.items())]
               + [{'metric': f'reason:{key}', 'value': value}
                  for key, value in sorted(summary['topology_reasons'].items())])
    repairs = summary['geometry_repairs']
    csv_report('geometry_repair_summary.csv', ['metric', 'count'],
               [{'metric': key, 'count': value} for key, value in sorted(repairs.items())])
    csv_report('null_summary.csv',
               ['field', 'null_value_rows', 'null_value_fraction', 'source_null_area_cells',
                'source_null_area_cell_fraction', 'source_null_area_fraction_sum',
                'source_null_area_fraction_mean', 'valid_area_fraction_sum'],
               [{'field': field,
                 'null_value_rows': record['null_value_rows'],
                 'null_value_fraction': record['null_value_rows'] / rows,
                 'source_null_area_cells': record['source_null_area_cells'],
                 'source_null_area_cell_fraction': record['source_null_area_cells'] / rows,
                 'source_null_area_fraction_sum': record['source_null_area_fraction_sum'],
                 'source_null_area_fraction_mean': record['source_null_area_fraction_sum'] / rows,
                 'valid_area_fraction_sum': record['valid_area_fraction_sum']}
                for field, record in summary['intensive_nulls'].items()])
    csv_report('categorical_null_summary.csv',
               ['field', 'source_null_area_fraction_sum', 'source_null_area_fraction_mean'],
               [{'field': field, 'source_null_area_fraction_sum': value,
                 'source_null_area_fraction_mean': value / rows}
                for field, value in summary['categorical_source_null_area_fraction_sums'].items()])
    csv_report('partition_hashes.csv', ['partition', 'sha256', 'bytes'], hashes)
    build = {'canonical_rows': rows, 'tile_count': summary['tile_count'],
             'partition_count': summary['partition_count'],
             'dataset_bytes': summary['dataset_bytes'],
             'wui_failure_rows': counts['wui_failures'],
             'tolerated_overlap_cells': topology.get('tolerated_overlap_cells', 0),
             'tolerated_overlap_pairs': topology.get('tolerated_overlap_pairs', 0),
             'accepted_repairs': repairs.get('accepted', 0),
             'rejected_repairs': repairs.get('rejected', 0)}
    (reports / 'build_summary.json').write_text(json.dumps(build, indent=2, sort_keys=True) + '\n')
    (reports / 'schema.json').write_bytes((run_reports / 'schema.json').read_bytes())
    manifest = json.loads((run_reports / 'manifest.json').read_text())
    manifest['constructor_commit'] = 'b4dc1bee7d9a870999323e8ca73eb63b40ad1060'
    manifest['tile_size'] = TILE_SIZE
    manifest['tile_count'] = summary['tile_count']
    manifest['partition_count'] = summary['partition_count']
    manifest['canonical_rows'] = rows
    manifest['partition_hashes_csv_sha256'] = _hash(reports / 'partition_hashes.csv')
    manifest.pop('pilot_windows')
    manifest.pop('base_commit')
    (reports / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (reports / 'validation_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/dataset/canonical/partitions'))
    parser.add_argument('--run-reports', type=Path, default=Path('artifacts/dataset/canonical/run_reports'))
    parser.add_argument('--pilot-output', type=Path, default=Path('artifacts/dataset/pilot'))
    parser.add_argument('--pilot-manifest', type=Path, default=Path('reports/dataset/pilot/manifest.json'))
    parser.add_argument('--reports', type=Path, default=Path('reports/dataset/canonical'))
    args = parser.parse_args()
    summary, hashes = validate(args.root, args.output, args.run_reports,
                               args.pilot_output, args.pilot_manifest)
    write_compact_reports(args.root, args.run_reports, args.reports, summary, hashes)
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
