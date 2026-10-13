"""Independent native-raster and internal overlay checks on built pilot partitions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

from wildfire_risk_uncertainty.dataset.create_dataset import (
    CATEGORICAL,
    FLP_TOLERANCE,
    GRID_WIDTH,
    INTENSIVE,
    SHARE_TOLERANCE,
    Tile,
    category_column,
)
from wildfire_risk_uncertainty.eda.dillon import DILLON_DIR, FILENAMES


def validate(root: Path, output: Path, reports: Path) -> dict:
    manifest = json.loads((reports / 'manifest.json').read_text())
    schema = json.loads((reports / 'schema.json').read_text())
    with (reports / 'geometry_repair_summary.csv').open(newline='') as handle:
        repairs = list(csv.DictReader(handle))
    accepted_ids = set()
    for repair in repairs:
        if repair['accepted'] == 'True':
            original = float(repair['original_projected_area_m2'])
            repaired = float(repair['repaired_projected_area_m2'])
            change = float(repair['relative_area_change'])
            if (repair['repair_result_type'] not in ('Polygon', 'MultiPolygon')
                    or not all(math.isfinite(v) and v > 0 for v in (original, repaired))
                    or change > manifest['tolerances']['geometry_repair_relative_area_change']):
                raise AssertionError(f'Accepted repair violates policy: {repair["objectid"]}')
            accepted_ids.add(int(repair['objectid']))
    expected_names = [c['name'] for c in schema['columns']]
    seen = set()
    counts = Counter()
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(root / DILLON_DIR / name))
                   for name in FILENAMES]
        for tile_spec in manifest['pilot_windows']:
            tile = Tile(**{key: tile_spec[key] for key in ('row', 'column', 'height', 'width')})
            table = pq.read_table(output / f'{tile.name}.parquet')
            if table.column_names != expected_names:
                raise AssertionError(f'Column order differs in {tile.name}')
            rows = table.to_pylist()
            ids = [row['cell_id'] for row in rows]
            if ids != sorted(ids):
                raise AssertionError(f'Unsorted cell_id in {tile.name}')
            window = Window(tile.column, tile.row, tile.width, tile.height)
            blocks = [source.read(1, window=window, masked=True) for source in sources]
            bp_valid = ~np.ma.getmaskarray(blocks[0]) & np.isfinite(blocks[0].data)
            if len(rows) != int(bp_valid.sum()):
                raise AssertionError(f'BP support mismatch in {tile.name}')
            for row in rows:
                cid = row['cell_id']
                if cid in seen or cid != row['row'] * GRID_WIDTH + row['column']:
                    raise AssertionError(f'Invalid/duplicate cell_id {cid}')
                seen.add(cid)
                dr, dc = row['row'] - tile.row, row['column'] - tile.column
                if not bp_valid[dr, dc] or not row['bp_valid']:
                    raise AssertionError(f'BP-invalid row {cid}')
                for index, field in enumerate(('BP', *(f'FLP{i}' for i in range(1, 7)))):
                    block = blocks[index]
                    valid = not bool(np.ma.getmaskarray(block)[dr, dc]) and math.isfinite(float(block.data[dr, dc]))
                    expected = float(block.data[dr, dc]) if valid else None
                    if row[field] != expected:
                        raise AssertionError(f'{field} raster mismatch at {cid}')
                flps = [row[f'FLP{i}'] for i in range(1, 7)]
                present = all(value is not None for value in flps)
                flp_sum = sum(float(v) for v in flps) if present else None
                valid = bool(row['BP'] > 0 and present and all(0 <= v <= 1 for v in flps)
                             and abs(flp_sum - 1) <= FLP_TOLERANCE)
                if row['flp_sum'] != flp_sum or row['flp_target_valid'] != valid:
                    raise AssertionError(f'FLP mask mismatch at {cid}')
                if row['BP'] == 0:
                    counts['bp_zero'] += 1
                    if not row['flp_not_applicable_bp_zero'] or row['flp_target_valid']:
                        raise AssertionError(f'BP zero mask mismatch at {cid}')
                if valid:
                    counts['flp_valid'] += 1
                repaired_ids = json.loads(row['wui_repaired_source_ids'])
                tolerated_ids = json.loads(row['wui_tolerated_overlap_source_ids'])
                if (repaired_ids != sorted(set(repaired_ids))
                        or not set(repaired_ids) <= accepted_ids
                        or tolerated_ids != sorted(set(tolerated_ids))):
                    raise AssertionError(f'Invalid WUI provenance at {cid}')
                if repaired_ids:
                    counts['repaired_cells'] += 1
                if tolerated_ids:
                    counts['tolerated_overlap_cells'] += 1
                if row['wui_failure']:
                    counts['wui_failure'] += 1
                    if not row['wui_failure_source_ids'] or row['wui_covered_fraction'] is not None:
                        raise AssertionError(f'WUI failure diagnostics missing at {cid}')
                    continue
                coverage = row['wui_covered_fraction']
                if coverage < -SHARE_TOLERANCE or coverage > 1 + SHARE_TOLERANCE:
                    raise AssertionError(f'Coverage out of bounds at {cid}')
                if coverage == 0:
                    counts['uncovered'] += 1
                    if any(row[field] is not None for field in INTENSIVE):
                        raise AssertionError(f'Uncovered intensive value at {cid}')
                if 0 < coverage < 1 - SHARE_TOLERANCE:
                    counts['partial_coverage'] += 1
                for field, levels in CATEGORICAL.items():
                    total = sum(row[category_column(field, level)] for level in levels)
                    total += row[f'{field}__source_null_share'] + row[f'{field}__uncovered_share']
                    if abs(total - 1) > SHARE_TOLERANCE:
                        raise AssertionError(f'Categorical shares fail at {cid}, {field}')
                for field in INTENSIVE:
                    valid_fraction = row[f'{field}__valid_area_fraction']
                    null_fraction = row[f'{field}__source_null_area_fraction']
                    if abs(valid_fraction + null_fraction - coverage) > SHARE_TOLERANCE:
                        raise AssertionError(f'Intensive area accounting fails at {cid}, {field}')
                    if (row[field] is None) != (valid_fraction == 0):
                        raise AssertionError(f'Intensive null mismatch at {cid}, {field}')
                    if null_fraction > 0:
                        counts[f'{field}_source_null_cells'] += 1
                counts['rows'] += 1
    counts['rows'] += counts['wui_failure']
    with (reports / 'tolerated_topology_summary.csv').open(newline='') as handle:
        tolerated = list(csv.DictReader(handle))
    with (reports / 'topology_summary.csv').open(newline='') as handle:
        material = list(csv.DictReader(handle))
    if (counts['tolerated_overlap_cells'] != sum(int(r['overlap_pair_count']) > 0 for r in tolerated)
            or counts['wui_failure'] != len(material)):
        raise AssertionError('WUI QA tables do not reconcile with Parquet rows')
    counts['accepted_repairs'] = len(accepted_ids)
    counts['rejected_repairs'] = sum(repair['accepted'] == 'False' for repair in repairs)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/dataset/pilot'))
    parser.add_argument('--reports', type=Path, default=Path('reports/dataset/pilot'))
    args = parser.parse_args()
    summary = validate(args.root, args.output, args.reports)
    (args.reports / 'validation_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
