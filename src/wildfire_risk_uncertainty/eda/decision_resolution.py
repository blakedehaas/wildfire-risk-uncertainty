"""Checkpoint 3D source-only accounting; no canonical dataset is built."""

import csv
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from pyogrio.raw import read
from rasterio.windows import Window

from . import cross_dataset, dillon, wui, wui_values

FIELDS = ('BLK20', 'HU2020', 'POP2020', 'OCCHU2020', 'VACHU2020',
          'HUDEN2020', 'POPDEN2020', 'OCCHUDEN2020', 'VACHUDEN2020',
          'PUBFLAG', 'BUFVEG', 'WATER20', 'AWATER20PC', 'WUIFLAG2020', 'WUICLASS2020')
COUNTS = ('HU2020', 'POP2020', 'OCCHU2020', 'VACHU2020')
DENSITIES = ('HUDEN2020', 'POPDEN2020', 'OCCHUDEN2020', 'VACHUDEN2020')


def classify_target(bp, bp_valid, flps, flp_valid):
    """Disjoint states including outside-canonical cells; sums use float64."""
    bp = np.asarray(bp)
    flps = np.asarray(flps, dtype=np.float64)
    present = np.all(flp_valid & np.isfinite(flps), axis=0)
    sums = np.sum(flps, axis=0, dtype=np.float64)
    in_range = np.all((flps >= 0) & (flps <= 1), axis=0)
    zero = bp_valid & (bp == 0)
    positive = bp_valid & (bp > 0)
    strict = positive & present & in_range & (np.abs(sums - 1) <= 1e-6)
    return {
        'excluded_bp_invalid_flp_present': ~bp_valid & present,
        'excluded_bp_invalid_flp_invalid': ~bp_valid & ~present,
        'retained_bp_zero_flp_present': zero & present,
        'retained_bp_zero_flp_invalid': zero & ~present,
        'retained_bp_positive_flp_valid': strict,
        'retained_bp_positive_flp_missing_invalid': positive & ~present,
        'retained_bp_positive_flp_present_non_normalized': positive & present & ~strict,
    }


def pattern(count_vectors, flags):
    """Classify repeated block payload without assuming each row is a census block."""
    values = np.asarray(count_vectors, dtype=np.float64)
    if np.all(values == values[0]):
        return 'repeated_unchanged'
    if np.any(~np.isfinite(values)):
        return 'missing_count'
    public = np.asarray(flags, dtype=np.float64) == 1
    private = np.asarray(flags, dtype=np.float64) == 0
    if public.any() and private.any() and np.all(values[public] == 0) and np.any(values[private] > 0):
        return 'zeroed_public_private_positive'
    if np.any(np.all(values == 0, axis=1)):
        return 'some_pieces_zeroed'
    return 'varied_nonzero'


def missingness_summary(values, groups):
    """Exact missing/present crosstab for a synthetic or source array."""
    absent = ~np.isfinite(np.asarray(values, dtype=np.float64))
    labels = np.asarray(groups).astype(str)
    return {label: (int(np.count_nonzero(absent & (labels == label))),
                    int(np.count_nonzero(~absent & (labels == label))))
            for label in sorted(set(labels))}


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def target_scan(root):
    counts = Counter()
    with ExitStack() as stack, rasterio.Env(GDAL_PAM_ENABLED='NO', GDAL_CACHEMAX=64 * 1024 * 1024):
        sources = [stack.enter_context(rasterio.open(root / dillon.DILLON_DIR / name))
                   for name in dillon.FILENAMES]
        height, width = sources[0].shape
        for top in range(0, height, 1024):
            for left in range(0, width, 1024):
                window = Window(left, top, min(1024, width-left), min(1024, height-top))
                blocks = [src.read(1, window=window, masked=True) for src in sources]
                bp_valid = ~np.ma.getmaskarray(blocks[0]) & np.isfinite(blocks[0].data)
                flps = np.stack([block.data for block in blocks[1:]])
                valid = np.stack([~np.ma.getmaskarray(block) for block in blocks[1:]])
                states = classify_target(blocks[0].data, bp_valid, flps, valid)
                counts.update({key: int(np.count_nonzero(mask)) for key, mask in states.items()})
        counts['total_raster_cells'] = width * height
    assert sum(v for k, v in counts.items() if k != 'total_raster_cells') == counts['total_raster_cells']
    return counts


def target_rows(c):
    canonical = sum(v for k, v in c.items() if k.startswith('retained_'))
    positive = sum(v for k, v in c.items() if k.startswith('retained_bp_positive_'))
    present = c['retained_bp_zero_flp_present'] + c['retained_bp_positive_flp_valid'] + c['retained_bp_positive_flp_present_non_normalized']
    invalid = c['retained_bp_zero_flp_invalid'] + c['retained_bp_positive_flp_missing_invalid']
    bp_invalid = c['total_raster_cells'] - canonical
    rows = [('total_raster_cells', c['total_raster_cells'], c['total_raster_cells']),
            ('bp_valid_canonical', canonical, c['total_raster_cells']),
            ('bp_invalid_excluded', bp_invalid, c['total_raster_cells']),
            ('bp_valid_zero', c['retained_bp_zero_flp_present'] + c['retained_bp_zero_flp_invalid'], canonical),
            ('bp_valid_positive', positive, canonical),
            ('bp_valid_all_flps_valid', present, canonical),
            ('bp_valid_any_flp_invalid', invalid, canonical),
            ('strict_flp_valid', c['retained_bp_positive_flp_valid'], positive),
            ('bp_valid_masked_from_flp', canonical - c['retained_bp_positive_flp_valid'], canonical),
            ('bp_positive_missing_invalid_flps', c['retained_bp_positive_flp_missing_invalid'], positive),
            ('bp_positive_present_non_normalized', c['retained_bp_positive_flp_present_non_normalized'], positive),
            ('bp_invalid_flp_valid_excluded', c['excluded_bp_invalid_flp_present'], bp_invalid)]
    rows += [(key, value, c['total_raster_cells'] if key.startswith('excluded_') else canonical)
             for key, value in sorted(c.items()) if key.startswith(('excluded_', 'retained_'))]
    return [{'category': key, 'count': value, 'denominator': denominator,
             'fraction': value / denominator} for key, value, denominator in rows]


def wui_scan(root, output):
    source = root / wui.GDB
    total = int(pyogrio.read_info(source, layer=wui_values.LAYER)['features'])
    missing = defaultdict(int)
    tails = defaultdict(lambda: {'max': -np.inf, 'fid': None, 'count': None, 'small_area_proxy': None,
                                  'water': None, 'pubflag': None, 'rows_above_100000': 0})
    flag_counts = defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix='checkpoint3d-') as tmp:
        con = sqlite3.connect(str(Path(tmp) / 'groups.sqlite'))
        con.execute('PRAGMA journal_mode=OFF')
        con.execute('PRAGMA synchronous=OFF')
        con.execute('CREATE TABLE records (id TEXT, hu REAL, pop REAL, occ REAL, vac REAL, pub REAL)')
        for offset, count in wui_values.batch_offsets(total, 100_000):
            info, _, _, arrays = read(source, layer=wui_values.LAYER, columns=list(FIELDS),
                                      skip_features=offset, max_features=count, read_geometry=False)
            block = dict(zip(info['fields'], arrays, strict=True))
            n = len(block['BLK20'])
            con.executemany('INSERT INTO records VALUES (?,?,?,?,?,?)',
                            zip(*(block[f].tolist() for f in ('BLK20', *COUNTS, 'PUBFLAG')), strict=True))
            for field in ('PUBFLAG', 'BUFVEG'):
                flag_counts[field].update(wui_values.category_key(field, x) or 'null' for x in block[field])
            for field in DENSITIES[1:]:
                vals = np.asarray(block[field], dtype=np.float64)
                bad = ~np.isfinite(vals)
                for other in ('WATER20', 'PUBFLAG', 'WUIFLAG2020', 'WUICLASS2020'):
                    x = block[other]
                    for key, (n_missing, n_present) in missingness_summary(vals, x).items():
                        missing[field, other, key, 'missing'] += n_missing
                        missing[field, other, key, 'present'] += n_present
                for count_field in COUNTS:
                    x = np.asarray(block[count_field], dtype=np.float64)
                    for label, selector in (('zero', x == 0), ('positive', x > 0), ('missing', ~np.isfinite(x))):
                        missing[field, count_field, label, 'missing'] += int(np.count_nonzero(bad & selector))
                        missing[field, count_field, label, 'present'] += int(np.count_nonzero(~bad & selector))
                for label, selector in (('zero', np.asarray(block['AWATER20PC']) == 0),
                                        ('positive', np.asarray(block['AWATER20PC']) > 0)):
                    missing[field, 'AWATER20PC', label, 'missing'] += int(np.count_nonzero(bad & selector))
                    missing[field, 'AWATER20PC', label, 'present'] += int(np.count_nonzero(~bad & selector))
            for field, count_field in zip(DENSITIES, COUNTS, strict=True):
                vals = np.asarray(block[field], dtype=np.float64)
                tail = tails[field]
                tail['rows_above_100000'] += int(np.count_nonzero(vals > 100000))
                if np.isfinite(vals).any():
                    i = int(np.nanargmax(vals))
                    if vals[i] > tail['max']:
                        tail.update(max=float(vals[i]), fid=offset+i+1, count=float(block[count_field][i]),
                                    water=str(block['WATER20'][i]), pubflag=str(block['PUBFLAG'][i]))
            if offset % 1_000_000 == 0:
                print(f'WUI scan {offset+n}/{total}', flush=True)
        con.commit()
        con.execute('CREATE INDEX records_id ON records(id)')
        group_counts = Counter()
        examples = {}
        cur = con.execute('SELECT id, hu, pop, occ, vac, pub FROM records WHERE id IN '
                          '(SELECT id FROM records GROUP BY id HAVING COUNT(*)>1) ORDER BY id')
        prior = None
        group = []
        def finish(rows):
            if not rows:
                return
            label = pattern([r[1:5] for r in rows], [r[5] for r in rows])
            group_counts[label] += 1
            if label not in examples:
                examples[label] = {'BLK20': rows[0][0], 'rows': len(rows),
                                   'values': [list(r[1:]) for r in rows[:5]]}
        for row in cur:
            if prior is not None and row[0] != prior:
                finish(group)
                group = []
            group.append(row)
            prior = row[0]
        finish(group)
    tail_rows = []
    for field, data in tails.items():
        fid = data['fid']
        info, _, geometries, arrays = read(source, layer=wui_values.LAYER, columns=['BLK20'],
                                          fids=[fid], read_geometry=True)
        area = cross_dataset.projected_polygon_area(geometries[0]) / 1e6
        tail_rows.append({'field': field, 'maximum': data['max'], 'fid': fid,
                          'BLK20': arrays[0][0], 'count_at_max': data['count'],
                          'projected_area_km2': area, 'count_div_area': data['count']/area if area else '',
                          'PUBFLAG': data['pubflag'], 'WATER20': data['water'],
                          'rows_above_100000': data['rows_above_100000']})
    write_csv(output / 'density_tail_diagnostics.csv', tail_rows)
    write_csv(output / 'density_missingness.csv',
              [{'density': field, 'stratifier': other, 'value': key,
                'missing': missing[field, other, key, 'missing'],
                'present': missing[field, other, key, 'present']}
               for field, other, key in sorted({k[:3] for k in missing})])
    write_csv(output / 'blk20_count_semantics.csv',
              [{'pattern': label, 'groups': n, 'example': json.dumps(examples[label], sort_keys=True)}
               for label, n in sorted(group_counts.items())])
    write_csv(output / 'pubflag_bufveg_semantics.csv',
              [{'field': field, 'value': key, 'rows': n} for field in ('PUBFLAG', 'BUFVEG')
               for key, n in sorted(flag_counts[field].items())])
    return group_counts


def run(root):
    output = root / 'reports/eda/raw/decision_resolution'
    output.mkdir(parents=True, exist_ok=True)
    counts = target_scan(root)
    write_csv(output / 'target_support_accounting.csv', target_rows(counts))
    print('Target counts:', dict(counts), flush=True)
    groups = wui_scan(root, output)
    print('Repeated BLK20 groups:', dict(groups), flush=True)
    group_detail_scan(root)
    tail_population_scan(root)


def group_detail_scan(root):
    """Second attribute-only pass for per-field repeated-block and multiplicity evidence."""
    source = root / wui.GDB
    total = int(pyogrio.read_info(source, layer=wui_values.LAYER)['features'])
    output = root / 'reports/eda/raw/decision_resolution'
    tail_ids = {row['BLK20'] for row in csv.DictReader((output / 'density_tail_diagnostics.csv').open())}
    field_patterns = Counter()
    multiplicity_missing = Counter()
    tail_multiplicity = {}
    with tempfile.TemporaryDirectory(prefix='checkpoint3d-groups-') as tmp:
        con = sqlite3.connect(str(Path(tmp) / 'groups.sqlite'))
        con.execute('PRAGMA journal_mode=OFF')
        con.execute('PRAGMA synchronous=OFF')
        con.execute('CREATE TABLE records (id TEXT, hu REAL, pop REAL, occ REAL, vac REAL, pub REAL, missing INTEGER)')
        fields = ('BLK20', *COUNTS, 'PUBFLAG', 'POPDEN2020')
        for offset, count in wui_values.batch_offsets(total, 150_000):
            info, _, _, arrays = read(source, layer=wui_values.LAYER, columns=list(fields),
                                      skip_features=offset, max_features=count, read_geometry=False)
            block = dict(zip(info['fields'], arrays, strict=True))
            missing_density = ~np.isfinite(np.asarray(block['POPDEN2020'], dtype=np.float64))
            con.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',
                            zip(*(block[f].tolist() for f in fields[:-1]),
                                missing_density.astype(int).tolist(), strict=True))
        con.commit()
        con.execute('CREATE INDEX records_id ON records(id)')
        cursor = con.execute('SELECT * FROM records ORDER BY id')
        previous = None
        group = []
        def finish(rows):
            if not rows:
                return
            n = len(rows)
            tail_multiplicity.update({rows[0][0]: n} if rows[0][0] in tail_ids else {})
            multiplicity_missing[n, 'missing'] += sum(row[-1] for row in rows)
            multiplicity_missing[n, 'present'] += n - sum(row[-1] for row in rows)
            if n == 1:
                return
            public = np.array([row[5] == 1 for row in rows])
            private = np.array([row[5] == 0 for row in rows])
            for i, field in enumerate(COUNTS, start=1):
                vals = np.array([row[i] for row in rows], dtype=np.float64)
                if np.all(vals == vals[0]):
                    label = 'repeated_unchanged'
                elif public.any() and private.any() and np.all(vals[public] == 0) and np.any(vals[private] > 0):
                    label = 'zeroed_public_private_positive'
                elif np.any(vals == 0):
                    label = 'some_pieces_zeroed'
                else:
                    label = 'varied_nonzero'
                field_patterns[field, label] += 1
        for row in cursor:
            if previous is not None and row[0] != previous:
                finish(group)
                group = []
            group.append(row)
            previous = row[0]
        finish(group)
    expected_groups = 989_016
    for field in COUNTS:
        assert sum(n for (name, _), n in field_patterns.items() if name == field) == expected_groups
    assert sum(multiplicity_missing[n, 'missing'] for n in range(1, 6)) == 933_788
    write_csv(output / 'blk20_field_patterns.csv',
              [{'field': field, 'pattern': label, 'groups': field_patterns[field, label]}
               for field in COUNTS for label in ('repeated_unchanged', 'zeroed_public_private_positive',
                                                   'some_pieces_zeroed', 'varied_nonzero')])
    write_csv(output / 'density_missingness_by_blk20_multiplicity.csv',
              [{'multiplicity': n, 'missing': multiplicity_missing[n, 'missing'],
                'present': multiplicity_missing[n, 'present']}
               for n in sorted({key[0] for key in multiplicity_missing})])
    path = output / 'density_tail_diagnostics.csv'
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row['BLK20_multiplicity'] = tail_multiplicity.get(row['BLK20'], '')
    write_csv(path, rows)
    return field_patterns


def tail_population_scan(root, threshold=100_000):
    """Inspect geometry for every 2020 density row above a fixed extreme threshold."""
    source = root / wui.GDB
    total = int(pyogrio.read_info(source, layer=wui_values.LAYER)['features'])
    fields = (*DENSITIES, *COUNTS, 'BLK20', 'PUBFLAG', 'WATER20')
    selected = []
    for offset, count in wui_values.batch_offsets(total, 150_000):
        info, _, _, arrays = read(source, layer=wui_values.LAYER, columns=list(fields),
                                  skip_features=offset, max_features=count, read_geometry=False)
        block = dict(zip(info['fields'], arrays, strict=True))
        for field, count_field in zip(DENSITIES, COUNTS, strict=True):
            values = np.asarray(block[field], dtype=np.float64)
            for i in np.flatnonzero(values > threshold):
                selected.append({'field': field, 'fid': offset + int(i) + 1,
                                 'density': float(values[i]), 'count': float(block[count_field][i]),
                                 'BLK20': str(block['BLK20'][i]),
                                 'PUBFLAG': str(block['PUBFLAG'][i]),
                                 'WATER20': str(block['WATER20'][i])})
    ids = sorted({item['fid'] for item in selected})
    areas = {}
    for start in range(0, len(ids), 100):
        info, fids, geometries, _ = read(source, layer=wui_values.LAYER, columns=[],
                                         fids=ids[start:start+100], read_geometry=True,
                                         return_fids=True)
        for fid, geometry in zip(fids, geometries, strict=True):
            areas[int(fid)] = cross_dataset.projected_polygon_area(geometry) / 1e6
    for row in selected:
        area = areas[row['fid']]
        row['projected_area_km2'] = area
        row['count_div_area'] = row['count'] / area if area else ''
    selected.sort(key=lambda row: (row['field'], -row['density'], row['fid']))
    out = root / 'reports/eda/raw/decision_resolution'
    write_csv(out / 'density_tail_population.csv', selected)
    summaries = []
    for field in DENSITIES:
        subset = [row for row in selected if row['field'] == field]
        areas = np.array([row['projected_area_km2'] * 1e6 for row in subset])
        error = np.array([abs(row['count_div_area'] - row['density']) / row['density'] for row in subset])
        summaries.append({'field': field, 'rows_above_100000': len(subset),
                          'median_projected_area_m2': float(np.median(areas)),
                          'area_below_100_m2': int(np.count_nonzero(areas < 100)),
                          'median_count': float(np.median([row['count'] for row in subset])),
                          'median_relative_count_per_projected_area_error': float(np.median(error)),
                          'count_per_area_within_1pct': int(np.count_nonzero(error <= 0.01)),
                          'water_rows': sum(row['WATER20'] == '1' for row in subset),
                          'public_flag_1_rows': sum(row['PUBFLAG'] in {'1', '1.0'} for row in subset),
                          'public_flag_null_rows': sum(row['PUBFLAG'] == 'nan' for row in subset)})
    write_csv(out / 'density_tail_summary.csv', summaries)
    return selected


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve())
