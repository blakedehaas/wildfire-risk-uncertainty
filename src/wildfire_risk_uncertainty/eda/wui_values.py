"""Batched, attribute-first empirical SILVIS WUI analysis."""

import csv
import math
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from pyogrio.raw import read

from .wui import GDB, metadata

LAYER = 'CONUS_WUI_block_1990_2020_change_v4_gcs_na83'
YEARS = (1990, 2000, 2010, 2020)
VEG_YEARS = (1992, 2001, 2011, 2019)
BATCH = 100_000
SAMPLE_STRIDE = 97
NUMERIC = (
    *(f'HU{y}' for y in YEARS), *(f'HUDEN{y}' for y in YEARS),
    *(f'VEG{y}PC' for y in VEG_YEARS), 'POP2020', 'POPDEN2020',
    'OCCHU2020', 'VACHU2020', 'OCCHUDEN2020', 'VACHUDEN2020', 'AWATER20PC',
)
CATEGORICAL = (
    *(f'WUIFLAG{y}' for y in YEARS), *(f'WUICLASS{y}' for y in YEARS),
    'WATER20', 'PUBFLAG', 'BUFVEG', 'STATE', 'fips',
)
FIELDS = ('BLK20', *NUMERIC, *CATEGORICAL)
TRANSITIONS = ((1990, 2000), (2000, 2010), (2010, 2020), (1990, 2020))
QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def batch_offsets(total, size=BATCH):
    if total < 0 or size < 1:
        raise ValueError('Invalid batch size or count')
    return ((offset, min(size, total - offset)) for offset in range(0, total, size))


def missing(values):
    values = np.asarray(values)
    if values.dtype.kind in 'f':
        return ~np.isfinite(values)
    if values.dtype.kind in 'iu':
        return np.zeros(values.shape, dtype=bool)
    return np.array([v is None or isinstance(v, float) and math.isnan(v) for v in values])



def category_key(field, value):
    """Stable string keys for numeric flags despite batch-dependent null upcasts."""
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if field in {'WATER20', 'PUBFLAG', 'BUFVEG'} or field.startswith('WUIFLAG'):
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(value)
    return str(value)

def flag_codes(values):
    values = np.asarray(values)
    absent = missing(values)
    codes = np.full(len(values), 4, dtype=np.int8)
    codes[absent] = 3
    for i in range(3):
        codes[~absent & (values == i)] = i
    return codes


def transition_counts(left, right):
    return np.bincount(flag_codes(left) * 5 + flag_codes(right), minlength=25).reshape(5, 5)


class NumericSummary:
    def __init__(self, domain=None):
        self.domain = domain
        self.rows = self.valid = self.zero = self.outside = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total = 0.0
        self.samples = []

    def update(self, values, row_start, stride=SAMPLE_STRIDE):
        values = np.asarray(values, dtype=np.float64)
        good = np.isfinite(values)
        finite = values[good]
        self.rows += len(values)
        self.valid += len(finite)
        if not len(finite):
            return
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        self.total += float(finite.sum(dtype=np.float64))
        self.zero += int(np.count_nonzero(finite == 0))
        if self.domain is not None:
            lo, hi = self.domain
            self.outside += int(np.count_nonzero((finite < lo) | (finite > hi)))
        chosen = (row_start + np.arange(len(values))) % stride == 0
        self.samples.extend(values[chosen & good].tolist())

    def row(self, name):
        sample = np.asarray(self.samples, dtype=np.float64)
        result = {'field': name, 'rows': self.rows, 'missing': self.rows - self.valid,
                  'missing_fraction': (self.rows - self.valid) / self.rows,
                  'valid': self.valid, 'min': self.minimum if self.valid else None,
                  'max': self.maximum if self.valid else None,
                  'mean': self.total / self.valid if self.valid else None,
                  'zero_count': self.zero,
                  'zero_fraction_valid': self.zero / self.valid if self.valid else None,
                  'out_of_domain_count': self.outside if self.domain is not None else None,
                  'domain': str(self.domain) if self.domain is not None else '',
                  'sample_valid': len(sample), 'quantile_method': f'every {SAMPLE_STRIDE}th feature row; exact extrema'}
        result.update({f'q{q:02d}': float(np.quantile(sample, q / 100)) if len(sample) else None
                       for q in QUANTILES})
        return result


def domain(field):
    if field.startswith('VEG') or field == 'AWATER20PC':
        return (0, 100)
    return (0, math.inf)


def csv_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)



def summarize_ids(connection, total_rows, missing_ids):
    """Exact disk-backed ID multiplicities, excluding null identifiers."""
    multiplicity = Counter()
    unique = duplicate = duplicate_rows = max_multiplicity = 0
    for _, n in connection.execute('SELECT id, COUNT(*) FROM ids GROUP BY id'):
        unique += 1
        multiplicity[n] += 1
        if n > 1:
            duplicate += 1
            duplicate_rows += n
        max_multiplicity = max(max_multiplicity, n)
    return ({'total_rows': total_rows, 'missing_BLK20': missing_ids,
             'unique_nonmissing_BLK20': unique, 'IDs_once': multiplicity[1],
             'IDs_multiple': duplicate, 'rows_in_duplicated_IDs': duplicate_rows,
             'maximum_multiplicity': max_multiplicity}, multiplicity)

def analyze(root: Path, batch_size=BATCH):
    source = root / GDB
    total = int(pyogrio.read_info(source, layer=LAYER, force_feature_count=False)['features'])
    definitions = metadata(root)['attributes']
    documented = {d['edomv'] for d in definitions['WUICLASS2020']['domains']}
    numeric = {field: NumericSummary(domain(field)) for field in NUMERIC}
    changes = {f'{kind}{a}_to_{b}': NumericSummary()
               for kind, pairs in (('HU', TRANSITIONS), ('HUDEN', TRANSITIONS),
                                   ('VEG', ((1992, 2001), (2001, 2011), (2011, 2019), (1992, 2019))))
               for a, b in pairs}
    categories = {field: Counter() for field in CATEGORICAL}
    transitions = {pair: np.zeros((5, 5), dtype=np.int64) for pair in TRANSITIONS}
    with tempfile.TemporaryDirectory(prefix='wui-ids-') as temporary:
        connection = sqlite3.connect(str(Path(temporary) / 'ids.sqlite'))
        connection.execute('PRAGMA journal_mode=OFF')
        connection.execute('PRAGMA synchronous=OFF')
        connection.execute('PRAGMA temp_store=FILE')
        connection.execute('CREATE TABLE ids (id TEXT NOT NULL)')
        missing_ids = 0
        for offset, count in batch_offsets(total, batch_size):
            info, _, _, arrays = read(source, layer=LAYER, columns=list(FIELDS),
                                      skip_features=offset, max_features=count,
                                      read_geometry=False)
            block = dict(zip(info['fields'], arrays, strict=True))
            actual = len(block['BLK20'])
            if actual != count:
                raise RuntimeError(f'Expected {count} WUI rows at {offset}, got {actual}')
            for field, stat in numeric.items():
                stat.update(block[field], offset)
            for name, stat in changes.items():
                left, b = name.split('_to_')
                kind = next(k for k in ('HUDEN', 'VEG', 'HU') if left.startswith(k))
                a = left.removeprefix(kind)
                field_a = f'{kind}{a}' + ('PC' if kind == 'VEG' else '')
                field_b = f'{kind}{b}' + ('PC' if kind == 'VEG' else '')
                stat.update(np.asarray(block[field_b], dtype=np.float64) -
                            np.asarray(block[field_a], dtype=np.float64), offset)
            for field, counter in categories.items():
                values = block[field]
                absent = missing(values)
                counter.update(category_key(field, value) for value in values)
            for a, b in TRANSITIONS:
                transitions[a, b] += transition_counts(block[f'WUIFLAG{a}'], block[f'WUIFLAG{b}'])
            ids = block['BLK20']
            absent = missing(ids)
            missing_ids += int(np.count_nonzero(absent))
            connection.executemany('INSERT INTO ids VALUES (?)',
                                   ((str(value),) for value, bad in zip(ids, absent, strict=True) if not bad))
            if offset % (batch_size * 10) == 0:
                connection.commit()
                print(f'WUI attribute rows processed: {offset + actual}/{total}', flush=True)
        connection.commit()
        uniqueness, multiplicity = summarize_ids(connection, total, missing_ids)
        connection.close()
    return {'total': total, 'numeric': numeric, 'changes': changes, 'categories': categories,
            'transitions': transitions, 'documented_classes': documented,
            'id_summary': uniqueness,
            'multiplicity': multiplicity}


def wkb_bounds(wkb):
    """Bounding box of a two-dimensional WKB MultiPolygon, without Shapely."""
    import struct

    view = memoryview(wkb)
    endian = '<' if view[0] == 1 else '>'
    if struct.unpack_from(endian + 'I', view, 1)[0] != 6:
        raise ValueError('Expected a 2D WKB MultiPolygon')
    count = struct.unpack_from(endian + 'I', view, 5)[0]
    offset = 9
    minimum = np.array([np.inf, np.inf])
    maximum = np.array([-np.inf, -np.inf])
    for _ in range(count):
        polygon_endian = '<' if view[offset] == 1 else '>'
        if struct.unpack_from(polygon_endian + 'I', view, offset + 1)[0] != 3:
            raise ValueError('Expected WKB Polygon member')
        rings = struct.unpack_from(polygon_endian + 'I', view, offset + 5)[0]
        offset += 9
        for _ in range(rings):
            points = struct.unpack_from(polygon_endian + 'I', view, offset)[0]
            offset += 4
            coordinates = np.frombuffer(view, dtype=polygon_endian + 'f8',
                                        count=points * 2, offset=offset).reshape(-1, 2)
            if points:
                minimum = np.minimum(minimum, coordinates.min(axis=0))
                maximum = np.maximum(maximum, coordinates.max(axis=0))
            offset += points * 16
    return (*minimum, *maximum)


def spatial_sample(root: Path, total: int, requested=10_000, batch=500):
    """Deterministic sparse FID sample; bounded geometry batches."""
    source = root / GDB
    selected = np.unique(np.linspace(1, total, min(total, requested), dtype=np.int64))
    points = []
    for start in range(0, len(selected), batch):
        fids = selected[start:start + batch]
        info, returned, geometry, arrays = read(source, layer=LAYER,
                                                 columns=['WUIFLAG2020'], fids=fids,
                                                 read_geometry=True, return_fids=True)
        flags = dict(zip(info['fields'], arrays, strict=True))['WUIFLAG2020']
        if len(returned) != len(flags):
            raise RuntimeError('Geometry and attribute samples differ in length')
        for wkb, code in zip(geometry, flag_codes(flags), strict=True):
            if wkb is None:
                continue
            xmin, ymin, xmax, ymax = wkb_bounds(wkb)
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            if np.isfinite(x) and np.isfinite(y):
                points.append((float(x), float(y), int(code)))
    return np.asarray(points)

def write_products(result, root: Path):
    base = root / 'reports/eda/raw/wui'
    tables, figures = base / 'tables', base / 'figures'
    for path in (tables, figures):
        if path.resolve() != path.absolute():
            raise ValueError(f'Output traverses symlink: {path}')
        path.mkdir(parents=True, exist_ok=True)
    total = result['total']
    csv_rows(tables / 'value_summary.csv', [stat.row(field) for field, stat in result['numeric'].items()])
    csv_rows(tables / 'row_change_summary.csv', [stat.row(name) for name, stat in result['changes'].items()])
    frequencies = []
    for field, counts in sorted(result['categories'].items()):
        for value, n in sorted(counts.items(), key=lambda item: str(item[0] if item[0] is not None else '<NULL>')):
            frequencies.append({'field': field, 'value': value if value is not None else '<NULL>',
                                'row_count': n, 'row_fraction': n / total,
                                'documented_class': value in result['documented_classes']
                                if field.startswith('WUICLASS') and value is not None else ''})
    csv_rows(tables / 'categorical_frequencies.csv', frequencies)
    flags = []
    for year in YEARS:
        counts = result['categories'][f'WUIFLAG{year}']
        for value, label in (('0', 'non-WUI'), ('1', 'intermix'), ('2', 'interface'),
                             (None, 'missing'), ('other', 'other')):
            n = counts[value] if value != 'other' else sum(v for k, v in counts.items()
                                                          if k not in ('0', '1', '2', None))
            flags.append({'year': year, 'state': label, 'feature_rows': n,
                          'fraction_of_all_feature_rows': n / total})
    csv_rows(tables / 'wuiflag_composition.csv', flags)
    classes = []
    for year in YEARS:
        counts = result['categories'][f'WUICLASS{year}']
        for value in sorted(result['documented_classes'] | set(counts) - {None}):
            classes.append({'year': year, 'class': value, 'feature_rows': counts[value],
                            'fraction_of_all_feature_rows': counts[value] / total,
                            'documented': value in result['documented_classes'],
                            'observed': bool(counts[value])})
        classes.append({'year': year, 'class': '<NULL>', 'feature_rows': counts[None],
                        'fraction_of_all_feature_rows': counts[None] / total,
                        'documented': False, 'observed': bool(counts[None])})
    csv_rows(tables / 'wuiclass_composition.csv', classes)
    state_names = ('non-WUI', 'intermix', 'interface', 'missing', 'other')
    rows = [{'from_year': a, 'to_year': b, 'from_state': state_names[i],
             'to_state': state_names[j], 'feature_rows': int(matrix[i, j]),
             'fraction_of_all_feature_rows': int(matrix[i, j]) / total}
            for (a, b), matrix in result['transitions'].items()
            for i in range(5) for j in range(5)]
    csv_rows(tables / 'wuiflag_transitions.csv', rows)
    csv_rows(tables / 'blk20_uniqueness.csv', [result['id_summary']])
    csv_rows(tables / 'blk20_multiplicity.csv', [
        {'multiplicity': n, 'ID_count': count, 'feature_rows': n * count}
        for n, count in sorted(result['multiplicity'].items())])
    import json
    provenance = {'row_count': total, 'batch_size': BATCH, 'attribute_geometry': 'disabled',
                  'quantile_sampling': f'global zero-based feature row index modulo {SAMPLE_STRIDE} == 0',
                  'numeric_extrema_counts': 'full coverage',
                  'BLK20_uniqueness': 'disk-backed SQLite GROUP BY on all nonmissing IDs',
                  'spatial_sample': '10,000 deterministic FIDs across source range, read in geometry batches of 500; bounding-box centers',
                  'transition_states': list(state_names), 'row_unit': 'feature/polygon row'}
    (tables / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    draw_figures(result, root, figures)


def draw_figures(result, root, figures):
    total = result['total']
    with plt.rc_context({'figure.dpi': 140, 'savefig.dpi': 140, 'font.size': 9}):
        def save(fig, name):
            fig.savefig(figures / name, bbox_inches='tight', metadata={'Software': 'wildfire-risk-uncertainty'})
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottom = np.zeros(4)
        for value, label in (('0', 'non-WUI'), ('1', 'intermix'), ('2', 'interface'),
                             (None, 'missing')):
            amounts = np.array([result['categories'][f'WUIFLAG{y}'][value] / total for y in YEARS])
            ax.bar(YEARS, amounts, bottom=bottom, width=7, label=label)
            bottom += amounts
        ax.set(xlabel='WUI year', ylabel='Fraction of feature/polygon rows', ylim=(0, 1),
               title='WUIFLAG composition by feature row')
        ax.legend()
        save(fig, 'wuiflag_composition.png')
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for value, label in (('0', 'non-WUI'), ('1', 'intermix'), ('2', 'interface')):
            ax.plot(YEARS, [result['categories'][f'WUIFLAG{y}'][value] for y in YEARS],
                    marker='o', label=label)
        ax.set(xlabel='WUI year', ylabel='Feature/polygon row count',
               title='WUIFLAG feature-row counts')
        ax.legend()
        save(fig, 'wuiflag_counts.png')
        labels = sorted(result['documented_classes'])
        data = np.array([[result['categories'][f'WUICLASS{y}'][label] / total for y in YEARS]
                         for label in labels])
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        ax.set(xticks=range(4), xticklabels=YEARS, yticks=range(len(labels)), yticklabels=labels,
               xlabel='WUI year', title='WUICLASS fraction of feature/polygon rows')
        fig.colorbar(im, ax=ax, label='Fraction of all rows')
        save(fig, 'wuiclass_composition.png')
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
        names = ('non-WUI', 'intermix', 'interface', 'missing', 'other')
        for ax, ((a, b), matrix) in zip(axes.flat, result['transitions'].items(), strict=True):
            im = ax.imshow(matrix[:3, :3], cmap='Blues')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=names[:3], yticklabels=names[:3],
                   xlabel=f'{b} state', ylabel=f'{a} state', title=f'{a} → {b} feature-row transitions')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{matrix[i, j]:,}', ha='center', va='center', fontsize=7,
                            color='white' if matrix[i, j] > matrix[:3, :3].max() / 2 else 'black')
            fig.colorbar(im, ax=ax, label='Feature rows')
        save(fig, 'wuiflag_transitions.png')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for ax, prefix, title in ((axes[0], 'HU', 'Housing units'), (axes[1], 'HUDEN', 'Housing density')):
            stats = [result['numeric'][f'{prefix}{y}'].row('') for y in YEARS]
            ax.plot(YEARS, [s['q50'] for s in stats], marker='o', label='sample median')
            ax.plot(YEARS, [s['q95'] for s in stats], marker='o', label='sample p95')
            ax.set(xlabel='Year', ylabel='Units' if prefix == 'HU' else 'Units / km²', title=title)
            ax.legend()
        fig.suptitle('Row distributions; fixed systematic quantile sample')
        save(fig, 'housing_longitudinal.png')
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, label in (('HU1990_to_2020', 'Housing units'),
                           ('HUDEN1990_to_2020', 'Housing density')):
            stat = result['changes'][key]
            values = np.asarray(stat.samples)
            ax.hist(np.sign(values) * np.log1p(np.abs(values)), bins=80, histtype='step',
                    label=label, density=True)
        ax.set(xlabel='Signed log1p(row change), sample', ylabel='Sample density',
               title='1990–2020 row-level housing changes')
        ax.legend()
        save(fig, 'housing_changes.png')
        fig, ax = plt.subplots(figsize=(8, 4))
        stats = [result['numeric'][f'VEG{y}PC'].row('') for y in VEG_YEARS]
        ax.plot(VEG_YEARS, [s['q50'] for s in stats], marker='o', label='sample median')
        ax.plot(VEG_YEARS, [s['q95'] for s in stats], marker='o', label='sample p95')
        ax.set(xlabel='Vegetation source year', ylabel='Wildland vegetation (%)',
               title='Wildland vegetation row distributions')
        ax.legend()
        save(fig, 'vegetation_longitudinal.png')
        points = spatial_sample(root, total)
        fig, ax = plt.subplots(figsize=(9, 5))
        for code, label, color in ((0, 'non-WUI', '#9e9e9e'), (1, 'intermix', '#268f4e'),
                                   (2, 'interface', '#b3432c'), (3, 'missing', '#593f9a'),
                                   (4, 'other', '#111111')):
            subset = points[points[:, 2] == code]
            if len(subset):
                ax.scatter(subset[:, 0], subset[:, 1], s=2, alpha=.65, c=color,
                           label=f'{label} (n={len(subset):,})', rasterized=True)
        ax.set(xlabel='Longitude (EPSG:4269)', ylabel='Latitude (EPSG:4269)',
               title='2020 WUIFLAG; 10,000 deterministic FID-sampled polygon bounding-box centers')
        ax.legend(markerscale=4, fontsize=8)
        save(fig, 'wui2020_spatial_sample.png')
