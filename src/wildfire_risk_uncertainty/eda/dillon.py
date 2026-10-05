"""Dillon header inventory and bounded-memory native-grid EDA."""

import math
from pathlib import Path

import rasterio

DILLON_DIR = Path("data/interim/dillon_2023/Data/I_FSim_CONUS_LF2020_270m")
FILENAMES = ("CONUS_BP.tif", *(f"CONUS_FLP{i}.tif" for i in range(1, 7)))


def inventory(root: Path) -> list[dict]:
    rows = []
    for name in FILENAMES:
        path = root / DILLON_DIR / name
        with rasterio.open(path, "r") as src:
            rows.append({
                "filename": name, "path": path.relative_to(root).as_posix(),
                "crs": src.crs.to_string() if src.crs else None,
                "crs_wkt": src.crs.to_wkt() if src.crs else None,
                "width": src.width, "height": src.height,
                "transform": list(src.transform)[:6],
                "resolution": list(src.res), "bounds": list(src.bounds),
                "dtypes": list(src.dtypes), "band_count": src.count,
                "nodata": [
                    "NaN" if v is not None and math.isnan(v) else v
                    for v in src.nodatavals
                ],
                "overviews": [src.overviews(i) for i in src.indexes],
            })
    return rows


def alignment(rows: list[dict]) -> dict[str, bool]:
    """Exact comparisons, without rounding; grid means CRS + shape + transform."""
    if not rows:
        raise ValueError("Cannot compare an empty raster inventory")
    groups = {
        "crs": ("crs_wkt",), "shape": ("width", "height"),
        "transform": ("transform",), "bounds": ("bounds",),
        "nodata_convention": ("nodata",),
        "grid": ("crs_wkt", "width", "height", "transform"),
    }
    return {
        name: all(all(row[key] == rows[0][key] for key in keys) for row in rows)
        for name, keys in groups.items()
    }


# Fixed analysis settings; all are written to canonical provenance.
WINDOW_SIZE = 1024
REQUESTED_SAMPLE_SIZE = 1_000_000
COVERAGE_FACTOR = 32


def sample_grid(height, width, requested=REQUESTED_SAMPLE_SIZE):
    """Centered systematic lattice, at most requested cells; no RNG/seed."""
    import numpy as np

    if min(height, width, requested) < 1:
        raise ValueError('Shape and requested sample size must be positive')
    stride = max(1, math.ceil(math.sqrt(height * width / requested)))
    while True:
        rows = np.arange(min(stride // 2, height - 1), height, stride)
        cols = np.arange(min(stride // 2, width - 1), width, stride)
        if len(rows) * len(cols) <= requested:
            break
        stride += 1
    return rows, cols, stride


def analyze(root: Path, requested=REQUESTED_SAMPLE_SIZE, window_size=WINDOW_SIZE):
    """One full-resolution pass over aligned rasters, independent of overviews.

    Valid = GDAL mask valid AND finite. Memory bounded by seven windows plus
    a spatial sample of about requested cells and a small coverage grid.
    """
    from contextlib import ExitStack

    import numpy as np
    from rasterio.windows import Window

    from .statistics import Accumulator

    headers = inventory(root)
    if not alignment(headers)['grid']:
        raise ValueError('Dillon raster grids must match before joint window reads')
    if window_size % COVERAGE_FACTOR:
        raise ValueError('Window size must be a multiple of coverage factor')
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(root / DILLON_DIR / n)) for n in FILENAMES]
        height, width = sources[0].shape
        rows, cols, stride = sample_grid(height, width, requested)
        samples = np.full((7, len(rows), len(cols)), np.nan, dtype=np.float32)
        accumulators = [Accumulator() for _ in sources]
        patterns = np.zeros(128, dtype=np.int64)
        coverage_shape = (math.ceil(height / COVERAGE_FACTOR), math.ceil(width / COVERAGE_FACTOR))
        coverage_all = np.zeros(coverage_shape, dtype=np.int64)
        coverage_any = np.zeros(coverage_shape, dtype=np.int64)
        coverage_total = np.zeros(coverage_shape, dtype=np.int64)
        nonfinite = np.zeros(7, dtype=np.int64)
        for top in range(0, height, window_size):
            for left in range(0, width, window_size):
                h, w = min(window_size, height - top), min(window_size, width - left)
                window = Window(left, top, w, h)
                bits = np.zeros((h, w), dtype=np.uint8)
                ri = np.flatnonzero((rows >= top) & (rows < top + h))
                ci = np.flatnonzero((cols >= left) & (cols < left + w))
                for i, source in enumerate(sources):
                    block = source.read(1, window=window, masked=True)
                    finite = np.isfinite(block.data)
                    gdal_valid = ~np.ma.getmaskarray(block)
                    valid = gdal_valid & finite
                    nonfinite[i] += np.count_nonzero(gdal_valid & ~finite)
                    accumulators[i].update(block.data[valid])
                    bits |= valid.astype(np.uint8) << i
                    values = np.where(valid, block.data, np.nan)
                    samples[i][np.ix_(ri, ci)] = values[np.ix_(rows[ri] - top, cols[ci] - left)]
                patterns += np.bincount(bits.ravel(), minlength=128)
                ys, xs = np.arange(0, h, COVERAGE_FACTOR), np.arange(0, w, COVERAGE_FACTOR)
                target = np.s_[top // COVERAGE_FACTOR:math.ceil((top + h) / COVERAGE_FACTOR),
                               left // COVERAGE_FACTOR:math.ceil((left + w) / COVERAGE_FACTOR)]
                for out, mask in ((coverage_all, bits == 127), (coverage_any, bits != 0),
                                  (coverage_total, np.ones((h, w), dtype=bool))):
                    out[target] = np.add.reduceat(np.add.reduceat(mask.astype(np.int64), ys, axis=0), xs, axis=1)
            print(f'Dillon full-resolution rows processed: {min(top + window_size, height)}/{height}', flush=True)
        summaries = []
        for i, acc in enumerate(accumulators):
            summary = acc.summary(height * width, samples[i])
            summary.update(raster=FILENAMES[i].removesuffix('.tif'), width=width, height=height,
                           unmasked_nonfinite=int(nonfinite[i]))
            summaries.append(summary)
        flat = samples.reshape(7, -1)
        common = flat[:, np.all(np.isfinite(flat), axis=0)]
        corr = np.corrcoef(common.astype(np.float64)) if common.shape[1] > 1 else np.full((7, 7), np.nan)
        return {
            'summaries': summaries, 'accumulators': accumulators, 'samples': samples,
            'common_sample': common, 'correlation': corr, 'patterns': patterns,
            'coverage_all': coverage_all, 'coverage_any': coverage_any,
            'coverage_total': coverage_total, 'bounds': list(sources[0].bounds),
            'sample_rows': rows, 'sample_cols': cols,
            'provenance': {
                'window_size': window_size, 'validity': 'GDAL valid mask AND finite numeric value',
                'sample_algorithm': 'centered regular row/column lattice in native grid',
                'seed': None, 'seed_reason': 'no randomness', 'requested_sample_size': requested,
                'actual_sample_locations': len(rows) * len(cols), 'sample_stride_pixels': stride,
                'sample_row_offset': int(rows[0]), 'sample_col_offset': int(cols[0]),
                'common_valid_sample_size': common.shape[1],
                'quantiles': 'numpy.quantile method=linear on per-raster valid lattice sample; endpoints exact',
                'correlation': 'Pearson on common-valid spatial lattice; all 7 variables pairwise only',
                'map_strategy': 'native full-resolution lattice values collected during streaming; no supplied overviews',
                'coverage_map': f'exact valid proportions in {COVERAGE_FACTOR}x{COVERAGE_FACTOR} native-cell bins; edge bins smaller; disagreement locations enlarged with logarithmic fraction colors',
                'std_convention': 'population ddof=0; float64 parallel central-moment merge',
                'histogram_convention': 'left-closed/right-open except final bin includes right edge; exact counts',
            },
        }


def write_products(result, root: Path):
    """Write only small canonical tables and figures; no full-size arrays."""
    import csv
    import json
    from importlib.metadata import version

    from . import plotting
    from .statistics import LINEAR_EDGES, LOG_EDGES

    base = root / 'reports/eda/raw/dillon'
    tables, figures = base / 'tables', base / 'figures'
    for folder in (tables, figures):
        if folder.resolve() != folder.absolute():
            raise ValueError(f'Output must not traverse symlinks: {folder}')
        folder.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in folder.iterdir()):
            raise ValueError(f'Output directory contains symlinks: {folder}')

    def csv_table(name, rows):
        with (tables / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    csv_table('distribution_summary.csv', result['summaries'])
    patterns = result['patterns']
    csv_table('mask_overlap.csv', [{
        'total_pixels': int(patterns.sum()), 'valid_all_seven': int(patterns[127]),
        'valid_at_least_one': int(patterns[1:].sum()), 'invalid_all_seven': int(patterns[0]),
        'mask_disagreements': int(patterns[1:127].sum()),
        'masks_identical': not bool(patterns[1:127].sum()), 'method': 'exact full-resolution counts',
    }])
    csv_table('mask_patterns.csv', [
        {'validity_bits_BP_FLP1_to_FLP6': format(i, '07b')[::-1], 'pixel_count': int(n)}
        for i, n in enumerate(patterns) if n
    ])
    histograms = []
    for s, acc in zip(result['summaries'], result['accumulators'], strict=True):
        for kind, edges, counts in [('linear', LINEAR_EDGES, acc.linear), ('log', LOG_EDGES, acc.logarithmic)]:
            for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True):
                histograms.append({'raster': s['raster'], 'bins': kind, 'left': float(left),
                                   'right': float(right), 'pixel_count': int(count)})
    csv_table('exact_histograms.csv', histograms)
    names = [s['raster'] for s in result['summaries']]
    csv_table('sample_correlations.csv', [
        {'raster': name, **dict(zip(names, result['correlation'][i], strict=True))}
        for i, name in enumerate(names)
    ])
    provenance = result['provenance'] | {
        'packages': {p: version(p) for p in ('numpy', 'rasterio', 'matplotlib')},
        'rasterio_gdal': rasterio.__gdal_version__,
        'linear_histogram_edges': LINEAR_EDGES.tolist(), 'log_histogram_edges': LOG_EDGES.tolist(),
        'histogram_exclusions': 'linear: below 0/above 1; log: nonpositive/below 1e-12/above 1; counts in summary',
        'map_scales': 'FLPs common linear [min(0,all FLP minima), max(1,all FLP maxima)]; BP linear [min(0,min),max]',
        'relationship_plot': 'common-valid lattice hexbin, gridsize=55; separate logarithmic count scales',
    }
    (tables / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    plotting.render(result, figures)
