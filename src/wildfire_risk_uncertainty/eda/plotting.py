"""Canonical headless numerical figures; no source previews or interactive state."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

STYLE = {'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
         'figure.dpi': 140, 'savefig.dpi': 140, 'font.family': 'DejaVu Sans'}


def save(fig, path):
    fig.savefig(path, bbox_inches='tight', metadata={'Software': 'wildfire-risk-uncertainty'})
    plt.close(fig)


def render(result, directory):
    """Maps use lattice samples; histograms use exact full-resolution bin counts."""
    from .statistics import LINEAR_EDGES, LOG_EDGES

    names = [s['raster'].removeprefix('CONUS_') for s in result['summaries']]
    bounds = result['bounds']
    extent = [bounds[0] / 1000, bounds[2] / 1000, bounds[1] / 1000, bounds[3] / 1000]
    # Match sample pixel centers/spacing, rather than stretch to the source extent.
    rows, cols = result['sample_rows'], result['sample_cols']
    summary = result['summaries'][0]
    dx = (bounds[2] - bounds[0]) / summary['width']
    dy = (bounds[3] - bounds[1]) / summary['height']
    stride = result['provenance']['sample_stride_pixels']
    map_extent = [(bounds[0] + (cols[0] + .5 - stride / 2) * dx) / 1000,
                  (bounds[0] + (cols[-1] + .5 + stride / 2) * dx) / 1000,
                  (bounds[3] - (rows[-1] + .5 + stride / 2) * dy) / 1000,
                  (bounds[3] - (rows[0] + .5 - stride / 2) * dy) / 1000]
    flp_min = min([0] + [s['min'] for s in result['summaries'][1:] if s['min'] is not None])
    flp_max = max([1] + [s['max'] for s in result['summaries'][1:] if s['max'] is not None])
    with plt.rc_context(STYLE):
        for i, name in enumerate(names):
            s = result['summaries'][i]
            low, high = (min(0, s['min'] or 0), max(s['max'] or 1, 1e-12)) if i == 0 else (flp_min, flp_max)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.set_facecolor('#e5e5e5')
            cmap = plt.get_cmap('viridis').with_extremes(bad='#e5e5e5')
            im = ax.imshow(np.ma.masked_invalid(result['samples'][i]), extent=map_extent,
                           origin='upper', cmap=cmap, vmin=low, vmax=high, interpolation='nearest')
            ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='EPSG:5070 easting (km)',
                   ylabel='EPSG:5070 northing (km)',
                   title=f'{name} — CONUS probability (systematic lattice map)')
            fig.colorbar(im, ax=ax, label='Probability (unitless; linear scale)')
            fig.text(.12, .01, 'Gray: invalid sampled cells. Native-grid lattice; fine features may be missed.', fontsize=8)
            save(fig, directory / f'{name.lower()}_map.png')
            fig, ax = plt.subplots(figsize=(8, 4.5))
            acc = result['accumulators'][i]
            ax.stairs(acc.linear, LINEAR_EDGES, fill=True, alpha=.7)
            ax.set(yscale='log', xlabel='Probability (unitless)', ylabel='Pixel count (log scale)',
                   title=f'{name} — exact full-raster histogram (100 equal-width bins)')
            fig.text(.12, .01, f'Outside [0, 1]: {s["below_0"] + s["above_1"]:,}; invalid cells excluded.', fontsize=8)
            save(fig, directory / f'{name.lower()}_distribution.png')
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, acc in zip(names[1:], result['accumulators'][1:], strict=True):
            ax.stairs(acc.linear / max(1, acc.count), LINEAR_EDGES, label=name)
        ax.set(yscale='log', xlabel='Probability (unitless)', ylabel='Fraction of valid pixels per bin (log scale)',
               title='FLP comparison — exact 0.01-wide probability bins')
        ax.legend(ncol=3)
        save(fig, directory / 'flp_distributions_comparison.png')
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for i, (name, ax) in enumerate(zip(names, axes.flat, strict=False)):
            acc = result['accumulators'][i]
            ax.stairs(acc.logarithmic, LOG_EDGES)
            positive = np.flatnonzero(acc.logarithmic)
            if positive.size:
                ax.set_xlim(LOG_EDGES[positive[0]], LOG_EDGES[positive[-1] + 1])
            ax.set(xscale='log', yscale='log', title=f'{name}; zeros={acc.zero:,}',
                   xlabel='Positive probability (log)', ylabel='Pixel count (log)')
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0, .8, 'Exact streaming log-bin counts.\nZero is reported separately.\nBins: 1e-12 to 1.\nUnder/overflow in summary table.\nInvalid cells excluded.', va='top')
        fig.suptitle('Bulk and tails — exact histograms of positive probabilities')
        fig.tight_layout()
        save(fig, directory / 'probability_log_distributions.png')
        disagreement = result['coverage_any'] - result['coverage_all']
        different = bool(disagreement.sum())
        fig, axes = plt.subplots(1, 2 if different else 1, figsize=(12 if different else 9, 5), squeeze=False)
        xedges = (bounds[0] + np.minimum(np.arange(result['coverage_total'].shape[1] + 1) * 32,
                                        summary['width']) * dx) / 1000
        yedges = (bounds[3] - np.minimum(np.arange(result['coverage_total'].shape[0] + 1) * 32,
                                        summary['height']) * dy) / 1000
        ax = axes.flat[0]
        im = ax.pcolormesh(xedges, yedges, result['coverage_all'] / result['coverage_total'],
                           vmin=0, vmax=1, cmap='cividis', shading='flat', rasterized=True)
        ax.set_title('Valid in all seven — fraction per bin')
        fig.colorbar(im, ax=ax, label='Native-cell fraction (linear)')
        if different:
            ax = axes.flat[1]
            ax.set_facecolor('#e5e5e5')
            yy, xx = np.nonzero(disagreement)
            fractions = disagreement[yy, xx] / result['coverage_total'][yy, xx]
            im = ax.scatter((xedges[xx] + xedges[xx + 1]) / 2,
                            (yedges[yy] + yedges[yy + 1]) / 2,
                            c=fractions, s=9, cmap='magma', norm=LogNorm(vmin=1/1024, vmax=1))
            ax.set_title('Disagreement locations (markers enlarged)')
            fig.colorbar(im, ax=ax, label='Disagreement fraction per bin (log)')
        for ax in axes.flat:
            ax.set(xlim=extent[:2], ylim=extent[2:], aspect='equal',
                   xlabel='EPSG:5070 easting (km)', ylabel='EPSG:5070 northing (km)')
        fig.suptitle('Dillon coverage — exact counts in 32 × 32 cell bins; marker size is not area')
        fig.tight_layout()
        save(fig, directory / 'valid_coverage.png')
        common = result['common_sample']
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
        for i, ax in enumerate(axes.flat, start=1):
            if common.shape[1]:
                ax.hexbin(common[0], common[i], gridsize=55, mincnt=1, norm=LogNorm(), cmap='magma')
            ax.set(xlabel='BP (probability)', ylabel=f'{names[i]} (probability)',
                   title=f'BP vs {names[i]} (r={result["correlation"][0, i]:.3f})')
        fig.suptitle(f'Common-valid spatial sample: n={common.shape[1]:,}; hexagon counts on log color scale')
        if common.shape[1]:
            # Each panel has its own count normalization; label this explicitly.
            for ax in axes.flat:
                fig.colorbar(ax.collections[0], ax=ax, label='Sample count (log)')
        save(fig, directory / 'bp_flp_relationships.png')
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(result['correlation'], vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set(xticks=range(7), yticks=range(7), xticklabels=names, yticklabels=names,
               title=f'Pearson correlations — common-valid lattice (n={common.shape[1]:,})')
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{result["correlation"][i,j]:.2f}', ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=ax, label='Sample Pearson r')
        save(fig, directory / 'sample_correlations.png')


def render_flp_semantics(result, directory):
    """Exact streaming histograms and native-lattice spatial diagnostics."""
    from .flp_semantics import (
        CATEGORICAL_TOLERANCE,
        DEVIATION_EDGES,
        ENTROPY_EDGES,
        SUM_EDGES,
        vector_metrics,
    )

    diag = result['flp_diagnostics']
    sums = result['flp_sum_sample']
    bp = result['samples'][0]
    bounds = result['bounds']
    rows, cols = result['sample_rows'], result['sample_cols']
    shape = result['summaries'][0]
    dx = (bounds[2] - bounds[0]) / shape['width']
    dy = (bounds[3] - bounds[1]) / shape['height']
    stride = result['provenance']['sample_stride_pixels']
    extent = [(bounds[0] + (cols[0] + .5 - stride / 2) * dx) / 1000,
              (bounds[0] + (cols[-1] + .5 + stride / 2) * dx) / 1000,
              (bounds[3] - (rows[-1] + .5 + stride / 2) * dy) / 1000,
              (bounds[3] - (rows[0] + .5 - stride / 2) * dy) / 1000]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.stairs(diag.sum_hist, SUM_EDGES, fill=True)
        ax.set(yscale='log', xlabel='Sum of six FLPs', ylabel='Pixel count (log)',
               title='FLP vector sums — full raster')
        save(fig, directory / 'flp_sum_distribution.png')
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.stairs(diag.deviation_hist[1:], DEVIATION_EDGES[1:], fill=True)
        ax.set(xscale='log', yscale='log', xlabel='Positive absolute deviation from one',
               ylabel='Pixel count (log)', title='FLP sum deviation — full raster')
        ax.text(.03, .95, f'Exact zero deviation: {diag.deviation.zero:,} pixels',
                transform=ax.transAxes, va='top')
        save(fig, directory / 'flp_sum_deviation_from_one.png')
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor('#e5e5e5')
        im = ax.imshow(np.ma.masked_invalid(sums), origin='upper', extent=extent,
                       cmap=plt.get_cmap('viridis').with_extremes(bad='#e5e5e5'),
                       vmin=0, vmax=max(1, diag.sums.maximum), interpolation='nearest')
        ax.set(xlabel='EPSG:5070 easting (km)', ylabel='EPSG:5070 northing (km)',
               title='FLP sum — native-grid lattice')
        fig.colorbar(im, ax=ax, label='Sum of six FLPs')
        save(fig, directory / 'flp_sum_map.png')
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for mask, label in ((bp == 0, 'BP = 0'), (bp > 0, 'BP > 0')):
            data = sums[mask & np.isfinite(sums)]
            ax.hist(data, bins=SUM_EDGES, histtype='step', label=f'{label} (n={len(data):,})')
        ax.set(yscale='log', xlabel='Sum of six FLPs (lattice)', ylabel='Sample count (log)',
               title='BP zero state versus FLP sum')
        ax.legend()
        save(fig, directory / 'bp_zero_vs_flp_sum.png')
        if diag.categorical_justified():
            flps = result['samples'][1:].astype(np.float64)
            valid = np.all(np.isfinite(flps), axis=0)
            safe = np.where(valid, flps, 0)
            _, deviation, zero, _, classes, entropy = vector_metrics(safe)
            support = valid & (bp > 0) & (deviation <= CATEGORICAL_TOLERANCE) & ~zero
            mapped = ((np.where(support, classes, np.nan), 'dominant_flp_class.png',
                       'Dominant FLP class', 'FLP class', 1, 6, 'tab10'),
                      (np.where(support, np.max(safe, axis=0), np.nan),
                       'maximum_flp_probability.png', 'Maximum FLP probability',
                       'Probability', 0, 1, 'viridis'),
                      (np.where(support, entropy, np.nan), 'flp_entropy.png',
                       'FLP entropy — simulated intensity dispersion', 'Nats', 0,
                       np.log(6), 'magma'))
            for data, filename, title, label, low, high, cmap_name in mapped:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.set_facecolor('#e5e5e5')
                im = ax.imshow(np.ma.masked_invalid(data), origin='upper', extent=extent,
                               cmap=plt.get_cmap(cmap_name).with_extremes(bad='#e5e5e5'),
                               vmin=low, vmax=high, interpolation='nearest')
                ax.set(xlabel='EPSG:5070 easting (km)', ylabel='EPSG:5070 northing (km)',
                       title=title)
                fig.colorbar(im, ax=ax, label=label)
                save(fig, directory / filename)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.stairs(diag.entropy_hist, ENTROPY_EDGES, fill=True)
            ax.set(xlabel='Shannon entropy (nats)', ylabel='Pixel count',
                   title='Conditional FLP entropy — full raster')
            save(fig, directory / 'flp_entropy_distribution.png')
