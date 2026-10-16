# Checkpoint 5A: target spatial dependence

**HITL acceptance:** Run 2 is the final spatial-dependence characterization for Checkpoint 5A. Run 1 is retained only as a superseded diagnostic. The [original checkpoint prompt](../../prompts/checkpoints/05a_target_spatial_dependence.md) and [Run 2 HITL amendment](../../prompts/checkpoints/05a_target_spatial_dependence_run2_hitl.md) preserve the instructions that produced these analyses.

## Scientific purpose

This checkpoint estimates isotropic spatial dependence of the frozen canonical targets before any spatial folds exist. The later evaluation goal is **geographic transferability to spatially separated locations**. [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) argue that validation should reflect dependence and the intended prediction task; random pixel CV is therefore not the intended design. [Wadoux et al. (2021)](https://doi.org/10.1016/j.ecolmodel.2021.109692) distinguish transfer-oriented spatial CV from design-unbiased accuracy assessment of a fixed map. Future spatial-CV results here must not be described as design-unbiased fixed-map accuracy.

## Frozen input and analysis aliases

The canonical Parquet partitions at `artifacts/dataset/canonical/partitions/` remain unchanged. The committed Checkpoint 4B manifest, schema, validation summary, and mask counts establish 110,817,231 BP-valid EPSG:5070 rows on a 270 m grid with pixel-center coordinates `x_5070`, `y_5070`. Run 1 checked those reports before streaming the partitions. Run 2 again checks the canonical reports and verifies the hashes, support counts, seed, and ordering of the three frozen sample files before fitting; it does not resample.

The physical canonical target columns are mapped at read time, with no value conversion or modification:

| Canonical column | Analysis alias |
| --- | --- |
| `BP` | `bp` |
| `FLP1` | `flp1` |
| `FLP2` | `flp2` |
| `FLP3` | `flp3` |
| `FLP4` | `flp4` |
| `FLP5` | `flp5` |
| `FLP6` | `flp6` |

The canonical `flp_target_valid` mask is used as stored. No FLP target is filled, clipped, transformed, or renormalized. `conditional_flep8 = flp5 + flp6` is evaluated on the strict FLP support. `unconditional_flep8` is zero where BP is zero, `bp * (flp5 + flp6)` where `flp_target_valid` is true, and undefined for positive BP with invalid FLPs. The three population sizes are 110,817,231 BP cells, 81,068,773 strict FLP cells, and 110,804,580 cells with defined unconditional FLEP8.

## Sampling and variograms

[Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107) provide a pre-model autocorrelation workflow using isotropic empirical variograms and random raster samples of 5,000 by default; they discuss raw responses as a pre-model option. Run 1 independently sampled 5,000 cells uniformly without replacement from each of the three supports. All six FLPs and conditional FLEP8 share one sample. The streaming priority sampler used seeded NumPy generators, sorted partition paths, and stable cell-ID output ordering. Seed and `PYTHONHASHSEED` were 42. Run 2 reuses the **same exact three Parquet sample files**. Their SHA-256 digests are fixed in the sample manifest; no target or mask values change.

## Run 1: superseded full-extent diagnostic

Run 1 used ten equal-width bins through the full sampled pair-distance extent, giving a first bin roughly 460 km wide. It also used SciKit-GStat `stable` in place of automap's `Ste` family and selected by unweighted RSS. Those are methodological defects for the intended shorter-range assessment. All Run 1 ranges were marked unresolved. Its tables, figures, and runtime are retained in [run1_full_extent](../../reports/spatial_dependence/05a/run1_full_extent/) as **superseded provenance**, not as evidence that target ranges are intrinsically unresolved.

## Run 2: automap-aligned lag and fit protocol

For each fixed sample population, compute the planar bounding-box diagonal `D = sqrt((xmax − xmin)² + (ymax − ymin)²)` in EPSG:5070 metres. The twelve upper boundaries are `[2, 4, 6, 9, 12, 15, 25, 35, 50, 65, 80, 100] × (0.35D)/100`. These coefficients, the 0.35 factor, and the default `min.np.bin = 5` come from [automap's `autofitVariogram` source](https://github.com/cran/automap/blob/master/R/autofitVariogram.r), the workflow used by Valavi. If any bin has fewer than five pairs, drop the earliest boundary and recompute all bins, repeating until every retained bin has at least five pairs. This reproduces automap's short-distance-bin merging rule; no semivariance is interpolated. The report records initial boundaries, retained boundaries, mean pair distances, and pair counts. All targets on the strict FLP sample use identical boundaries and pair populations.

Raw targets, Euclidean distance, isotropy, and SciKit-GStat's Matheron semivariance estimator remain unchanged. [Mälicke et al. (2022)](https://doi.org/10.5194/gmd-15-2505-2022) describe SciKit-GStat's estimators and model parameterization. Run 2 fits spherical, exponential, Gaussian, and **Matérn** families with a free nugget. Automap's `Ste` is a Matérn/Stein parameterization, not SciKit-GStat's `stable` model. SciKit-GStat's Matérn is a Python Matérn-family analogue and is **not claimed numerically identical** to gstat's Stein parameterization. The SciKit smoothness parameter is optimized continuously, initialized at 1; automap tests a kappa grid. Other initial values adapt automap's starting rules: nugget is the minimum empirical semivariance, range is 0.1D, and initial total sill is the mean of the maximum and median empirical semivariances.

For lag class `k`, let `N_k` be its pair count and `h_k` its **mean pair separation in metres**. Set SciKit-GStat fitting `sigma_k = (h_k / sqrt(N_k)) / median_j(h_j / sqrt(N_j)) × median_j(|γ_j|)`. The final two factors are common across bins and only set numerical scale, keeping optimizer residuals near unit scale without changing target values or the minimizer. SciKit-GStat passes sigma to SciPy's bounded least-squares fit, minimizing `Σ_k [(γ_k − model(b_k)) / sigma_k]²`, where `b_k` is SciKit-GStat's upper lag boundary. This is proportional to the unscaled `Σ_k (N_k / h_k²)[γ_k − model(b_k)]²`. The latter **weighted SSE** is reported and used for model selection. [gstat's `fit.variogram` documentation](https://r-spatial.github.io/gstat/reference/fit.variogram.html) confirms `N_k/h_k²` as its default method 7; automap calls that fitter and selects lowest fitted SSE. SciKit-GStat fits at upper boundaries; automap/gstat uses class mean distances. This is a documented Python implementation difference. Ordinary unweighted RSS, RMSE, and empirical-bin R² are also reported but do not select the family.

SciKit-GStat's `sill` key is the **partial sill**. The total sill is partial sill plus nugget; the nugget/total-sill ratio is reported. `effective_range` is the reported effective range, not an internal mathematical scale parameter. For asymptotic models this corresponds to about 95% of the partial-sill rise. A range is resolved only if parameters are finite, partial sill is positive, nugget is nonnegative, the effective range is below 90% of the last lag boundary, at least two bins with five or more pairs lie beyond the range, unweighted R² is nonnegative, and the mean absolute discrepancy of the last two supported empirical bins from the total sill is no more than twice the unweighted RMSE. These conservative diagnostics guard against a boundary-pinned or unsupported sill. Otherwise the target is `RANGE_UNRESOLVED` with all empirical lags and candidate fits preserved; no range is extrapolated.

The isotropic workflow assumes approximate stationarity, as Valavi et al. note. No detrending, regional variograms, anisotropy, or other treatment is added. A remaining rising semivariogram is reported as possible nonstationarity or multiscale continental structure for HITL review.

## Accepted scientific conclusion

Under the selected global stationary isotropic fits, BP has a supported effective range of approximately **211.151 km** (Matérn), FLP2 approximately **173.810 km** (exponential), and unconditional FLEP8 approximately **82.059 km** (spherical). The unconditional FLEP8 fit has materially weaker unweighted quality (R² **0.476**) than BP (**0.956**) and FLP2 (**0.972**); its resolved range must be read with that caveat. FLP1, FLP3, FLP4, FLP5, FLP6, and conditional FLEP8 remain **RANGE_UNRESOLVED**. These findings do not support assuming one universal target-derived autocorrelation range for the complete multi-output prediction problem. A fitted variogram nugget describes near-zero-distance semivariance in this raw-target analysis; it must not be interpreted as model aleatoric uncertainty.

[Ploton et al. (2020)](https://doi.org/10.1038/s41467-020-18321-y) measured response autocorrelation with a semivariogram before constructing spatial validation units slightly larger than its range. This checkpoint stops at measurement; it does not copy their clustering or select unit size. The raw-response variogram may reflect broad geographic trends and zero inflation as well as local dependence, and it cannot substitute for a future model-residual variogram.

## Decision state

**FROZEN**

- The evaluation goal is a later geographic-transferability assessment.
- Random pixel CV is not the intended validation design.
- Empirical target semivariograms characterize dependence before fold construction.
- The three deterministic 5,000-cell samples, automap-aligned lag construction, global isotropic semivariograms, and target-specific resolved/unresolved Run 2 findings are validated and accepted for this checkpoint.

**NOT YET FROZEN**

- Which spatial quantity determines validation separation, including whether BP, predictors, residuals, or another literature-backed rule governs scale.
- Square blocks, spatial clusters, or a buffered validation scheme.
- Number of outer folds and number of inner folds.
- Fold allocation method.
- Buffer distance, if any.
- How inner spatial resampling will be implemented for hyperparameter tuning.
- Final train/tuning/test proportions, if such proportions are used.

[Schratz et al. (2019)](https://doi.org/10.1016/j.ecolmodel.2019.06.002) motivate separate spatial resampling loops for tuning and final performance estimation. That is documented for a later checkpoint; no nested CV or folds are implemented here.
