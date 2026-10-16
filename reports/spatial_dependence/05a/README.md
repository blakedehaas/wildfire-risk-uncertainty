# Checkpoint 5A: target spatial dependence

## Scope and provenance

Run 2 is the **HITL-accepted final spatial-dependence characterization for Checkpoint 5A**. It uses the literature-aligned automap-style lag protocol on the **exact same three deterministic 5,000-cell samples** as Run 1. The [sample manifest](tables/sample_manifest.csv) includes their SHA-256 digests. The approved read-time aliases are `BP` → `bp` and `FLP1`–`FLP6` → `flp1`–`flp6`; neither canonical files nor sample values or masks changed. Run 2 is reproduced with `PYTHONHASHSEED=42 uv run wildfire-spatial-dependence`, which verifies and reads the frozen samples without resampling. The [original prompt](../../../prompts/checkpoints/05a_target_spatial_dependence.md) and [Run 2 HITL amendment](../../../prompts/checkpoints/05a_target_spatial_dependence_run2_hitl.md) preserve instruction provenance.

[Run 1's full-extent tables and figures](run1_full_extent/README.md) are preserved as **superseded diagnostics**. Its roughly 460 km first lag bin and stable-for-Stein model mismatch prevent its use for range inference. The [methodology](../../../docs/methodology/checkpoint_05a_spatial_dependence.md) and [evidence ledger](../../../docs/methodology/methodology_evidence_ledger.csv) give the literature lineage, implementation details, and deviations from R automap/gstat.

## Empirical result: Run 2

The populations are 110,817,231 BP-valid cells, 81,068,773 strict conditional-FLP cells, and 110,804,580 cells with defined unconditional FLEP8. Each frozen sample has 5,000 unique cells. The [empirical lag table](tables/variogram_empirical.csv) preserves the exact upper and lower boundaries, mean pair separation, Matheron semivariance, and pair counts for every target. FLP1–FLP6 and conditional FLEP8 share one sample, so their boundaries and pair counts are identical. All 12 original boundaries met automap's minimum of five pairs; no bins were merged.

| Lag | BP boundary km | BP pairs | Conditional boundary km | Conditional pairs | Unconditional boundary km | Unconditional pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.537 | 6,877 | 37.285 | 7,610 | 37.690 | 6,723 |
| 2 | 75.074 | 19,787 | 74.570 | 22,057 | 75.381 | 19,793 |
| 3 | 112.611 | 32,645 | 111.855 | 34,863 | 113.071 | 32,510 |
| 4 | 168.916 | 71,082 | 167.782 | 76,080 | 169.607 | 71,608 |
| 5 | 225.222 | 95,967 | 223.709 | 102,851 | 226.142 | 96,740 |
| 6 | 281.527 | 120,387 | 279.637 | 127,348 | 282.678 | 120,537 |
| 7 | 469.212 | 552,858 | 466.061 | 571,286 | 471.130 | 554,908 |
| 8 | 656.897 | 749,448 | 652.486 | 748,561 | 659.582 | 754,853 |
| 9 | 938.424 | 1,394,291 | 932.123 | 1,355,950 | 942.260 | 1,398,082 |
| 10 | 1,219.951 | 1,560,916 | 1,211.759 | 1,520,063 | 1,224.938 | 1,573,947 |
| 11 | 1,501.479 | 1,580,492 | 1,491.396 | 1,531,475 | 1,507.616 | 1,598,636 |
| 12 | 1,876.848 | 1,919,876 | 1,864.245 | 1,833,991 | 1,884.520 | 1,942,561 |

The exact metre and kilometre boundaries, sample bounding-box diagonals, and merge flags are in [lag_boundaries.csv](tables/lag_boundaries.csv). The selected fits below minimize **weighted SSE** with `N_k/h_k²` weights, where `h_k` is mean pair separation. RMSE and R² are ordinary unweighted empirical-bin diagnostics. The [all-fits table](tables/variogram_fits.csv) also records alternative families, unweighted RSS, Matérn smoothness, and numerical fit status. Effective ranges are reported only where the fit passes the supported-sill checks.

| Target | Selected family | Nugget | Partial sill | Total sill | Nugget / total | Effective range km | Weighted SSE | RMSE | R² | Supported sill |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bp | matern | 1.690e-5 | 2.144e-5 | 3.834e-5 | 0.441 | 211.151 | 1.993e-17 | 1.098e-6 | 0.956 | yes |
| flp1 | spherical | 0.078996 | 0.041230 | 0.120226 | 0.657 | RANGE_UNRESOLVED | 4.145e-10 | 0.005255 | 0.895 | no |
| flp2 | exponential | 0.050124 | 0.018485 | 0.068609 | 0.731 | 173.810 | 4.409e-12 | 0.000494 | 0.972 | yes |
| flp3 | matern | 0.002088 | 0.041024 | 0.043111 | 0.048 | RANGE_UNRESOLVED | 2.674e-11 | 0.001464 | 0.930 | no |
| flp4 | matern | 0.000197 | 0.013670 | 0.013868 | 0.014 | RANGE_UNRESOLVED | 5.861e-13 | 0.000178 | 0.985 | no |
| flp5 | matern | 0.009179 | 0.002596 | 0.011775 | 0.780 | RANGE_UNRESOLVED | 1.032e-12 | 0.000272 | 0.906 | no |
| flp6 | spherical | 0.008124 | 0.003132 | 0.011256 | 0.722 | RANGE_UNRESOLVED | 3.988e-12 | 0.000471 | 0.807 | no |
| conditional_flep8 | spherical | 0.026591 | 0.009257 | 0.035848 | 0.742 | RANGE_UNRESOLVED | 2.217e-11 | 0.001310 | 0.824 | no |
| unconditional_flep8 | spherical | ≈0 | 1.856e-5 | 1.856e-5 | ≈0 | 82.059 | 4.846e-17 | 1.697e-6 | 0.476 | yes |

The exact values are in [variogram_selected_ranges.csv](tables/variogram_selected_ranges.csv). The [figures](figures/) show empirical semivariance at mean pair separation, the selected theoretical fit, and a range marker only when resolved. The [runtime telemetry](runtime_telemetry.json) records Run 2 duration and peak resident memory.

## Literature precedent and interpretation

Valavi et al. used 5,000 sampled raster locations and isotropic variograms before spatial blocking. Automap's nonuniform boundary rule and five-pair bin-merging default are used here. Automap's default gstat fitting weights are `N_k/h_k²`; SciKit-GStat receives the corresponding sigma array. Its Matérn family is an analogue of the automap Stein family, not an exact numerical duplicate. SciKit-GStat fits at upper class boundaries, while gstat uses mean class distances; that difference is explicit in the methodology.

The short-lag BP and FLP2 empirical semivariances approach supported sills, giving effective ranges of approximately 211.151 km and 173.810 km. Unconditional FLEP8 also passes the predefined diagnostics at approximately 82.059 km, though its unweighted R² (0.476) is materially weaker than BP (0.956) and FLP2 (0.972). FLP1 and FLP3 continue rising toward the last examined lag, and FLP4 nearly reaches the range boundary. FLP5, FLP6, and conditional FLEP8 have fitted interior ranges but fail the empirical tail-to-sill check and remain **RANGE_UNRESOLVED**. These unresolved targets may reflect nonstationarity or multiscale continental structure; this checkpoint does not add a treatment for it. The evidence does not support one universal target-derived autocorrelation range for the complete multi-output prediction problem. All results are raw-target diagnostics rather than model-residual ranges or fixed-map accuracy estimates. The variogram nugget must not be interpreted as model aleatoric uncertainty.

## Unresolved decision

Future HITL review must decide which spatial quantity determines validation separation, including whether BP, predictors, residuals, or another literature-backed rule should govern scale. Block, cluster, or buffered validation; fold count and allocation; inner spatial resampling for hyperparameter tuning; buffer distance; and any final train/validation/test proportions remain unfrozen. No folds, block size, or spatial split are constructed here.
