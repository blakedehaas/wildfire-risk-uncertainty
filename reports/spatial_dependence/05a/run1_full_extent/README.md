# Run 1: superseded full-extent diagnostic

These results used ten equal-width bins over the full sampled pair-distance extent. The first bin was about 460 km wide, so this run was superseded for range inference by Run 2. These files are retained only as diagnostic provenance. See the [revised methodology](../../../../docs/methodology/checkpoint_05a_spatial_dependence.md). The three frozen samples remain in the parent report's `samples/` directory.

The approved read-time aliases are `BP` → `bp` and `FLP1`–`FLP6` → `flp1`–`flp6`, respectively. The canonical files and float values are unchanged.

## Empirical result

The [sample manifest](tables/sample_manifest.csv) records each support and sample hash; the unchanged [sample Parquet files](../samples/) contain exact sampled cells and values. The [empirical lag table](tables/variogram_empirical.csv) contains distance, Matheron semivariance, and pair count for every target. [All family fits](tables/variogram_fits.csv), [selected results](tables/variogram_selected_ranges.csv), and [figures](figures/) preserve diagnostics and unresolved results.

Each sample contains exactly 5,000 distinct cells. The BP support contains 110,817,231 cells, strict conditional-FLP support 81,068,773, and defined unconditional FLEP8 support 110,804,580. Every lowest-RSS family reaches the examined-distance boundary. Consequently **all nine effective ranges are `RANGE_UNRESOLVED`**; the boundary parameter values in the all-fits table are not reported as spatial ranges.

| Target | Lowest-RSS family | Range | Fit RMSE | Empirical-bin R² | Maximum lag (km) |
| --- | --- | --- | ---: | ---: | ---: |
| bp | stable | RANGE_UNRESOLVED | 8.51e-5 | -3.336 | 4,624.7 |
| flp1 | stable | RANGE_UNRESOLVED | 0.01311 | 0.923 | 4,593.4 |
| flp2 | spherical | RANGE_UNRESOLVED | 0.00851 | 0.229 | 4,593.4 |
| flp3 | stable | RANGE_UNRESOLVED | 0.01217 | -5.340 | 4,593.4 |
| flp4 | stable | RANGE_UNRESOLVED | 0.00349 | -10.416 | 4,593.4 |
| flp5 | stable | RANGE_UNRESOLVED | 0.00409 | -9.118 | 4,593.4 |
| flp6 | stable | RANGE_UNRESOLVED | 0.00697 | -4.484 | 4,593.4 |
| conditional_flep8 | stable | RANGE_UNRESOLVED | 0.01727 | -5.284 | 4,593.4 |
| unconditional_flep8 | stable | RANGE_UNRESOLVED | 4.40e-5 | -3.627 | 4,621.0 |

The first equal-width lag spans roughly 0–459/462 km because the maximum sampled separation is about 4,600 km. This setup cannot distinguish shorter dependence scales within that first interval. The last lag has 21,663 pairs for BP, 25,698 for conditional FLPs, and 24,458 for unconditional FLEP8, against more than 2 million pairs in several middle bins. The full lag table should therefore be read with the coarse first bin and unequal pair counts in view. The [runtime telemetry](runtime_telemetry.json) records the first-run duration and peak resident memory.

## Literature precedent

Valavi et al. used 5,000 random raster samples and isotropic variograms to characterize autocorrelation before spatial blocking. Ploton et al. measured a response range before constructing validation units. Mälicke et al. provide the SciKit-GStat implementation; Hiemstra et al. provide the automatic residual-based model selection precedent. Full DOI-level links and implementation differences are in the methodology evidence ledger.

## Interpretation

These are exploratory raw-target dependence diagnostics on three explicitly different target supports. They are not model-residual ranges, proof of stationarity, or estimates of fixed-map accuracy. The empirical BP semivariance rises from 3.73e-5 in the first lag to 1.72e-4 in the last, and the unconditional FLEP8 semivariance rises from 1.87e-5 to 8.49e-5. FLP3 and FLP4 are nonmonotone. These patterns and the boundary fits prevent a defensible finite spatial-dependence range under this analysis setup.

## Unresolved decision

The target or range that will govern spatial resampling, the geometry and allocation of spatial units, fold counts, buffer distance, and any split proportions require later HITL review. The unresolved fits also require HITL review before any alternative lag window, binning, estimator, model-fitting method, or local/regional analysis is chosen. No spatial folds or final block size are created here.
