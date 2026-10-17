# Checkpoint 5B: frozen outer spatial CV

This scheme estimates geographic-transferability performance under a controlled minimum geographic separation from training observations.
Spatial evaluation is used because nearby observations can share BP response structure; the exclusion radius is the exact accepted Checkpoint 5A BP effective range. BP is the primary wildfire-risk response, has complete canonical support, and had the strongest clearly resolved global variogram.
It is not random-CV performance, ordinary interpolation performance, design-unbiased CONUS map accuracy, or evaluation of the later all-data production model.
No model is fitted in this checkpoint.
Following Schratz et al. (2019), five spatially coherent outer folds are formed by Euclidean coordinate k-means on EPSG:5070 `x_5070` and `y_5070`. The five-fold count follows that study's outer assessment design. Full-batch fitting over 110,817,231 rows is costly, so a deterministic uniform-without-replacement sample of 1,097,262 canonical coordinates is fitted with MiniBatchKMeans, then every canonical row is assigned to its nearest fitted centroid. This sampling and mini-batch computation are engineering adaptations, not additional scientific partitioning rules.
Fold membership uses coordinates only. No target, predictor, WUI, or row-count balancing is performed: those differences are reported as diagnostics rather than used to change the spatial partition. The corrected folds were frozen by HITL approval before model development. Inner spatial validation for hyperparameter tuning is deferred to a later literature-grounded checkpoint.

The exact accepted Checkpoint 5A BP radius is **211150.99027309995 m**. Every canonical row is TEST in exactly one of five folds.
For a fold, TRAIN contains only non-test canonical cells strictly farther than the radius from every TEST cell; nearer cells form DEAD_ZONE.
For each TEST cell, `oof_nearest_training_distance_m` is the exact Euclidean cell-center distance to its nearest usable TRAIN cell in that fold; every value exceeds the radius.
The full-grid Euclidean distance transform treats raster holes as background geometry, never as movement barriers or model rows.

| Fold | TEST rows (%) | DEAD_ZONE rows (%) | TRAIN rows (%) | OOF minimum / median / p95 / maximum (km) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 19,595,079 (17.68%) | 6,722,755 (6.07%) | 84,499,397 (76.25%) | 211.153 / 582.490 / 1,170.163 / 1,401.206 |
| 2 | 19,594,646 (17.68%) | 9,379,655 (8.46%) | 81,842,930 (73.85%) | 211.151 / 490.633 / 874.225 / 1,030.108 |
| 3 | 21,984,912 (19.84%) | 9,312,621 (8.40%) | 79,519,698 (71.76%) | 211.151 / 517.721 / 1,001.335 / 1,186.557 |
| 4 | 22,597,733 (20.39%) | 10,456,172 (9.44%) | 77,763,326 (70.17%) | 211.151 / 494.117 / 918.447 / 1,048.446 |
| 5 | 27,044,861 (24.40%) | 5,994,262 (5.41%) | 77,778,108 (70.19%) | 211.151 / 754.527 / 1,330.274 / 1,547.042 |

The smallest exact OOF distance is 211151.04830428856 m, which exceeds the radius by 0.05803118861513212 m; the rounded kilometre display above obscures that positive margin.

The [centroid table](tables/fold_centroids.csv) records ordered EPSG:5070 coordinates. The [target audit](tables/target_summary_by_fold.csv) reports full-support BP-zero, BP-positive, FLP-valid, WUI, and mean summaries for TEST and TRAIN. Every BP and FLEP8 quantile column is explicitly labeled `sample_estimated`; these diagnostic quantiles retain the deterministic `cell_id % 101 == 0` audit subset. Distance quantiles use all TEST cells. This audit subset is separate from the uniform fitting sample.
The [independent separation checks](tables/separation_validation.csv) recompute status and OOF distances from the written artifact. [Partition hashes](tables/artifact_hashes.csv), the [manifest](manifest.json), the [reproducibility check](reproducibility.json), the [fold map](figures/outer_folds.png), five status maps, and the [distance distribution](figures/nearest_training_distance_distribution.png) complete the audit.

The large row-level artifact is in the ignored `artifacts/spatial_validation/05b/partitions/` directory, aligned to the 3,060 canonical tiles. It stores cell ID, one outer fold, one OOF nearest-training distance, and five training-eligibility flags; no target or predictor is duplicated.

Rebuild with `PYTHONHASHSEED=42 uv run wildfire-spatial-folds`, validate with `PYTHONHASHSEED=42 uv run wildfire-validate-folds`, and verify deterministic centroids and assignments with `PYTHONHASHSEED=42 uv run wildfire-reproduce-folds`. See [the method](../../../docs/methodology/checkpoint_05b_outer_spatial_cv.md), [the original prompt](../../../prompts/checkpoints/05b_construct_outer_spatial_cv.md), [the uniform-sampling HITL amendment](../../../prompts/checkpoints/05b_uniform_sampling_hitl.md), and [the corrected-geometry HITL decision](../../../prompts/checkpoints/05b_accept_corrected_geometry_hitl.md). The [pre-HITL comparison](correction_comparison.md) records only superseded implementation provenance.

MiniBatchKMeans on an exact uniform-without-replacement coordinate sample approximates ordinary full-batch coordinate k-means for scale. Every row is nevertheless assigned to its nearest fitted centroid. The native-grid distance transform is exact for Euclidean cell-center geometry; projected distance is not terrain-path or geodesic distance. Folds were not rebalanced using targets or predictors.
