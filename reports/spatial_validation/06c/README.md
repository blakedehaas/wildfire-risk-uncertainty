# Checkpoint 6C: nested inner spatial folds

**Status:** independently validated, HITL approved, and frozen. No model, preprocessing fit, HPO, W&B, prediction, or outer TEST performance evaluation was run. Base commit: `173b85fc2a3be4d8b87f224ab14dbf57a65e569f` on `dev`.

Each of the five accepted 5B outer TRAIN populations receives its **own** five coordinate-only inner folds. Exactly 1,097,262 eligible TRAIN coordinates per study were sampled uniformly without replacement with seed 42; the accepted 5B MiniBatchKMeans settings fit five centroids, ordered by centroid x then y. Every eligible row was assigned to its nearest centroid. Outer TEST and DEAD_ZONE rows are null in that study. No inner dead zone, balancing, targets, predictors, or masks determined membership. The [methodology](../../../docs/methodology/checkpoint_06c_inner_spatial_cv.md) distinguishes the spatial model-selection method from the scalable 5B engineering adaptation.

## Source populations and fitting samples

| Outer study | Exact 5B TRAIN rows | Uniform sample rows | Sample-ID Parquet SHA-256 |
| ---: | ---: | ---: | --- |
| 1 | 84,499,397 | 1,097,262 | `1d962f878ebddd6e824e5fa66dad19e6343e8405e9c734560c349e820d714d15` |
| 2 | 81,842,930 | 1,097,262 | `d0ffbbd4a8604f2023f5a9d470164a6beb9eba6c754f91c85140f8208f07f02e` |
| 3 | 79,519,698 | 1,097,262 | `6295365aac046dc4b9547c91ae2b8ec38d9b32c1325989e1eaeece2126a18983` |
| 4 | 77,763,326 | 1,097,262 | `463b3c779e1dfe71fbb04069bdb49e69a9fe5122a33c5636988608ece2768366` |
| 5 | 77,778,108 | 1,097,262 | `1c7fef3c0474f01ddf4686e26c976d2d66984512ed8e6672c76e88de878aa0b7` |

The sample hashes identify files in ignored `artifacts/spatial_validation/06c/fitting_sample_outer_{j}.parquet`. The [manifest](manifest.json) records canonical and 5B provenance, seed, sampler, cluster configuration, all 25 centroids, software versions, and the hashes of all 3,060 output partitions.

## All inner-fold locations and support

Centroids and bounding boxes use EPSG:5070 metres. `BP0`, `BP+`, and `FLP` are BP-zero, BP-positive, and strict FLP-valid row counts. Each INNER TRAIN count and support is the complement of that fold’s validation group within the same outer TRAIN population.

| Outer | Inner | Centroid x | Centroid y | Validation rows (%) | Inner TRAIN rows | Validation BP0 / BP+ / FLP | Inner TRAIN BP0 / BP+ / FLP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | -1,246,205.3 | 1,496,699.8 | 16,365,629 (19.37%) | 68,133,768 | 2,389,944 / 13,975,685 / 13,975,685 | 23,473,225 / 44,660,543 / 44,647,906 |
| 1 | 2 | -88,822.6 | 1,112,691.7 | 17,825,210 (21.10%) | 66,674,187 | 4,065,721 / 13,759,489 / 13,758,087 | 21,797,448 / 44,876,739 / 44,865,504 |
| 1 | 3 | 53,933.0 | 2,303,087.4 | 20,885,589 (24.72%) | 63,613,808 | 10,451,828 / 10,433,761 / 10,422,823 | 15,411,341 / 48,202,467 / 48,200,768 |
| 1 | 4 | 1,047,792.7 | 1,371,046.5 | 17,594,488 (20.82%) | 66,904,909 | 4,932,719 / 12,661,769 / 12,661,475 | 20,930,450 / 45,974,459 / 45,962,116 |
| 1 | 5 | 1,546,918.7 | 2,275,979.3 | 11,828,481 (14.00%) | 72,670,916 | 4,022,957 / 7,805,524 / 7,805,521 | 21,840,212 / 50,830,704 / 50,818,070 |
| 2 | 1 | -1,762,782.2 | 2,636,201.5 | 12,333,575 (15.07%) | 69,509,355 | 1,665,815 / 10,667,760 / 10,667,759 | 23,398,288 / 46,111,067 / 46,098,430 |
| 2 | 2 | -580,067.6 | 2,569,929.4 | 10,553,005 (12.89%) | 71,289,925 | 3,175,022 / 7,377,983 / 7,377,825 | 21,889,081 / 49,400,844 / 49,388,364 |
| 2 | 3 | 290,865.0 | 1,078,976.0 | 23,228,440 (28.38%) | 58,614,490 | 5,381,325 / 17,847,115 / 17,845,684 | 19,682,778 / 38,931,712 / 38,920,505 |
| 2 | 4 | 339,676.4 | 2,200,580.6 | 17,308,483 (21.15%) | 64,534,447 | 9,621,848 / 7,686,635 / 7,675,590 | 15,442,255 / 49,092,192 / 49,090,599 |
| 2 | 5 | 1,453,126.3 | 1,932,392.0 | 18,419,427 (22.51%) | 63,423,503 | 5,220,093 / 13,199,334 / 13,199,331 | 19,844,010 / 43,579,493 / 43,566,858 |
| 3 | 1 | -1,525,328.3 | 2,577,819.4 | 18,476,910 (23.24%) | 61,042,788 | 2,567,269 / 15,909,641 / 15,909,640 | 19,299,365 / 41,743,423 / 41,732,573 |
| 3 | 2 | -1,512,464.6 | 1,666,438.6 | 17,977,312 (22.61%) | 61,542,386 | 2,968,629 / 15,008,683 / 15,008,670 | 18,898,005 / 42,644,381 / 42,633,543 |
| 3 | 3 | -60,348.5 | 2,410,797.4 | 18,335,229 (23.06%) | 61,184,469 | 8,084,224 / 10,251,005 / 10,240,171 | 13,782,410 / 47,402,059 / 47,402,042 |
| 3 | 4 | 1,273,781.8 | 1,285,747.5 | 10,286,263 (12.94%) | 69,233,435 | 2,387,529 / 7,898,734 / 7,898,734 | 19,479,105 / 49,754,330 / 49,743,479 |
| 3 | 5 | 1,378,887.7 | 2,231,804.1 | 14,443,984 (18.16%) | 65,075,714 | 5,858,983 / 8,585,001 / 8,584,998 | 16,007,651 / 49,068,063 / 49,057,215 |
| 4 | 1 | -1,674,590.9 | 2,593,260.1 | 15,950,702 (20.51%) | 61,812,624 | 2,293,749 / 13,656,953 / 13,656,952 | 13,656,141 / 48,156,483 / 48,155,125 |
| 4 | 2 | -1,525,508.3 | 1,678,385.2 | 17,181,580 (22.09%) | 60,581,746 | 2,899,392 / 14,282,188 / 14,282,175 | 13,050,498 / 47,531,248 / 47,529,902 |
| 4 | 3 | -362,473.3 | 1,080,709.3 | 15,521,694 (19.96%) | 62,241,632 | 2,583,992 / 12,937,702 / 12,936,363 | 13,365,898 / 48,875,734 / 48,875,714 |
| 4 | 4 | 908,306.9 | 1,178,678.7 | 16,014,314 (20.59%) | 61,749,012 | 4,016,128 / 11,998,186 / 11,998,183 | 11,933,762 / 49,815,250 / 49,813,894 |
| 4 | 5 | 1,546,950.1 | 2,122,333.0 | 13,095,036 (16.84%) | 64,668,290 | 4,156,629 / 8,938,407 / 8,938,404 | 11,793,261 / 52,875,029 / 52,873,673 |
| 5 | 1 | -1,827,679.8 | 2,516,764.2 | 14,306,310 (18.39%) | 63,471,798 | 1,996,739 / 12,309,571 / 12,309,563 | 16,406,629 / 47,065,169 / 47,052,905 |
| 5 | 2 | -1,463,336.7 | 1,538,934.3 | 15,054,801 (19.36%) | 62,723,307 | 2,366,129 / 12,688,672 / 12,688,666 | 16,037,239 / 46,686,068 / 46,673,802 |
| 5 | 3 | -883,068.0 | 2,337,057.5 | 14,369,102 (18.47%) | 63,409,006 | 2,571,954 / 11,797,148 / 11,797,148 | 15,831,414 / 47,577,592 / 47,565,320 |
| 5 | 4 | -163,137.8 | 1,127,940.0 | 19,814,456 (25.48%) | 57,963,652 | 4,340,499 / 15,473,957 / 15,472,557 | 14,062,869 / 43,900,783 / 43,889,911 |
| 5 | 5 | 25,661.1 | 2,344,495.1 | 14,233,439 (18.30%) | 63,544,669 | 7,128,047 / 7,105,392 / 7,094,534 | 11,275,321 / 52,269,348 / 52,267,934 |

The [inner-fold summary](tables/inner_fold_summary.csv) contains exact row fractions, TRAIN complements, and coordinate bounding boxes. The [centroid table](tables/inner_fold_centroids.csv) contains full-precision coordinates. The [target-support audit](tables/target_support_by_inner_fold.csv) additionally reports BP means and conditional and unconditional FLEP8 support counts and means. These target measurements were calculated **after** coordinate assignments and did not alter them. Different regional target distributions are expected.

## Geographic maps

- [Outer study 1: inner folds and excluded TEST/DEAD_ZONE](figures/outer_1_inner_folds.png)
- [Outer study 2: inner folds and excluded TEST/DEAD_ZONE](figures/outer_2_inner_folds.png)
- [Outer study 3: inner folds and excluded TEST/DEAD_ZONE](figures/outer_3_inner_folds.png)
- [Outer study 4: inner folds and excluded TEST/DEAD_ZONE](figures/outer_4_inner_folds.png)
- [Outer study 5: inner folds and excluded TEST/DEAD_ZONE](figures/outer_5_inner_folds.png)

The five maps were visually checked for gross implementation defects. They show contiguous nearest-centroid regions, clipped by each outer study’s excluded TEST and DEAD_ZONE area. Fold sizes were not optimized or balanced for appearance.

## Eligibility and full-artifact reproduction

The output has **110,817,231 unique canonical cell IDs** in **3,060 aligned partitions**. Each row contains `cell_id` and five nullable `uint8` labels; a label 1–5 exists exactly for the corresponding 5B TRAIN-eligible row, and otherwise is null. Coordinates, targets, predictors, outer distances, and 5B flags remain only in authoritative source artifacts. The [eligibility checks](tables/eligibility_validation.csv) pass for all five studies. All 25 validation and corresponding TRAIN sets are nonempty and have BP-positive and FLP-valid support; validation and TRAIN are disjoint and exhaustive over the applicable outer TRAIN population.

Independent full reproduction passed **7 checks**: all five sample hashes, all 25 refitted centroids, every eligible nearest-centroid label, all 3,060 regenerated partition hashes, unique IDs, exact eligibility, and independently recomputed validation and INNER TRAIN support counts. Reproduction took **88.73 s** with peak RSS **0.56 GiB**. See [reproducibility.json](reproducibility.json) and [independent checks](tables/independent_validation.csv).

Construction took **92.94 s** with peak RSS **0.70 GiB**. The large row-level artifact and five fitting samples remain ignored under `artifacts/spatial_validation/06c/`. The frozen 5B outer artifact and 211.151 km radius were not modified.

Inner folds support model/hyperparameter selection only. The final geographic-transferability estimand belongs to the frozen 5B outer TEST sets; no outer TEST performance was inspected here.
