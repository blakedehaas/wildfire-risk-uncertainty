# Checkpoint 6C: nested inner spatial folds

**Status:** HITL approved and frozen. No model, preprocessing fit, HPO, outer TEST evaluation, or prediction was run. Base commit: `173b85fc2a3be4d8b87f224ab14dbf57a65e569f` on `dev`.

## Scientific purpose and hierarchy

For **each** of the five frozen 5B outer studies, the usable outer TRAIN population is exactly the rows with `train_eligible_fold_j == True`. This population receives its own five spatial inner folds. One group is INNER VALIDATION for a particular fit; the other four groups together are INNER TRAIN. The inner groups are used only for model and hyperparameter selection within that outer TRAIN population. Outer TEST and DEAD_ZONE cells have no inner label for that study and never enter inner model selection. No inner 211.151 km exclusion zone is applied.

```text
full canonical population
└── outer study j
    ├── OUTER TEST              excluded from inner selection
    ├── OUTER DEAD_ZONE         excluded from inner selection
    └── OUTER TRAIN             exactly five inner validation groups
        ├── inner fold 1
        ├── inner fold 2
        ├── inner fold 3
        ├── inner fold 4
        └── inner fold 5
```

The frozen 5B outer TEST populations with the 211.151 km exclusion radius estimate final geographic-transferability performance. The inner folds **do not** estimate that performance. The independent study-specific construction prevents one outer study's TEST population from entering its own model-selection procedure through a national reused fold design.

The scientific method is five spatial inner folds, grouped on projected Euclidean `x_5070` and `y_5070` coordinates only, nested separately inside each outer TRAIN population. [Schratz et al. (2019)](https://doi.org/10.1016/j.ecolmodel.2019.06.002) implemented spatial hyperparameter tuning and spatial assessment and motivates spatially structured tuning for spatial data. That paper did **not** prescribe this project's exact five-group implementation, fitting sample, MiniBatchKMeans settings, or absence of an inner dead zone. Those details are accepted project design and engineering choices. BP, FLP, WUI, predictors, null/coverage diagnostics, target masks, state boundaries, and row-count balancing do not determine assignment.

## Scalable construction

The implementation reuses 5B's exact uniform row-position sampler and its coordinate MiniBatchKMeans configuration. For each outer study independently, sorted canonical tile and row order induces a bijection from eligible TRAIN cells to ranks `0..N_j-1`. `random.Random(42).sample(range(N_j), 1_097_262)` selects exactly that many ranks uniformly without replacement, in bounded memory. A streaming pass maps those ranks to canonical cell IDs and two coordinates. The fitting sample is permuted by NumPy seed 42 and fitted using the accepted 5B MiniBatchKMeans settings: five Euclidean clusters, `random_state=42`, batch size 16,384, `n_init=10`, `max_iter=100`, and `reassignment_ratio=0`. Clusters are relabeled by ascending centroid x, then y. Every eligible row is assigned to its nearest fitted centroid. No membership is changed to equalize sizes or target distributions.

The 1,097,262-row sample size and MiniBatchKMeans are scalability adaptations reused from validated 5B, **not** a new scientific sampling rule or methods attributed to Schratz et al. The rejected cell-ID modulo sampling method is not used. The five saved sample-ID Parquet files and hashes, 25 centroids, software versions, source manifest hashes, and row-level partition hashes are recorded in the ignored artifact manifest and the committed report manifest.

The row-level artifact has 3,060 partitions aligned to canonical and 5B tiles. Each row stores exactly `cell_id` and five nullable `uint8` inner labels. A label is 1–5 exactly when 5B marks that row TRAIN-eligible for that outer study; otherwise it is null. No coordinate, predictor, target, distance, or 5B eligibility column is duplicated. The accepted 5B artifact remains the authoritative source of eligibility. The 6C artifact and its manifest live under ignored `artifacts/spatial_validation/06c/`.

## Validation and interpretation

Construction checks each tile's canonical and 5B cell IDs, verifies the frozen 5B partition hash, and verifies every study's assigned total against the accepted 5B TRAIN count. Each of the 25 INNER VALIDATION sets is checked to be nonempty and to contain BP-positive and strict FLP-valid support; its complementary INNER TRAIN set is checked the same way. All TEST and DEAD_ZONE rows remain null. The five validation groups are disjoint and exhaustive over the applicable outer TRAIN population. [The 6C report](../../reports/spatial_validation/06c/README.md) provides centroids, counts, fractions, coordinate bounding boxes, and maps with excluded regions masked.

Target audits report BP-zero, BP-positive, strict FLP-valid counts, BP mean, and conditional/unconditional FLEP8 support means for each validation set and its training complement. They are computed **after** coordinate assignments and never influence membership. Regional differences are expected; no target balancing or fold rejection is performed on their basis.

An independent validation command resamples all five eligible populations from ranks, reproduces five sample-file hashes, refits all 25 centroids, reassigns every eligible canonical row, serializes all 3,060 partitions anew, and compares every partition hash. This full reproduction is stronger than refitting on saved samples alone. The validator also checks unique canonical IDs, nullable schema, exact 5B eligibility, exclusion of outer TEST/DEAD_ZONE cells, nearest-centroid assignment, and target-support counts. The results are in the report's `reproducibility.json`.

The maps are implementation diagnostics, not an objective for optimizing fold appearance. If geography suggests a defect, methodology review is required before any seed, balancing, or k change. Projected Euclidean centroid regions need not have equal row counts, and the inner partitions do not impose the outer geographic transfer distance.
