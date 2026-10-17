Checkpoint 5B — Construct and validate outer spatial cross-validation

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

Required base commit:
  db0364f654cff53850984ea2a7760b8734828bcc

Before changing anything:
1. Verify HEAD == required base commit.
2. Verify origin/dev == required base commit.
3. Verify branch == dev.
4. Verify working tree is clean.
5. Verify accepted Checkpoint 5A reports are present.
6. Do not modify main.
7. Save this exact prompt verbatim as:
   prompts/checkpoints/05b_construct_outer_spatial_cv.md

SCIENTIFIC PURPOSE

Construct and validate the frozen OUTER spatial evaluation design for
later geographic-transferability experiments and national out-of-fold
predictions.

This checkpoint is model-free.

Do NOT:
- train models
- preprocess predictors
- tune hyperparameters
- create inner CV folds
- run SHAP or uncertainty estimation
- create an 80/10/10 split
- modify canonical data
- use targets or predictors to balance/optimize spatial folds

The intended interpretation is:

  Every future OOF prediction is produced by a model that did not train
  on that cell or on any canonical training observation within the
  literature- and data-derived geographic exclusion radius.

LITERATURE-GROUNDED DESIGN

D1 — Spatial k-fold with dead zone
Primary source:
  Pohjankukka et al. (2017)
  International Journal of Geographical Information Science
  DOI: 10.1080/13658816.2017.1346255

Implementation:
For held-out outer fold j:
- TEST_j = cells assigned to fold j
- DEAD_ZONE_j = non-test canonical cells within distance <= r of TEST_j
- TRAIN_j = remaining canonical cells with distance > r from TEST_j

Thus every usable training observation must be strictly farther than r
from every held-out test observation.

D2 — Geographic exclusion radius
Primary source:
  Ploton et al. (2020)
  Nature Communications
  DOI: 10.1038/s41467-020-18321-y

Supporting source:
  Valavi et al. (2019)
  Methods in Ecology and Evolution
  DOI: 10.1111/2041-210X.13107

Implementation:
Use the accepted Checkpoint 5A BP effective range as r because BP is the
primary wildfire-risk response, has complete canonical support, and had
the strongest clearly resolved global variogram.

Programmatically read the exact accepted BP effective range from the
committed Checkpoint 5A selected-range report.

Expected approximately:
  211.151 km

Do not manually round or invent a new radius.

Fail and stop if the committed BP result is missing, unresolved, or does
not match the accepted Checkpoint 5A result.

D3 — Spatially coherent outer folds
Primary source:
  Schratz et al. (2019)
  Ecological Modelling
  DOI: 10.1016/j.ecolmodel.2019.06.002

Implementation:
Construct spatially coherent folds using k-means clustering of geographic
coordinates only.

Use:
  x_5070
  y_5070

Do NOT use:
  BP
  FLPs
  WUI variables
  any predictor
  state boundaries
  target distributions
  any target-balancing or stratification rule

D4 — Number of folds
Primary source:
  Schratz et al. (2019)

Use exactly:
  5 outer spatial folds

Every canonical row must belong to exactly one test fold and therefore
be eligible for exactly one future OOF prediction.

D5 — Hyperparameter tuning
Schratz et al. also motivates spatial inner resampling for tuning, but
DO NOT implement inner folds in this checkpoint.

INPUT DATA

Use the frozen canonical dataset:

  artifacts/dataset/canonical/partitions/

Expected:
- 110,817,231 canonical rows
- EPSG:5070
- 270 m grid
- width 17,372
- height 11,283
- stable cell_id
- x_5070 / y_5070 cell centers

Validate these assumptions from committed reports first.

Do not modify canonical files.

REPRODUCIBILITY

Use:
  seed = 42
  PYTHONHASHSEED=42

Require explicit deterministic ordering.

The full dataset is too large for naive in-memory full-batch k-means.
Use a deterministic bounded-memory scalable implementation, e.g.
MiniBatchKMeans or equivalent, provided that:

- the scientific operation remains Euclidean coordinate k-means
- only x_5070/y_5070 enter clustering
- seed 42 controls initialization
- every canonical row is ultimately assigned to its nearest fitted centroid
- centroids are persisted
- behavior is deterministic within the pinned environment
- the approximation relative to ordinary full-batch k-means is documented
  as an engineering scalability adaptation, not a new scientific method

After fitting, deterministically relabel arbitrary cluster IDs using a
stable centroid-ordering convention without changing membership.

DEAD-ZONE GEOMETRY

Use EPSG:5070 Euclidean cell-center distance in metres.

Exploit the regular native raster rather than all-pairs calculations.

For each fold:
1. Rasterize that fold's canonical TEST cells onto the full native grid.
2. Compute exact Euclidean distance to the nearest TEST cell using a
   deterministic Euclidean distance transform or equivalent exact method.
3. For non-test canonical cells:
     distance <= r -> DEAD_ZONE
     distance > r  -> TRAIN
4. Noncanonical grid cells are geometry/background only and must never
   become training or test rows.

Raster holes must not act as movement barriers.

For every TEST cell also compute:

  oof_nearest_training_distance_m

meaning the Euclidean distance to the nearest usable TRAIN cell for the
fold in which that cell is held out.

Every such value must be strictly > r.

ROW-LEVEL ARTIFACT

Write large row-level output under:

  artifacts/spatial_validation/05b/

Keep it ignored by Git.

Prefer partitioning aligned with the canonical Parquet tiles.

At minimum persist:

  cell_id
  outer_fold
  oof_nearest_training_distance_m
  train_eligible_fold_1
  train_eligible_fold_2
  train_eligible_fold_3
  train_eligible_fold_4
  train_eligible_fold_5

For fold j:
- rows in TEST_j must have train_eligible_fold_j = false
- DEAD_ZONE_j rows = false
- TRAIN_j rows = true

Do not duplicate canonical targets or predictors.

The artifact must contain exactly 110,817,231 unique canonical cell_ids.

Create an artifact manifest recording:
- schema
- row count
- partition count
- hashes
- seed
- exact radius
- fitted centroids
- deterministic fold-label convention
- source commit provenance
- relevant software versions

AUDIT EACH FOLD

Report:

- test rows and fraction
- dead-zone rows and fraction
- usable training rows and fraction
- BP-zero and BP-positive counts in test/train
- FLP-target-valid counts in test/train
- mean and quantiles of BP
- conditional FLEP8 summary where valid
- unconditional FLEP8 summary where defined
- useful WUI coverage summaries
- minimum nearest-training distance
- p5 / p25 / median / p75 / p95 nearest-training distance
- maximum nearest-training distance

These are diagnostics only.

Do NOT modify folds because target distributions or fold sizes differ.

GEOMETRIC VALIDATION

Independently verify:

1. Exactly 110,817,231 unique canonical rows are represented.
2. Every canonical row belongs to exactly one outer fold.
3. Union of five test folds equals the full canonical support exactly once.
4. TEST / DEAD_ZONE / TRAIN are exhaustive and mutually exclusive per fold.
5. Test rows cannot train their own fold.
6. Every TRAIN row has distance to that fold's TEST set > r.
7. Every DEAD_ZONE row has distance to TEST <= r.
8. Every TEST row's nearest usable training distance is > r.
9. No noncanonical cell appears as a model row.
10. Fold assignment depends only on coordinates.
11. Canonical source files remain unchanged.

Use independent validation code where practical rather than validating
only with the same functions that generated the artifacts.

REPORTS

Create:

  reports/spatial_validation/05b/
    README.md
    manifest.json
    tables/
    figures/

Tables should include at minimum:
  fold_centroids.csv
  fold_summary.csv
  target_summary_by_fold.csv
  nearest_training_distance_summary.csv
  separation_validation.csv
  artifact_hashes.csv

Figures:
1. outer_folds.png
   - CONUS map showing all five spatial folds

2. one diagnostic map per outer fold showing:
   - TEST
   - DEAD_ZONE
   - usable TRAIN
   - background/noncanonical area

3. nearest-training-distance distribution by fold

METHODOLOGY DOCUMENTATION

Create:

  docs/methodology/checkpoint_05b_outer_spatial_cv.md

Update:

  docs/methodology/methodology_evidence_ledger.csv

Record for each decision:
- decision ID
- primary citation
- journal
- DOI
- what the paper actually implemented
- repository implementation
- adaptation
- status

Evidence mapping:
- D1: Pohjankukka et al. 2017 — dead-zone spatial k-fold semantics
- D2: Ploton et al. 2020 — response autocorrelation informs spatial
      validation scale
- D3: Schratz et al. 2019 — coordinate k-means spatial folds
- D4: Schratz et al. 2019 — five outer folds
- Valavi et al. 2019 — supporting evidence for D2

Explicitly document the scalable k-means and raster distance-transform
implementations as engineering adaptations required by dataset scale.

README/METHOD INTERPRETATION

State clearly:

This scheme estimates geographic-transferability performance under a
controlled minimum geographic separation from training observations.

Do NOT describe it as:
- random-CV performance
- ordinary interpolation performance
- design-unbiased CONUS map accuracy
- evaluation of the later all-data production model

SOURCE CODE

Extend the existing package cleanly, e.g.:

  src/wildfire_risk_uncertainty/spatial_validation/
    folds.py
    dead_zone.py
    validate_folds.py
    cli.py

Expose an appropriate CLI such as:

  uv run wildfire-spatial-folds

Do not use notebooks.

TESTS

Add tests for at least:

- exact extraction of accepted BP radius
- failure when BP range is unresolved
- deterministic coordinate clustering
- stable fold relabeling
- clustering uses coordinates only
- exactly one fold per point
- Euclidean dead-zone classification on synthetic grid
- strict > r TRAIN rule
- distance exactly == r is excluded
- raster holes do not act as barriers
- test rows cannot train their own fold
- exact nearest-training-distance calculation
- all OOF nearest-training distances > r
- fold-status exhaustiveness and mutual exclusivity
- stable output ordering
- row-level artifact schema
- report schema

Run:
  full pytest suite
  Ruff
  git diff --check

REPRODUCIBILITY

Record hashes for all large assignment partitions.

Where computationally reasonable, verify deterministic regeneration of:
- centroids
- fold assignments
- reports
- hashes

Do not repeat expensive work unnecessarily if determinism can be
demonstrated more directly.

HITL STOP CONDITIONS

Stop and ask before changing methodology if:

- the accepted 5A BP radius cannot be recovered
- scalable coordinate k-means would require changing the scientific
  clustering rule
- the five-fold geometry is pathological or unstable
- any fold has no practical usable training population after buffering
- exact > r distance semantics cannot be guaranteed
- memory/runtime limitations require approximate geographic distances
- canonical support cannot be reconstructed consistently
- any new scientific decision is required

Do NOT invent:
- alternate radii
- target stratification
- fold balancing
- regional exceptions
- new thresholds

FINAL HITL REPORT

When complete, report:

1. Exact BP-derived radius used
2. Five fitted centroids
3. Test count/fraction per fold
4. Dead-zone count/fraction per fold
5. Usable training count/fraction per fold
6. Minimum / median / p95 / maximum nearest-training distance per fold
7. BP/FLP support summaries by fold
8. All geometric validation results
9. Confirmation every canonical row is test exactly once
10. Runtime and peak RSS for clustering and distance processing
11. Reproducibility result
12. Full tests / Ruff / diff-check status
13. Files changed
14. Any methodological or numerical caveat

Then STOP for HITL review.

Do NOT:
- construct inner folds
- train any model
- tune hyperparameters
- change the accepted radius
- commit
- push
