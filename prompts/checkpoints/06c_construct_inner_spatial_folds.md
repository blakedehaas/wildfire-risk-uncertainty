Checkpoint 6C — Construct and validate nested inner spatial folds

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

Required base commit:
  173b85fc2a3be4d8b87f224ab14dbf57a65e569f

PREFLIGHT

Before changing anything:

1. Verify branch == dev.
2. Verify HEAD == required base commit.
3. Verify origin/dev == required base commit.
4. Verify working tree is clean.
5. Verify accepted/frozen Checkpoints 5B, 6A, and 6B are present.
6. Verify the local ignored 5B outer-fold artifact is present and valid.
7. Verify the canonical dataset is present.
8. Do not modify main.
9. Save this exact prompt verbatim as:

   prompts/checkpoints/06c_construct_inner_spatial_folds.md

PURPOSE

Construct and independently validate the nested spatial folds that will
later be used for hyperparameter selection inside each frozen outer TRAIN
population.

Checkpoint 6A already froze the scientific rule:

- five inner spatial folds
- coordinate-only grouping
- constructed separately inside each usable outer TRAIN population
- no inner 211.151 km dead zone
- no target balancing
- no predictor balancing
- outer TEST and DEAD_ZONE rows are completely excluded from inner model
  selection

Checkpoint 6C implements that frozen design.

This checkpoint is entirely model-free.

DO NOT:

- train a neural network
- fit preprocessing
- run Optuna
- run ASHA
- run W&B
- select hyperparameters
- inspect outer TEST performance
- generate predictions
- modify outer folds
- modify the outer exclusion radius
- modify the 33 frozen model inputs
- commit or push before HITL review

AUTHORITATIVE OUTER ELIGIBILITY

Use the accepted Checkpoint 5B artifact.

For outer study j:

  eligible population =
    exactly the rows for which train_eligible_fold_j == True

These rows form that study's complete OUTER TRAIN population.

Rows belonging to that study's:

  OUTER TEST
  OUTER DEAD_ZONE

must never receive an inner-fold assignment for that study.

Verify the five eligible counts against the committed 5B reports before
constructing anything.

Expected approximate OUTER TRAIN populations:

  outer 1: 84,499,397
  outer 2: 81,842,930
  outer 3: 79,519,698
  outer 4: 77,763,326
  outer 5: 77,778,108

Read exact values programmatically rather than hard-coding them.

INNER FOLD SCIENTIFIC RULE

For each outer study j independently:

1. Restrict the population to usable OUTER TRAIN_j.
2. Construct exactly five spatial groups using Euclidean k-means on:

     x_5070
     y_5070

   only.

3. Assign every eligible OUTER TRAIN row to exactly one inner fold.
4. For inner fold k:

     INNER_VALIDATION_jk =
       eligible rows assigned to inner fold k

     INNER_TRAIN_jk =
       all other eligible OUTER TRAIN_j rows

5. Do NOT introduce an inner geographic exclusion/dead zone.

Thus:

  OUTER TEST / DEAD_ZONE
      never participate in inner selection

  OUTER TRAIN
      partitioned exactly once among five INNER VALIDATION groups

For a particular inner fit:

  one group = validation
  other four groups = training

FOLD ASSIGNMENT MUST USE ONLY COORDINATES

Forbidden inputs to inner clustering include:

- BP
- FLP1...FLP6
- FLEP8
- WUI variables
- any of the 33 model inputs
- missingness diagnostics
- target masks
- row-density balancing objectives
- state boundaries
- outer-fold target distributions

Only x_5070 and y_5070 may determine memberships.

SCALABLE K-MEANS IMPLEMENTATION

Reuse the already validated Checkpoint 5B scalable coordinate-k-means
engineering pattern rather than inventing a new clustering procedure.

For EACH outer study:

- uniformly sample exactly 1,097,262 eligible OUTER TRAIN rows without
  replacement
- seed = 42
- every eligible row must have equal sampling probability
- use bounded-memory deterministic sampling
- persist sample provenance / sample hash
- fit MiniBatchKMeans with:
    k = 5
    Euclidean coordinate inputs only
    random_state / seed = 42
- assign EVERY eligible OUTER TRAIN row to its nearest fitted centroid
- deterministically relabel resulting clusters by:
    ascending centroid x_5070,
    then ascending centroid y_5070

Using the same 1,097,262 fitting-sample size as 5B is an engineering
reuse of the validated scalable approximation.

It is NOT a new scientific sampling rule.

If the existing 5B implementation can be safely reused/refactored,
prefer reuse over duplicating clustering logic.

Do not use the rejected cell_id modulo sampling method.

Do not alter fold memberships to obtain equal row counts.

If exact reuse of the validated 5B sampling/clustering machinery is not
possible without changing its scientific behavior, STOP for HITL.

INDEPENDENT OUTER STUDIES

There are FIVE separate inner-fold designs:

  outer study 1
  outer study 2
  outer study 3
  outer study 4
  outer study 5

Do not construct one national inner-fold map and reuse it across all
outer studies.

A canonical cell may:

- be eligible in several different outer studies
- receive a different inner-fold label in each eligible outer study
- be ineligible for another outer study because it is OUTER TEST or
  DEAD_ZONE there

This is expected.

ROW-LEVEL ARTIFACT

Write the large output under:

  artifacts/spatial_validation/06c/

Keep it ignored by Git.

Prefer one artifact aligned with the existing 3,060 canonical partitions
and containing exactly one row per canonical cell.

Minimum schema:

  cell_id
  inner_fold_outer_1
  inner_fold_outer_2
  inner_fold_outer_3
  inner_fold_outer_4
  inner_fold_outer_5

Each inner-fold field should contain:

  1..5
    iff that row is usable OUTER TRAIN for the corresponding outer study

  null
    iff that row is not usable OUTER TRAIN for that outer study

Do not duplicate:

- coordinates
- predictors
- targets
- outer TEST distances
- 5B training eligibility columns

Those remain in their authoritative source artifacts.

The 06C artifact must contain:

  exactly 110,817,231 unique canonical cell_ids

and exactly 3,060 aligned partitions unless an existing project
constraint makes that impossible.

MANIFEST

Create an ignored artifact manifest recording at minimum:

- canonical provenance
- 5B artifact provenance
- 6B/base git SHA
- seed
- exact sampling algorithm
- fitting sample size
- per-outer-study sample hash
- per-outer-study centroids
- deterministic fold-label convention
- row count
- partition count
- schema
- partition hashes
- relevant software versions

VALIDATION — ELIGIBILITY

For each outer study j independently verify:

1. Number of non-null inner labels exactly equals accepted
   OUTER TRAIN_j count.

2. Every 5B train_eligible_fold_j == True row has exactly one inner label.

3. Every train_eligible_fold_j == False row has a null inner label.

4. No OUTER TEST row participates.

5. No OUTER DEAD_ZONE row participates.

6. Inner-fold labels are only integers 1..5.

7. The five INNER VALIDATION sets are:
   - exhaustive over OUTER TRAIN
   - mutually exclusive

8. Union of five validation groups exactly equals OUTER TRAIN_j.

INNER TRAIN / VALIDATION VALIDATION

For every outer study j and inner fold k:

  INNER_VALIDATION_jk =
    rows where inner_fold_outer_j == k

  INNER_TRAIN_jk =
    eligible OUTER TRAIN_j rows where inner_fold_outer_j != k

Verify:

- validation and training are disjoint
- validation union training == complete OUTER TRAIN_j
- no outer TEST / DEAD_ZONE rows appear
- both sets are nonempty
- both contain BP-positive support
- both contain strict FLP-valid support

Do not modify folds because their sizes or target distributions differ.

GEOGRAPHIC VALIDATION

Persist the 5 centroids for each of the 5 outer studies:

  25 total centroids

For each inner fold report:

- centroid
- row count
- fraction of that outer TRAIN population
- coordinate bounding box

Create a map for each outer study showing:

- its five inner folds
- its excluded outer TEST/DEAD_ZONE region as background/masked area

The purpose is to visually detect implementation or sampling pathology.

Do NOT optimize fold appearance.

If the corrected coordinate clustering produces clearly pathological
geometry suggesting an implementation defect, STOP for HITL rather than
trying alternative seeds or balancing methods.

TARGET AUDIT — DIAGNOSTIC ONLY

For every outer-study / inner-fold validation population report:

- row count
- BP-zero count
- BP-positive count
- strict FLP-valid count
- BP mean
- optional compact BP quantiles
- conditional FLEP8 support summary where valid
- unconditional FLEP8 support summary where defined

Also report corresponding INNER TRAIN support counts.

These are diagnostics only.

Do not:

- stratify folds
- optimize target similarity
- rebalance memberships
- reject a geographically valid fold because its wildfire distribution
  differs from another region

Differences are expected from spatial partitioning.

REPORTS

Create:

  reports/spatial_validation/06c/
    README.md
    manifest.json
    reproducibility.json
    tables/
    figures/

At minimum:

  tables/outer_study_summary.csv
  tables/inner_fold_centroids.csv
  tables/inner_fold_summary.csv
  tables/target_support_by_inner_fold.csv
  tables/eligibility_validation.csv
  tables/artifact_hashes.csv

Figures:

  figures/outer_1_inner_folds.png
  figures/outer_2_inner_folds.png
  figures/outer_3_inner_folds.png
  figures/outer_4_inner_folds.png
  figures/outer_5_inner_folds.png

METHODOLOGY DOCUMENTATION

Create:

  docs/methodology/checkpoint_06c_inner_spatial_cv.md

Update:

  docs/methodology/methodology_evidence_ledger.csv

The methodology should explicitly distinguish:

SCIENTIFIC METHOD:
- five spatial inner folds
- coordinate-only grouping
- nested independently inside each outer TRAIN population
- no inner dead zone

Primary methodological precedent:
  Schratz et al. (2019)
  Ecological Modelling
  DOI: 10.1016/j.ecolmodel.2019.06.002

ENGINEERING ADAPTATION:
- deterministic uniform coordinate fitting sample
- MiniBatchKMeans for scalability
- 1,097,262 fitting rows
- nearest-centroid assignment of all eligible rows
- lexicographic centroid relabeling
- reuse of the already validated 5B implementation pattern

Do not claim Schratz et al. prescribed these engineering details.

INTERPRETATION

State clearly:

The inner folds are used only for model/hyperparameter selection within
an outer training population.

They do NOT estimate the paper's final geographic-transferability
performance.

That estimand remains the role of the frozen 5B outer TEST sets with the
211.151 km exclusion radius.

The hierarchy is:

  full canonical population

      outer study j
      ├── OUTER TEST
      ├── OUTER DEAD_ZONE
      └── OUTER TRAIN
             ├── inner fold 1
             ├── inner fold 2
             ├── inner fold 3
             ├── inner fold 4
             └── inner fold 5

For inner validation k:

  INNER VALIDATION = fold k
  INNER TRAIN = other four folds

No outer TEST or DEAD_ZONE information participates.

SOURCE CODE

Extend/reuse the spatial-validation package cleanly, for example:

  src/wildfire_risk_uncertainty/spatial_validation/
    inner_folds.py
    validate_inner_folds.py
    reproduce_inner_folds.py
    inner_cli.py

Prefer reuse/refactoring of existing validated 5B coordinate-sampling and
clustering utilities rather than copying them.

Expose an appropriate CLI such as:

  uv run wildfire-inner-folds

Do not use notebooks.

TESTS

Add focused tests for at least:

- uniform-without-replacement eligible-row sampling
- deterministic sample reproduction
- clustering uses x/y only
- stable lexicographic centroid relabeling
- nearest-centroid assignment
- exactly five inner folds
- ineligible outer rows remain null
- every eligible outer TRAIN row receives exactly one label
- no outer TEST / DEAD_ZONE leakage
- validation/train disjointness
- validation/train exhaustiveness
- artifact schema
- stable output ordering
- deterministic hashes on a synthetic fixture
- target audit does not affect assignment

Run:

  full pytest suite
  Ruff
  git diff --check

REPRODUCIBILITY

For all five outer studies, independently reproduce:

- uniform fitting sample IDs/hash
- fitted centroids
- deterministic centroid labels
- every eligible row's inner-fold assignment

Where computationally reasonable, regenerate the complete 3,060
partition artifact and verify partition hashes.

Given the measured 5B/6B throughput and the simplicity of this operation,
full assignment reproducibility is expected unless a concrete resource
constraint appears.

Do not claim determinism solely from repeating centroid fitting on the
same saved sample.

HITL STOP CONDITIONS

STOP before changing methodology if:

- 5B train eligibility cannot be reconstructed exactly
- accepted OUTER TRAIN counts do not match
- canonical/5B artifacts disagree on cell_id
- coordinate sampling would not be uniform without replacement
- scalable clustering requires changing the coordinate-k-means objective
- an inner fold is empty
- an inner TRAIN or validation population lacks usable BP or FLP support
- fold geometry indicates a likely implementation defect
- exact deterministic regeneration cannot be obtained
- memory/runtime constraints require approximate final assignment
- a new scientific rule is needed

Do NOT respond to a stop condition by:

- changing seed
- balancing folds
- target stratification
- adding an inner dead zone
- changing k
- trying alternative clustering until a map looks better

FINAL HITL REPORT

When complete, report:

1. Base commit SHA
2. Exact outer TRAIN count for each of five studies
3. Sample size and hash for each outer study
4. All 25 final inner centroids
5. Inner validation count/fraction for all 25 folds
6. Corresponding inner TRAIN counts
7. BP-zero / BP-positive / FLP-valid support for every inner validation
   and training population
8. Confirmation all OUTER TRAIN rows are assigned exactly once per study
9. Confirmation all OUTER TEST / DEAD_ZONE rows remain unassigned
10. Eligibility validation results
11. Reproducibility results
12. Artifact row count and partition count
13. Runtime and peak RSS
14. Full pytest status
15. Ruff status
16. git diff --check status
17. Files changed
18. Any geographic or methodological caveat

Then STOP for HITL review.

Do not train a model.
Do not fit preprocessing.
Do not run Optuna.
Do not run ASHA.
Do not run W&B.
Do not inspect outer TEST performance.
Do not commit.
Do not push.
