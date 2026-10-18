Checkpoint 6F pre-launch gate — Optimize the HPO input pipeline without
changing scientific behavior

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

Frozen scientific base:
  fb1cf795983a34e5e81b3a4bc662751a7a358235

CONTEXT

Checkpoint 6F has NOT begun real HPO.

The 6F preflight previously stopped before trial 0 because active-trial
pause/resume was not yet safe. User-controlled pause/resume has now also
been authorized.

Before launching real HPO, optimize the measured input bottleneck.

The current training hot path repeatedly performs, for every fold/epoch:

  canonical Parquet read
  + 5B eligibility read
  + 6C inner-fold read
  + column decode
  + eligibility filtering
  + NumPy stacking/casting
  + preprocessing
  + GPU transfer

Checkpoint 6B/6E showed the small MLP is not GPU-memory constrained and
that host/data preparation is the principal bottleneck.

This checkpoint is ENGINEERING ONLY.

It must not change:
- examples seen
- example order
- preprocessing
- targets
- loss mathematics
- model mathematics
- HPO search space
- spatial folds
- seed
- ASHA behavior
- any scientific estimand

Do NOT launch real HPO.

PREFLIGHT

1. Verify dev and repository provenance.
2. Verify no real 6F Optuna trial has executed.
3. Preserve the saved 6F execution prompt and any currently authorized
   pause/resume implementation work.
4. Verify canonical, 5B, and 6C artifacts.
5. Verify CUDA sees the RTX 5080.
6. Record:
   - system RAM available to the container
   - CPU logical/core information
   - free workspace storage
   - GPU/VRAM
7. Run the current test suite before modifying the data path.
8. Save this prompt as:

   prompts/checkpoints/06f_prelaunch_pipeline_optimization.md

If repository state cannot be reconciled safely, STOP for HITL.

PURPOSE

Build one reusable, uncompressed, memory-mapped HPO data cache that
materializes the expensive immutable Parquet decoding/join work once.

Then prove that the optimized reader is scientifically identical to the
frozen Parquet reader and benchmark its end-to-end effect.

Use the optimized path for 6F only if equivalence passes and it is
demonstrably faster.

CACHE DESIGN

Create an ignored artifact under approximately:

  artifacts/training/hpo_cache_v1/

Build it once from the authoritative canonical + 5B + 6C artifacts.

Use the exact frozen model-input order returned by the existing
model_inputs() implementation.

Materialize row-aligned arrays in canonical row order:

  cell_id.npy
    int64
    shape [N]

  features.npy
    float32
    shape [N, 33]

  bp.npy
    float32
    shape [N]

  flp.npy
    float32
    shape [N, 6]

  flp_valid.npy
    bool or uint8
    shape [N]

  inner_fold.npy
    uint8
    shape [N, 5]

For inner_fold:

  0 = not eligible for that outer study
  1..5 = frozen 6C inner-fold label

Also persist:

  partition_offsets.npy
    exact start/end offsets for the 3,060 canonical partitions

  partition_names.json
    exact sorted canonical partition names

  manifest.json

Use NumPy .npy files compatible with:

  np.load(..., mmap_mode="r")

or an equivalently simple uncompressed memory-mappable representation.

Do NOT introduce another database, distributed datastore, Zarr service,
or bespoke storage framework.

Do not compress the hot-path cache.

Do not duplicate separate feature matrices for the 25 inner fits.

One canonical row-aligned cache is preferred.

CACHE BUILD SEMANTICS

The cache must reproduce the CURRENT reader's numeric representation.

In particular:

- features are the same float32 values produced by the current Parquet
  reader before preprocessing
- BP is the same float32 target
- FLPs are the same float32 targets
- invalid/missing FLP values and their mask semantics are preserved
- wui_failure has the same float32 0/1 representation used by training
- no preprocessing transformation is baked into the cache
- no imputation is baked into the cache
- no standardization is baked into the cache

Preprocessing remains fold-specific and governed by the frozen
PreprocessingState.

Build into a temporary artifact directory and rename into place only
after successful validation so an interrupted build cannot masquerade
as a complete cache.

Record hashes for every cache file.

Check predicted cache size against free storage before building.
If insufficient space exists, STOP for HITL.

ELIGIBILITY VALIDATION

During cache construction independently verify, for every outer study:

  inner_fold[:, j] != 0

matches exactly:

  frozen 5B train_eligible_fold_j

For every canonical row.

Verify:
- N == 110,817,231
- unique cell_id count == 110,817,231
- canonical ordering agrees
- exactly 3,060 partition ranges
- partition row counts agree with canonical source
- inner labels are only 0..5
- nonzero counts equal frozen OUTER TRAIN counts
- each 6C inner membership agrees exactly

Do not infer eligibility solely from one artifact without cross-checking
5B and 6C.

OPTIMIZED READER

Implement a cache-backed reader behind the existing reusable training
interface.

Do not create a second trainer.

The cache reader must support:

  outer_study
  inner_fold
  train / validation role
  epoch
  seed

using:

  label = inner_fold[:, outer_study - 1]

Selection:

  TRAIN:
    label != 0
    AND label != inner_fold

  VALIDATION:
    label == inner_fold

PRESERVE EXACT 6D DATA ORDER

The optimized path must reproduce the accepted bounded-memory traversal.

For epoch e:

1. begin from the same sorted 3,060 partition ordering;
2. initialize the same NumPy SeedSequence([42, e]);
3. perform the same partition permutation;
4. within each visited partition, select the same eligible rows in the
   same original row order;
5. call the same RNG permutation for that selected-row count;
6. yield rows in exactly that resulting order.

Validation remains sorted partition order with no epoch shuffle.

Do not replace this with a new global random permutation.

Do not use mmap as justification for changing stochastic order.

Add a diagnostic ability to stream/hash cell_ids in traversal order.

For the accepted 6D case:

  outer study 3
  inner fold 1

prove the complete TRAIN epoch cell-id sequence is identical between:

  current Parquet reader
  cache reader

for at least epoch 1.

Also prove the complete validation cell-id sequence is identical.

PREPROCESSING EQUIVALENCE

Using outer study 3 / inner fold 1:

Fit preprocessing independently through:

  old Parquet reader
  new cache reader

Require identical:

- inputs
- row counts
- FLP-valid counts
- missing counts
- means
- standard deviations
- preprocessing digest/hash

If the frozen implementation produces any difference, investigate.

Do not loosen scientific tolerances merely to accept the cache.

TRAINING REGRESSION

Run a fixed regression using the frozen 6D configuration:

  outer study = 3
  inner fold = 1
  depth = 2
  width = 128
  learning rate = 1e-3
  weight decay = 1e-4
  batch size = 65536
  BF16
  seed = 42

Start both paths from identical initialization.

Run exactly:

  one complete training epoch
  one complete validation pass

Compare old Parquet path versus optimized cache path.

Require equality of:

- rows traversed
- FLP-valid rows
- optimizer steps
- BP training loss sum
- FLP training loss sum
- BP validation loss sum
- FLP validation loss sum
- derived means/objectives
- final model parameters
- optimizer state

First require exact equality.

If an implementation detail produces only floating-point-level
differences despite identical data/order, STOP for HITL and quantify
them rather than silently accepting a new trajectory.

The preferred result is exact reproduction of the frozen 6D trajectory.

PERFORMANCE BENCHMARK

After equivalence passes, measure old versus optimized paths on the same
fixed 6D case.

Measure separately:

1. preprocessing scan
2. raw train traversal / host preparation
3. one full training epoch
4. one full validation pass
5. rows/s
6. CPU/RSS
7. GPU utilization if reliable
8. GPU allocated/reserved VRAM

Use identical model configuration and batch size.

Measure enough repeated warm runs to distinguish real improvement from
timing noise, without turning this into a large benchmark study.

Report:
- individual timings
- median timing
- speedup ratio
- end-to-end projected 6F impact

Do not use loss/model quality for selecting the data path.

ADOPTION RULE

Adopt the cache-backed path as the frozen 6F execution path only if:

1. all scientific-equivalence checks pass; AND
2. it produces a clear end-to-end runtime improvement beyond ordinary
   run-to-run measurement noise.

If equivalence fails:
  keep the Parquet path and STOP for HITL.

If equivalence passes but runtime does not improve clearly:
  keep the existing Parquet path and report that result.

Do not add more complex optimizations merely to manufacture a speedup.

NO SECOND OPTIMIZATION PHASE YET

For this gate, do NOT additionally introduce:

- torch.compile
- custom CUDA kernels
- architecture changes
- multi-GPU/DDP
- global GPU-resident dataset
- multiprocessing redesign
- distributed storage
- transformed copies for all 25 folds

Do not add pinned-memory/prefetch complexity unless the mmap cache itself
fails to address the bottleneck and HITL explicitly authorizes another
optimization pass.

OS PAGE CACHE

Do not manually load the entire dataset into Python RAM.

Use memory mapping normally and allow the operating system to use
available RAM as filesystem/page cache.

Record host/container RAM so the behavior is interpretable.

25 PREPROCESSING STATES

After the optimized reader is accepted, construct/verify the complete:

  5 outer studies × 5 inner folds = 25

fold-specific preprocessing states using the accepted data path.

Each must remain fitted only from its INNER TRAIN population.

Persist them under ignored artifacts.

Create a compact report table containing:

- outer study
- inner fold
- TRAIN row count
- FLP-valid count
- preprocessing hash
- cache/source provenance
- fit time
- validation status

For already-existing 6E states, verify rather than assume equivalence.

All 25 must pass provenance/hash validation before real HPO launch.

PAUSE/RESUME INTEGRATION

The separately authorized active-trial pause/resume implementation must
bind checkpoints to the accepted data-cache manifest/hash.

A resumed trial must fail rather than continue if:

- cache version differs
- cache hash differs
- partition layout differs
- feature order differs
- preprocessing hash differs

Resume must preserve the exact data-order version.

Do not launch real HPO in this checkpoint.

SOURCE ORGANIZATION

Prefer extending the existing training data module cleanly, e.g.:

  training/data.py
  training/cache.py

or similarly minimal organization.

There must remain one trainer and one loss implementation.

Avoid duplicating scientific logic between Parquet and cache readers.

CLI

Provide a clear cache command, for example:

  uv run wildfire-training-cache build

and:

  uv run wildfire-training-cache validate

Exact project-conventional spelling is acceptable.

The HPO runner should later select the validated cache through explicit
configuration/provenance, not silently fall back between data sources.

REPORTS

Create:

  reports/training/06f_prelaunch/

At minimum:

  README.md
  environment.json
  cache_manifest_summary.json
  cache_validation.json
  preprocessing_states.csv
  equivalence_validation.json
  throughput_comparison.csv
  projected_hpo_runtime.json

Report:

- RAM / CPU / storage / GPU environment
- cache dimensions/dtypes/size
- cache construction time
- exact eligibility validation
- traversal-order validation
- preprocessing equivalence
- training trajectory equivalence
- old/new performance
- measured speedup
- revised 6F runtime estimate
- whether cache path was accepted

Do not portray projected runtime as guaranteed.

TESTS

Add focused tests for:

- cache schema/dtypes
- cache manifest/hash validation
- null/NaN preservation
- 0 versus 1..5 inner-label semantics
- exact 5B eligibility equivalence
- exact 6C label equivalence
- partition-offset correctness
- mmap reader filtering
- train/validation disjointness
- deterministic partition shuffle
- deterministic within-partition shuffle
- traversal order equivalence on synthetic fixture
- preprocessing equivalence
- cache provenance mismatch rejection
- resume/checkpoint cache-hash rejection

Run:

  full pytest suite
  Ruff
  git diff --check

HITL STOP CONDITIONS

STOP if:

- disk space is inadequate
- cache cannot reproduce canonical float32 values
- 5B/6C eligibility disagrees
- traversal order differs
- preprocessing hash differs
- training trajectory differs
- optimized path changes any frozen scientific behavior
- cache provenance cannot be made strict
- a new scientific decision is required

Do not respond by weakening validation.

FINAL HITL REPORT

Report:

1. repository/base state
2. hardware:
   - RAM
   - CPUs
   - workspace free storage
   - GPU/VRAM
3. cache size and build time
4. exact row/partition counts
5. 5B/6C eligibility validation
6. full traversal-order equivalence
7. preprocessing equivalence
8. one-epoch training/validation trajectory equivalence
9. old Parquet timings
10. mmap-cache timings
11. measured speedup
12. revised 6F runtime estimate
13. all 25 preprocessing-state validation results
14. cache provenance/hash scheme
15. pause/resume cache-binding validation
16. full pytest status
17. Ruff status
18. git diff --check status
19. files changed
20. whether optimized path is recommended for 6F

Then STOP.

Do not:
- launch real HPO
- run any 30-trial study
- select hyperparameters
- inspect OUTER TEST
- generate OOF predictions
- commit
- push