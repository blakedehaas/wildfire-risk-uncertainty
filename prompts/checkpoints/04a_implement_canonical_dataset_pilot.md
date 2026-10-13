Continue the Wildfire Risk Uncertainty project on `dev`.

Base commit:
d519589a43f342d29abf4c223cc9bc5e5f331bf9

Intended model: gpt-6-sol
Reasoning: high

Implement ONLY Checkpoint 4A:

Implement the production canonical dataset construction pipeline and validate it on a small deterministic set of real pilot tiles.

Do NOT:
- run the full-CONUS build
- create train/validation/test splits
- train models
- change scientific decisions in `docs/dataset_specification.md`

The binding scientific contract is:

docs/dataset_specification.md

If implementation reality conflicts with that specification, stop and report the conflict rather than inventing a new scientific rule.

Save THIS EXACT PROMPT verbatim as:

prompts/checkpoints/04a_implement_canonical_dataset_pilot.md

GOAL

Build production-quality `create_dataset.py` that deterministically converts the raw Dillon + SILVIS sources into the canonical pre-split dataset.

Checkpoint 4A must validate that constructor on representative real pilot tiles before the full ~110.8M-row CONUS build is authorized.

1. IMPLEMENT THE CANONICAL CONSTRUCTOR

Create:

src/wildfire_risk_uncertainty/dataset/create_dataset.py

and supporting package modules only where useful.

Provide a reproducible CLI.

Canonical rows:

- one BP-valid native Dillon EPSG:5070 270 m cell
- exclude BP-invalid cells
- retain BP=0 cells

Use:

cell_id = row * 17_372 + column

Preserve:

- cell_id
- row
- column
- x_5070
- y_5070
- BP
- FLP1–FLP6
- target-validity diagnostics required by `docs/dataset_specification.md`

Read BP/FLPs on the native raster grid with no resampling and no FLP renormalization.

FLP target-valid is true exactly when:

- BP > 0
- all six FLPs are present and finite
- all six FLPs are in [0,1]
- abs(float64_sum(FLP1..FLP6) - 1) <= 1e-6

BP-valid rows with unusable FLPs remain in the dataset and are masked from FLP loss.

2. POLYGON-TO-CELL AGGREGATION

Use true polygon/cell intersection area in EPSG:5070.

Categorical retained fields:

- WUIFLAG2020
- WUICLASS2020
- WATER20

Store deterministic per-category area shares plus source-null share and uncovered share.

Intensive retained fields:

- HUDEN2020
- POPDEN2020
- OCCHUDEN2020
- VACHUDEN2020
- VEG2019PC
- AWATER20PC

Compute:

sum(intersection_area * finite_source_value)
/
sum(intersection_area for finite_source_value)

Also preserve:
- valid-area fraction
- source-null area fraction

Do not:
- impute
- clip
- winsorize
- log-transform
- replace null with zero
- recompute density from counts
- allocate raw extensive counts
- use centroid, dominant-polygon, largest-overlap, or last-write-wins approximations

Do not include PUBFLAG or BUFVEG as initial predictors.

3. TOPOLOGY / GEOMETRY SAFETY

Ordinary positive-area polygon/cell intersection is expected.

Detect and report:

- positive-area overlap between distinct source polygons that would double-count source coverage
- invalid/unusable source geometry
- failed intersections
- coverage exceeding cell area beyond numerical tolerance
- categorical-share reconciliation failure

Do not silently repair, normalize, apply precedence, or invent another resolution.

Preserve implicated source IDs.

If topology or geometry failures are material in the real pilot, surface them for HITL review before 4B.

4. BOUNDED-MEMORY EXECUTION

The production constructor must be suitable for the eventual full-CONUS build.

Requirements:

- deterministic spatial tiles/windows
- bounded memory
- fixed tile ordering
- restartable/idempotent where practical
- no requirement to load the full raster or WUI dataset into RAM
- explicit CRS and units
- no notebook-only logic

Prefer correctness and auditability over premature optimization.

5. PROGRESS BAR

Dataset construction must display a production-quality terminal progress bar.

Use a lightweight implementation such as `tqdm` unless an existing dependency is equally suitable.

The primary progress bar must:

- use the known number of requested tiles/windows as its total
- advance once per completed tile/window
- show completed/total, percent, elapsed time, rate, and ETA
- use a clear description such as `Building canonical dataset`
- update in place rather than printing one line per tile
- reach 100% only after successful completion

Support:

--no-progress

for tests, CI, or redirected logs.

Progress reporting must not alter scientific outputs or determinism.

6. FULL REPRODUCIBILITY

Full reproducibility is a binding requirement.

Use project-wide seed:

42

Prefer deterministic algorithms over randomness. Do not use randomized pilot selection.

Given identical:
- source files/checksums
- Git commit
- configuration
- locked dependency environment
- seed

the pipeline must reproduce identical:
- canonical rows
- row order
- tile/partition membership
- column order
- schema
- masks
- spatial aggregations
- QA summaries
- output filenames
- Parquet files and hashes within the pinned runtime environment

Define:

REPRODUCIBILITY_SEED = 42

If randomness is used:
- seed Python `random`
- use explicit seeded NumPy generators
- seed any other RNG used
- never use unseeded random APIs

Explicitly sort:

- pilot windows
- tiles/windows
- canonical rows
- polygon candidates
- source IDs
- intersections
- categorical levels
- output columns
- Parquet partitions
- QA rows
- diagnostic source-ID lists

Do not rely on filesystem order, set order, spatial-index order, database order without `ORDER BY`, or parallel completion order.

Canonical rows should be ordered by ascending `cell_id`.

When multiple polygons contribute to a cell, reduce them in stable source-ID/FID order.

Use float64 for:

- FLP sums
- projected intersection areas
- coverage accumulation
- weighted numerators/denominators

Use fixed accumulation order.

Do not use nondeterministic parallel floating-point reductions.

Parallel execution, if used, must not alter outputs or ordering. Do not introduce concurrency merely for speed during 4A.

Use fixed Parquet:
- schema
- column order
- partition naming
- row ordering
- compression codec/settings
- row-group policy
- writer configuration

Do not embed timestamps, random UUIDs, temporary paths, hostnames, or other run-specific values in canonical files or deterministic manifests.

Document reproducible invocation with:

PYTHONHASHSEED=42

but correctness must come from explicit sorting, not hash behavior.

Expose `--seed`, defaulting to 42.

Changing the seed must not affect logically deterministic operations.

7. STORAGE AND SCHEMA

Use deterministic partitioned Parquet.

Generated pilot data belongs under an ignored path such as:

artifacts/dataset/pilot/

Do not commit generated large Parquet files.

Commit compact QA/reproducibility outputs under:

reports/dataset/pilot/

At minimum:

- schema.json
- manifest.json
- pilot_summary.csv
- target_mask_summary.csv
- coverage_summary.csv
- topology_summary.csv
- null_summary.csv

The schema must document for every column:

- name
- dtype
- units where applicable
- source
- aggregation rule
- role: target / predictor / provenance / diagnostic
- null semantics where established

Categorical-share column names must be deterministic and stable.

Do not derive a dominant WUI class.

The deterministic manifest must record:

- seed = 42
- base/Git commit context
- source checksums/version identifiers
- Python version
- locked package versions
- GDAL version
- GEOS/PROJ versions if used
- PyArrow/Parquet version if used
- CRS/grid definition
- pilot windows
- numerical tolerances
- writer configuration
- schema version
- target-mask rules
- aggregation rules

Do not place wall-clock timestamps in deterministic artifacts.

8. PILOT WINDOWS

Do NOT run the full-CONUS build.

Choose and persist a small deterministic set of real raster windows that exercises, where source data permit:

- substantial WUI coverage
- mixed WUI classes
- water/density-null behavior
- partial or low WUI coverage
- BP=0
- BP>0 with valid FLPs
- a BP-positive FLP anomaly/missing case if it can be identified without another expensive nationwide search

Do not choose pilot windows randomly.

Keep the pilot small enough for rapid iteration.

9. PILOT QA

For every pilot tile verify:

Targets:
- emitted rows are BP-valid only
- cell_id is unique
- cell_id exactly matches row/column
- BP/FLP values match source rasters
- FLP validity uses float64 and 1e-6
- BP=0 rows remain and are FLP-masked

Spatial:
- intersection area is computed in EPSG:5070
- coverage is within valid bounds
- categorical shares + source-null + uncovered reconcile to 1 within an explicit named tolerance
- intensive aggregation uses only finite contributing source area
- uncovered cells remain present
- uncovered intensive values remain null

Safety:
- count topology overlaps
- count invalid geometry/intersection failures
- count pre-failure coverage > cell area
- preserve implicated source IDs

Missingness:
- source null area remains distinguishable from uncovered area
- water-associated density nulls remain null

10. TESTS

Add focused synthetic tests for:

- cell_id uniqueness/endpoints
- BP-invalid exclusion
- BP=0 retention
- strict FLP masking
- missing FLP preservation
- categorical area shares
- mixed-category cell
- partial/uncovered cell
- categorical source-null share
- intensive weighted mean
- intensive null-area accounting
- multiple legitimate polygons contributing to one cell
- genuine positive-area source-polygon overlap
- zero-area boundary contact
- invalid geometry/intersection failure
- deterministic schema/order/output
- progress enabled
- `--no-progress`
- seeded behavior
- progress does not alter output contents

No unit test may require a nationwide source scan.

11. REPRODUCIBILITY VALIDATION

Run the complete real pilot twice from clean output directories using:

seed = 42

For both runs compare:

- schema
- partition names
- row counts
- row ordering
- QA artifacts
- Parquet file hashes
- deterministic report hashes

The two runs must match byte-for-byte within the same pinned environment.

If file hashes differ despite identical scientific values, identify and remove nondeterministic writer metadata/settings rather than accepting unexplained differences.

12. PERFORMANCE

Measure and report:

- runtime
- peak memory
- number of tiles
- rows emitted
- throughput
- pilot Parquet size

Use this to estimate whether the full-CONUS build is operationally reasonable.

Do not introduce distributed-computing infrastructure unless the pilot demonstrates a need.

13. VALIDATION

Run the real pilot with progress enabled.

Then run:

uv run pytest
uv run ruff check .
git diff --check
git status --ignored

Verify:

- source files are unchanged and unstaged
- generated Parquet is not accidentally committed
- `docs/dataset_specification.md` is scientifically unchanged
- QA tables reconcile with pilot outputs
- deterministic rerun hashes match
- progress reaches completion correctly

14. STOP / HITL CONDITIONS

STOP after pilot validation.

Do not:
- run full CONUS
- create train/validation/test splits
- train models

Explicitly surface any:

- genuine source-polygon overlap
- invalid geometry
- intersection failure
- share reconciliation failure
- unexpected null semantics
- schema ambiguity
- unacceptable runtime/memory behavior
- reproducibility failure
- conflict with `docs/dataset_specification.md`

Do not invent scientific resolutions.

15. COMMIT

If implementation and pilot validation succeed:

git commit -m "Implement canonical dataset pilot pipeline"

Push ONLY `origin/dev`.

Do not modify `main`.

STOP after Checkpoint 4A.

Final report:

- actual model/reasoning if exposed
- modules/files added
- dependencies added and why
- reproducibility seed
- exact pilot windows
- number of tiles
- pilot row count
- BP/FLP mask counts
- categorical-share reconciliation
- coverage/null results
- topology/geometry failure counts
- progress-bar implementation
- runtime
- peak memory
- throughput
- pilot data size
- deterministic rerun result
- whether all regenerated hashes matched
- pinned environment/version provenance
- generated artifact/report paths
- tests/lint status
- HITL issues
- commit SHA
- confirmation that only origin/dev was pushed
