Continue the Wildfire Risk Uncertainty project on `dev`.

Base commit:
b4dc1bee7d9a870999323e8ca73eb63b40ad1060

Intended model: gpt-6-sol
Reasoning: medium

Implement ONLY Checkpoint 4B:

Build and validate the full pre-split CONUS canonical dataset using the production pipeline validated in Checkpoint 4A.

Do NOT:
- change scientific methodology
- create train/validation/test splits
- perform model preprocessing
- train models

The binding specification is:

docs/dataset_specification.md

The validated constructor is:

src/wildfire_risk_uncertainty/dataset/create_dataset.py

If the full build exposes a new scientific ambiguity or a failure mode not covered by the specification, stop for HITL review rather than inventing a rule.

Save THIS EXACT PROMPT verbatim as:

prompts/checkpoints/04b_build_full_canonical_dataset.md

GOAL

Produce the complete deterministic canonical dataset containing all 110,817,231 BP-valid Dillon cells and validate it independently before any data splitting or modeling.

1. REPRODUCIBILITY

Use:

REPRODUCIBILITY_SEED = 42
PYTHONHASHSEED=42

Use the frozen deterministic methodology from 4A.

Do not introduce randomness or change:
- target masks
- spatial aggregation
- geometry repair
- topology tolerance
- schema
- Parquet writer settings

Use tile size:

256

The full native grid therefore contains 3,060 deterministic processing tiles.

Progress reporting must remain enabled and show full-build tile progress, rate, elapsed time, and ETA.

2. PREFLIGHT

Before the full build:

- verify HEAD is the expected `dev` state
- verify source files/checksums match established provenance
- verify tests and lint pass
- verify canonical output paths are ignored by Git
- verify sufficient disk space for the projected full output
- verify no train/validation/test split logic is involved

Do not proceed if source provenance is inconsistent.

3. FULL BUILD

Start from a clean canonical output directory and run:

PYTHONHASHSEED=42 uv run wildfire-create-dataset \
  --full \
  --tile-size 256 \
  --seed 42 \
  --output artifacts/dataset/canonical/partitions \
  --reports artifacts/dataset/canonical/run_reports

Keep:
- canonical Parquet partitions
- detailed row-level topology diagnostics
- detailed repair diagnostics
- other large generated outputs

under ignored `artifacts/`.

Do not commit generated canonical Parquet or large row-level diagnostics.

Do not change the constructor merely to optimize runtime unless a correctness problem requires it.

4. EXACT TARGET VALIDATION

Independently validate the completed dataset.

The following full-source counts are required:

- canonical BP-valid rows: 110,817,231
- BP-zero rows: 29,735,807
- BP-positive rows: 81,081,424
- all-six-FLP-present rows: 110,809,720
- strict FLP-target-valid rows: 81,068,773
- BP-positive FLP missing/invalid: 18
- BP-positive FLPs present but invalid under the 1e-6 rule: 12,633
- BP-zero / FLP-not-applicable: 29,735,807

These must reconcile exactly with Checkpoint 3D.

Also verify:

- every emitted row is BP-valid
- `cell_id = row * 17_372 + column`
- partitions represent non-overlapping native raster tiles
- rows are ordered deterministically
- all partitions use the canonical schema
- target values remain unmodified
- no FLP renormalization occurred

5. FULL SPATIAL QA

Aggregate and report nationally:

- WUI aggregation failures
- tolerated numerical-overlap cells and polygon pairs
- accepted geometry repairs
- rejected geometry repairs
- uncovered cells
- partially covered cells
- source-null counts/fractions
- maximum categorical-share reconciliation error
- maximum pairwise overlap
- maximum duplicate coverage
- maximum coverage excess

Apply the already-approved rules:

- numerical geometry tolerance = 1e-6 of cell area
- invalid geometry repair = deterministic `make_valid` with the approved simple 1e-6 relative-area guard

Do not introduce new repair or overlap rules.

Existing specification-defined WUI failures may remain as null WUI predictors and must be quantified.

STOP for HITL only if:
- a new failure type appears
- the existing rules cannot represent a case
- source or target semantics conflict with the specification
- an implementation correctness problem is discovered

6. INDEPENDENT VALIDATION

Add or extend an independent full-dataset validator as needed without duplicating the constructor's implementation.

Validate the completed dataset in bounded memory.

As a tiling-invariance check, extract the exact six 4A pilot windows from the full canonical dataset, sort by `cell_id`, and compare them column-for-column against the committed 4A pilot results.

They must match exactly in scientific values, masks, and diagnostics. Partition file hashes are not expected to match because the full build uses different partition boundaries.

Do not rerun the entire full-CONUS build solely to demonstrate reproducibility; Checkpoint 4A already established byte-for-byte deterministic regeneration. Instead, record SHA256 hashes for every full canonical partition so a future rerun can be compared exactly.

7. COMPACT CANONICAL REPORTS

Commit compact reproducibility and QA products under:

reports/dataset/canonical/

At minimum create:

- schema.json
- manifest.json
- build_summary.csv or build_summary.json
- target_mask_summary.csv
- coverage_summary.csv
- topology_summary.csv
- geometry_repair_summary.csv
- partition_hashes.csv
- validation_summary.json

These committed reports must summarize the full dataset, not contain millions of row-level diagnostic records.

The manifest must record:

- seed = 42
- constructor commit
- source checksums/version identifiers
- Python/package/GDAL/GEOS/PROJ versions
- native grid definition
- tile size and tile count
- canonical schema version
- target-mask rule
- topology tolerance
- geometry-repair policy
- Parquet writer configuration
- total partition count
- deterministic partition hashes

Do not include nondeterministic timestamps in canonical reproducibility artifacts.

8. PERFORMANCE

Record:

- wall-clock runtime
- peak memory
- total canonical rows
- number of tiles
- number of Parquet partitions
- total dataset size
- average throughput

Runtime telemetry may be stored separately from deterministic canonical metadata if necessary.

9. FINAL VALIDATION

Run:

uv run pytest
uv run ruff check .
git diff --check
git status --ignored

Verify:

- exact target counts match Checkpoint 3D
- 4A pilot rows match their corresponding full-build rows exactly
- canonical schema is consistent across partitions
- source data are unchanged
- canonical Parquet remains ignored
- large detailed diagnostics remain ignored
- compact reports are internally consistent
- no splits or model artifacts were created

10. COMMIT AND PUSH

If the full build and validation pass:

git commit -m "Build full canonical wildfire dataset"

Commit only:
- code needed for independent full validation
- exact 4B prompt
- compact canonical reports/documentation

Do NOT commit:
- canonical Parquet
- large row-level diagnostic artifacts
- raw/interim data

Push ONLY origin/dev.

Do not modify main.

STOP after Checkpoint 4B.

Final report:

- full canonical row count
- exact target-mask counts
- tile/partition count
- total dataset size
- WUI failure count and fraction
- tolerated overlap count
- accepted/rejected geometry repairs
- coverage/null summary
- maximum topology/reconciliation discrepancies
- 4A-to-4B pilot comparison result
- partition-hash manifest result
- runtime
- peak memory
- throughput
- tests/lint
- generated artifact/report paths
- any remaining HITL issues
- commit SHA
- confirmation that only origin/dev was pushed
