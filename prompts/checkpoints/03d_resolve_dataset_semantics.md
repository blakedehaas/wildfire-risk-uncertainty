Continue the Wildfire Risk Uncertainty project on `dev`.

Base commit:
7b38294b01ed03132e6cdec225373231d0ad3f07

Intended model: gpt-6-sol
Reasoning: medium

Implement ONLY Checkpoint 3D:

Resolve canonical dataset semantics before `create_dataset.py`.

Do not:
- build the canonical dataset
- implement create_dataset.py
- implement train/validation/test splits
- train models
- change previously reviewed findings except where needed to resolve the questions below

Save THIS EXACT PROMPT verbatim as:

prompts/checkpoints/03d_resolve_dataset_semantics.md

LOCKED HUMAN DECISIONS

1. Canonical unit:
   one BP-valid Dillon EPSG:5070 270 m raster cell.

   Preserve:
   - stable cell ID
   - raster row/column
   - EPSG:5070 center x/y

   Coordinates are provenance/splitting information, not automatically model features.

2. Preserve source information without inventing information.

3. For retained categorical polygon variables, preserve polygon-cell area shares rather than collapsing to one class.

4. Canonical support is strictly BP-valid cells.
   BP-invalid cells are excluded, including the previously observed 2,275 BP-invalid / FLP-valid cells.

5. BP head:
   train wherever BP is valid, including BP=0 cells.

6. FLP head:
   train only when:

   - BP > 0
   - all six FLPs are valid
   - all six FLPs are in [0,1]
   - abs(sum(FLP1..FLP6) - 1) <= 1e-6

   Do not renormalize FLPs.

   A BP-valid row with unusable FLPs remains in the canonical dataset and is masked from FLP loss.

7. Preserve enough target metadata to distinguish:
   - BP valid
   - FLPs present
   - FLP sum
   - FLP target valid
   - FLP missing/invalid
   - FLP present but non-normalized
   - BP=0 / FLP not applicable

QUESTIONS TO RESOLVE

1. TARGET SUPPORT

Compute exact, mutually reconciling counts and fractions for:

- total raster cells
- BP-valid canonical cells
- BP-invalid cells
- BP-valid BP=0
- BP-valid BP>0
- BP-valid with all FLPs valid
- BP-valid with any FLP invalid
- strict FLP-valid support under the 1e-6 rule
- BP-valid rows retained for BP but masked from FLP
- BP-positive rows masked because FLPs are missing/invalid
- BP-positive rows masked because FLPs are present but fail the 1e-6 rule
- BP-invalid / FLP-valid rows excluded from canonical support

Report percentages relative to the appropriate BP-valid and BP-positive supports.

All categories must reconcile exactly without double counting.

2. REPEATED BLK20 / COUNT SEMANTICS

Use authoritative Version 4 metadata plus empirical inspection of repeated-BLK20 groups to determine how these fields behave across derived polygons:

- HU2020
- POP2020
- OCCHU2020
- VACHU2020

Repeated BLK20 values may be legitimate derived polygon pieces, not duplicate records.

Determine whether counts are:
- repeated unchanged
- partitioned
- moved between public/private pieces
- zeroed
- otherwise transformed

Report representative patterns and their frequencies.

Determine whether these extensive counts can be safely allocated to Dillon cells.

If semantics cannot be established confidently, recommend excluding raw counts from the initial predictive feature set.

3. DENSITY TAILS

Investigate extreme values in:

- HUDEN2020
- POPDEN2020
- OCCHUDEN2020
- VACHUDEN2020

Test relationships with:
- projected polygon area
- count value
- BLK20 multiplicity
- PUBFLAG
- WATER20
- other relevant geometry/source conditions

Do not clip or transform values.

Recommend how raw density should be stored canonically and what preprocessing should be deferred to model training.

4. DENSITY MISSINGNESS

Resolve, where possible, missing values in:

- POPDEN2020
- OCCHUDEN2020
- VACHUDEN2020

Cross-tab missingness against useful variables such as:
- counts
- WATER20 / AWATER20PC
- PUBFLAG
- BLK20 multiplicity
- polygon area
- WUIFLAG/WUICLASS

Do not replace null with zero unless source semantics support it.

5. PUBFLAG

Determine:
- exact meaning
- null meaning if recoverable
- role in public-land adjustment
- relationship to repeated BLK20/count allocation
- scientific usefulness as a wildfire predictor

Default recommendation:
exclude from the initial model unless there is a clear scientific reason to include it.

6. BUFVEG

Determine:
- exact meaning
- null meaning
- role in interface-WUI construction
- explanation of 2.14 km vs 2.414 km metadata conflict
- redundancy with WUIFLAG/WUICLASS and VEG2019PC

Default recommendation:
exclude if it is primarily an intermediate WUI-construction variable or remains semantically ambiguous.

7. INITIAL FEATURE RECOMMENDATION

Recommend include / exclude / provenance-only status for:

Categorical:
- WUIFLAG2020
- WUICLASS2020
- WATER20

Intensive:
- HUDEN2020
- POPDEN2020
- OCCHUDEN2020
- VACHUDEN2020
- VEG2019PC
- AWATER20PC

Extensive:
- HU2020
- POP2020
- OCCHU2020
- VACHU2020

Provisional:
- PUBFLAG
- BUFVEG

Provenance/admin:
- BLK20
- fips
- STATE
- OBJECTID/FID

Explicitly discuss redundancy, especially:
- WUIFLAG vs WUICLASS
- counts vs densities
- occupied/vacant values vs totals
- WATER20 vs AWATER20PC
- BUFVEG vs WUI/vegetation

Do not train models or perform predictive feature selection.

DATASET SPECIFICATION

Create:

docs/dataset_specification.md

This is the scientific contract for the next checkpoint.

It must specify:

- canonical spatial unit
- spatial identity/provenance
- targets
- exact BP and FLP validity rules
- 1e-6 FLP tolerance
- target masking semantics
- temporal alignment:
  - Dillon LF2020
  - 2020 WUI/housing/population
  - VEG2019PC
- aggregation rule for every retained predictor
- categorical area-share representation
- intensive area-weighted representation
- count allocation rule only if scientifically established
- coverage/uncovered-cell policy recommendation
- missing-data policy
- initial predictor include/exclude/provenance-only list
- distinction between canonical source values and later model preprocessing
- decisions deferred to later checkpoints

It must be precise enough that `create_dataset.py` can be implemented without making new scientific decisions.

OUTPUTS

Write concise supporting results under:

reports/eda/raw/decision_resolution/

At minimum:

- target_support_accounting.csv
- blk20_count_semantics.csv
- density_tail_diagnostics.csv
- density_missingness.csv
- pubflag_bufveg_semantics.csv
- initial_feature_recommendation.csv

Create figures only if they resolve a decision.

Update the Checkpoint 3D section of:

reports/eda/raw/README.md

with findings, final recommendations, remaining uncertainty, and a link to `docs/dataset_specification.md`.

TESTS

Add focused synthetic tests for:

- target-mask classification
- exact category reconciliation
- strict 1e-6 FLP validation
- BP=0 retained / FLP masked
- BP-valid / FLP-invalid retained
- BP-invalid excluded
- repeated-BLK20 pattern classification
- missingness summaries
- deterministic outputs

Do not put full-source scans in unit tests.

VALIDATION

Run the targeted 3D command, then:

uv run pytest
uv run ruff check .
git diff --check
git status --ignored

Verify:
- all counts reconcile
- all new figures were inspected
- raw/interim/processed sources are unchanged and unstaged
- docs/dataset_specification.md agrees with the decision tables

COMMIT

If validation passes:

git commit -m "Resolve canonical dataset semantics"

Push ONLY origin/dev.
Do not modify main.

STOP.

Do not implement create_dataset.py.

Final report:
- actual model/reasoning if exposed
- runtime / peak memory if measured
- tests/lint
- exact BP support
- exact strict FLP support
- exact masked/excluded target counts
- BLK20 count semantics
- density-tail findings
- density-missingness findings
- PUBFLAG recommendation
- BUFVEG recommendation
- initial predictor recommendation
- unresolved issues
- dataset specification path
- commit SHA
- confirmation only origin/dev was pushed
