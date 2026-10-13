Continue the Wildfire Risk Uncertainty project on `dev`.

Base commit:
6b23c7dd72adade17f1ee88e88352fb711e6d7d2

Implement ONLY Checkpoint 3E:

Correct two specification defects found during the human audit of Checkpoint 3D.

Do not implement `create_dataset.py`, perform new EDA, change scientific decisions, or begin Checkpoint 4A.

Save THIS EXACT PROMPT verbatim as:

prompts/checkpoints/03e_correct_dataset_specification.md

1. FIX RASTER DIMENSIONS AND CELL ID

In `docs/dataset_specification.md`, the Dillon raster dimensions are currently reversed.

The correct native raster dimensions are:

- height = 11,283 rows
- width = 17,372 columns

Rows and columns are zero-based.

The stable row-major cell identifier must therefore be:

cell_id = row * 17_372 + column

with:

0 <= row < 11_283
0 <= column < 17_372

Correct the specification anywhere it states or implies otherwise.

Make clear that this formula yields a unique stable ID over the full native Dillon grid.

Do not change the previously approved canonical support rule: only BP-valid cells become canonical rows.

2. FIX POLYGON-OVERLAP WORDING

In `docs/dataset_specification.md`, the current wording incorrectly says that any positive-area polygon overlap with a Dillon cell should cause an aggregation failure.

That is wrong because positive-area polygon/cell intersection is the normal operation required for spatial aggregation.

Replace this with an unambiguous rule:

- Ordinary positive-area intersection between a source WUI polygon and a Dillon cell is expected and should be aggregated normally.
- A failure condition occurs when two or more distinct source polygons overlap each other by positive area within the same Dillon cell in a way that would double-count source coverage, or when a source geometry/intersection cannot be computed reliably.
- Zero-area boundary contacts do not contribute area and are not failures.
- Do not silently resolve genuine source-polygon overlap by precedence, last-write-wins, normalization, or another invented rule.
- Preserve implicated source IDs for diagnosis.
- If such topology problems prove material during Checkpoint 4A pilot construction, stop and return them for human review rather than inventing a policy.

Preserve the approved use of true polygon-cell intersection area for categorical area shares and intensive area-weighted aggregation.

VALIDATION

Review the resulting `docs/dataset_specification.md` and verify:

- raster dimensions are correct
- cell_id uses width 17,372 as the row-major stride
- no cell_id collision is possible over the native grid
- ordinary polygon/cell intersection is clearly distinguished from source-polygon overlap
- no other scientific decisions changed

Run:

uv run pytest
uv run ruff check .
git diff --check
git status --short

If validation passes, commit:

Correct canonical dataset specification

Push ONLY `origin/dev`.

Do not modify `main`.

STOP after Checkpoint 3E.

Final report:
- files changed
- exact corrected cell_id definition
- exact corrected overlap/failure rule
- tests/lint status
- commit SHA
- confirmation that only `origin/dev` was pushed
