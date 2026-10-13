# Checkpoint 3C: Cross-dataset compatibility analysis and final raw-EDA synthesis

- Base scientific commit: `d990f5c9114b543898107d8266328697a754ac68`
- Intended model: `gpt-6-sol`
- Intended reasoning level: `medium`
- Branch: `dev`
- Scope: Assess Dillon/WUI spatial and temporal compatibility, sampled WUI polygon scale, candidate spatial associations, target support, and predictor implications. Generate concise deterministic reports, figures, and synthetic tests; update the raw EDA synthesis. Recommend a design for later human approval.
- Stop condition: Commit and push Checkpoint 3C to `origin/dev` after validation. Do not build the canonical dataset, implement `create_dataset.py`, make splits, begin predictive modeling, or alter earlier reviewed Dillon/WUI findings beyond synthesis.

## Task specification

1. **Spatial compatibility.** Verify Dillon EPSG:5070, 270 m grid, dimensions, transform, bounds, and valid support; WUI EPSG:4269, MultiPolygon geometry, empirical bounds, and 9,158,401 rows. Transform only the WUI extent to EPSG:5070, compare envelopes and coverage, and state that bounding-box overlap does not imply matching valid-data support. Do not reproject the full WUI dataset.
2. **Polygon scale.** Use deterministic bounded-memory WUI geometry sampling; project sampled geometries and compute physical areas from projected coordinates, never stored `Shape_Area`. Report area, equivalent linear scale/radius, cell area 72,900 m², fractions on either side of a cell, and area/cell quantiles.
3. **Association candidates, assessment only.** Compare (A) cell-center point-in-polygon, (B) WUI attribute rasterization, and (C) polygon/raster area-overlap aggregation. Assess determinism, cost, boundaries, small and multiple polygons, duplicate `BLK20`, loss, and suitability separately for categorical (`WUIFLAG`, `WUICLASS`, `WATER20`, possibly `PUBFLAG`, `BUFVEG`), intensive (densities, vegetation/water percentages), extensive/count (housing/population counts), and identifiers (`BLK20`, `fips`, `STATE`). Do not implement any join or require a single rule for all variables.
4. **Time.** Synthesize LF2020 as circa-2020 landscape conditions for hypothetical contemporary seasons, WUI/housing snapshots 1990/2000/2010/2020, and vegetation 1992/2001/2011/2019. Assess 2020 WUI/housing with 2019 vegetation as an initial pairing, without claiming simultaneity.
5. **Targets.** Keep BP burn-probability support separate from conditional FLP distribution support. Record BP=0/zero FLPs, validated BP-positive approximately normalized FLPs, 12,627 BP-positive/common-valid cells outside the 1e-4 zero/near-one pattern, and 9,786 BP/FLP mask disagreements. Leave cleaning policy open; do not renormalize. State alternatives for later dataset construction.
6. **Predictors.** Classify WUIFLAG2020/WUICLASS2020, HU2020/HUDEN2020, POP2020/POPDEN2020, OCCHU2020/VACHU2020 and densities, VEG2019PC, WATER20/AWATER20PC, PUBFLAG, BUFVEG, BLK20, fips, STATE. Preserve duplicate IDs, extreme density tails, missing densities, null flag semantics, and BUFVEG distance conflict. Do not freeze features.
7. **Products.** Write decision-relevant deterministic tables and two figures under `reports/eda/raw/cross_dataset/`; update `reports/eda/raw/README.md` with geographic, CRS/scale, temporal, join, target, predictor and design implications while preserving prior reviewed sections.
8. **Decision.** End Checkpoint 3C with a labeled decision-ready recommendation for future `create_dataset.py`, including variable-specific association rules and remaining human decisions. Do not implement it.
9. **Tests.** Add focused synthetic tests for CRS transformation, overlap, sampling, projected area, scale summary, variable types, and output determinism. No complete WUI scans in unit tests.
10. **Validation.** Run `time uv run wildfire-eda cross-dataset`, `uv run pytest`, `uv run ruff check .`, `git diff --check`, and `git status --ignored`. Inspect figures, reconcile units/totals, and verify source files unchanged and unstaged.
11. **Commit.** Commit as `Assess Dillon and WUI spatial compatibility`; push only `origin/dev`; stop after Checkpoint 3C. Report actual model/reasoning, runtime/peak memory if measured, findings, recommendation, unresolved decisions, outputs, prompt files, commit SHA, and push confirmation.
