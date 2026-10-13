# Canonical dataset specification (Checkpoint 3D)

## Spatial support and identity

One row represents one GDAL-valid, finite Dillon `CONUS_BP` raster cell on the native 270 m EPSG:5070 grid. The rectangular grid has height 11,283 rows and width 17,372 columns. Exclude every BP-invalid cell, including 2,275 cells with otherwise valid FLPs. Define the stable row-major `cell_id = row * 17_372 + column`, with zero-based coordinates `0 <= row < 11_283` and `0 <= column < 17_372`, ordered from the raster's upper left. Using the full grid width as the stride gives every native Dillon cell a unique stable ID; only BP-valid cells become canonical rows. Persist `row`, `column`, and pixel-center `x_5070`, `y_5070` calculated from the native affine transform; persist grid/source identifiers. These are provenance and possible spatial splitting fields, not initial model predictors. Do not crop the support to WUI coverage.

## Targets and loss masks

Read BP and FLP1–FLP6 at the identical native grid; do not resample targets. A value is valid only if its GDAL mask says valid and its stored value is finite. Keep original BP and FLP float values and missing indicators; do not renormalize the FLPs. BP validity defines canonical inclusion and BP loss eligibility, including BP=0.

Persist `bp_valid`, `flps_present` (all six GDAL-valid and finite), `flp_sum` (float64 sum where all six are present; null otherwise), `flp_in_range` (all six within inclusive [0,1]), `flp_target_valid`, `flp_missing_invalid`, `flp_present_non_normalized`, and `flp_not_applicable_bp_zero`. The FLP loss mask is true exactly when BP>0, all six FLPs are present and in [0,1], and `abs(flp_sum - 1) <= 1e-6`, using a float64 sum. `flp_present_non_normalized` means BP>0, all six are present, but the range or sum criterion fails; retain the reason separately if useful. `flp_not_applicable_bp_zero` applies to all BP=0 cells regardless of FLP presence. BP-valid rows failing FLP criteria remain in the dataset with BP loss enabled and FLP loss disabled. Absent FLP values remain null, not zero. No dominant class or renormalized vector is a canonical target.

Exact full-source reconciliation is in [target_support_accounting.csv](../reports/eda/raw/decision_resolution/target_support_accounting.csv). It partitions 196,008,276 cells into 110,817,231 BP-valid rows and 85,191,045 excluded rows. Of BP-valid rows, 29,735,807 have BP=0 and 81,081,424 have BP>0. The strict FLP loss support is 81,068,773 cells. BP loss includes all 110,817,231 rows; FLP loss masks 29,748,458 rows: 29,735,807 BP=0, 18 BP-positive with missing/invalid FLPs, and 12,633 BP-positive with present but unusable FLPs. The 2,275 BP-invalid/FLP-present cells are excluded.

## Predictor source and temporal alignment

Dillon targets use the LF2020 product. WUI, housing, and population fields use the SILVIS Version 4 2020 layer; vegetation percentage is `VEG2019PC`. Preserve this one-year offset in metadata. This is cross-sectional alignment, not a claim that all inputs preceded the hazard simulations. Source geometries are NAD83 geographic; transform polygon vertices to EPSG:5070 before area intersection with raster cells. Preserve source GDB identity, source metadata version, and per-cell coverage diagnostics.

## Polygon-to-cell aggregation

Use true intersection area of each WUI polygon with each 270 m cell, in EPSG:5070. Ordinary positive-area intersection between a source polygon and a Dillon cell is aggregated normally. Set `AREA_FRACTION_TOLERANCE = 1e-6`, equal to 0.0729 m² of a 72,900 m² cell. Positive-area overlap, coverage excess, or categorical-share reconciliation discrepancy at or below this fraction is tolerated as numerical geometry error and recorded in QA; do not renormalize. A discrepancy above this tolerance or a failed intersection is a WUI aggregation failure. Zero-area boundary contacts contribute no area. For a failure, preserve implicated source IDs and leave the affected cell's WUI predictor values and shares null. Do not resolve material overlap by precedence, last-write-wins, normalization, centroid, or largest-overlap assignment.

For invalid source geometry, apply deterministic `shapely.make_valid` in EPSG:5070. Use the repaired geometry with unchanged source attributes only if it is a non-empty Polygon or MultiPolygon, both original and repaired projected areas are finite and positive, and their relative area change is at most `1e-6`. Record its OBJECTID as repaired. Otherwise, record the repair rejection and leave affected cells' WUI-derived values null. Do not add further geometry-repair heuristics.

For retained categorical predictors `WUIFLAG2020`, `WUICLASS2020`, and `WATER20`, write one area-share column per observed source category: area of cell intersected by polygons with that category divided by the full 72,900 m² cell area. Preserve a separate source-null share and uncovered share. Shares must reconcile to 1 within numerical overlay tolerance after topology handling. Do not collapse a cell to one class. Category labels and flag code meanings come from Version 4 metadata; preserve even categories absent in some local areas.

For retained intensive predictors `HUDEN2020`, `POPDEN2020`, `OCCHUDEN2020`, `VACHUDEN2020`, `VEG2019PC`, and `AWATER20PC`, calculate the area-weighted mean of finite source values: `sum(intersection_area * value) / sum(intersection_area of finite-valued polygons)`. Store the contributing valid-area fraction of the cell and source-null area fraction for each field. When valid area is zero, value is null. The weighted value describes covered source area only; it must never silently stand for an uncovered whole cell. Store source density values as supplied in their documented units, without clipping, winsorization, log transform, or null-to-zero conversion. Keep water percent as percent, not fraction, in the canonical intensive value.

The three nullable density fields have 933,788 identical null rows, all classified as water with zero population and housing counts; 949 other water rows have present densities. Preserve these nulls. The maximum density rows are on roughly 10 m² polygons and closely match count divided by projected polygon area. Across all supplied density values above 100,000, that relationship is not universal: derived polygon area and count do not reproduce most stored densities. Preserve their raw values and area coverage rather than recalculating or clipping them.

Do not allocate `HU2020`, `POP2020`, `OCCHU2020`, or `VACHU2020` to cells. Repeated `BLK20` rows can be adjusted polygon pieces. Version 4 metadata describes housing movement away from protected land, but does not provide a generally valid cell allocation rule for all four counts. Retain counts only in an optional polygon provenance/audit table; they are absent from the initial cell predictor matrix.

Uncovered cells remain in BP support. They receive zero WUI coverage, categorical uncovered share 1, and null intensive values. Partial coverage is measured against cell area and disclosed through coverage fields. Do not infer zeros from nulls or uncovered area. A later model may require complete cases, imputation, missing indicators, or a coverage threshold; that policy is a training decision and cannot alter canonical target support.

## Initial predictor status

| Status | Source fields | Cell representation |
| --- | --- | --- |
| Include | `WUIFLAG2020`, `WUICLASS2020`, `WATER20` | Per-category area shares, null and uncovered shares |
| Include | `HUDEN2020`, `POPDEN2020`, `OCCHUDEN2020`, `VACHUDEN2020`, `VEG2019PC`, `AWATER20PC` | Area-weighted raw value and valid-area fraction |
| Exclude from initial model | `HU2020`, `POP2020`, `OCCHU2020`, `VACHU2020`, `PUBFLAG`, `BUFVEG` | Source audit/provenance only; no count allocation |
| Provenance only | `BLK20`, `fips`, `STATE`, `OBJECTID`/FID, cell coordinates | Source crosswalk and spatial/admin checks |

This list is a source-preserving initial matrix, not predictive feature selection. `WUIFLAG` is coarser than `WUICLASS`; retain both source descriptions but review collinearity during model design. Counts overlap densities and need unsupported allocation. Occupied and vacant quantities relate to housing totals, and `WATER20` overlaps `AWATER20PC` in meaning; retain their distinct categorical/continuous source information, then decide regularization or removal during training. `BUFVEG` was used to construct interface WUI and overlaps WUI and vegetation. `PUBFLAG` is a public-land adjustment indicator, not an established wildfire predictor. Null meanings of `PUBFLAG` and `BUFVEG` are not established; do not treat null as zero.

## Deferred decisions

Choose train/validation/test splits, model preprocessing, transforms or clipping for heavy density tails, imputation, missing-indicator strategy, coverage thresholds, collinearity handling, and model features during later checkpoints. Do not use coordinates as model inputs by default. The Version 4 `BUFVEG` definition says 2.14 km, while its lineage and WUI class definitions say 2.414 km; a typo in the single field definition is plausible, but source implementation remains unresolved. Density null causes and some repeated-block count transformations also remain incompletely documented. These uncertainties do not authorize filling or changing canonical source values.
