# Raw-source discovery — Checkpoint 1

This is a structural/source-discovery report, **not the finished raw-data EDA**.
No pixel distributions, feature values, predictive models, training dataset, or
WUI-to-Dillon spatial join have been computed or selected.

Run from the repository root with the locally extracted sources listed in
[data/SOURCES.md](../../../data/SOURCES.md):

```sh
uv sync --locked
uv run wildfire-eda raw
uv run pytest
uv run ruff check .
```

The exact inventory command executed was `uv run wildfire-eda raw`.
`raw --root /path/to/repository` supports invocation from another directory.
The command regenerates the small JSON/CSV inventories, not this interpretive
README. It fails on missing required sources. Tests require those local sources;
no data download occurs. Dependencies are locked in `uv.lock`; library and GDAL
versions and the XML checksum are recorded in [inventory_run.json](inventory_run.json).
No sampling or random seed is needed: the inventory is deterministic, with no
run timestamps or absolute paths. Future EDA can extend the existing CLI and
modules; statistics, plotting, and cross-dataset analysis are explicitly deferred.

`reports/eda/raw/` holds intentionally committed scientific products.
`artifacts/eda/raw/` exists locally for disposable/large output and remains ignored.
Empty figure directories and cross-dataset tables are retained with `.gitkeep`;
there are no figures or cross-dataset results yet.

## Empirical Dillon inventory

Source directory: `data/interim/dillon_2023/Data/I_FSim_CONUS_LF2020_270m/`.
Each file was independently opened read-only with Rasterio. The full per-file
inventory, including unrounded transforms, bounds, and CRS WKT, is in
[rasters.json](dillon/tables/rasters.json).

| Filename | Width | Height | Bands | Dtype | CRS | Resolution |
| --- | ---: | ---: | ---: | --- | --- | --- |
| CONUS_BP.tif | 17372 | 11283 | 1 | float32 | EPSG:5070 | 270 × 270 m |
| CONUS_FLP1.tif | 17372 | 11283 | 1 | float32 | EPSG:5070 | 270 × 270 m |
| CONUS_FLP2.tif | 17372 | 11283 | 1 | float32 | EPSG:5070 | 270 × 270 m |
| CONUS_FLP3.tif | 17372 | 11283 | 1 | float32 | EPSG:5070 | 270 × 270 m |
| CONUS_FLP4.tif | 17372 | 11283 | 1 | float32 | EPSG:5070 | 270 × 270 m |
| CONUS_FLP5.tif | 17372 | 11283 | 1 | float32 | EPSG:5070 | 270 × 270 m |
| CONUS_FLP6.tif | 17372 | 11283 | 1 | float32 | EPSG:5070 | 270 × 270 m |

All seven have the following identical header values:

- Affine coefficients `(a,b,c,d,e,f)`:
  `(270.0, 0.0, -2362635.000000001, 0.0, -270.0, 3267674.999999998)`.
- Bounds `(left,bottom,right,top)`:
  `(-2362635.000000001, 221264.99999999814, 2327804.999999999, 3267674.999999998)`.
- NoData: `-3.4028230607370965e+38`.
- Band 1 overview factors: `2, 4, 8, 16, 32, 64`.

[alignment.json](dillon/tables/alignment.json) reports exact equality for CRS WKT,
shape, transform, bounds, NoData convention, and grid (CRS + shape + transform).
No rounding or tolerance masks differences. All checks pass. This establishes
structural alignment only; it does not establish identical valid-data masks,
probability semantics, or suitability of any aggregation. Overviews were listed,
not used to compute statistics.

## Empirical WUI database/schema inventory

Source: `data/interim/silvis_wui_2020/CONUS_WUI_block_1990_2020_change_v4_gcs_na83.gdb`.
Pyogrio lists **one logical layer**:
`CONUS_WUI_block_1990_2020_change_v4_gcs_na83`.
The OpenFileGDB driver reports MultiPolygon geometry, EPSG:4269 (NAD83 geographic),
9,158,401 features, FID column `OBJECTID`, and geometry column `Shape`.
Bounds `(xmin,ymin,xmax,ymax)` are
`(-124.84897399999994, 24.396308000000033, -66.88544399999995, 49.384479000000056)`.
Both fast feature count and fast total bounds are supported. Inspection uses
`list_layers` and `read_info` with forced counts/bounds disabled, never a feature
read. Physical FileGDB filenames are used only for the file-state mutation guard,
never as schema or semantic evidence.

[layers.json](wui/tables/layers.json) records the complete layer inventory.
[fields.csv](wui/tables/fields.csv) records all **35 attribute fields** in source
order, NumPy dtype, OGR type/subtype, schema years, matched XML label, and definition.
`OBJECTID` and `Shape` are separate FID/geometry columns, not omitted attributes.
The complete attribute list is:

```text
BLK20 WATER20 AWATER20PC POP2020 HU2020 OCCHU2020 VACHU2020
POPDEN2020 HUDEN2020 OCCHUDEN2020 VACHUDEN2020 PUBFLAG STATE
HU1990 HU2000 HU2010 HUDEN1990 HUDEN2000 HUDEN2010
VEG1992PC VEG2001PC VEG2011PC VEG2019PC
WUIFLAG1990 WUICLASS1990 WUIFLAG2000 WUICLASS2000
WUIFLAG2010 WUICLASS2010 WUIFLAG2020 WUICLASS2020
BUFVEG fips Shape_Length Shape_Area
```

All attribute names match XML labels case-insensitively; actual `fips` matches
XML `FIPS`. String fields have Pyogrio dtype `object` and OGR type `OFTString`.
`WATER20` and `BUFVEG` are int16; population/housing counts, `PUBFLAG`, and
`WUIFLAG*` are int32; density, vegetation percentage, water percentage, and
shape measurement fields are float64. These are schema types, not observed
value distributions or verified domains.

## Temporal evidence and documented semantics

The supplied XML is
`data/raw/silvis_wui_2020/CONUS_WUI_block_1990_2020_change_v4_metadata.xml`.
[metadata.json](wui/tables/metadata.json) preserves its attribute definitions,
class domains, process steps, documented bounds, and contextual year mentions
with XML paths. Definitions are source statements, not independently verified
properties of every record.

- Explicit census/housing/WUI field years: **1990, 2000, 2010, 2020**.
  The XML `idinfo/timeperd/current` states the same four snapshots.
- Explicit vegetation fields: **1992, 2001, 2011, 2019**. Thus the full set of
  schema years is **1990, 1992, 2000, 2001, 2010, 2011, 2019, 2020**.
- XML definitions associate `BLK20`, `WATER20`, and `AWATER20PC` with 2020 block/
  water information (the last through the block context); the two-digit suffix
  alone is not used to infer a year automatically.
- Additional XML context years: **1998** (metadata standard), **2008, 2009**
  (retrofit source citation), **2016** (PAD-US source), **2021** (NLCD citation),
  **2023** (publication, processing, and metadata dates). These are not additional
  observation snapshots. The complete XML year set is the eight schema years
  plus these six context years. ISO 19115 is a standard identifier, not a year.
- XML `eainfo/detailed/attr` defines `HU1990/2000/2010` as allocated housing
  units; housing densities are units/km², population density is persons/km².
  Wildland vegetation percentages comprise forests, grasslands, and wetlands.
- `WUIFLAG*` definitions specify 0 non-WUI, 1 intermix, 2 interface. The 14
  enumerated `WUICLASS2020` labels and their definitions are preserved in JSON;
  earlier class definitions refer to that domain. Actual values remain unchecked.
- `PUBFLAG` is described as a protected-area flag (1 = public land). XML process
  step 1 describes moving housing units from protected portions to private
  portions sharing a block ID, with an exception when no private portion exists.
  These polygons therefore require care before treating rows as unique blocks.
- XML process step 4 states that **2020** dense-vegetation buffers were used
  for interface assignment in **all four decades**. This is a documented
  temporal dependency, not a decision to use a particular modeling year.

## Anomalies and unresolved scientific questions

- `WUICLASS2020` is described as “2010” in its XML definition. The schema label
  and document scope point to 2020, but this conflict is retained, not corrected.
- `BUFVEG` says 2.14 km; interface domains and process step 4 say 2.414 km
  (2,414 m). Which distance was actually implemented needs clarification.
- XML geographic bounds (-127.977107, 22.768690, -65.254885, 51.649519) are broader
  than the empirical layer bounds. No assumption is made about the cause.
- Shape length/area are documented only as internal units and internal units
  squared. Do not infer that stored measurements have suitable physical units
  from the current geographic CRS; lineage and values need verification.
- NoData-mask overlap, missing/sentinel WUI values, class completeness, block-ID
  duplication, geometry validity, and numeric ranges have not been evaluated.
- Dillon EPSG:5070 and WUI EPSG:4269 differ. Projection, polygon/raster support,
  area weighting, boundary treatment, and temporal compatibility require review.
  No final WUI-to-Dillon spatial join is selected.
- The Dillon directory includes `LF2020` and the source citation is 2023; these
  labels do not independently establish the modeled fire-probability period.
  Dillon probability definitions and temporal provenance need metadata review.

## Validation and next work

Tests verify required files, actual raster alignment, deliberately mismatched
headers, byte-preserving reads of synthetic rasters, schema/XML matching,
output-path restrictions, reproducible output bytes, and unchanged source file
sizes/mtimes across execution. The real-data guard is not a content checksum;
full hashing of multi-gigabyte data is intentionally avoided. The XML checksum
matches the source manifest. Real datasets are opened read-only.

Next checkpoint, subject to review: establish documented variable semantics,
then implement bounded-memory raster statistics and masks, WUI value/domain/
missingness and geometry diagnostics, temporal summaries, and scientific plots.
Evaluate spatial/temporal integration options explicitly before choosing any
join. No full statistical EDA or dataset construction is part of this checkpoint.
