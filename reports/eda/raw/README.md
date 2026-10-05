# Raw-data EDA — Checkpoints 1, 2, and 3A

This report retains Checkpoint 1 structural discovery and Checkpoint 2
Dillon statistical/spatial EDA, and adds **Checkpoint 3A FLP-vector semantics**.
It is not the finished raw-data EDA.

Run from the repository root with the locally extracted sources listed in
[data/SOURCES.md](../../../data/SOURCES.md):

```sh
uv sync --locked
uv run wildfire-eda raw
uv run pytest
uv run ruff check .
```

The exact command executed was `uv run wildfire-eda raw`: it now regenerates
Checkpoint 1 inventory and Checkpoint 2 Dillon products. `uv run wildfire-eda
dillon` regenerates Dillon analysis alone, without requiring WUI sources.
`raw --root /path/to/repository` supports invocation from another directory.
The command regenerates canonical JSON/CSV tables and PNG figures, not this
interpretive README. It fails on missing required sources. Tests require those local sources;
no data download occurs. Dependencies are locked in `uv.lock`; library and GDAL
versions and the XML checksum are recorded in [inventory_run.json](inventory_run.json).
The structural inventory uses no sampling; Checkpoint 2 sampling is documented
below. Canonical outputs omit run timestamps and absolute paths. Statistics and
plotting modules now implement Dillon analysis; cross-dataset analysis remains deferred.

`reports/eda/raw/` holds intentionally committed scientific products.
`artifacts/eda/raw/` exists locally for disposable/large output and remains ignored.
Empty figure directories and cross-dataset tables are retained with `.gitkeep`;
Dillon figures now exist; WUI figures and cross-dataset results remain deferred.

## Checkpoint 1: empirical Dillon inventory

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

## Checkpoint 1: empirical WUI database/schema inventory

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

## Checkpoint 1: temporal evidence and documented semantics

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

## Checkpoint 1: anomalies and unresolved scientific questions

- `WUICLASS2020` is described as “2010” in its XML definition. The schema label
  and document scope point to 2020, but this conflict is retained, not corrected.
- `BUFVEG` says 2.14 km; interface domains and process step 4 say 2.414 km
  (2,414 m). Which distance was actually implemented needs clarification.
- XML geographic bounds (-127.977107, 22.768690, -65.254885, 51.649519) are broader
  than the empirical layer bounds. No assumption is made about the cause.
- Shape length/area are documented only as internal units and internal units
  squared. Do not infer that stored measurements have suitable physical units
  from the current geographic CRS; lineage and values need verification.
- At Checkpoint 1, masks and values were unexamined. Checkpoint 2 below resolves
  Dillon mask/range questions. WUI missing/sentinel values, class completeness,
  block-ID duplication, geometry validity, and numeric ranges remain unexamined.
- Dillon EPSG:5070 and WUI EPSG:4269 differ. Projection, polygon/raster support,
  area weighting, boundary treatment, and temporal compatibility require review.
  No final WUI-to-Dillon spatial join is selected.
- The Dillon directory includes `LF2020` and the source citation is 2023; these
  labels do not independently establish the modeled fire-probability period.
  Dillon probability definitions and temporal provenance need metadata review.

## Checkpoint 2: methods and provenance

The analysis reads the seven aligned rasters once at full resolution in
1,024 × 1,024 windows, using a 64 MiB GDAL block cache. No complete raster is
loaded. All source files under `data/interim/dillon_2023/`, including TIFF,
world-file, XML, auxiliary XML and overview sidecars, are protected by a recursive
size/mtime snapshot before and after execution. Additions and deletions are also
detected. GDAL persistent auxiliary metadata writes are disabled during analysis.
This guard does not claim content-hash assurance for large real sources. Synthetic
tests use byte hashes and explicitly exercise sidecar addition/modification/removal.

**Full-raster quantities:** valid/invalid counts, extrema, range violations,
zero counts, mask overlap patterns, and histogram bin counts use every native
pixel. Means and population standard deviations use float64 merged central
moments (`ddof=0`); “exact” here means full coverage rather than sampling, not
infinite-precision arithmetic. Validity means GDAL mask valid **and finite**.
NoData and any unmasked NaN/infinity are excluded; no unmasked nonfinite values
were found in these sources.

**Sampled quantities:** a fixed centered regular lattice uses every 15th row and
column, starting at zero-based row/column 7 (4,050 m spacing). Requested maximum:
1,000,000 locations; actual: **870,816**. Per-raster valid sample sizes are
**492,525 for BP** and **492,491 for each FLP**. Common-valid sample:
**492,489**. There is no RNG and no seed (recorded as null), so repeated runs select
identical native pixels. The lattice can alias spatial structure; no sampling
error bounds or independence assumptions are claimed.

Interior quantiles (1, 5, 25, 50, 75, 95, 99%) use NumPy linear interpolation
on float64-converted valid sample values. Quantile endpoints (0%, 100%) remain
**exact full-raster extrema**. Pearson correlations and hexbin relationships use
the common-valid lattice only. Choosing common-valid support for this diagnostic
does not prescribe a future mask reconciliation or cross-dataset join.

Histograms have exact counts in 100 linear bins spanning [0,1] and 120 logarithmic
bins spanning [1e-12,1]. Bins are left-closed/right-open except the final bin,
which includes its right edge. Zero counts are reported separately in log views;
negative, above-one, and positive-below-1e-12 counts are explicit in the summary.
All such overflow/underflow counts are zero here. Log panels show the occupied
positive-bin range and do not change underlying values. Log-bin heights are counts,
not probability densities. Histograms and means use each raster's own valid mask.

Numerical maps use the same native-grid lattice collected during the streaming
pass, never supplied PNG previews or precomputed overview values. Sampled invalid
cells are gray. FLP maps share [0,1]; BP uses its exact [0,max] range. No valid
values are clipped. Pixel-center coordinates determine the displayed lattice
extent. These maps are representative decimations and can miss fine features or
unsampled extremes. Coverage instead aggregates **every pixel** into 32 × 32
native-cell bins (smaller at edges); disagreement markers enlarge affected-bin
locations, with logarithmic fraction colors. Marker size does not represent area.

All settings, histogram edges, versions, and methods are in
[analysis_provenance.json](dillon/tables/analysis_provenance.json).
`inventory_run.json` continues to describe only the Checkpoint 1 inventory.

## Checkpoint 2: empirical results

Every raster has **196,008,276** total pixels (17,372 × 11,283).
Below are full-raster statistics except the explicitly sampled median/p99:

| Raster | Valid | Invalid | Mean | Population std | Sample median | Sample p99 | Exact zero fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BP | 110,817,231 | 85,191,045 | 0.00299723 | 0.00716324 | 0.00050000 | 0.03542933 | 26.833% |
| FLP1 | 110,811,995 | 85,196,281 | 0.19882098 | 0.33673480 | 0.00000000 | 1.00000000 | 51.961% |
| FLP2 | 110,811,995 | 85,196,281 | 0.25331120 | 0.27681751 | 0.16666667 | 1.00000000 | 38.651% |
| FLP3 | 110,811,995 | 85,196,281 | 0.16220482 | 0.20358870 | 0.02323009 | 0.66666669 | 47.467% |
| FLP4 | 110,811,995 | 85,196,281 | 0.06259012 | 0.10792450 | 0.00000000 | 0.41237113 | 60.809% |
| FLP5 | 110,811,995 | 85,196,281 | 0.03509779 | 0.09259334 | 0.00000000 | 0.44230768 | 73.094% |
| FLP6 | 110,811,995 | 85,196,281 | 0.01965852 | 0.09495477 | 0.00000000 | 0.58182849 | 88.977% |

BP is 56.537016% valid; each FLP is 56.534345% valid over the rectangular raster
extent. These fractions are not fractions of CONUS land area. BP spans
**0 to 0.13678333163261414**; each FLP spans **0 to 1**. All seven have
**zero values below 0, zero above 1, and zero outside-range fraction**.
The full table includes every requested quantile and all exact counts/method labels.

### Exact mask comparison

- Valid in all seven: **110,809,720**.
- Valid in at least one: **110,819,506**.
- Invalid in all seven: **85,188,770**.
- Disagreement: **9,786** cells; masks are **not identical**.
- **7,511** cells are BP-valid and invalid in all FLPs.
- **2,275** cells are BP-invalid and valid in all FLPs.
- The six FLP valid masks are exactly identical to each other.

The mask-pattern table preserves all observed seven-bit patterns in BP, FLP1–6
order. No mask reconciliation was applied to the per-raster statistics.
The discrepancy locations are sparse at this national scale; their causes are
unresolved and are not inferred from their appearance.

### Distributions and spatial patterns

Direct distribution findings: BP is strongly concentrated near zero with a long
right tail: mean 0.00299723, sampled median 0.0005, sampled p95 0.01466667,
sampled p99 0.03542933, and maximum 0.13678333. Its exact zero fraction is 26.833%.
The log-bin view exposes low positive values that the linear histogram's first
bin combines. No extreme-event threshold is defined by these descriptive quantiles.

All FLPs include zeros and values up to one. FLP1 has sampled median zero and
p95/p99 equal to one. FLP2 has the largest mean (0.25331120) and sampled median
(0.16666667); FLP3 has sampled median 0.02323009. FLP4–6 concentrate increasingly
at zero (60.809%, 73.094%, 88.977% respectively); FLP6 has sampled p75 zero yet
p99 about 0.58182849, showing a sparse substantial right tail. These statements
characterize individual distributions only, without asserting FLP-vector normalization.

Direct map observations: higher BP patches are most visible in the western CONUS
and southern Florida on the national linear scale, with much of the central and
eastern map nearer zero. FLP1 has broad high-value patches in the eastern CONUS and upper Midwest;
FLP3–5 show more visible positive structure in the west/central CONUS. FLP6
has localized high-value patches in the west, along parts of the Gulf Coast,
and in southern Florida. The FLP maps share a common scale. These are visual descriptions of a
systematic decimation, not regional statistics or explanations of fire processes.

### Pairwise relationships (sampled)

BP Pearson r with FLP1–6 is respectively **−0.145905, +0.064683, +0.242198,
+0.314643, +0.337288, +0.303520**, using the 492,489 common-valid sample cells.
Thus BP has a weak negative linear relationship with FLP1, little linear
relationship with FLP2, and positive relationships with FLP3–6 in this sample.
Hexbins show strong concentration at small BP and broad/nonlinear spreads;
correlations do not imply causality, predictive performance, or independent
observations. All seven-variable pairwise correlations are preserved in the table.
At Checkpoint 2, no sums across the six FLPs, entropy, or dominant classes were calculated. Checkpoint 3A below adds vector diagnostics.

### Canonical tables and figures

Structural products retained:
[rasters](dillon/tables/rasters.json), [alignment](dillon/tables/alignment.json).
New tables:
[distribution summary](dillon/tables/distribution_summary.csv),
[mask overlap](dillon/tables/mask_overlap.csv),
[mask patterns](dillon/tables/mask_patterns.csv),
[exact histograms](dillon/tables/exact_histograms.csv),
[sampled correlations](dillon/tables/sample_correlations.csv), and
[analysis provenance](dillon/tables/analysis_provenance.json).

| Variable | Numerical CONUS map (sampled) | Distribution (exact counts) |
| --- | --- | --- |
| BP | [Map](dillon/figures/bp_map.png) | [Histogram](dillon/figures/bp_distribution.png) |
| FLP1 | [Map](dillon/figures/flp1_map.png) | [Histogram](dillon/figures/flp1_distribution.png) |
| FLP2 | [Map](dillon/figures/flp2_map.png) | [Histogram](dillon/figures/flp2_distribution.png) |
| FLP3 | [Map](dillon/figures/flp3_map.png) | [Histogram](dillon/figures/flp3_distribution.png) |
| FLP4 | [Map](dillon/figures/flp4_map.png) | [Histogram](dillon/figures/flp4_distribution.png) |
| FLP5 | [Map](dillon/figures/flp5_map.png) | [Histogram](dillon/figures/flp5_distribution.png) |
| FLP6 | [Map](dillon/figures/flp6_map.png) | [Histogram](dillon/figures/flp6_distribution.png) |

Additional canonical figures:

- [FLP distribution comparison — exact counts](dillon/figures/flp_distributions_comparison.png)
- [Positive-probability log distributions — exact counts](dillon/figures/probability_log_distributions.png)
- [Valid coverage and disagreements — exact aggregated counts](dillon/figures/valid_coverage.png)
- [BP–FLP relationships — common-valid sampled hexbin counts](dillon/figures/bp_flp_relationships.png)
- [Pearson correlation matrix — sampled](dillon/figures/sample_correlations.png)

## Validation and remaining scientific questions

Tests use small synthetic sources for NoData/nonfinite exclusion, exact counts,
mean/std, range violations, mask disagreement, deterministic selection and
quantiles across window sizes, headless plot creation/closure, output paths,
source content preservation, and byte-identical repeated products. Real-source
integration checks stay at header/schema/file-state level; they do not repeatedly
scan full rasters. The full CLI run separately verifies real source package
sizes/mtimes remain unchanged. Figures were visually inspected for orientation,
scales, coverage, and readable labels; tables were checked for count consistency.
Runtime and peak memory are recorded in the completion message, not canonical outputs.

The BP/FLP mask mismatch needs source-specific investigation before any choice
of joint support. Source zeros remain valid numeric values; this checkpoint does
not decide whether all zeros have identical physical meaning. Spatial lattice
aliasing and decimation limit fine-scale interpretations. Spatial dependence
precludes treating the sample as independent observations for inference.
Dillon probability definitions and modeled time context are reviewed against the local source metadata in Checkpoint 3A below. Checkpoint 1's WUI metadata conflicts
remain unresolved. No WUI values were analyzed.

Checkpoint 3A below addresses FLP-vector semantics. Other project work remains outside this checkpoint.

## Checkpoint 3A: Dillon FLP probability-distribution semantics

### Authoritative local metadata

The local USDA Forest Service [metadata XML](../../../data/raw/dillon_2023/_metadata_RDS-2016-0034-3.xml)
(`idinfo/descript/abstract`, `dataqual/logic`, `dataqual/complete`, and the BP/FLP
file descriptions in `eainfo/overview/eadetc`) and [file index](../../../data/raw/dillon_2023/_fileindex_RDS-2016-0034-3.html)
are the sources for the following interpretations:

- **BP** is the number of FSim annual iterations in which a pixel burned divided
  by total annual iterations. Burnable pixels never burned in the finite simulation
  were assigned nominal BP **0.000008**.
- **FLP1–FLP6** are conditional proportions of simulated fires burning a cell in
  six flame-length classes, given that a fire occurs there. Classes are in feet:
  FLP1 <2, FLP2 2–<4, FLP3 4–<6, FLP4 6–<8, FLP5 8–<12,
  FLP6 ≥12. The source uses shorthand such as “2 < 4 ft.”; the explicit inclusion
  of boundary values is not fully specified in the metadata.
- The metadata says BP-positive cells have a positive FLP total and six FLPs sum
  to one. BP-zero cells should have zero FLP total and correspond to nonburnable
  FBFM40 fuels after 270 m resampling. Burnable cells without a simulated burn
  were assigned zonal mean FLPs from similar pixels, checked and adjusted to sum
  to one. These are documented rules, tested below rather than presumed exact.
- `LF2020` refers to circa-2020 static LANDFIRE landscape conditions, with
  contemporary weather, ignition patterns, and fire management. FSim uses tens
  of thousands of hypothetical contemporary seasons. Historical 2006–2020
  records informed weather generation and calibration. The probabilities are
  neither a 2020 observed burn map nor a dated future forecast.

### Direct empirical results

All figures below use the same bounded-memory native-raster pass as Checkpoint 2.
Sums are float64 additions of the original six float32 values, without
renormalization. Validity is all six GDAL-valid and finite FLP cells, **before**
BP restriction. Interior quantiles come from the existing deterministic lattice;
min/max and all counts/moments are full coverage.

Across **110,811,995** six-FLP-valid pixels, FLP sum ranges **0 to
1.0000001909211278**, mean **0.7316834138**, population std **0.4430831083**.
There are **29,732,679 exact-zero** sums, **81,079,316 positive** sums, and
**20,028,181 exactly-one** sums. Lattice q25/q50/q75/q95/q99 are respectively
0, 1, 1.0000000186264515, 1.0000000497093424, and
1.000000074505806. Thus **not all valid FLP vectors sum to one**.

The absolute deviation from one has full-coverage mean **0.2683166079**, maximum
**1**, lattice q25 **7.4505805969e-9**, q50 **2.9802322388e-8**, and q75/q95/q99
**1**. Exact tolerance accounting over all six-FLP-valid pixels is:

| Absolute tolerance | Within count | Fraction |
| ---: | ---: | ---: |
| 1e-6 | 81,068,773 | 0.7315884260 |
| 1e-4 | 81,068,779 | 0.7315884801 |
| 1e-3 | 81,070,454 | 0.7316035958 |
| 1e-2 | 81,079,316 | 0.7316835691 |

For descriptive state labels, “near one” means `abs(sum-1) <= 1e-4`:
**29,732,679 zero**, **10,537 positive below 0.9999**, **81,068,779 near one**,
and **0 above 1.0001**. These states partition all six-FLP-valid pixels.
The positive low sums are real supplied vectors, not rounded into the near-one
category. Small deviations around one are compatible with float32 rounding;
the low sums and BP-positive zero vectors below are a substantive conflict with
metadata claims whose cause is unresolved.

On the **110,809,720** BP-and-six-FLP-valid pixels, the exact crosstab is:

| BP | FLP zero | FLP near one (1e-4) | FLP other |
| --- | ---: | ---: | ---: |
| BP = 0 | 29,728,314 | 0 | 0 |
| BP > 0 | 2,090 | 81,068,779 | 10,537 |

Therefore BP-positive/zero-FLP is **2,090**, BP-zero/nonzero-FLP is **0**,
BP-positive/not-near-one is **12,627**, and BP-zero/near-one is **0**.
Separately, BP and FLP valid masks differ at **9,786** pixels: **7,511** BP-valid
with FLPs invalid and **2,275** BP-invalid with all six FLPs valid. The FLP masks
match each other. The crosstab does not assign a BP state to mask-mismatched cells.

### Conditional categorical interpretation

The metadata supports conditional categorical meaning, and **81,068,773**
BP-positive cells have all six FLPs valid, each in [0,1], and sum within **1e-6**
of one. This explicitly defined support is treated as an approximately normalized
six-class distribution. The **12,633** other BP-positive/common-valid cells are
excluded: 2,090 zero vectors, 10,537 materially low sums, and six with deviations
between 1e-6 and 1e-4. No values are renormalized, and zero vectors receive no
class or entropy. The tolerance leaves tiny departures from exact normalization;
normalized entropy can consequently exceed one by float32-scale rounding.

Within that support, dominant classes FLP1–FLP6 have respectively **22,045,985**,
**30,742,739**, **20,358,784**, **2,469,340**, **3,135,895**, and **2,316,030**
pixels. Ties use the first class in FLP order. Maximum supplied class probability
has mean **0.6341600** and population std **0.2059668**. Shannon entropy
`-sum(p log p)`, with `0 log 0 = 0`, has mean **0.7976603 nats**, population std
**0.4270676**, and range **0 to 1.79175949 nats**. Mean `H/log(6)` is
**0.4451827**. FLP entropy describes concentration or dispersion of the supplied
simulated wildfire intensity distribution. It is **not** epistemic uncertainty,
aleatoric ML uncertainty, predictive uncertainty, or model confidence.

### Products and unresolved questions

New tables: [sum summary](dillon/tables/flp_sum_summary.csv),
[deviation summary](dillon/tables/flp_sum_deviation_summary.csv),
[tolerances](dillon/tables/flp_sum_tolerances.csv),
[states](dillon/tables/flp_sum_states.csv),
[BP crosstab](dillon/tables/bp_flp_sum_crosstab.csv),
[categorical summary](dillon/tables/flp_distribution_summary.csv), and
[dominant-class counts](dillon/tables/dominant_flp_classes.csv).

New figures: [sum distribution](dillon/figures/flp_sum_distribution.png),
[deviation](dillon/figures/flp_sum_deviation_from_one.png),
[sum map](dillon/figures/flp_sum_map.png),
[BP zero comparison](dillon/figures/bp_zero_vs_flp_sum.png),
[dominant class](dillon/figures/dominant_flp_class.png),
[maximum probability](dillon/figures/maximum_flp_probability.png),
[entropy map](dillon/figures/flp_entropy.png), and
[entropy distribution](dillon/figures/flp_entropy_distribution.png).
Maps and BP comparison use the deterministic native-grid lattice; histogram counts
are full coverage. The categorical map gray area includes zero vectors, invalid
pixels, and excluded nonnormalized vectors.

The source does not explain the 2,090 BP-positive zero vectors or 10,537 materially
low positive sums, nor identify whether these reflect processing edges, exceptions
in source fuels, or another cause. The metadata's class-boundary shorthand and
how conditional FLPs should be interpreted for pixels assigned nominal BP and
zonal mean FLPs also merit source clarification. The categorical summaries are
valid descriptive calculations on the stated support, not a claim that all
burnable cells have normalized vectors or that these probabilities measure
uncertainty in a fitted prediction model.
