Checkpoint 5A — Literature-grounded target spatial-dependence analysis

Repository:
  /workspace/wildfire-risk-uncertainty

Base branch:
  dev

Required base commit:
  77897143ceec6ec224c623229302067d68d5934e

Before making any changes:
1. Verify that HEAD is exactly the required base commit.
2. Verify the working tree is clean.
3. Do not modify or push main.
4. Save this exact prompt verbatim at:
   prompts/checkpoints/05a_target_spatial_dependence.md

SCIENTIFIC PURPOSE

This checkpoint characterizes the spatial autocorrelation of the canonical wildfire-risk targets before any train/validation/test or cross-validation folds are constructed.

Do NOT construct spatial folds.
Do NOT choose a train/validation/test ratio.
Do NOT choose a final block size.
Do NOT perform preprocessing or model training.
Do NOT implement buffers, nested CV, SHAP, or uncertainty estimation.

The sole scientific objective is:

  Estimate and report the empirical spatial-dependence scale of the
  wildfire-risk targets so that the spatial-resampling design can be
  selected in a subsequent HITL checkpoint from literature and evidence.

The intended evaluation estimand for later checkpoints is geographic
transferability to spatially separated locations, not design-unbiased
accuracy of the fixed CONUS map.

LITERATURE BASIS

Record these methodological connections in the repository.

1. Roberts et al. (2017), Ecography
   DOI: 10.1111/ecog.02881

   Use:
   Cross-validation for spatially structured data should reproduce the
   dependence structure and intended prediction task rather than use
   ordinary random folds.

2. Valavi et al. (2019), Methods in Ecology and Evolution
   DOI: 10.1111/2041-210X.13107

   Use:
   Spatial autocorrelation can be characterized before model fitting
   using empirical variograms to inform spatial block/buffer scale.
   Their autocorrelation workflow uses random raster samples (5,000 by
   default), isotropic variograms, and explicitly discusses fitting a
   variogram to the raw response as a pre-model option.

3. Ploton et al. (2020), Nature Communications
   DOI: 10.1038/s41467-020-18321-y

   Use:
   Large-scale environmental mapping example in which response spatial
   autocorrelation was measured using an empirical semivariogram and
   spatial validation units were subsequently constructed at a scale
   slightly larger than the response autocorrelation range.

   Important: this checkpoint only measures the ranges. Do not yet
   reproduce Ploton's clustering or choose a cluster/block scale.

4. Mälicke et al. (2022), Geoscientific Model Development
   DOI: 10.5194/gmd-15-2505-2022

   Use:
   Peer-reviewed Python implementation precedent for empirical
   variograms, the Matheron semivariance estimator, theoretical
   variogram fitting, and effective-range reporting.

5. Hiemstra et al. (2009), Computers & Geosciences
   DOI: 10.1016/j.cageo.2008.10.011

   Use:
   Automatic theoretical variogram fitting. The automap workflow used
   by Valavi tests standard variogram families and selects the model
   with the smallest residual sum of squares.

Also document, but DO NOT implement yet:

6. Schratz et al. (2019), Ecological Modelling
   DOI: 10.1016/j.ecolmodel.2019.06.002

   Later methodological implication:
   hyperparameter tuning and final performance estimation should use
   separate spatial resampling loops (nested spatial CV).

7. Wadoux et al. (2021), Ecological Modelling
   DOI: 10.1016/j.ecolmodel.2021.109692

   Interpretation constraint:
   future spatial-CV results must not be described as design-unbiased
   fixed-map accuracy.

CANONICAL INPUT

Use the frozen Checkpoint 4B canonical dataset:

  artifacts/dataset/canonical/partitions/

Expected properties:
- 110,817,231 canonical BP-valid rows
- EPSG:5070
- x_5070 and y_5070 are pixel-center coordinates in meters
- BP is valid for every canonical row
- flp_target_valid identifies the strict conditional-FLP support
- no target values may be changed or renormalized

Validate these assumptions against the committed canonical reports before
running the analysis. Fail rather than silently proceeding if the
canonical contract differs.

TARGETS

Analyze the following spatial fields.

A. Full BP support
- bp

B. Strict conditional-FLP support (`flp_target_valid == True`)
- flp1
- flp2
- flp3
- flp4
- flp5
- flp6
- conditional_flep8 = flp5 + flp6

C. Unconditional high-intensity support
- unconditional_flep8

Define unconditional_flep8 exactly as:
- 0 when bp == 0
- bp * (flp5 + flp6) when flp_target_valid is true
- unavailable when bp > 0 and flp_target_valid is false

Do not fill, repair, renormalize, or otherwise modify FLP targets.

SAMPLING

The full dataset is much too large for an all-pairs empirical variogram.

Follow the 5,000-raster-location sampling scale used in the Valavi et al.
autocorrelation workflow.

Construct deterministic uniform-without-replacement samples using seed 42.

Use:
1. 5,000 cells from the complete BP-valid canonical support for BP.
2. 5,000 cells from flp_target_valid support, shared by FLP1-FLP6 and
   conditional_flep8 so those ranges are estimated from identical
   locations.
3. 5,000 cells from the defined unconditional_flep8 support.

The implementation must be streaming / bounded-memory over the existing
Parquet partitions; do not load 110.8 million rows into RAM merely to
sample them.

Persist the exact sampled cell_id, x_5070, y_5070 and relevant target
values in compact committed CSV/Parquet analysis inputs so the published
variograms are exactly reproducible.

Use seed 42 and PYTHONHASHSEED=42.
Do not rely on Python hash ordering for determinism.

VARIOGRAM METHOD

Use SciKit-GStat in Python. Add the dependency through the existing uv
workflow and update uv.lock.

For each target:

- coordinates: x_5070, y_5070
- metric: Euclidean, because EPSG:5070 coordinates are projected meters
- empirical semivariance estimator: Matheron
- isotropic spatial dependence
- raw target values; do not transform the target for this analysis

Fit standard theoretical variogram models independently:
- spherical
- exponential
- gaussian
- stable

Select the primary fitted model using the smallest residual sum of
squares to the empirical semivariogram, mirroring the automatic
model-selection principle used by automap.

Report at minimum for every target/model:
- target
- sample size
- variogram model
- nugget
- partial sill / sill according to library semantics
- effective range in meters
- effective range in kilometers
- fit RSS
- fit RMSE if available
- fit R^2 if available
- maximum lag examined
- whether the fitted range is well resolved within the examined lags
- selected/not selected

For asymptotic models, use/document SciKit-GStat's effective-range
definition (approximately 95% of the sill), rather than confusing the
effective range with the model's internal range parameter.

Do not silently accept a pathological fit.

If a target's best fit:
- is non-finite,
- fails optimization,
- clearly does not approach a sill over the examined lags,
- or places the effective range effectively at/above the analysis
  boundary,

mark that target as RANGE_UNRESOLVED and preserve all empirical results.
Do not invent or extrapolate a final spatial range.

Because variogram fitting choices can matter, preserve the empirical
lag table (distance, semivariance, pair count) in addition to fitted
parameters.

REPORTS AND FIGURES

Create a standardized committed structure such as:

reports/spatial_dependence/05a/
  README.md
  tables/
    sample_manifest.csv
    variogram_empirical.csv
    variogram_fits.csv
    variogram_selected_ranges.csv
  figures/
    bp_variogram.png
    flp1_variogram.png
    ...
    conditional_flep8_variogram.png
    unconditional_flep8_variogram.png
    effective_range_comparison.png
    sample_locations.png

Plots must be generated by Python source code, not notebooks.

Each target variogram figure should show:
- empirical semivariance by distance
- selected theoretical fit
- estimated effective range when resolved
- units in km on the displayed distance axis
- sample size

The summary report must clearly distinguish:
- empirical result
- literature precedent
- interpretation
- unresolved decision

METHODOLOGY DOCUMENTATION

Create:

  docs/methodology/README.md
  docs/methodology/checkpoint_05a_spatial_dependence.md
  docs/methodology/methodology_evidence_ledger.csv

Do not merely list citations.

For each methodological decision record:
- decision ID
- scientific decision
- citation
- journal
- DOI
- what the cited paper actually implemented
- what this repository implements
- any adaptation
- status

Use statuses such as:
- LITERATURE_SUPPORTED
- HITL_APPROVED
- IMPLEMENTED
- VALIDATED

The Checkpoint 5A document must explicitly state:

FROZEN:
- evaluation goal is later geographic-transferability assessment
- random pixel CV is not the intended validation design
- empirical target semivariograms are used to characterize spatial
  dependence before fold construction

NOT YET FROZEN:
- which target/range governs final spatial unit size
- square blocks vs spatial clusters vs buffered scheme
- number of outer folds
- number of inner folds
- block/fold allocation method
- buffer distance, if any
- final train/tuning/test proportions, if such proportions are used

Do not resolve those items autonomously.

IMPLEMENTATION

Create clean reusable source code under an appropriate package namespace,
for example:

src/wildfire_risk_uncertainty/spatial_validation/
  __init__.py
  sampling.py
  variogram.py
  cli.py

Expose a project CLI, e.g.:

  uv run wildfire-spatial-dependence

The analysis must be deterministic and repeatable from the existing
canonical partitions.

Tests must cover at least:
- deterministic sampling
- correct support masks for all three sampling populations
- exact conditional_flep8 derivation
- exact unconditional_flep8 derivation
- no FLP renormalization
- semivariance calculation on a small known example
- effective-range/result-table handling
- RANGE_UNRESOLVED behavior
- stable output ordering
- report schema

Run:
- full pytest suite
- Ruff
- git diff --check

Do not create or modify model-training code in this checkpoint.

HITL STOP CONDITIONS

Stop and ask before proceeding if:
- required canonical target columns differ from the expected schema
- SciKit-GStat cannot support the required reproducible calculation
  without materially changing the literature-backed method
- a numerical or memory limitation requires a scientifically different
  sampling method
- the empirical results reveal that the variogram method itself is
  invalid or uninterpretable for one or more primary targets
- any scientific choice not specified above is required

Ordinary engineering details are Codex's responsibility and do not
require HITL.

FINAL RESPONSE BEFORE COMMIT

When the analysis is complete, report:
- exact sample sizes/supports
- selected variogram family for each target
- estimated effective range for each target, or RANGE_UNRESOLVED
- fit-quality diagnostics
- run time and peak memory
- tests/lint status
- files changed
- any methodological caveats

Then STOP for HITL review.

Do not choose a final spatial split or block size.
Do not commit or push until I approve the empirical results.
