HITL decision: revise and rerun Checkpoint 5A using an automap-aligned
variogram protocol. Do not construct spatial folds or choose a final
spatial scale.

The first run is not accepted as evidence that the target correlation
ranges are truly unresolved. Its 10 equal-width lag bins spanned the
full ~4,600 km sampled pair-distance extent, causing the first empirical
bin to cover roughly 0–460 km and therefore preventing interpretation
of shorter-range spatial dependence.

Preserve all three existing deterministic 5,000-cell samples exactly.
Do not resample them and do not alter any target values or support masks.

METHODOLOGICAL CORRECTION

The revised calculation should more faithfully reproduce the
Valavi et al. (2019) -> automap / Hiemstra et al. variogram workflow.

Literature lineage:

- Valavi et al. (2019), Methods in Ecology and Evolution,
  DOI 10.1111/2041-210X.13107:
  5,000 sampled raster cells, isotropic variograms, fitted using automap.

- Hiemstra et al. (2009), Computers & Geosciences,
  DOI 10.1016/j.cageo.2008.10.011:
  automatic empirical/theoretical variogram fitting, including
  distance-sensitive fitting and nugget estimation.

- Mälicke (2022), Geoscientific Model Development,
  DOI 10.5194/gmd-15-2505-2022:
  Python/SciKit-GStat implementation and effective-range semantics.

1. LAG BOUNDARIES

Replace the current 10 equal-width full-extent lag bins.

Use the automap-style nonuniform boundary rule:

    boundaries =
        [2, 4, 6, 9, 12, 15, 25, 35, 50, 65, 80, 100]
        * (0.35 * D) / 100

where D is the planar bounding-box diagonal of the exact sampled
locations for that sampling population, matching the logic of
autofitVariogram for projected spatial data.

Because the existing samples are fixed, these boundaries must also be
deterministic.

Report the resulting boundary distances in metres and kilometres for
each sampling population.

For reference only, the full canonical raster extent would imply
approximately:
39, 78, 118, 176, 235, 294, 489, 685, 979, 1272, 1566, 1958 km.
Do not hard-code these reference numbers; derive boundaries from the
fixed sample coordinates.

Preserve pair counts for every empirical lag bin.

If a bin contains too few pairs to support fitting, follow the
automap principle of merging short-distance bins until the minimum
pair-count requirement is satisfied. Document the exact rule used.
Do not invent observations or interpolate empirical semivariances.

2. VARIOGRAM MODEL FAMILIES

Correct the earlier model-family mistake.

Remove the SciKit-GStat `stable` model from this checkpoint.

Fit:
- spherical
- exponential
- gaussian
- matern

The previous prompt incorrectly treated automap's `Ste` model as the
SciKit-GStat stable model. `Ste` is a Matérn/Stein parameterization,
not a stable variogram.

Document that SciKit-GStat's Matérn implementation is used as the
Python Matérn-family analogue; do not claim it is numerically identical
to gstat's Stein parameterization.

3. NUGGET

Enable nugget fitting.

Do not silently fix the nugget to zero.

Report:
- nugget
- partial sill
- total sill
- nugget-to-total-sill ratio
- effective range

Use SciKit-GStat's documented effective-range semantics and make clear
that the reported range is the effective range, not an internal model
range parameter.

4. FITTING WEIGHTS

Align the theoretical-model fitting more closely with the published
automap/Hiemstra procedure.

Use lag-class fitting weights proportional to:

    N_k / h_k^2

where:
- N_k is the number of sampled point pairs in lag bin k
- h_k is the representative/mean separation distance for that bin

Implement the equivalent SciKit-GStat fitting uncertainty/sigma
representation carefully and test it on a small synthetic example.

Record the exact fitting objective in the methodology document.

For transparency, report both:
- the weighted fitting objective used to select the fitted model
- ordinary unweighted fit diagnostics such as RMSE and R²

Select the primary model according to the same weighted residual-fit
criterion used by the fitting procedure, rather than mixing a weighted
fit with an unrelated unweighted model-selection criterion.

5. RANGE RESOLUTION

Keep the existing conservative RANGE_UNRESOLVED policy.

A fitted range is accepted only if:
- all fitted parameters are finite and scientifically valid;
- the theoretical fit is not pinned to the permitted maximum range;
- the effective range occurs within the examined lag domain with
  sufficient empirical support;
- the fit represents an actual sill rather than a monotonic continental
  trend forced against the fitting boundary.

If those conditions fail, report RANGE_UNRESOLVED.

Do not extrapolate beyond the analyzed lag domain.

6. STATIONARITY / CONTINENTAL TREND

Do not add detrending, regional variograms, anisotropy, or any other new
methodological treatment in this rerun.

Valavi et al. explicitly note the stationarity assumption for their
isotropic variogram workflow. If the revised short-lag analysis still
fails because the target semivariance does not approach a sill, report
that as evidence of possible nonstationarity / multi-scale continental
structure and stop for another HITL decision.

Do not autonomously "fix" that problem.

7. OUTPUT / PROVENANCE

Update the existing Checkpoint 5A reports and methodology documentation
to distinguish:

- Run 1: diagnostic equal-width full-extent attempt
- Run 2: literature-aligned automap-style lag analysis

Do not delete the first-run diagnostics; preserve them as provenance,
but clearly mark them as superseded for range inference.

Update tests for:
- deterministic automap-style boundary construction
- correct model-family list
- nugget-enabled fitting
- N_k / h_k^2 weighting behavior
- RANGE_UNRESOLVED behavior
- stable output ordering

Run the full test suite, Ruff, and git diff --check.

Then report for each target:
- empirical lag boundaries
- pair counts
- selected variogram family
- nugget
- partial sill
- total sill
- nugget/sill ratio
- effective range or RANGE_UNRESOLVED
- weighted fit criterion
- unweighted RMSE and R²
- whether the fit approached a supported sill

STOP for HITL review again.

Do not select a spatial split.
Do not choose a block or buffer size.
Do not commit or push.
