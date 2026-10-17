HITL review: the scientific Checkpoint 5B design is approved, but make
one engineering correction before the checkpoint can be accepted.

DO NOT change:
- the five-fold scientific design
- the BP-derived radius
- the dead-zone semantics
- seed 42
- coordinate-only k-means
- MiniBatchKMeans as the scalable fitting implementation
- deterministic centroid relabeling
- any validation rule

ISSUE TO CORRECT

The current 1,097,262-row MiniBatchKMeans fitting sample was selected
using cell_id % 101.

The native raster width is 17,372, and:

    17,372 = 101 * 172

Therefore this selection aliases with the raster geometry and samples
the same column residue pattern on every row. It produces periodic
north-south spatial stripes rather than a spatially neutral sample of
canonical coordinates.

This is an avoidable engineering artifact and should not determine the
frozen outer-fold geometry.

CORRECTION

Replace the cell_id % 101 fitting sample with a deterministic uniform
sample without replacement from the full canonical row population.

Requirements:

- preserve the existing fitting sample size: 1,097,262 canonical cells
- seed = 42
- every canonical row must have equal sampling probability
- no target or predictor may influence sampling
- sampling must be bounded-memory / streaming
- do not load all 110,817,231 rows merely to sample them
- persist the sampled cell_ids or a deterministic sample manifest
- explicitly document the sampling algorithm
- explicitly verify that the resulting sample does not exhibit the
  previous raster-column aliasing
- do not use Python's built-in hash for deterministic selection

Use an exact or algorithmically uniform streaming sampling procedure
such as reservoir sampling or another mathematically equivalent
uniform-without-replacement method.

This is an engineering correction only. Do not introduce a new
scientific sampling method or change the coordinate k-means objective.

RERUN

Using the corrected fitting sample:

1. refit the five coordinate centroids;
2. deterministically relabel folds using the already approved convention;
3. assign all 110,817,231 canonical rows to the nearest centroid;
4. regenerate all five dead zones using the unchanged exact
   211150.99027309995 m radius;
5. regenerate OOF nearest-training distances;
6. regenerate the row-level artifact, hashes, reports, and figures;
7. rerun the independent validator;
8. rerun the deterministic reproducibility check.

COMPARISON TO THE CURRENT PRE-HITL RESULT

Before discarding the current temporary fold result, report the effect
of the correction:

- old vs new centroids
- centroid displacement in km after optimal/stable fold-label matching
- number and fraction of canonical rows whose fold assignment changed
- old vs new TEST row counts per fold
- old vs new DEAD_ZONE and TRAIN counts per fold
- old vs new nearest-training-distance summaries

Do not impose an arbitrary acceptance threshold.

If the corrected sampling changes the broad geographic fold geometry
materially, STOP for HITL review.

If it produces the same qualitative five-region geometry, proceed with
the complete validation and report the quantitative differences.

AUDIT QUANTILES

The current report notes that BP/FLEP8 audit quantiles are estimated
from a deterministic subset rather than full support.

That is acceptable for diagnostic audit purposes, but label those
quantiles explicitly as sample-estimated diagnostics in all tables and
documentation. Counts and means that are calculated on full support
must remain clearly distinguished from sample-estimated quantiles.

Do not add a new quantile methodology merely to make them exact.

VALIDATION

Retain all existing validation requirements.

In particular verify again:

- exactly 110,817,231 unique canonical cells
- every cell belongs to exactly one TEST fold
- TEST/DEAD_ZONE/TRAIN are exhaustive and mutually exclusive
- TRAIN distance to TEST is strictly > r
- DEAD_ZONE distance is <= r
- every OOF nearest-training distance is strictly > r
- canonical source hashes are unchanged
- all artifact hashes validate

Run:
- full pytest suite
- Ruff
- git diff --check

Update the methodology document to describe the corrected uniform
sampling implementation and remove the cell_id % 101 approach from the
accepted method. Preserve the first attempt only as implementation
provenance if useful; do not present it as the final methodology.

Then STOP again for HITL review.

Do not commit or push.
