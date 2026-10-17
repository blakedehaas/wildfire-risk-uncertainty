HITL decision: proceed with the corrected uniform-sample fold geometry.

The material change relative to the pre-HITL geometry is accepted.

SCIENTIFIC INTERPRETATION

The original cell_id % 101 fitting sample is rejected as an implementation
artifact because the modulus aliases with the native raster width and
therefore does not provide a spatially neutral approximation to the
literature-backed coordinate k-means operation.

The corrected deterministic uniform-without-replacement sample is the
accepted scalable implementation.

Do not attempt to preserve similarity to the old folds.

Do not rebalance the corrected folds by row count, target distribution,
WUI distribution, or any other variable.

The corrected five-fold coordinate geometry is now the candidate final
Checkpoint 5B design.

PROCEED

Using the already computed corrected uniform sample and corrected fitted
centroids:

1. Assign all 110,817,231 canonical rows to the corrected five folds.
2. Apply the unchanged exact exclusion radius:

   211150.99027309995 m

3. Rebuild all five exact TEST / DEAD_ZONE / TRAIN geometries.
4. Recompute and persist every OOF nearest-usable-training distance.
5. Replace the old pre-HITL row-level artifact with the corrected artifact.
6. Regenerate all artifact hashes, manifests, compact reports, tables,
   and figures.
7. Run the full independent validator against the corrected artifact.
8. Run the complete reproducibility checks required by the checkpoint.
9. Preserve the old geometry only as superseded implementation provenance
   where useful; it must not appear as the accepted 5B methodology.

METHODOLOGY DOCUMENTATION

The final methodology must state clearly:

- Outer folds are produced by five-fold Euclidean k-means on EPSG:5070
  coordinates, following the spatial coordinate-clustering precedent of
  Schratz et al. (2019).

- Because full-batch k-means on 110,817,231 rows is unnecessarily costly,
  the centroids are fitted from a deterministic uniform-without-replacement
  sample of 1,097,262 canonical coordinates and all canonical rows are then
  assigned to their nearest fitted centroid.

- The sampling step is an engineering scalability adaptation, not an
  additional scientific partitioning rule.

- The earlier cell_id % 101 implementation was rejected because it aliased
  with the native raster width and produced periodic spatial sampling.

- Fold membership is based exclusively on geographic coordinates.

- No target stratification, target balancing, predictor balancing, or
  row-count balancing is performed.

- The dead-zone radius remains exactly the accepted Checkpoint 5A BP
  effective range.

VALIDATION

Re-run and require all existing 5B invariants, including:

- exactly 110,817,231 unique canonical cells
- exactly one outer TEST fold per canonical cell
- union of TEST folds equals canonical support exactly once
- TEST / DEAD_ZONE / TRAIN exhaustive and mutually exclusive per fold
- every TRAIN observation strictly farther than r from TEST
- every DEAD_ZONE observation at distance <= r
- every OOF nearest-training distance strictly > r
- persisted OOF distances independently reproduce
- no noncanonical model rows
- all canonical source hashes unchanged
- all row-level artifact hashes valid

REPRODUCIBILITY

Repeat enough of the corrected workflow to establish that:

- the uniform sample IDs reproduce exactly
- corrected centroids reproduce exactly
- all 110,817,231 fold assignments reproduce exactly
- row-level artifact hashes reproduce exactly, if computationally
  reasonable within the existing checkpoint requirements
- compact reports reproduce deterministically

Do not compare reproducibility against the rejected old geometry.

REPORT THE FINAL CORRECTED DESIGN

Report:

1. Exact radius
2. Final five centroid coordinates
3. Final TEST / DEAD_ZONE / TRAIN counts and percentages
4. Final nearest-training-distance min / p5 / p25 / median / p75 / p95 / max
5. BP-positive / BP-zero / FLP-valid support for TEST and TRAIN
6. Confirmation every canonical cell is TEST exactly once
7. Confirmation every distance invariant passes
8. Full corrected artifact row and partition counts
9. Reproducibility results
10. Runtime and peak RSS
11. Full pytest / Ruff / git diff --check results
12. Final files changed
13. Any remaining methodological caveat

Then STOP for final Checkpoint 5B HITL approval.

Do not construct inner folds.
Do not train models.
Do not commit.
Do not push.
