# Checkpoint 5B uniform fitting sample correction: HITL comparison

**Status: superseded implementation provenance.** This comparison was prepared at the pre-HITL stop, before the corrected artifact rebuild. The uniform fitting sample changes broad fold geography materially. The corrected row-level artifact, reports, validator, and reproducibility results are now complete and accepted as the frozen 5B design; this comparison records why the earlier stripe-sample implementation was rejected.

The fitting sample uses `random.Random(42).sample(range(110817231), 1097262)` on sorted canonical row positions, then streams selected `cell_id`, `x_5070`, and `y_5070`. Every canonical row has equal inclusion probability. The [persisted sampled IDs](../../../artifacts/spatial_validation/05b/fitting_sample_cell_ids.parquet) have SHA-256 `63aac6b0696caea32c0ac55ff6db054a3d39b73f5a01f761521ad46742a53245`. All 101 native column residues are present (counts 10659–11111); the old fitting sample occupied one residue. No targets or predictors enter fitting.

A second corrected refit reproduced the persisted sample IDs and all five centroids byte for byte. All 3,060 canonical source partition hashes still match the committed canonical report.

The old fold map remains in the ignored pre-HITL archive. The [corrected preflight map](figures/corrected_outer_folds_preflight.png) and [accepted fold map](figures/outer_folds.png) show the corrected geometry. The optimal one-to-one centroid match is old folds 1→1, 2→2, 3→4, 4→3, 5→5. After that match, **40,583,471 rows (36.62%) change test fold**. Under the literal stable numeric labels, 68,014,648 rows change. The same one-to-one matching both minimizes summed Euclidean centroid displacement and maximizes row overlap; no acceptance threshold was imposed.

## Centroids and displacement

| Old fold → matched new fold | Old centroid x, y (m) | New centroid x, y (m) | Displacement (km) |
| --- | --- | --- | ---: |
| 1 → 1 | -1627464.079806, 2220278.157856 | -1594791.677018, 2546779.779235 | 328.132 |
| 2 → 2 | -555099.597840, 1253920.215279 | -1370228.650342, 1617291.145590 | 892.454 |
| 3 → 4 | -141928.471404, 2365931.832182 | -35054.014688, 2335507.896173 | 111.120 |
| 4 → 3 | 801586.991544, 1225707.115610 | -56960.075483, 1148939.591968 | 861.972 |
| 5 → 5 | 1376551.693965, 2155826.965566 | 1283881.519011, 1757125.418691 | 409.330 |

## Fold populations

| Old fold → matched new fold | TEST old → new | DEAD_ZONE old → new | TRAIN old → new |
| --- | ---: | ---: | ---: |
| 1 → 1 | 29,856,560 → 19,595,079 | 6,219,106 → 6,722,755 | 74,741,565 → 84,499,397 |
| 2 → 2 | 22,098,501 → 19,594,646 | 9,780,620 → 9,379,655 | 78,938,110 → 81,842,930 |
| 3 → 4 | 22,129,527 → 22,597,733 | 10,644,383 → 10,456,172 | 78,043,321 → 77,763,326 |
| 4 → 3 | 19,693,685 → 21,984,912 | 8,923,233 → 9,312,621 | 82,200,313 → 79,519,698 |
| 5 → 5 | 17,038,958 → 27,044,861 | 6,174,123 → 5,994,262 | 87,604,150 → 77,778,108 |

## OOF nearest usable TRAIN distance

All distances below are kilometres. The corrected figures were calculated in the full-grid diagnostic before the row-level rebuild and agree with the accepted corrected artifact.

| Old fold → matched new fold | Version | Min | p5 | p25 | Median | p75 | p95 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 → 1 | old | 211.151 | 263.244 | 477.321 | 756.762 | 1,067.951 | 1,385.098 | 1,602.133 |
| 1 → 1 | corrected | 211.153 | 244.135 | 384.231 | 582.490 | 834.279 | 1,170.163 | 1,401.206 |
| 2 → 2 | old | 211.151 | 237.016 | 345.771 | 497.947 | 673.458 | 931.119 | 1,109.730 |
| 2 → 2 | corrected | 211.151 | 235.692 | 340.867 | 490.633 | 663.633 | 874.225 | 1,030.108 |
| 3 → 4 | old | 211.151 | 235.475 | 338.199 | 487.827 | 680.101 | 908.683 | 1,030.776 |
| 3 → 4 | corrected | 211.151 | 236.250 | 342.356 | 494.117 | 685.030 | 918.447 | 1,048.446 |
| 4 → 3 | old | 211.151 | 237.575 | 352.440 | 518.308 | 719.423 | 958.136 | 1,306.690 |
| 4 → 3 | corrected | 211.151 | 238.851 | 355.604 | 517.721 | 736.120 | 1,001.335 | 1,186.557 |
| 5 → 5 | old | 211.151 | 242.099 | 374.664 | 567.093 | 983.912 | 1,537.596 | 1,758.182 |
| 5 → 5 | corrected | 211.151 | 261.251 | 467.644 | 754.527 | 1,051.470 | 1,330.274 | 1,547.042 |

The corrected geometry retains five spatially coherent regions and a usable TRAIN population in every fold, but large regional boundaries and centroid locations changed. The requested material-change review accepted this corrected geometry. The old artifact and reports remain only in the ignored pre-HITL archive; the current `05b` README and row-level artifact describe the approved geometry. [Machine-readable comparison](correction_comparison.json) contains exact values and the full assignment confusion matrix.
