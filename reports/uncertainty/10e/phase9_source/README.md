# Phase 9: final held-out geographic ensemble evaluation

Frozen protocol: `evaluation_manifest.json` (SHA-256 `a99f06f93e33cb6361cdd1502e04dbf96aaf606a198af746912317a4e11889fd`). It was written after verifying all 25 epoch-100 Phase 8 checkpoints and before any outer TEST target or prediction was read. The TRAIN-only real-checkpoint smoke test passed for all five ensembles. The five outer TEST runs then completed with exact frozen fold counts. No model was trained, tuned, selected, or changed during evaluation.

| Study | BP cells | FLP-valid cells | BP MAE | BP RMSE | BP CE skill vs TRAIN constant | FLP CE | FLP excess CE/KL | FLP mean TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19,595,079 | 16,698,069 | 0.006486 | 0.012899 | 0.0514 | 1.6833 | 0.7065 | 0.4578 |
| 2 | 19,594,646 | 16,787,874 | 0.005555 | 0.007982 | −0.0146 | 1.6296 | 0.7278 | 0.4565 |
| 3 | 21,984,912 | 16,889,664 | 0.002786 | 0.004858 | 0.0226 | 1.3782 | 0.5688 | 0.4051 |
| 4 | 22,597,733 | 12,197,428 | 0.002410 | 0.003143 | 0.1140 | 1.5191 | 0.6525 | 0.4186 |
| 5 | 27,044,861 | 18,495,738 | 0.002162 | 0.004030 | 0.1645 | 1.3001 | 0.8148 | 0.5053 |
| Pooled cells (descriptive) | 110,817,231 | 81,068,773 | 0.003701 | 0.007166 | 0.0494 | 1.4965 | 0.6988 | 0.4515 |

The pooled BP bias is +0.000235, Pearson correlation is 0.171, and R² is −0.00066. The pooled BP excess CE/KL is 0.003694. Pooled values weight cells, whereas the five study values show geographic variation. Study 2 has negative BP CE skill against its TRAIN-derived constant. FLP targets are conditional soft distributions; modal confusion and dominant-class accuracy describe agreement between their modal classes and the predicted modal classes, not observed fire outcomes.

The ensemble beats the mean member in all five studies for BP MAE, BP CE, FLP CE, and FLP TV, with small average improvements. It does not beat the best member consistently: zero of five for BP MAE and FLP TV, one of five for BP CE and FLP CE. No member was chosen using TEST. Between-member BP variance has positive full-support Pearson association with absolute BP error in each study (0.021, 0.046, 0.089, 0.231, 0.089). On the fixed 1/101 cell-ID sample, the pooled Spearman association is 0.513, but this can reflect shared geographic risk structure and is not evidence of calibrated uncertainty or a causal error detector. Within-member and between-member variance are descriptive predictive-mixture components, not identified aleatoric and epistemic effects.

All additive metrics use full held-out support. Sampled scatter and Spearman diagnostics use `cell_id % 101 == 0` (1,097,262 cells). Risk strata are defined by **target** BP ≥0.01 and ≥0.02 and retain their support counts. The spatial maps aggregate true EPSG:5070 cell centers into 100 km bins. They show geography at that resolution, not individual-cell rasters. There are no independent-pixel confidence intervals; the geographic studies, not cells, are the replication units.

Rebuild numerical evaluation with `python -m wildfire_risk_uncertainty.training.phase9 evaluate`; it checks the frozen preflight marker and verifies partition recovery hashes. Rebuild tables and figures with `python -m wildfire_risk_uncertainty.training.phase9_figures`. `provenance.json` lists checkpoint, code, manifest, support, and artifact hashes. The `summaries/` directory has complete machine-readable results, including raw additive sums, target bins, risk strata, uncertainty components, and sampled associations. `tables/` contains CSV summaries; `figures/` contains presentation figures; `samples/` contains bounded diagnostic samples. `predictions/` contains full held-out per-cell ensemble BP/FLP means and between-member variances with cell IDs (1,971 nonempty Parquet partitions). Rebuild or verify them with `python -m wildfire_risk_uncertainty.training.phase9 export`. Existing Phase 8 artifacts remain intact.
