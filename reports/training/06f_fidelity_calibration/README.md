# Checkpoint 6F HPO fidelity calibration

**Status: epoch-16 calibration complete; epoch-32 extension suspended and incomplete.** This diagnostic used only outer study 1 TRAIN and its five frozen inner spatial folds. The 11 distinct configurations are precisely the diagnostic trials that reached epoch 4: 0, 1, 2, 7, 13, 15, 16, 19, 22, 28, and 29. Each was trained from scratch through epoch 16 without pruning, using the frozen production two-worker executor and common epoch barrier. No outer TEST population was inspected, no real HPO study was launched, and no HPO methodology or loss/objective definition was changed.

## Historical equivalence gate

The reruns exactly reproduced the diagnostic aggregate J and all five fold-level validation support counts, raw BP and FLP loss sums, component means, and equal-task objectives at epochs 1 and 4. This covers **22 aggregate epochs and 110 fold-epochs**, with no numerical tolerance. The full gate passed before any configuration advanced beyond epoch 4. See [the machine-readable gate](historical_equivalence_gate.json) and its individual `gate_checks/` records.

All 11 configurations completed 16 common epochs, yielding 176 aggregate records and 880 fold records. The observed end-to-end runtime was **16,658.794 seconds (4 h 37 min 39 s)**, including worker startup and orchestration. The sum of measured epoch barrier times was 16,577.953 seconds. Local per-fold checkpoints and full epoch records are preserved under ignored `artifacts/hpo/06f_fidelity_calibration/`; the compact, machine-readable outputs are here.

## Rank stability

Lower J is better. Correlations use all 11 configurations and their values at the specified common epochs.

| Epoch pair | Spearman ρ | Kendall τ |
|---|---:|---:|
| 1–4 | −0.564 | −0.527 |
| 1–8 | 0.864 | 0.673 |
| 1–16 | 0.527 | 0.382 |
| 4–8 | −0.464 | −0.418 |
| 4–16 | **−0.136** | **−0.127** |
| 8–16 | 0.591 | 0.491 |

| Earlier epoch vs 16 | Top 1 overlap | Top 3 overlap | Top 5 overlap |
|---:|---:|---:|---:|
| 1 | 0/1 | 2/3 | 4/5 |
| 4 | **0/1** | **0/3** | **1/5** |
| 8 | 1/1 | 2/3 | 4/5 |

Trial 7 is best at epoch 16 with `J₁₆ = 0.7030805870782386`. Its ranks at epochs 1, 4, 8, and 16 were **3, 10, 1, 1**. It survived the original epoch-1 decision but was pruned at epoch 4. Within this cohort, selecting the best configuration at epoch 1 would have chosen trial 19 and incurred epoch-16 regret **0.0005215614660700751**; selecting at epoch 4 would have chosen trial 0 and incurred regret **0.00041049802144221825**. Epoch 8 selected trial 7, for zero regret against the epoch-16 reference.

Epoch 4 is **not sufficiently predictive of epoch 16 to support pruning** in this observed cohort. The near-zero, slightly negative rank correlations, zero top-3 overlap, and loss of the eventual winner under the original epoch-4 decision support that conclusion. The original 30-trial study was diagnostic only: one trial completed 100 epochs and 29 were pruned. Trial 0, its sole completion, ranked 20/30 at epoch 1 yet first among the 11 epoch-4 survivors. Trial 1 was only about 8.7 × 10⁻⁶ worse at epoch 4 but was pruned. No implementation bug was found in that audit.

The cohort is conditioned on reaching epoch 4 in the original diagnostic study, so it cannot measure outcomes for the 19 trials pruned earlier. Epoch 16 is a calibration endpoint, not proof of ranking stability through the frozen 100-epoch maximum. The current equal-task objective and loss formulation remain unchanged. Further fidelity calibration is suspended until loss balancing is finalized; if that work changes the training objective or trajectory, later-fidelity rankings from the current formulation may have limited value for subsequent pruning decisions. No replacement pruning schedule has been selected.

## Suspended epoch-32 extension

The [extension manifest](epoch32/manifest.json) records **SUSPENDED_INCOMPLETE**. The [checkpoint validation](epoch32/checkpoint_validation.json) passed for all 55 epoch-16 fold checkpoints, including file hashes, cache and preprocessing binding, configuration, trial/fold identity, seed, data-order version, optimizer step, and saved model/AdamW/RNG fields. Trial 0 began epoch 17, but the user paused the run before a common epoch completed. No epoch-17 fold record, aggregate, or replacement checkpoint exists; all 11 trials remain complete through epoch 16 only. The continuation script is retained for provenance, with no authorization to run it during this freeze.

Large model checkpoints and full per-fold records remain local under ignored `artifacts/hpo/06f_fidelity_calibration/` and are not part of this commit. The committed [checkpoint validation](epoch32/checkpoint_validation.json) preserves their SHA-256 hashes and provenance, but a fresh clone cannot resume at epoch 16 without those local checkpoint files. The [environment snapshot](environment.json) hashes the frozen repository scripts and compact evidence; the scripts received only output-format and rerun guards after the training run. The CSV summaries, plots, manifest, exact historical gate, and scripts provide the compact repository evidence.

## Outputs

- [Complete analysis](analysis.json), [manifest](manifest.json), [runtime](runtime_progress.json), [freeze audit](freeze_audit.json), and [environment snapshot](environment.json).
- [All-epoch metrics](epoch_metrics.csv) and [selected-fidelity metrics](fidelity_metrics.csv): J, five fold J values, BP/FLP components, fold standard deviation, configuration, and common-epoch wall time.
- [Ranks](rankings.csv), [rank correlations](rank_correlations.csv), [top-k overlaps](top_k_overlap.csv), and [selection regret](selection_regret.csv).
- [Rank trajectories](rank_trajectories.png), [learning curves](learning_curves.png), [correlation heatmaps](rank_correlations.png), and [top-k overlap plot](top_k_overlap.png).
- Reproduction/resume driver: [run_06f_fidelity_calibration.py](../../../scripts/run_06f_fidelity_calibration.py).
- Suspended continuation driver: [continue_06f_fidelity_to_32.py](../../../scripts/continue_06f_fidelity_to_32.py).
