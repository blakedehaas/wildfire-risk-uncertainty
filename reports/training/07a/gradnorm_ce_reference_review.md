# CE-rate GradNorm bounded validation, 2026-10-22

**Historical three-epoch decision point.** Its hold on full-fidelity training
was followed by the approved and completed
[100-epoch continuation](gradnorm_full_fold_review.md). The HPO ranking and BP
geographic predictive-adequacy questions remain open in
[Phase 7B](../../../docs/methodology/checkpoint_07b_model_selection.md). The
three-epoch measurements and contemporary interpretation below are preserved.

## Method and provenance

This revision uses each task's existing masked cross entropy for both the model update and GradNorm's relative training-rate measurement. The first complete inner-TRAIN epoch uses weights `(1, 1)`. Its exact BP and FLP separate-support mean training cross entropies become fixed references; adaptive weight updates start in epoch 2. This follows the scheduling and first-epoch loss-buffer convention of [pinned LibMTL GradNorm](https://github.com/median-research-group/LibMTL/blob/4336804847eaa5e0b924b743d76beec7ac3fdc97/LibMTL/weighting/GradNorm.py), with the same detached-target equations as [Chen et al. (2018), Algorithm 1](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf). The existing separate Adam optimizer at learning rate 0.025, direct positive projection, sum-to-two normalization, network AdamW, and frozen global FLP-support correction remain. LibMTL's softmax parameterization and common optimizer are documented differences.

The [machine-readable result](gradnorm_ce_reference_fold1.json) comes from one three-epoch CUDA run on outer study 3, inner fold 1, using the frozen 3×128 model, BF16, batch size 65,536, seed 42, four CPU threads, and **inner TRAIN and VALIDATION only**. The [runner](../../../scripts/validate_gradnorm_ce_reference.py) wrapped the production GradNorm step for logging but called the production `train_epoch` and `validate` functions. It saved and reloaded a provenance-bound checkpoint after each epoch, verifying exact model, network optimizer, GradNorm weights/references, and weight-optimizer state. All three checkpoint SHA-256 values match their records. The first attempt failed *after* epoch-1 checkpoint verification because the diagnostic script formatted a relative path incorrectly; it did not write a completed result. The script was corrected and the completed run used a fresh checkpoint directory. Historical KL-rate artifacts were not overwritten or resumed.

Predeclared descriptive BP target strata were `[0, .001)`, `[.001, .01)`, `[.01, .02)`, and `[.02, 1]`; the thresholds were chosen before this run from the frozen outer-TRAIN target summary and were not used for model or algorithm selection. The outer-TRAIN sampled BP 95th percentile was about 0.01595. These are probabilistic simulation targets, so the reported correlations are descriptive associations with the supplied surface, not observed-fire discrimination statistics.

## Three-epoch results

| Epoch | BP / FLP train CE | BP / FLP validation KL | BP / FLP weight at end | Mean BP / FLP relative rate | Median BP share of weighted shared norm | BP floor contacts |
|---|---|---|---|---|---:|---:|
| 1 | 0.02376 / 1.46495 | 0.00730 / 0.67539 | 1.000 / 1.000 | warmup | — | 0 |
| 2 | 0.01320 / 1.43850 | 0.00955 / 0.62524 | 1.594 / 0.406 | 0.555 / 1.445 | 10.6% | 0 |
| 3 | 0.01314 / 1.43805 | 0.00901 / 0.60722 | 1.537 / 0.463 | 0.553 / 1.447 | 12.8% | 0 |

The initial references exactly match epoch-1 training CE `(0.02376445, 1.46494725)`. All 932 steps in epoch 1 kept equal weights; all 1,864 steps in epochs 2–3 updated them. BP's minimum post-step weight was 1.025 in epoch 2 and 1.521 in epoch 3, far from the `1e-8` floor. Weighted BP shared-layer gradients were material, though smaller than FLP's. This removes the earlier **weight-boundary failure mode**; a larger BP weight alone does not demonstrate successful BP prediction.

An independent trace audit found all recorded losses and weights finite, maximum sum-to-two error `2.4e-7`, maximum relative-rate sum-to-two error `1.8e-7`, and maximum reconstruction error in the GradNorm objective `7.0e-8`. The result records hashes for the runner and production implementation files; the implementation hashes were added after execution with those files unchanged.

FLP validation excess log loss improved each epoch (`0.6754 → 0.6252 → 0.6072`). BP training CE declined, but BP validation excess log loss worsened after warmup (`0.00730 → 0.00955`) and remained higher at epoch 3 (`0.00901`). Its validation CE likewise rose from `0.04536` to `0.04708` by epoch 3. This one-fold three-epoch result does not establish a long-run trend, but it does **not** show both tasks improving in validation.

| BP target stratum | Rows | Mean target | Mean prediction, epochs 1 → 2 → 3 | MAE, epochs 1 → 2 → 3 | Pearson with target, epochs 1 → 2 → 3 |
|---|---:|---:|---|---|---|
| `[.01, .02)` | 2,583,327 | 0.01422 | 0.00576 → 0.00299 → 0.00328 | 0.00847 → 0.01122 → 0.01094 | −0.015 → 0.027 → −0.016 |
| `[.02, 1]` | 1,841,578 | 0.03643 | 0.00584 → 0.00303 → 0.00330 | 0.03059 → 0.03340 → 0.03314 | 0.089 → 0.018 → 0.016 |

Mean predictions in elevated strata were much lower than their soft targets and changed little between the two elevated bands. The near-zero within-stratum correlations and increasing elevated-stratum MAE suggest weak risk ordering on this held geographic validation fold. These target-defined strata are descriptive; they do not replace full-support validation KL or prove that the model cannot improve at later epochs.

The completed run took **80.40 s** including three full training traversals, three production validations, three descriptive validation passes, checkpoint save/load, startup, and monitoring. Training traversals took 19.41, 20.43, and 20.25 s; production validations took 3.51, 3.01, and 3.22 s. Peak sampled process RSS was **14.42 GiB**, minimum system available RAM **26.95 GiB**, and swap use **zero**. Maximum PyTorch allocated CUDA memory during training was **218.49 MB** (per-epoch training peak; the final process-wide counter was reset by validation). No numerical or memory failure occurred. These single-worker figures do not independently revalidate the earlier two-worker resource envelope.

## Interpretation and next gate

The previous KL-rate variant used the first minibatch's KL as the reference and reached BP weights near zero, with a 0.28% median BP shared-gradient share by epoch 3 of its fold-1 continuation. The revised run uses first-epoch CE means and retains a roughly 11–13% BP gradient share after warmup. The changes to both **rate loss** and **reference schedule** were made together to align with the published reference; this bounded validation is not a controlled comparison that attributes the improvement in weight stability to either change alone.

**Hold full 100-epoch training and HPO for review.** Algorithmic weight stability is now demonstrated for one representative fold over three epochs, and checkpointing is sound. BP's held-fold excess log loss and elevated-risk errors did not improve beyond warmup, and geographic risk ordering remained weak over this short window. A full-fidelity run may change that assessment, but the present evidence does not establish that both equally important outputs continue learning well. The HPO ranking objective's numerical CE-scale imbalance is also unresolved. No outer TEST data, equal-weight comparison, Pareto sweep, or HPO was used.
