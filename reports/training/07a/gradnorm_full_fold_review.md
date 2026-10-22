# Phase 7A full-fidelity GradNorm training validation

## Run and recovery

Experiment `07a-20261022T054112Z-5270cc63` completed 100 epochs on outer study
3, inner fold 1 **TRAIN and geographic VALIDATION only**. The first three epochs
come from [the CE-reference run](gradnorm_ce_reference_fold1.json); epochs 4–100
continued from its exact epoch-3 checkpoint. The detached run's
[`config.json`](../../../artifacts/benchmark/07a_detached/07a-20261022T054112Z-5270cc63/config.json)
fixes the 33-input, three-layer width-128 MLP; BF16 CUDA; batch size 65,536;
model AdamW learning rate 0.001 and weight decay 0.000316227766; seed 42; four
CPU threads; and the CE-rate GradNorm method. GradNorm uses asymmetry 1.5,
separate Adam learning rate 0.025, first-epoch equal weights and exact
first-epoch separate-support CE references, masked conditional FLP loss, positive
weights normalized to sum to two, and gradients at the final shared Linear
weight matrix. No model or GradNorm hyperparameter changed during continuation.

The continuation preflight verified source SHA-256, implementation hashes,
frozen cache/preprocessing identity, architecture and optimizer configuration,
model and optimizer state, GradNorm state, epoch/global step, and Python, NumPy,
and PyTorch RNG restoration. Every epoch produced independent full-support BP
and FLP validation excess log losses, BP target-stratum metrics, a hash-checked
checkpoint, a successful immediate restoration, resource samples, and a JSON
record. The detached supervisor retained process logs, progress, configuration,
and resource measurements. The run reached epoch 100 and restored its checkpoint,
then the worker failed in finalization because a broad `*.json` glob also
selected `.pt.sha256.json` sidecars. Commit
`d08a42b99400a6e29f1e30b0966eff3c9d0c56ba` narrowed epoch discovery to
exact metric filenames and added recovery validation. Recovery verified all
100 checkpoint checksums, all epoch records and step continuity, validation
support, finite metrics, and a fresh load of the epoch-100 checkpoint. The
experiment status is `complete`, with the original worker exit code 1 preserved
in its recovery history. No epoch was retrained or artifact overwritten.

## Selected observations

| Epoch | BP validation KL | FLP validation KL | Final BP/FLP weights | BP ≥ 0.02 target/predicted mean | BP ≥ 0.02 cell Pearson |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.007296 | 0.675394 | 1.000 / 1.000 | 0.03643 / 0.00584 | 0.0890 |
| 3 | 0.009013 | 0.607224 | 1.537 / 0.463 | 0.03643 / 0.00330 | 0.0159 |
| 100 | 0.008574 | 0.613337 | 1.736 / 0.264 | 0.03643 / 0.00343 | 0.0636 |

At epoch 100 the full validation BP KL was 0.008574 versus 0.011518 for a
constant BP forecast derived from the same inner TRAIN fold. This is positive
full-support excess-log-loss skill (0.002944), while the 1,841,578 validation
cells with target BP ≥ 0.02 remained strongly underpredicted. Their equal-tile
mean correlation was −0.099 across 139 frozen cache partitions with at least
100 eligible cells. These target-defined strata describe the geographic
VALIDATION surface; they are not new model-selection criteria or outer TEST
estimates. The BP and FLP validation trajectories are nonmonotonic, so the
single final epoch should not stand in for the full recorded trajectory.

Across epochs 4–100, no BP weight contacted the positive floor; the minimum
recorded post-step BP weight was 1.172. All training/validation metrics in the
100-epoch artifact audit were finite. The continuation recorded 97 epoch
records, 97 restorable checkpoints, and 2,492 one-second resource samples.
Maximum sampled worker RSS was 14.41 GiB; minimum sampled system available RAM
was 25.97 GiB; maximum sampled GPU memory use was 2,837 MiB; and swap use was
zero. The 97 continuation epochs took 2,575.8 seconds in summed epoch timing.
These observations establish this one-fold run's execution and numerical
stability, not general performance across all geographic folds or HPO settings.

## Interpretation and handoff

Phase 7A's probabilistic loss formulation, literature-standard GradNorm
implementation, and production training validation are complete. The weak
elevated-risk BP calibration and spatial discrimination are **predictive
adequacy and geographic transfer** questions for [Phase 7B](../../../docs/methodology/checkpoint_07b_model_selection.md),
not evidence that the validated GradNorm equations, loss masks, or checkpoint
implementation failed. The first Phase 7B diagnostic should evaluate the
existing epoch checkpoints on the same inner TRAIN and geographic VALIDATION
supports, with matched BP metrics and frozen risk strata; it must not retrain or
inspect outer TEST. The HPO ranking objective remains undecided.

Metrics and checkpoints: `artifacts/benchmark/07a_detached/07a-20261022T054112Z-5270cc63/`.
The recovered `fold_1.json` lists epochs 4–100, and
`source_epochs_001_to_003.json` binds the first three epochs to their original
result. Use `scripts/benchmark_07a_detached.py status|results` with the
experiment ID to inspect status and the latest record.
