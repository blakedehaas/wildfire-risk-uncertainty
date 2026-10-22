# Phase 7A: full 100-epoch inner-fold continuation

This run is a **single diagnostic** on outer study 3, inner fold 1 TRAIN and
VALIDATION. It uses the frozen 33-feature, 3×128 MLP, BF16 CUDA, GradNorm CE-rate
algorithm (`gradnorm_alpha=1.5`, weight Adam learning rate 0.025), model AdamW
learning rate 0.001 and weight decay 0.000316227766, batch size 65,536, seed
42, and four CPU threads. It is not a model-selection experiment. Outer TEST,
HPO, other folds, and alternative loss weights are excluded.

The first three complete epochs are in
`reports/training/07a/gradnorm_ce_reference_fold1.json`. The epoch-3 checkpoint
`artifacts/benchmark/07a_gradnorm_ce_reference_fold1_validated/fold_1_epoch_003.pt`
is resumed only after SHA-256, source-code, cache, preprocessing, configuration,
model and optimizer state, GradNorm state, epoch/step, and RNG restoration checks
pass. The original three epoch results and hashes are copied into the detached
run's provenance. Training continues from epoch 4 through 100, using the saved
optimizer, GradNorm, and random states. On each subsequent epoch the worker
saves and reloads a complete, hash-checked checkpoint before writing the epoch
record. A stopped or failed run resumes from its last complete epoch record;
the epoch's checkpoint and metrics must agree.

Every epoch runs the production full validation with independent BP and masked
FLP excess log losses. A second full-support BP pass reports observed and
predicted means, errors, excess log loss, and prediction-target correlation in
the same fixed target intervals as the three-epoch validation: `[0, .001)`,
`[.001, .01)`, `[.01, .02)`, and `[.02, 1.000001)`. The last is the declared
BP ≥ 0.02 elevated-risk subset. It also reports correlation between observed
and predicted **partition means**, with equal weighting for frozen cache
partitions containing at least 100 stratum cells. This is descriptive spatial
discrimination on inner VALIDATION, not a spatially independent test. A
constant BP prediction of `0.001848224540038233`, derived only from the same
inner TRAIN support, provides climatological MAE, RMSE, and excess log loss;
the difference in excess log loss is reported as skill. Strata are used for
diagnosis only and do not tune GradNorm or the model.

Each epoch record includes train CE, validation CE/KL, BP strata, GradNorm
weights and relative rates, sampled weight/gradient trajectories, floor
contacts, median BP weighted shared-gradient share, stage timings, RSS, peak
CUDA allocation/reservation, global step, and checkpoint hash. The detached
supervisor persists process status, progress, stdout/stderr, one-second GPU/RAM/
swap/RSS/I/O resource samples, and failure status. Its existing safety gates
remain active. A clean committed worktree is required at launch and resume.

```bash
.venv/bin/python scripts/benchmark_07a_detached.py launch --kind gradnorm_full_fold
.venv/bin/python scripts/benchmark_07a_detached.py status EXPERIMENT_ID
.venv/bin/python scripts/benchmark_07a_detached.py results EXPERIMENT_ID
.venv/bin/python scripts/benchmark_07a_detached.py stop EXPERIMENT_ID
.venv/bin/python scripts/benchmark_07a_detached.py resume EXPERIMENT_ID
```

The historical run `07a-20261022T054112Z-5270cc63` has completed and was
recovered after a post-epoch-100 finalization error; see the
[full-fidelity review](gradnorm_full_fold_review.md). The launch command above
documents the original reproducible protocol and is **not** a request to run
another experiment. Phase 7B begins with read-only evaluation of existing
checkpoints.
