# Checkpoint 6F: production HPO execution optimization

**Status: ENGINEERING_DECISION / VALIDATED. HITL accepted and frozen.**

Production HPO keeps Optuna trials sequential (`n_jobs=1`). Within one trial, exactly two persistent CUDA worker processes own the five inner-fold training states. The parent process alone owns Optuna, ASHA reporting and pruning, W&B orchestration, authoritative records, and the common-epoch objective. It waits for all five folds to finish each epoch before calculating or reporting J_e. This execution change does not alter the search space, sampler, resource schedule, folds, preprocessing, seeds, row order, losses, batch semantics, model, AdamW configuration, BF16 behavior, or objective calculation.

The fixed outer-study-1 trial-0 configuration was run for four epochs through the sequential reference and production concurrent path. All 20 fold-epochs were exactly equal for row and support counts, optimizer steps, raw BP and FLP loss sums, fold objectives, aggregate J_e, every final model tensor, complete AdamW state, RNG state, and checkpoint provenance. No numerical tolerance was used.

The two-worker interrupted/resumed run exactly reproduced the uninterrupted trajectory and final state. It retained Optuna trial 0 and W&B run ID `mock-run`, produced 20 unique fold-epoch records and four aggregate records, and produced no duplicates. The validation used a controlled common-epoch pause and mock tracker; the production signal handler uses the same checkpoint boundary, while live W&B same-run restoration was validated separately during 6F prelaunch.

A four-trial mocked study produced identical sequential and concurrent rung reports and pruning decisions under the frozen sampler and pruner. Production timing fell from 577.155 seconds sequentially to 397.241 seconds with two workers, a **1.453× end-to-end speedup**. Concurrency three crossed the recorded VRAM safety gate and was rejected; concurrency above two is outside the accepted execution design.

The machine-readable comparisons and resource measurements are preserved in the [6F resource-optimization report](../../reports/training/06f_resource_optimization/PRODUCTION_INTEGRATION.md). The completed outer-study-1 HPO run remains diagnostic evidence only. No outer TEST data were inspected and no real HPO was launched while validating this optimization.
