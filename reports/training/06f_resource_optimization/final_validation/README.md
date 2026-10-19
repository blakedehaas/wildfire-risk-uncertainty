# Checkpoint 6F concurrency-2 final validation

Status: **PASS — stop for HITL.** Recommend integrating concurrency 2 into the production HPO runner, subject to review. Production code has not been modified.

## Warm-cache timing confirmation

Each measured run performed an explicit full read of the mmap training inputs immediately before timing. Cache-warming time was excluded. Runs 1 and 3 executed c1 before c2; run 2 reversed the order.

| Repetition | c1 wall time | c2 wall time | Speedup |
|---:|---:|---:|---:|
| 1 | 185.533 s | 112.270 s | 1.653× |
| 2 | 173.035 s | 114.254 s | 1.514× |
| 3 | 171.059 s | 114.189 s | 1.498× |
| Median | 173.035 s | 114.189 s | 1.514× paired median |

All 30 timing fold executions exactly matched the accepted sequential epoch-1 reference for supports, batch/optimizer-step counts, raw BP and FLP loss sums, derived objectives, model-state digest, and complete AdamW-state digest. The speedup remained material in every repetition.

## Four-epoch epoch-barrier regression

The definitive regression used persistent worker processes and a common barrier after every epoch. Each fold retained its model and AdamW instance across epochs. C1 used one worker for all five folds. C2 used two workers, assigned folds `(1, 3, 5)` and `(2, 4)`. Wall times below are observed common-epoch elapsed times.

| Epoch | c1 wall time | c2 wall time | Speedup |
|---:|---:|---:|---:|
| 1 | 157.744 s | 98.497 s | 1.602× |
| 2 | 157.618 s | 98.217 s | 1.605× |
| 3 | 157.111 s | 98.459 s | 1.596× |
| 4 | 157.787 s | 98.451 s | 1.603× |
| End to end¹ | 633.051 s | 396.652 s | **1.596×** |

¹ End-to-end time includes worker initialization and shutdown. The four measured epoch barriers sum to 630.260 seconds for c1 and 393.624 seconds for c2. Explicit cache warming took 0.870 and 0.710 seconds respectively and was excluded.

Exact equality passed independently for every epoch and fold: **20/20 fold-epochs**. The comparison covers training and validation row/support counts, per-epoch training optimizer steps, validation batch counts, raw BP/FLP loss sums, task means, equal-task objectives, every final model tensor, and the complete AdamW state after each epoch. Exact AdamW digests also establish equal cumulative step counters and moment tensors. No tolerance was used.

The earlier continuous four-epoch regression independently passed the same 20 fold-epoch comparisons and measured 627.004 seconds at c1 versus 405.481 seconds at c2 (1.546×). It is retained as corroborating evidence; the barrier regression supplies the requested observed per-epoch wall times and matches the HPO common-epoch structure.

## Resources and scope

During the barrier regression, peak sampled device VRAM was 2,259 MiB at c1 and 2,704 MiB at c2. Minimum available system RAM was 26.43 GiB and 25.24 GiB; swap remained at 8 KiB. No OOM, swap pressure, deadlock, numerical mismatch, or severe regression occurred. GPU samples are device-wide and may include unrelated activity.

The run used outer study 1, trial-0 hyperparameters, seed 42, the validated mmap cache, and the 25 frozen preprocessing states at source commit `8db4238c5f9b141ea67a5a8a746488d1ff2cc0ed`. It did not evaluate outer TEST. No real HPO was launched. No production HPO code, commit, or remote branch was changed.

Machine-readable evidence: [result.json](result.json), [barrier result](barrier_result.json), [barrier c1](barrier_c1.json), [barrier c2](barrier_c2.json), [timing repetitions](timing_repetitions.csv), and [epoch times](epoch_barrier_times.csv). The isolated harnesses are [validate_06f_concurrency2.py](../../../../scripts/validate_06f_concurrency2.py) and [validate_06f_epoch_barrier.py](../../../../scripts/validate_06f_epoch_barrier.py).

## HITL recommendation

Integrate concurrency 2 into the production HPO runner using persistent workers and a common epoch barrier. Preserve the current per-fold seed, traversal order, preprocessing state, checkpoint identity, and aggregate/pruning decision only after all five fold records exist for the epoch. Retain the existing resource and exact-regression gates during integration validation.
