# Checkpoint 6F production concurrency integration

Status: **ENGINEERING_DECISION / VALIDATED — HITL accepted and frozen.**

## Implementation

The production HPO path now creates exactly two persistent CUDA worker processes for one Optuna trial. Worker 1 owns inner folds 1, 3, and 5; worker 2 owns folds 2 and 4. Each worker initializes its assigned models once and retains their model, AdamW, and independent RNG states across epochs. CPU, BLAS, and PyTorch worker thread counts are limited to one.

The parent remains the only process that owns the Optuna trial, W&B tracker, authoritative JSON records, checkpoint barrier, five-fold aggregate, ASHA report, and pruning decision. It waits for all five fold results at every epoch before performing any aggregate or pruning operation. Optuna trial execution remains strictly sequential with `n_jobs=1`; separate trials are never run concurrently.

SIGINT and SIGTERM are handled by the parent as a graceful pause request. Workers ignore those terminal signals and finish the active common epoch. After both workers return, the parent persists all five checkpoints and records, writes the aggregate, completes any rung decision, finishes the W&B session, and leaves the Optuna trial `RUNNING`. Worker checkpoints retain the existing cache, preprocessing, data-order, configuration, fold, trial, seed, optimizer, and RNG provenance checks.

## Real-data exact regression

The integrated production provider was compared with the sequential resumable provider for four epochs using outer study 1 and the fixed trial-0 configuration. Explicit cache warming was excluded from timing.

| Epoch | Sequential | Production concurrency 2 | Speedup |
|---:|---:|---:|---:|
| 1 | 145.413 s | 97.937 s | 1.485× |
| 2 | 143.863 s | 98.919 s | 1.454× |
| 3 | 143.390 s | 98.654 s | 1.453× |
| 4 | 143.257 s | 97.918 s | 1.463× |
| End to end | 577.155 s | 397.241 s | **1.453×** |

Exact equality passed for all 20 fold-epochs. The comparison covered supports, optimizer steps, raw BP and FLP loss sums, derived objectives, aggregate `J_e`, every model tensor, complete AdamW state, per-fold Python/NumPy/PyTorch CPU/CUDA RNG state, global step, and checkpoint provenance. No numerical tolerance was used. Median epoch speedup was **1.459×**.

Machine-readable result: [production integration validation](production_integration_validation.json).

## Pause and resume

The production concurrent provider ran epochs 1–2, paused at the completed common-epoch boundary, closed both worker processes, and left Optuna trial 0 `RUNNING`. Two fresh workers restored all five checkpoints and completed epochs 3–4.

The resumed run exactly matched the uninterrupted concurrent trajectory and final checkpoints. It retained trial number 0 and W&B run ID `mock-run`, with exactly 20 unique fold-epoch records and four aggregate records. Cache/checkpoint mismatch rejection remains covered by the existing checkpoint provenance tests.

## Pruning equivalence

A four-trial mocked ASHA study compared the sequential provider with the production batch-provider interface using the frozen sampler, pruner, search space, and `n_jobs=1`. Both modes reported identical rung values and made identical decisions: trial 0 continued at epochs 1 and 4 and completed; trials 1–3 pruned at epoch 1. No aggregate or pruning call occurred before five fold results existed.

Machine-readable result: [pruning equivalence](pruning_equivalence.json).

## Scope

No real HPO or outer-TEST evaluation was launched. The search space, sampler, ASHA schedule, folds, preprocessing, seed and data order, losses, batches, architecture, AdamW configuration, BF16 mode, and objective arithmetic are unchanged. Concurrency 3 and above remain excluded.

## Verification gates at acceptance

- Full pytest: **112 passed**.
- Ruff: **passed** for the repository.
- `git diff --check`: **passed**.
- `git diff --cached --check`: **passed**; no staged changes were present.
