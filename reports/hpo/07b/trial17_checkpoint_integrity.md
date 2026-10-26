# Trial 17 checkpoint integrity and disposition

This report is a read-only inspection of outer study 3, trial 17 on 2026-10-26.
The existing Optuna trial remains `FAIL`; no training, trial insertion, budget change,
or artifact migration was performed.

## Evidence and likely failure mechanism

- SQLite records trial 17 as `FAIL`, with no objective value. Its failure attribute
  contains `FoldWorkerError` caused by `ValueError: Checkpoint file hash mismatch`
  during worker restoration. The trial has 307 fold-epoch records: all five folds
  through epoch 61, plus folds 1 and 2 at epoch 62. There are 16 `COMPLETE`
  trials among 18 attempts, and the current lifetime cap is 21.
- Both `fold_1_epoch_062.pt` and `fold_2_epoch_062.pt` are zero bytes with zero
  allocated data blocks. Both SHA sidecars are 78 bytes and contain nonempty
  payload hashes; the empty-file SHA-256 is
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  File birth, modification, and change times are clustered at 03:35:51 UTC.
  The corresponding fold records were persisted later. The sidecars establish
  that nonempty bytes were hashed after `torch.save` and atomic replacement;
  the later fold records establish that the worker save barrier completed.
- The save path previously used `torch.save` to a temporary path followed by
  `Path.replace`, but had no `fsync` for checkpoint data or directory entries.
  JSON files fsynced their contents but not the parent directory. A crash or
  filesystem durability event could therefore lose checkpoint data despite
  completed save calls. This is a plausible mechanism, not a proven cause.
  Both workers save distinct fold paths, and retention only removes epoch
  `N-2`; neither normal concurrency nor retention targets epoch 62 here.
  There is no surviving write trace that distinguishes data loss from an
  external truncation. Free space was ample (ext4 `/workspace`, 7% used at
  inspection). We cannot assign a more specific root cause from these files.

## Epoch 61 validation

All five epoch 61 payloads match their sidecars and load with `torch.load` on
CPU. Each has the expected trial/fold/configuration, GradNorm version and
parameters, complete cache binding, and frozen preprocessing hash. The cache
manifest and all five preprocessing hashes match the frozen trial provenance.
Strict `RiskMLP`, AdamW, and GradNorm state loading succeeded on CPU. The
Python, NumPy, Torch CPU, and one CUDA RNG states are present. Model and
GradNorm tensors are finite, GradNorm is initialized, and the largest AdamW
step equals the corresponding persisted global step. The read-only common
epoch audit returns **61**.

| Fold | SHA-256 | Global step |
| --- | --- | ---: |
| 1 | `d33673a0e766cc8c24cb9c5c303c699dae772e372845be8a8d3fce0e055a943b` | 113643 |
| 2 | `aca7f3d04db4cc647f745c423a90ef93d67ae2512bb96c2a94d8e8d02250f8d1` | 114619 |
| 3 | `a6b9729ee6ae3c2c6e8fd0189e73375e3df0f79265291efa85e9d4a550d2ae0a` | 113948 |
| 4 | `ec5e96e78b66ea5c50f0b89376e0dfa9e8694444fd3d349bf41ea006478529e0` | 128893 |
| 5 | `1b2d203ef2609d268b5d5f8627b1bc8305c28af9a9b8db0c56d26fd31cd57c92` | 121146 |

## Recovery decision

Epoch 61 can seed a deterministic continuation in principle, but trial 17
cannot be resumed under its original Optuna identity: Optuna 5 rejects
transitioning a finished `FAIL` trial back to `RUNNING`. Two epoch 62 fold
records already exist; replaying those epochs in place would conflict with
the authoritative records. A replacement would need a new Optuna trial number
created through supported `enqueue_trial`/`ask` operations, an explicit
trial-17 lineage declaration, verified checkpoint migration that changes
embedded trial identity and hashes, and careful reconciliation of epoch 62
records, sampler state, and W&B identity. That work is disproportionate to
two remaining successful trials. Do not relabel trial 17, copy checkpoints
under a new trial silently, or insert a fabricated objective.

**Recommended disposition:** keep trial 17 as an excluded infrastructure
failure and, after review, raise the lifetime cap from 21 to 22 attempts.
With 16 completed trials this leaves four slots for four successes, assuming
no further failures. The two previous `FAIL` trials remain excluded.

## Safe restart after review

1. Confirm the committed repair is the reviewed source, the worktree is clean,
   no detached controller is active, trial 17 is still `FAIL`, the trial count
   is 18, the cap is 21, and the epoch 61/62 evidence above is unchanged.
2. Register the reviewed infrastructure revision without changing trial 17:
   `python -m wildfire_risk_uncertainty.hpo.operations adopt-checkpoint-repair --outer-study 3`.
   This command checks source lineage, the changed-file allowlist, frozen
   methodology/environment, sampler state, trial history, and checkpoint
   integrity before writing the study revision attribute.
3. Extend the lifetime cap:
   `python -m wildfire_risk_uncertainty.hpo.operations extend --outer-study 3 --new-limit 22`.
   Inspect `python -m wildfire_risk_uncertainty.hpo.operations status --outer-study 3`.
4. Only after those checks and a separate training authorization, restart the
   existing GradNorm controller from the repository root:
   `python -m wildfire_risk_uncertainty.hpo.cli real-hpo --outer-study 3 --cache-dir artifacts/training/hpo_cache_v1 --trial-limit 22 --gradnorm-objective-reviewed`.
   The controller will create trial 18 through the normal sampler; it will
   not restore or rewrite trial 17.

For any future `RUNNING` trial with a restore error, the controller now exits
without marking the trial `FAIL`. Inspect its immutable fold records and
checkpoints before any restart. The read-only audit can identify a lower
common epoch, but continuation from that epoch requires an explicit migration
if records at later epochs already exist.
