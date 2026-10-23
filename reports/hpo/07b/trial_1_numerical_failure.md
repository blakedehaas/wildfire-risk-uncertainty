# Outer-study-3 trial 1: BP loss failure and recovery boundary

## Persisted evidence

The original detached controller `07a-20261023T074958Z-d84cabfb` stopped with
Optuna trial 1 in `FAIL`. Trial 0 remains `COMPLETE` with objective
`0.10449623942546031`. Trial 1 used depth 3, width 64, AdamW learning rate
`0.00023270677083837802`, weight decay `8.17949947521167e-05`, and batch
size 32,768. There are 375 completed fold-epoch records: epochs 1–75 for each
of the five inner geographic folds. All five epoch-74 and five epoch-75
checkpoint files matched their SHA-256 sidecars. There is no epoch-76 fold
record or checkpoint. The last verified fold-1 checkpoint is epoch 75, global
optimizer step 139,725, SHA-256
`96ea5a4321f721988d836e2e2fb9f4842da0353848240af6a184b8392d5a3eb7`.
The other epoch-75 checkpoint hashes are:

| Fold | SHA-256 |
| --- | --- |
| 2 | `96c5c4833b4740cf87b4f761f2b38a3b6af0abf81f1ea61b38d06cf3465307a6` |
| 3 | `cb249e2c60115349278feae64fb8e4349cd49a235b894dc269cf84484ed4975e` |
| 4 | `2a361e6f3f78ae96365db959fdfdaa135e341c1dd801941f75d496e760104393` |
| 5 | `a4afd432a5cda5067b9103de300e3d5ead3be8fde729b5b079df740bb71847ae` |

The original worker traceback raised `FloatingPointError: Invalid BP CE rate
loss` in `GradNorm.step`. Controller cleanup then waited for a `closed` message
from the failed worker, hit the 900-second queue timeout, and recorded
`_queue.Empty` as the trial failure. This masked the primary error.

## Bounded replay and numerical cause

The read-only [replay script](../../../scripts/replay_07b_trial1_failure.py)
restores the hash-checked epoch-75 model, optimizer, GradNorm state, and RNG
from the existing checkpoint. It verifies the frozen trial configuration and
step count, then executes at most epoch 76 for one specified fold. It writes
only its JSON report; it does not write Optuna state or HPO checkpoints.

The [failure replay](trial1_replay_fold1.json) reproduced the exception in
**fold 1, epoch 76, attempted optimizer step 141,546**. The batch contained
32,768 BP targets, all zero, no FLP-valid rows, and finite BP logits all equal
to `-15.0625`. PyTorch float32
`binary_cross_entropy_with_logits(..., reduction="sum")` returned `0.0`.
The equivalent stable Bernoulli identity, `y·softplus(−z) +
(1−y)·softplus(z)`, gives a positive batch sum
`0.0094164926558732986`, or about `2.87e-7` per BP row. The rejected
`rate_ce[0]` was **exactly `0.0`**, while the FLP rate was absent for this
minibatch. A finite, confidently correct negative logit can have a very small
positive cross entropy. The zero arose from float32 cancellation in this BCE
implementation, not nonfinite logits, parameters, or divergence.

The fix recomputes BP training CE with that stable identity **only when** the
ordinary BP minibatch mean rounds exactly to zero. It applies the resulting
sum consistently to the network loss, GradNorm BP rate, and epoch training
metric. It does not clamp the value, change the Bernoulli objective, alter
GradNorm equations or reference losses, or change validation scoring. The
[corrected replay](trial1_replay_fold1_stable_ce.json) restored the same
checkpoint and completed fold-1 epoch 76, all 1,863 steps. It produced BP
training CE `0.012773294029797877` and FLP training CE
`1.4208857547468374`. This verifies the immediate failure is repaired; it
does not establish full-trial convergence or model adequacy.

Failure-only diagnostics now include the rejected BP and FLP rates, task
losses, stable BP check, fold, epoch, optimizer step, batch support and target
sum, logit range, and the worker's trial configuration. The controller aborts
and reaps workers after a primary error, without awaiting a dead worker's
`closed` response. Future worker `FloatingPointError`s retain their traceback,
mark their suggested trial `FAIL`, consume one lifetime trial slot, and allow
the next suggestion. Other worker and infrastructure errors stop the study
for inspection. A future finite zero that remains zero after the stable
calculation is still explicitly rejected; it is not silently clamped or
mistaken for a proven nonfinite divergence.

## Recovery limit and next operation

Optuna 5.0.0 forbids changing a finished `FAIL` trial back to `RUNNING` via
its supported storage API. Trial 1 therefore **cannot be continued in place**.
Its 375 records and epoch-75 checkpoints remain intact, but the isolated
epoch-76 replay is not a persisted fold result. No trial state, checkpoint,
original detached run, or trial-0 artifact was edited in this investigation.
The approved lifetime cap remains **20 trial slots**, including failed trial
1. Creating a replacement trial from trial-1 checkpoints would require a
separate, explicit lineage and budget decision; it has not been done.

The original study froze its launch source SHA, so a future controller must
first register this reviewed implementation revision through the repository's
supported Optuna **study user attribute** operation. That operation checks a
clean committed worktree, the same study and frozen cache/preprocessing
identity, both replay reports, and the original checkpoint hash. It leaves
trial 0 and trial 1 states and values untouched. After reviewing this report
and the commit, the precise next **nontraining** operation is:

```bash
uv run python -m wildfire_risk_uncertainty.hpo.operations adopt-bp-ce-fix --outer-study 3
```

That command was **not run** during this investigation. A later authorized
HPO continuation would use a **new detached controller ID**, the existing
SQLite study, and `--trial-limit 20`; the failed original detached controller
cannot be resumed under a different source SHA. With no explicit replacement
protocol, the next suggestion would be trial 2, leaving trial 1 failed.
