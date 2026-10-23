# Outer-study-3 HPO extension fingerprint repair

## Cause

The approved BP CE correction was registered in study user attribute
`numerical_revision` at source
`e458da90488b1aee866a095a5e782b8fda162e5e`. The existing `extend`
command nevertheless rejected `--new-limit 21`. Its comparison built the
current fingerprint with `metadata()` alone, whereas the production HPO runner
had appended five fields before freezing the study:

| Omitted by old `extend()` | Stored in frozen study |
| --- | --- |
| `cache_version` | `hpo-canonical-row-aligned-v1` |
| `cache_manifest_sha256` | `7e3089929d53173b0181ba7245de9e228929dad38d3c4567d2cf5d154bbc196f` |
| `partition_offsets_sha256` | `d28e0edf101606a06324bf869cb64646053948845fcb57bb3d139f0da5db9423` |
| `inner_fold_constant_references` | Five frozen fold-specific BP/FLP climatologies |
| `preprocessing_hashes_by_fold` | Five frozen preprocessing digests |

The other two differences were the **registered and approved** source SHA and
`bp_ce_numerical_revision` marker. A read-only reconstruction of the full
fingerprint during edits matched every other frozen field except the transient
`worktree_clean_at_launch=True` versus `False` difference. There was no
unrelated cache, partition, feature, preprocessing, reference, runtime, or
methodology change. No alternative supported extension operation bypassed
the deficient comparison.

## Repair and controls

`extend` and revision registration now rebuild the **same complete
fingerprint** as the production runner, including independently recalculated
inner-fold references. The normal exact comparison remains strict. A source
revision is accepted only when its registered chain matches the original
study source, the approved BP CE source, the extension repair marker, the
current commit, and all other frozen fields. Cache and preprocessing identities
are rechecked; the Optuna sampler state is validated before the budget is
changed. No trial or sampler attribute is modified by registration or
extension, except that extension atomically updates `trial_budget`.

Because this repair changes source SHA again, `adopt-extension-repair` is an
explicit, auditable, **nontraining** registration step. It requires a clean
committed worktree, no active controller, the approved numerical revision,
the original trial states `COMPLETE` and `FAIL`, cap 20, valid sampler state,
complete matching provenance, and a Git diff from the approved BP CE commit
restricted to this HPO infrastructure, its tests, and this report. It writes
one study user attribute, `extension_repair_revision`, with the new source SHA,
approved ancestor, changed paths, original states and cap. It does not rewrite
the original `frozen_metadata` or either finished trial.

With exactly one existing failed attempt, cap 21 gives 19 remaining attempt
slots. If all 19 complete, the study will have **20 completed trials and one
failed trial**. Any further failed attempt would reduce the completed count
under this cap and require a separately approved extension. The failed trial 1
is never silently retried.

## Operator sequence after review

Run from the committed clean `dev` worktree. The commands below were **not
executed** during this repair. Registration and extension use the same study
controller lock and reject an active detached controller.

```bash
# Register this reviewed infrastructure-only source revision.
uv run python -m wildfire_risk_uncertainty.hpo.operations adopt-extension-repair --outer-study 3

# Increase the existing study's lifetime attempt cap from 20 to 21.
uv run python -m wildfire_risk_uncertainty.hpo.operations extend --outer-study 3 --new-limit 21

# Inspect the persisted cap and unchanged trial states before starting work.
uv run python -m wildfire_risk_uncertainty.hpo.operations status --outer-study 3

# In a separate authorized operation, start a new detached controller on the same study.
uv run python scripts/benchmark_07a_detached.py launch --kind gradnorm_hpo --outer-study 3 --trial-limit 21 --gradnorm-objective-reviewed
```

The failed original detached controller ID is tied to the old source SHA and
cannot be resumed with this commit. The new launch creates a controller ID,
not a new Optuna study. It should use existing trials 0–1 and begin with trial
2. No outer TEST access, HPO training, cap change, or manual database change
was performed in this repair task.
