# Phase 7B nested HPO production readiness

**Status:** ready for a bounded first outer study after review; no HPO has been
launched under the frozen GradNorm and relative-log-skill objective. This audit
uses committed implementation, frozen reports, inexpensive tests, and read-only
hardware inspection. It does not repeat a training benchmark.

## Frozen search and selection

Each of five outer TRAIN populations has its own Optuna study. Each trial trains
the same architecture on that outer study's five inner TRAIN folds and scores
only their geographically held-out inner VALIDATION folds. The shared network
searches depth `{2, 3}`, width `{64, 128, 256}`, AdamW learning rate
`[1e-4, 1e-2]` log-uniform, AdamW weight decay `[1e-5, 1e-2]` log-uniform,
and batch size `{32768, 65536, 131072}`. The 33 inputs, preprocessing,
supports, masks, partitions, seed 42, BF16 inference/training, 100 epochs,
and reference-aligned GradNorm remain frozen. No outer TEST row is selected by
the training or reference-calculation paths. In particular, the reference scan
now selects eligible outer TRAIN row indices **before** reading BP or FLP
targets; a guarded regression test fails if it reads an outer TEST target.

The [frozen objective](../../../docs/methodology/checkpoint_07b_hpo_objective.md)
is the equal BP/FLP mean of validation relative logarithmic skills, then the
equal mean over five inner folds. Optuna **maximizes** that score at epoch 100.
The existing TPE sampler is a literature-supported approach already adopted in
[Checkpoint 6A](../../../docs/methodology/checkpoint_06a_training_formulation.md),
which cites its use for an Earth-system surrogate by Lu and Ricciuto (2019).
Spatial tuning within outer TRAIN follows the previously cited Schratz et al.
(2019) precedent. These citations support the search and spatial evaluation
approach; none determines the trial count. With pinned Optuna 5.0.0, the
explicitly recorded `n_startup_trials=10` yields 10 initial random trials per
study before TPE uses completed results. Trials remain sequential (`n_jobs=1`);
within a trial, two persistent CUDA workers own fold sets `(1, 3, 5)` and
`(2, 4)`, with a five-fold epoch barrier. The new study uses `NopPruner`: epoch
1, 4, 16, and 64 reports are diagnostics, and `should_prune()` never rejects
a trial. The [06F fidelity result](../../training/06f_fidelity_calibration/README.md)
rejected epoch-4 pruning; its old ASHA study and records remain historical.

## Bounded production scope

**Proposed first launch: outer study 3, exactly 20 full 100-epoch trials.**
This is a project compute-budget choice. It gives the pinned TPE sampler 10
startup trials and at most 10 adaptive suggestions while limiting the first
commitment to one of the five outer studies. The launcher now requires an
explicit `--trial-limit` between 11 and the historical 30-trial planning
ceiling; `20` is stored in the detached configuration and immutable study
metadata. The study stops when it reaches its cap. A failed trial halts the
study for inspection, and a graceful stop leaves the current trial resumable.
The remaining four outer studies are **not** authorized by this initial scope;
their same-budget execution should follow a review of first-study runtime,
resource telemetry, failed-trial status, and the scientific adequacy of the
resulting candidate set. Review progress after each 100-epoch trial, especially
after the tenth startup trial. No interim epoch is used to rank or reject
models.

The 20-trial limit explores only a small part of 18 categorical
depth/width/batch combinations plus two continuous optimizer dimensions.
TPE may repeat categorical combinations. This is a bounded applied-science
search, not evidence that the hyperparameter optimum has been found. Five
independent studies at 20 trials each would require 100 full evaluations;
the previous 30-per-study ceiling of 150 is retained only as a hard upper
guard, not a launch target.

## Time and resource estimate

| Evidence | Measured value | Use and limitation |
|---|---:|---|
| Phase 7A 3×128, batch 65,536, one GradNorm fold, epochs 4–100 | 2,575.8 s total; 1,918.5 s training, 298.6 s validation, 357.1 s extra BP diagnostic | The HPO path omits the extra BP diagnostic; one-fold training plus validation averaged 22.9 s/epoch. It does not measure five-fold concurrency. |
| 06F two-worker equal-weight four-epoch barrier | 98.2–98.5 s/five-fold epoch | Exact fold-equivalence and 1.60× speedup were measured; this predates GradNorm. |
| 06F 11 configurations, epochs 5–16 | Batch 32,768: median 96.6–103.9 s/epoch; batch 65,536: 73.7–81.3 s/epoch | Covers selected depths 2–3 and widths 64–256, with the historical equal-weight loss. No comparable long trajectory for batch 131,072. |
| Historical no-pruning HPO projection | 482.8 h / 150 = 3.22 h per trial | Derived before GradNorm and two-worker full-fidelity runtime; planning cross-check only. |
| Historical 3×64, batch 32,768 completed 100-epoch trial | 4.53 h wall time | Shows long-run overhead can exceed a short-epoch extrapolation. |

The measured inputs are traceable to the [Phase 7A full-fold review](../../training/07a/gradnorm_full_fold_review.md),
[06F concurrency validation](../../training/06f_resource_optimization/final_validation/README.md),
[06F fidelity calibration](../../training/06f_fidelity_calibration/README.md),
and [prelaunch projection](../../training/06f_prelaunch/projected_hpo_runtime.json).

The **planning envelope is 2–6 wall hours per full five-fold trial** on the
single local GPU, not a measured bound. Twenty trials therefore imply
**40–120 hours** (about 1.7–5 days) for the first study; the conditional
five-study, 100-trial plan implies **200–600 hours** (about 8–25 days).
Architecture and batch size matter: batch 32,768 has about twice as many
optimizer steps as 65,536, while wider/deeper models may increase compute and
memory. The 06F short-run timings show the 32,768 candidates generally slower
than 65,536; the 131,072 candidates have no comparable 16- or 100-epoch
two-worker evidence. GradNorm overhead, page-cache behavior, W&B/network
latency, pauses, and unmeasured configurations can push actual duration beyond
the envelope. The trial count is the enforceable compute bound; wall time is
an estimate that must be updated from completed trials.

The audited host has an RTX 5080 with 16,303 MiB VRAM, 30 GiB RAM, 8 GiB
swap, and 896 GiB currently free on the workspace volume. The frozen mmap
cache uses about 18 GiB disk. In 06F the two-worker run peaked at 2,704 MiB
sampled device VRAM, retained at least 25.24 GiB available RAM, and showed
negligible swap; a three-worker run was rejected by the VRAM gate. The Phase 7A
one-fold continuation peaked at 2,837 MiB sampled device VRAM and 14.41 GiB
worker RSS. Its one-second GPU samples had median 10% utilization; they do
not isolate GPU, data, and host bottlenecks or predict two-worker utilization.
The detached supervisor stops on sampled device VRAM above 15,500 MiB,
available RAM below 2 GiB, or swap growth above 128 MiB. Larger search
configurations have not been verified end to end, so those gates remain
necessary. Checkpoints retain at most the recent per-fold epochs; local raw
records and SQLite are authoritative, and W&B is an online mirror.

## Launch and operations (commands for later approval)

Run from the repository root on a **clean committed `dev` worktree**. These
commands are documentation; none was executed in this readiness audit.

```bash
# Start only the approved first outer study, detached.
uv run python scripts/benchmark_07a_detached.py launch --kind gradnorm_hpo --outer-study 3 --trial-limit 20 --gradnorm-objective-reviewed

# Substitute the returned 07a-... experiment ID below.
uv run python scripts/benchmark_07a_detached.py status <experiment-id>
uv run python -m wildfire_risk_uncertainty.hpo.status --outer-study 3 --gradnorm
uv run python scripts/benchmark_07a_detached.py results <experiment-id>

# Request a graceful pause after the active five-fold common epoch.
uv run python scripts/benchmark_07a_detached.py stop <experiment-id>

# Resume the same study, trial, and checkpoint states after inspection.
uv run python scripts/benchmark_07a_detached.py resume <experiment-id>
```

The detached process checks online W&B authentication before starting HPO.
The parent tracks PID/start-time identity, worker logs, one-second resource
samples, and a stop-request file. At each common epoch it saves hash-sidecar
checkpoints, fold records, and an aggregate record; exact cache,
preprocessing, model, optimizer, GradNorm, RNG, and trial/fold provenance is
checked on resume. An interruption before a completed common epoch resumes
from its last persisted checkpoints without treating a partial epoch as a
result. The SQLite study and local trial records retain configurations,
component scores, references, maximum-resource score, and run identity.
W&B availability and the physical resource gates remain operational risks;
they do not alter the frozen scientific score.
