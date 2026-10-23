# Phase 7B nested HPO production readiness

**Status at this original prelaunch audit:** ready for a bounded first outer
study after review. Outer-study-3 HPO was subsequently launched; trial 0
completed and trial 1 failed after 75 epochs per fold. See the
[trial-1 numerical investigation](trial_1_numerical_failure.md) for current
status, recovery limits, and the source-revision registration step. The
original audit did not repeat a training benchmark.

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
commitment to one of the five outer studies. The launcher requires an
explicit lifetime `--trial-limit` between 11 and 150; **20 remains the approved
initial cap**. It is stored in the detached configuration and the study's
`trial_budget.initial` study attribute. The current cap and its history share
that one atomically updated study attribute and change only through an explicit
extension command. The study stops when
it reaches its cap. The original runner halted on a failed trial; the
trial-1 repair records future numerical failures as failed trial slots and
continues within the same cap. A graceful stop leaves a RUNNING trial resumable.
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
the previous 30-per-study ceiling of 150 is retained only as an upper guard
for an explicitly approved extension, not a launch target.

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

## Persistence, accounting, and extension

The stable study is `wildfire_outer_3_gradnorm_hpo_v3` in
`artifacts/hpo/outer_3_gradnorm_v3.sqlite3`; detached experiment IDs identify
controller attempts, not new Optuna studies. `trial_limit` is a **lifetime
total** including completed, running, and failed trials. Optuna's per-call
`n_trials` is not used to allocate the production budget. The single controller
holds an exclusive file lock while suggesting and training; an extension holds
the same lock and checks live detached supervisors. A cap increase from 20 to
30 therefore permits **at most 10 new trial numbers**. It does not reset trial
history, retry a FAILED trial, replace an objective, or rerun a completed
trial. An extension requires the original committed source revision, frozen
search and data identities, and the same Python/library/GPU environment. The
recorded budget history preserves initial and later caps. Review any failure
before extension; this interface deliberately refuses extension with FAILED
trials.

Frozen study metadata records the launch commit and clean-worktree flag;
canonical feature order, cache, geographic partition and preprocessing hashes;
Python, PyTorch, CUDA runtime, Optuna, NumPy and W&B versions; platform, CPU
availability, GPU model and memory; search space, seeds, TPE startup setting,
training configuration, GradNorm version, and objective identity. Trial
manifests bind that metadata to the suggested configuration and deterministic
`trial_NNN` artifact directory. Optuna records start/end timestamps, trial
state, parameters, intermediate values, and the final selection value.

The sampler is seeded at 42 with 10 startup trials. After each full suggestion,
both Optuna 5.0.0 TPE and startup random-generator states are stored in study
attributes and restored on reopening. The state is checked against the last
suggested trial. A crash between Optuna's parameter writes and the RNG-state
attribute can leave an ambiguous state; reopening then **stops for inspection**
instead of silently suggesting from a reset stream. Local SQLite plus these
states support reproducible sequential continuation under the pinned software
and unchanged history. Bitwise-identical continuation across library versions,
hardware changes, or asynchronous concurrent trial suggestions is not claimed.
Production trials remain sequential; only their fold training uses two workers.

Each fold's JSON epoch record contains BP and masked FLP training/validation
loss sums, support counts and means, BP/FLP validation excess log loss, the two
TRAIN-climatology reference cross-entropies, individual log skills, their
equal-weight objective, GradNorm diagnostics, optimizer steps, train/validation
time, and sampled process/GPU peaks. Five fold results and their population
standard deviation appear in each common-epoch aggregate. The epoch-100
aggregate alone supplies the Optuna selection value. Epoch 1/4/16/64 values
are diagnostic intermediates, with `NopPruner`. BP elevated-risk metrics are
retained in the earlier adequacy report; adding them to every HPO epoch would
require extra inference and is outside this execution audit. The scheduler is
absent in the frozen AdamW training setup. Local records are authoritative;
W&B mirrors them once per fold epoch and common epoch.

Checkpoint files use atomic replacement and SHA-256 sidecars. They include
model, optimizer, GradNorm and GradNorm optimizer, Python/NumPy/Torch CPU/CUDA
RNG state, epoch and step, plus cache/preprocessing/trial/fold identity. Each
two-worker epoch saves and records folds in batches of at most two. If three
fold records are valid when interrupted, resume restores and skips those three;
it trains only the missing folds and constructs the aggregate after all five
valid records exist. An incomplete in-flight fold may repeat from its last
epoch checkpoint. Missing, corrupt, or mismatched checkpoints stop recovery;
they are never silently accepted. Retention keeps the recent two checkpoints
per fold rather than every model snapshot. A FAILED Optuna trial is not
automatically retried. Graceful stop leaves a RUNNING trial with a resumable
local `PAUSED` record; hard termination can leave a RUNNING trial whose latest
valid common or fold records are inspected on resume.

Trial CSV export includes trial number, state, timestamps, configuration,
objective, five fold values/dispersion, epoch/fold count, training/validation/
checkpoint seconds, sampled process/GPU peaks, and failure text. Study status
reports counts by state, best completed objective, measured compute, remaining
slots, observed time per completed fit, and detached-run resource extrema. Its
unfinalized-compute total
includes resumable work and should not be interpreted as wasted GPU time.
Optuna trial wall time includes
setup, checkpointing, W&B and pauses; it is not a pure compute measure.
Detached `resource_samples.csv` retains one-second device/RAM/swap samples,
while its status file retains peak device VRAM, peak worker RSS, minimum
available RAM, maximum swap growth, and resource-gate errors. The gate records
its triggering sample and thresholds as a structured event. The worker RSS
sample includes the detached controller's nested fold-worker process tree;
shared mappings can be counted more than once, so host available RAM remains
the stronger overall memory gate. Data loading is
included in epoch training time and is not separately timed. Checkpoint time
is measured at the parent barrier; per-fold allocation within a batch is an
equal share. Resource samples are process-wide and cannot be attributed exactly
to a specific overlapping fold or trial. Setup overhead is inferred from wall
time minus timed work; no precise component timer is claimed. Hard termination
can waste an unfinished epoch or checkpoint write; the amount of that in-flight
work is not fully measurable from completed-epoch records.

### Operator commands

Run from a clean committed `dev` checkout. The initial launch below is
historical; consult the trial-1 investigation before any further launch.

```bash
# Initial approved study: exactly 20 lifetime trial slots.
uv run python scripts/benchmark_07a_detached.py launch --kind gradnorm_hpo --outer-study 3 --trial-limit 20 --gradnorm-objective-reviewed

# Use the returned detached experiment ID for controller status and stop/resume.
uv run python scripts/benchmark_07a_detached.py status <experiment-id>
uv run python scripts/benchmark_07a_detached.py stop <experiment-id>
uv run python scripts/benchmark_07a_detached.py resume <experiment-id>

# Read-only Optuna study progress and individual trial results.
uv run python -m wildfire_risk_uncertainty.hpo.operations status --outer-study 3
uv run python -m wildfire_risk_uncertainty.hpo.operations results --outer-study 3
uv run python -m wildfire_risk_uncertainty.hpo.operations export --outer-study 3 --output reports/hpo/07b/outer_3_trials.csv

# Only after the controller has stopped and a larger cap is separately approved:
uv run python -m wildfire_risk_uncertainty.hpo.operations extend --outer-study 3 --new-limit 30
uv run python scripts/benchmark_07a_detached.py launch --kind gradnorm_hpo --outer-study 3 --trial-limit 30 --gradnorm-objective-reviewed
```

For an interrupted trial, inspect study status and local worker logs, then
resume the same detached experiment ID under its original cap. If a completed
20-trial controller is extended, launch a new detached controller attempt with
the new cap; it opens the **same SQLite study and study name**. Never edit the
SQLite database, trial JSON, or budget attributes by hand. Exporting into a
tracked `reports/` path is an explicit operator action and should be committed
with the run's provenance if used for publication.
