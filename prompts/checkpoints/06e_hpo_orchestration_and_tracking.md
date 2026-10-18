Checkpoint 6E — Implement and validate HPO orchestration, full experiment
tracking, and training visualizations

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

Required base commit:
  d719dfafbbbe0475960e19a13149fa5d255ebcb0

PREFLIGHT

Before changing anything:

1. Verify branch == dev.
2. Verify HEAD == required base commit.
3. Verify origin/dev == required base commit.
4. Verify working tree is clean.
5. Verify frozen Checkpoints 5B and 6A–6D are present.
6. Verify canonical, 5B, and 6C local artifacts are present.
7. Verify the 6D reusable training pipeline passes its existing tests.
8. Verify CUDA/PyTorch still sees the RTX 5080.
9. Do not modify main.
10. Save this exact prompt verbatim as:

    prompts/checkpoints/06e_hpo_orchestration_and_tracking.md

PURPOSE

Implement and validate the complete orchestration layer required for the
real Checkpoint 6F nested HPO experiment:

  Optuna TPE
      +
  ASHA-style SuccessiveHalving pruning
      +
  five-fold spatial aggregation
      +
  reusable 6D training pipeline
      +
  persistent local experiment records
      +
  Weights & Biases observability
      +
  reproducible training/validation visualizations

Checkpoint 6E validates the INFRASTRUCTURE.

It must NOT perform the real 30-trial × 5-outer-study HPO search.

The real HPO remains Checkpoint 6F.

DO NOT:

- perform a 30-trial real-data study
- select scientific hyperparameters
- inspect outer TEST performance
- generate OOF predictions
- train final outer models
- train an ensemble
- run SHAP
- change the frozen HPO search space
- change the frozen loss
- change the frozen spatial folds
- change the frozen data traversal
- change the frozen preprocessing contract
- allow W&B Sweeps to control optimization
- commit or push before HITL review

FROZEN DESIGN TO IMPLEMENT

From 6A / 6B:

Sampler:
  Optuna TPESampler(seed=42)

Pruner:
  Optuna SuccessiveHalvingPruner

  min_resource = 1
  reduction_factor = 4
  min_early_stopping_rate = 0

Resource unit:
  epoch

Pruning comparison milestones:
  1, 4, 16, 64 epochs

Maximum resource:
  100 epochs

At each pruning milestone e:

  all five inner spatial folds for the trial must first reach epoch e

then compute:

  J_e(theta)
    = (1/5) * sum_f validation_L_tune(f, e)

Only J_e(theta) may be sent to:

  trial.report(J_e, step=e)

and only then may:

  trial.should_prune()

be evaluated.

Never prune from one geographic inner fold alone.

Search space:

  hidden depth:
    {2, 3}

  shared hidden width:
    {64, 128, 256}

  learning rate:
    log-distributed [1e-4, 1e-2]

  AdamW weight decay:
    log-distributed [1e-5, 1e-2]

  batch size:
    {32768, 65536, 131072}

Fixed:
  ReLU
  AdamW
  dropout = 0
  BF16 autocast
  33 frozen inputs
  frozen BP/FLP losses
  0.5 / 0.5 task weighting

Planned 6F budget:
  30 trials per outer study
  5 independent outer studies
  150 trials total
  n_jobs = 1 within each study

Do not execute that budget in 6E.

DEPENDENCIES

Add project dependencies required for:

- Optuna
- W&B
- plotting/visualization if not already available

Use stable versions compatible with the existing Python/PyTorch
environment and pin them through uv / uv.lock.

Do not add W&B Sweeps as an optimizer.

Prefer direct W&B SDK logging under our control rather than depending on
a generic Optuna-W&B callback because the project requires fold-level,
epoch-level, raw-component, and aggregate audit records.

OPTUNA STORAGE

Implement persistent study storage under ignored artifacts, e.g.:

  artifacts/hpo/

Use a persistent Optuna RDB storage appropriate for:

- one worker per study
- process restart
- study resume
- inspection after failure

SQLite is acceptable for the frozen single-worker local design.

Use explicit study names such as:

  wildfire_outer_1_hpo_v1
  ...
  wildfire_outer_5_hpo_v1

Do not overwrite an existing study silently.

Study creation must support:

  load_if_exists=True

with strict verification that frozen study metadata/search-space version
matches before continuation.

Persist study metadata including:

- outer study ID
- git SHA
- HPO schema/version
- seed
- sampler type/config
- pruner type/config
- search-space definition/hash
- maximum resource
- inner-fold artifact provenance
- 33-input hash
- training implementation version

A study with incompatible metadata must fail rather than resume.

OPTUNA RESUME SEMANTICS

Validate that process restart:

- preserves all completed/pruned/failed trial records
- does not duplicate trial numbers
- does not rerun completed trials
- allows continued valid TPE optimization
- preserves the realized trial history

Do not claim that recreating TPESampler after process restart guarantees
a bitwise-identical sequence of future suggestions unless this is
actually demonstrated.

The scientific record is the realized persisted trial sequence.

Every suggested configuration must be persisted before training begins.

LOCAL EXPERIMENT RECORD IS AUTHORITATIVE

W&B is not the sole record of the experiment.

For every study/trial/fold/epoch, persist local machine-readable records
under ignored artifacts/hpo/... sufficient to reconstruct:

- configuration
- training trajectory
- validation trajectory
- all objective calculations
- pruning calculations
- trial state
- timings
- provenance

Use append-safe / atomic writes where practical.

The repository's compact reports may summarize these records, but large
trial logs remain ignored.

FULL METRIC ACCOUNTING

Track all quantities used to derive training, validation, spatial-CV,
and pruning objectives.

FOR EVERY FOLD / EPOCH, record at minimum:

Identity:
- outer_study
- trial_number
- inner_fold
- epoch
- resource milestone status
- global optimizer step

Training support/calculation:
- train_rows
- train_bp_count
- train_flp_valid_count
- train_bp_loss_sum
- train_flp_loss_sum
- train_bp_loss_mean
- train_flp_loss_mean
- train_equal_task_objective

Validation support/calculation:
- validation_rows
- validation_bp_count
- validation_flp_valid_count
- validation_bp_loss_sum
- validation_flp_loss_sum
- validation_bp_loss_mean
- validation_flp_loss_mean
- validation_equal_task_objective

Optimization/config:
- learning_rate
- weight_decay
- batch_size
- depth
- width
- precision mode

Performance:
- training_wall_seconds
- validation_wall_seconds
- training_rows_per_second
- validation_rows_per_second
- peak_allocated_vram
- peak_reserved_vram where available
- peak_process_rss where available

Numerical diagnostics:
- all-finite status
- optimizer step count

If inexpensive and non-invasive, also record a clearly defined global
gradient-L2 diagnostic during training, but do not add material runtime
overhead or gradient clipping merely for tracking.

Do not record arbitrary metrics that materially slow HPO.

RAW SUMS AND COUNTS ARE REQUIRED.

The reported means must be recomputable exactly from:

  loss_sum / support_count

and the equal-task objective must be recomputable exactly from the two
task means.

TRIAL-LEVEL SPATIAL AGGREGATION

At every completed common resource level e across all five inner folds,
record:

- each fold's validation BP loss
- each fold's validation FLP loss
- each fold's validation equal-task objective
- five-fold mean validation BP loss
- five-fold mean validation FLP loss
- J_e = five-fold mean equal-task objective
- five-fold standard deviation of equal-task objective
- minimum fold objective
- maximum fold objective
- resource epoch
- Optuna pruning decision
- trial state after the decision

Only:

  J_e

is the Optuna pruning/optimization objective.

Dispersion statistics are diagnostics and must not influence pruning.

PRUNED TRIALS

When a trial is pruned:

- persist the resource level at which pruning occurred
- persist all five fold metrics used in that decision
- persist J_e
- persist pruning state before raising TrialPruned
- close/finalize W&B tracking cleanly
- do not mark it failed

When a trial fails due to an exception:

- distinguish FAILED from PRUNED
- preserve exception type/message and completed metrics
- do not silently retry with changed hyperparameters

W&B ROLE

Weights & Biases is the online observability/provenance mirror.

It does NOT:
- choose hyperparameters
- run a Sweep
- make pruning decisions
- change Optuna state
- change the optimization objective

Use one W&B run per Optuna trial.

Recommended organization:

  project:
    wildfire-risk-uncertainty

  group:
    outer-<j>

  job_type:
    hpo-trial

  run name:
    outer-<j>-trial-<NNN>

Use tags such as:
  checkpoint-6e or checkpoint-6f
  hpo
  tpe
  asha
  outer-<j>

Record in W&B config:

- complete hyperparameter configuration
- outer study
- Optuna study name
- Optuna trial number
- seed
- git SHA
- feature-order hash
- preprocessing/data provenance hashes
- search-space version/hash
- sampler/pruner config
- model parameter count
- software environment where practical

Record W&B run ID locally so the local trial record and online run are
bidirectionally identifiable.

W&B AUTHENTICATION PREFLIGHT

After installing W&B:

1. Detect whether online W&B authentication is available.
2. Never print an API key.
3. Never write an API key into:
   - repository files
   - reports
   - logs
   - manifests
   - command-line arguments
4. Do not commit W&B credential/config files.

If no authenticated online W&B session / WANDB_API_KEY is available:

  STOP for HITL and tell the user to authenticate securely.

Do not ask Codex to receive the API key in chat.

After authentication, verify a minimal online test run can be created
and finalized.

Use offline mode only as an explicitly tested fallback capability, not
as a substitute for the requested online 6E tracking validation.

W&B FAILURE BEHAVIOR

A transient W&B failure must not corrupt Optuna or the local
authoritative experiment record.

Implement explicit behavior:

- local metrics are written first or otherwise safely preserved
- W&B logging failure is surfaced clearly
- trial scientific calculations remain available locally
- do not silently discard tracking errors

For real 6F execution, whether a W&B outage should abort or continue
must be configurable.

Default proposed behavior:
  continue trial from the authoritative local record while marking
  W&B sync state as degraded/pending.

Do not change a trial's scientific objective because W&B is unavailable.

TRAINING / VALIDATION CURVES

Implement reusable visualization code.

For every completed or partially completed trial, support generation of:

1. Combined objective curve
   x = epoch
   y =
     train_equal_task_objective
     validation_equal_task_objective

2. BP loss curve
   x = epoch
   y =
     train_bp_loss_mean
     validation_bp_loss_mean

3. FLP loss curve
   x = epoch
   y =
     train_flp_loss_mean
     validation_flp_loss_mean

Generate these PER INNER FOLD.

Also generate trial-level spatial-CV figures:

4. Inner-fold validation objective overlay
   - five lines, one for each inner fold

5. Spatial mean validation curve
   - five-fold mean J_e at common resource epochs
   - include a representation of geographic fold dispersion
     (e.g. standard deviation band or min/max band)

6. Train-vs-validation spatial summary
   - comparable aggregated train and validation trajectories where
     mathematically defined

7. ASHA resource/pruning trace
   - resource milestone
   - J_e
   - prune/continue state

For study-level HPO results, implement functions for later 6F use:

8. Trial objective history
9. Best-so-far objective history
10. Trial duration / resource consumed
11. Completed vs pruned vs failed trial counts
12. Hyperparameter slice plots
13. Parallel-coordinate visualization
14. Hyperparameter importance visualization where statistically
    meaningful after enough completed trials

Do not infer importance from the tiny 6E smoke study.

Do not produce misleading importance figures when sample size is
insufficient.

LOCAL FIGURES

Generate deterministic static figures using the project's normal plotting
stack under:

  reports/hpo/06e/figures/

for the 6E validation run.

Figures for 6F should later be generated under its own report directory.

Use clear titles, axes, legends, trial/fold identifiers, and provenance.

Do not rely solely on screenshots from W&B.

W&B CURVES

Mirror the same key training/validation quantities to W&B so the hosted
UI provides interactive curves.

Because one W&B run represents one trial containing five sequential
inner-fold trajectories, define fold-specific epoch axes/step metrics
cleanly rather than depending on one ambiguous global W&B step.

For example conceptually:

  fold_1/epoch
  fold_1/train/objective
  fold_1/validation/objective

  ...

and:

  spatial_cv/resource_epoch
  spatial_cv/validation_objective_mean
  spatial_cv/validation_objective_std

Ensure W&B charts display correct epoch values even though inner folds
may be trained sequentially.

Do not allow W&B step handling to overwrite or reorder fold histories.

W&B SUMMARY

At trial completion/pruning, populate summary values such as:

- trial_state
- final_resource_epoch
- objective_at_last_common_resource
- best_observed_common_resource_objective
- total_runtime
- total_training_rows_processed
- total_validation_rows_processed
- prune_epoch if applicable
- W&B/local record identifiers

Do not label a pruned intermediate objective as a 100-epoch final result.

OPTUNA / W&B LINKAGE

Optuna remains source of truth for:

- trial number
- parameter suggestion
- trial state
- intermediate J_e values
- pruning decision
- final objective

W&B mirrors and visualizes those facts.

Validate for each smoke trial that:

  Optuna intermediate values
  ==
  local trial record J_e
  ==
  W&B logged spatial_cv objective

within serialization precision.

PERSISTED PREPROCESSING

Each inner fold must fit preprocessing only from its INNER TRAIN
population exactly as frozen.

For HPO efficiency, it is acceptable to cache deterministic preprocessing
state for a given:

  outer study + inner fold + frozen feature contract

because preprocessing does not depend on HPO hyperparameters.

If implemented:

- derive/cache it only from INNER TRAIN
- store provenance/hash
- verify repeated computation matches
- never cache from validation or outer TEST
- fail on provenance/hash mismatch

This optimization must not change scientific behavior.

6E SYNTHETIC ORCHESTRATION TEST

Implement a fast deterministic synthetic HPO fixture that exercises:

- TPESampler
- SuccessiveHalvingPruner
- persistent Optuna storage
- trial.report()
- trial.should_prune()
- completed trials
- pruned trials
- local metric persistence
- W&B logging
- W&B run finalization
- process/study resume
- plotting
- state reconciliation

Use enough synthetic trials to guarantee meaningful code-path coverage,
but do NOT treat their numerical results as scientific evidence.

Design the synthetic objective so that both:
  COMPLETE
and
  PRUNED
trial states occur deterministically.

Test all intended ASHA milestones without expensive real training.

REAL-PIPELINE TRACKING SMOKE TEST

After synthetic orchestration passes, run exactly ONE fixed real-data
smoke trial.

Use prospectively:

  outer study = 3

Use the same fixed integration hyperparameters as 6D:

  depth = 2
  width = 128
  learning rate = 1e-3
  weight decay = 1e-4
  batch size = 65536

This is NOT an Optuna-selected configuration.

Exercise all FIVE frozen inner folds.

Run exactly ONE epoch per inner fold.

For each fold:

- fit/load correctly provenance-checked INNER-TRAIN preprocessing
- train one complete epoch
- validate one complete pass
- persist all raw sums/counts/means
- log W&B training and validation metrics
- generate fold curves

After all five folds complete epoch 1:

- compute J_1
- compute fold dispersion
- persist aggregate record
- log aggregate W&B metrics

Do NOT use J_1 to make a scientific model-selection decision.

This real smoke trial exists only to validate five-fold orchestration,
aggregation, tracking, and visualization.

Do not continue it to epoch 4.

Do not compare it with alternative real-data hyperparameters.

OPTIONAL OPTUNA WRAPPING OF REAL SMOKE

If useful to validate exact plumbing, the single fixed real-data trial
may be represented as a fixed Optuna trial / enqueued configuration.

It must not permit TPE to choose an alternative configuration.

It must not constitute HPO.

REPORT CONSISTENCY VALIDATION

After the real smoke trial, independently recompute from persisted raw
records:

For every fold:
- BP mean = BP loss sum / BP count
- FLP mean = FLP loss sum / FLP-valid count
- equal-task objective = 0.5 BP mean + 0.5 FLP mean

Across five folds:
- J_1 = arithmetic mean of five validation equal-task objectives
- standard deviation
- min
- max

Compare independently recomputed values against:

- training orchestration state
- Optuna intermediate value if used
- W&B logged values
- report tables

Require agreement within documented serialization precision.

TRACKING REPORTS

Create:

  reports/hpo/06e/
    README.md
    environment.json
    tracking_schema.json
    search_space.json
    study_storage_validation.json
    synthetic_orchestration_validation.json
    real_smoke_trial_summary.json
    metric_reconciliation.json
    wandb_validation.json
    figures/

The README must explain:

- authoritative local record vs W&B mirror
- hierarchy:
    study -> trial -> inner fold -> epoch
- how J_e is calculated
- how pruning works
- how training/validation curves are generated
- how resume works
- what was and was not executed in 6E

Do not report the one-epoch real smoke loss as scientific performance.

SOURCE CODE

Build on the frozen 6D reusable training implementation.

Suggested organization:

  src/wildfire_risk_uncertainty/hpo/
    config.py
    storage.py
    objective.py
    study.py
    tracking.py
    wandb_tracking.py
    records.py
    visualization.py
    cli.py

Reuse:

  src/wildfire_risk_uncertainty/training/

Do not create a second trainer.

CLI

Provide explicit commands or modes for:

- synthetic orchestration validation
- real tracking smoke test
- future real HPO execution

The real-HPO command must require explicit outer-study selection and
must NOT default to launching all five studies accidentally.

Checkpoint 6E must not invoke the real-HPO mode.

TESTS

Add comprehensive tests for at least:

OPTUNA
- exact frozen search-space distributions
- deterministic seed configuration
- correct pruner configuration
- exactly J_e reported at common five-fold resource levels
- no one-fold pruning
- correct COMPLETE / PRUNED / FAIL states
- persistent study resume
- incompatible metadata rejection
- no duplicate completed trials on resume

METRICS
- raw sum/count -> task mean calculation
- task means -> equal-task objective
- five-fold objective mean
- fold standard deviation/min/max
- serialization/reload equality
- pruned-resource accounting

W&B
- tracker disabled mode
- offline/mock mode
- online adapter without exposing credentials
- correct run naming/grouping/config
- correct fold-specific epoch axes
- aggregate spatial-CV metrics
- correct summary/final state
- logging failure leaves local record intact

VISUALIZATION
- curve generation from deterministic fixture
- missing later epochs for pruned trials handled correctly
- no fabricated interpolation after pruning
- output files deterministic where practical

REAL PIPELINE
- frozen 33 inputs
- fold-specific preprocessing provenance
- outer TEST cannot enter HPO loaders
- 6D trainer reused
- one-fold metrics match reusable validator

Run:

  full pytest suite
  Ruff
  git diff --check

REPRODUCIBILITY

Record:

- git SHA
- Python/PyTorch/CUDA
- Optuna version
- W&B version
- seed
- search-space hash
- sampler/pruner configuration
- study schema/version
- 33-input hash
- 5B/6C provenance
- visualization configuration

Synthetic orchestration should reproduce deterministically in the pinned
environment.

Real-data smoke metrics should reproduce according to the already
validated 6D deterministic contract.

HITL STOP CONDITIONS

STOP if:

- W&B online authentication is unavailable after dependencies are installed
- W&B requires exposing credentials in repository/log output
- Optuna cannot persist/resume cleanly
- realized Optuna state disagrees with local records
- W&B metrics disagree with local records
- five-fold J_e cannot be independently reproduced
- trial pruning occurs before all five folds reach a common resource level
- implementation would require changing the frozen 6A–6D methodology
- outer TEST data enter any HPO calculation
- tracking materially alters the training trajectory
- a new scientific hyperparameter decision is required

Do not solve such a stop condition by changing methodology silently.

FINAL HITL REPORT

When complete, report:

1. Base commit SHA
2. Added dependency versions
3. W&B authentication status without revealing credentials
4. W&B entity/project/run identifiers or URLs for the validation runs
5. Exact Optuna sampler/pruner configuration
6. Persistent-storage/resume validation result
7. Synthetic COMPLETE/PRUNED trial results
8. Real smoke trial configuration
9. Exact five inner-fold row/support counts used
10. Per-fold epoch-1:
    - train BP/FLP/objective
    - validation BP/FLP/objective
    - raw loss sums/counts
    - runtimes/throughput
11. J_1 and fold dispersion
12. Independent metric reconciliation result
13. W&B/local/Optuna agreement
14. Training/validation curve files generated
15. W&B interactive curves confirmed
16. Tracking overhead / any operational caveat
17. Study-resume behavior
18. Full pytest status
19. Ruff status
20. git diff --check status
21. Files changed
22. Confirmation no real HPO/model selection occurred
23. Confirmation no outer TEST evaluation occurred

Then STOP for HITL review.

Do not execute the 30-trial studies.
Do not begin Checkpoint 6F.
Do not commit.
Do not push.
