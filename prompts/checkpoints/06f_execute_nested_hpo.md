Checkpoint 6F — Execute frozen nested spatial HPO and select one
configuration per outer fold

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

Required base commit:
  fb1cf795983a34e5e81b3a4bc662751a7a358235

PURPOSE

Execute the actual frozen hyperparameter-selection experiment established
by Checkpoints 6A–6E.

This is the FIRST checkpoint in which real scientific model selection
occurs.

For each of the five frozen outer studies:

  OUTER TRAIN_j
      ->
  five frozen inner spatial folds
      ->
  30-trial Optuna TPE study
      ->
  ASHA-style resource allocation
      ->
  select the best COMPLETE 100-epoch trial

The resulting configuration for each outer study will later be used to
fit that outer fold's scientific model on its complete OUTER TRAIN
population.

Checkpoint 6F does NOT evaluate OUTER TEST.

Do not generate OOF predictions in this checkpoint.

PREFLIGHT

Before changing or executing anything:

1. Verify:
     branch == dev

2. Verify:
     HEAD ==
     fb1cf795983a34e5e81b3a4bc662751a7a358235

3. Verify:
     origin/dev ==
     fb1cf795983a34e5e81b3a4bc662751a7a358235

4. Verify working tree is clean.

5. Verify main and origin/main remain unchanged.

6. Verify frozen Checkpoints:
     5B
     6A
     6B
     6C
     6D
     6E

7. Verify canonical dataset and frozen 5B/6C artifacts are locally
   available.

8. Verify CUDA and the RTX 5080 are available.

9. Verify:
     Optuna == 5.0.0
     W&B == 0.30.0

10. Verify online W&B authentication without displaying or storing
    credential contents.

11. Verify the exact frozen HPO search-space hash and 33-input feature
    hash match the accepted 6E records.

12. Save this exact prompt verbatim as:

    prompts/checkpoints/06f_execute_nested_hpo.md

If any frozen provenance does not match, STOP for HITL.

Do not modify main.

NO METHODOLOGY CHANGES AFTER HPO STARTS

Once the first real 6F trial begins, the experimental protocol is locked.

Do NOT change any setting because of:

- observed trial losses
- observed selected parameters
- W&B curves
- pruning frequency
- apparent convergence
- differences among geographic folds
- one study performing better or worse than another

Operational interruption may be repaired only by restoring the already
frozen experiment.

Do not inspect results from an early outer study and use them to change
later outer studies.

All five studies must execute under the same prospective protocol.

FROZEN SCIENTIFIC DESIGN

There are exactly five independent Optuna studies:

  outer study 1
  outer study 2
  outer study 3
  outer study 4
  outer study 5

For outer study j:

  eligible data =
    frozen OUTER TRAIN_j only

Its frozen five inner spatial folds define:

  INNER VALIDATION_jk =
    inner fold k

  INNER TRAIN_jk =
    other four inner folds

OUTER TEST and OUTER DEAD_ZONE must never enter:

- preprocessing
- training
- validation
- Optuna
- ASHA
- W&B metrics
- model selection

SEARCH SPACE

Use exactly:

  shared hidden depth:
    categorical {2, 3}

  shared hidden width:
    categorical {64, 128, 256}

  learning rate:
    log-uniform [1e-4, 1e-2]

  AdamW weight decay:
    log-uniform [1e-5, 1e-2]

  batch size:
    categorical {32768, 65536, 131072}

Fixed:

  activation = ReLU
  optimizer = AdamW
  dropout = 0
  precision = BF16 autocast
  inputs = frozen 33-column set
  BP loss = frozen soft BCE/log score
  FLP loss = frozen soft categorical log score
  task objective = 0.5 BP + 0.5 FLP

Do not add or remove HPO dimensions.

Do not narrow the search space based on 6D or 6E diagnostics.

SAMPLER

For EACH outer study instantiate independently:

  TPESampler(seed=42)

Use the pinned Optuna 5.0.0 behavior.

Record the complete realized sampler configuration, including Optuna
defaults that are not explicitly overridden.

Do not allow one outer study's trial history to influence another.

PRUNER

Use exactly:

  SuccessiveHalvingPruner(
      min_resource=1,
      reduction_factor=4,
      min_early_stopping_rate=0
  )

Resource unit:
  epoch

Pruning comparison resources:

  1
  4
  16
  64

Maximum training resource:

  100 epochs

FIVE-FOLD PRUNING CONTRACT

For each trial and resource level e:

1. Every one of the five inner spatial folds must reach epoch e.
2. Each fold performs validation using its frozen validation population.
3. Compute each fold's equal-task validation objective.
4. Calculate:

     J_e(theta)
       = (1/5) *
         sum over five inner folds of validation_L_tune(f, e)

5. Persist the five component values and J_e.
6. Report ONLY:

     trial.report(J_e, step=e)

7. Only after that may:

     trial.should_prune()

   be evaluated.

Never prune because of one inner fold.

Do not weight J_e by geographic fold row count.

The five inner-fold objectives receive equal weight exactly as frozen.

A pruned trial terminates after the common pruning resource at which
Optuna prunes it.

A surviving trial continues through the next resource interval.

A trial surviving epoch 64 continues through epoch 100.

At epoch 100:

- evaluate all five folds
- compute J_100
- return J_100 as the trial's final objective
- mark the trial COMPLETE

REAL HPO BUDGET

Exactly:

  30 realized trials per outer study

Therefore:

  5 studies
  × 30 trials
  = 150 trials total

Within each study:

  n_jobs = 1

Run trials sequentially.

A study is complete only when it contains exactly 30 trials belonging
to this frozen execution, with acceptable terminal states.

There must be no unexpected FAILED trials.

If any real 6F trial becomes FAILED due to:

- exception
- nonfinite training
- CUDA error
- storage error
- corrupted checkpoint
- W&B/orchestration inconsistency
- data/provenance mismatch
- any other unexpected condition

STOP for HITL.

Do not silently replace the failed trial with a 31st suggestion.

Do not reinterpret FAILED as PRUNED.

PREPROCESSING CACHE

Before launching real HPO, construct or verify the complete set of:

  5 outer studies × 5 inner folds
  = 25 preprocessing states

Each preprocessing state must be fitted only from that fold's
INNER TRAIN population.

Reuse the frozen 6A/6D preprocessing implementation.

For every cache record verify:

- outer study
- inner fold
- exact INNER TRAIN eligibility
- frozen 33-input ordering
- feature-order hash
- canonical provenance
- 5B provenance
- 6C provenance
- preprocessing implementation/version
- fitted row count
- imputation statistics
- scaling statistics
- cache hash

For any state already produced in 6E:

- verify rather than trust it
- require provenance/hash agreement

A mismatch must STOP for HITL.

Persist the 25-state cache under ignored artifacts.

Create a compact committed 6F preprocessing-cache manifest containing
only provenance, hashes, row counts, and validation status.

Do not refit preprocessing once per trial.

ACTIVE-TRIAL RESTART VALIDATION

Before the first real 6F HPO trial, explicitly validate one additional
engineering property:

An interrupted RUNNING Optuna trial must be recoverable without silently
creating a replacement trial or losing its already-completed fold/epoch
trajectory.

Use a cheap synthetic fixture or another non-scientific bounded
validation.

Test:

1. create a RUNNING trial;
2. advance through at least one persisted fold/resource state;
3. save its orchestration/checkpoint state;
4. terminate the process;
5. restart;
6. reconnect to the same Optuna trial identity;
7. reconnect/resume the same W&B run identity where applicable;
8. restore model/orchestration state;
9. continue the trial;
10. complete or prune it without creating a duplicate trial.

Verify:

- Optuna trial number unchanged
- W&B run ID unchanged
- already completed work not repeated as new scientific observations
- metric record remains append-consistent
- final Optuna state is correct

If the current implementation cannot safely resume an in-progress trial,
STOP for HITL BEFORE launching real HPO.

Do not solve this by marking interrupted trials FAILED and simply
replacing them unless explicitly approved.

TRAINING SEED AND DATA ORDER

Use the frozen project seed:

  42

Use the frozen deterministic 6D data traversal.

For a given fold and epoch:

- every eligible INNER TRAIN row appears exactly once
- partition ordering is deterministic
- eligible-row shuffling is deterministic
- no duplicates
- no omissions

Use the same frozen data-order semantics across hyperparameter trials.

Do not give different trials different stochastic training seeds.

This keeps the HPO comparison from adding an unnecessary source of
Monte Carlo variation.

PER-EPOCH TRACKING

Track every training epoch actually executed.

For every:

  outer study
  trial
  inner fold
  epoch

persist the full accepted 6E metric schema.

TRAINING:

- row count
- BP support count
- FLP-valid support count
- BP loss sum
- FLP loss sum
- BP loss mean
- FLP loss mean
- equal-task objective
- optimizer steps
- wall time
- rows/s
- peak allocated VRAM
- peak reserved VRAM
- process RSS
- finite-status diagnostics

VALIDATION:

- row count
- BP support count
- FLP-valid support count
- BP loss sum
- FLP loss sum
- BP loss mean
- FLP loss mean
- equal-task objective
- wall time
- rows/s
- finite-status diagnostics

CONFIGURATION:

- depth
- width
- learning rate
- weight decay
- batch size
- BF16 mode
- parameter count

Keep raw sums and counts.

Every reported mean must remain independently reconstructable.

VALIDATION FREQUENCY AND CURVES

Run validation after every completed epoch for each surviving fold.

These per-epoch validation values are for:

- tracking
- diagnostics
- training/validation curves

They do NOT alter ASHA behavior.

Only the frozen common five-fold resource values at:

  1
  4
  16
  64

and final:

  100

may be reported to Optuna as HPO intermediate/final objectives.

Do not call trial.should_prune() at any other epoch.

IMPORTANT CURVE INTERPRETATION

The training loss accumulated during an epoch is an online loss measured
while model parameters are changing.

The validation loss is evaluated after that epoch using the resulting
model state.

Therefore train and validation curves may be shown together for
diagnostic purposes, but documentation must not imply that the two are
identical fixed-model estimators.

Do not add a second full post-epoch INNER TRAIN evaluation merely to make
the curves symmetric.

That would materially increase compute without serving the frozen HPO
objective.

W&B

Use the accepted 6E tracking contract.

One W&B run per Optuna trial.

Project:
  wildfire-risk-uncertainty

Group:
  outer-<j>

Job type:
  hpo-trial

Suggested run name:
  outer-<j>-trial-<NNN>

Tags include:
  checkpoint-6f
  hpo
  tpe
  asha
  outer-<j>

Persist the W&B run ID locally.

Track:

- complete configuration
- fold/epoch training metrics
- fold/epoch validation metrics
- spatial-CV aggregates
- resource milestones
- pruning state
- runtime/resource use
- provenance

W&B must remain a mirror.

Local records and Optuna remain authoritative.

A transient W&B error after a trial has begun must not change:

- gradients
- training
- J_e
- Optuna state

Surface tracking degradation clearly.

If initial W&B authentication is unavailable:
  STOP for HITL.

STUDY STORAGE

Use the accepted persistent Optuna RDB storage.

Study names:

  wildfire_outer_1_hpo_v1
  wildfire_outer_2_hpo_v1
  wildfire_outer_3_hpo_v1
  wildfire_outer_4_hpo_v1
  wildfire_outer_5_hpo_v1

Before starting a study:

- verify no incompatible study exists
- verify study metadata
- verify search-space hash
- verify sampler/pruner configuration
- verify outer-study identity

If an expected study already contains trials because execution is being
resumed:

- validate all persisted state
- continue only the remaining work needed to reach exactly 30 trials

Never discard an existing valid study merely to obtain a different HPO
result.

OUTER-STUDY EXECUTION ORDER

Use deterministic ascending order:

  1
  2
  3
  4
  5

This order is operational only.

The studies are scientifically independent.

Do not alter later studies based on earlier study results.

If execution is interrupted by:

- terminal disconnect
- Codex usage limit
- process restart
- container interaction interruption

resume the SAME persisted study/trial under the frozen contract.

Do not restart the HPO experiment from scratch.

MODEL CHECKPOINTS DURING HPO

Persist enough ignored checkpoint state to resume each active
trial/fold trajectory.

At minimum preserve the accepted 6D checkpoint contract.

It is acceptable to retain only the most recent resumable checkpoint per
trial/fold rather than every historical model checkpoint.

HPO model weights are NOT scientific final models.

Do not commit them.

The scientific final outer models will be refit later from selected
configurations.

SELECTION RULE — FREEZE BEFORE VIEWING RESULTS

For each outer study j:

Eligible candidate for selection =
  trial state COMPLETE
  AND trial reached epoch 100
  AND J_100 is finite
  AND all five inner folds completed epoch 100 successfully.

Selected trial =
  eligible COMPLETE trial with MINIMUM J_100.

Do NOT select using:

- J_1
- J_4
- J_16
- J_64
- best single-fold loss
- best epoch before 100
- W&B visual appearance
- training loss
- runtime
- parameter count

If two eligible trials have exactly equal J_100 at stored numerical
precision:

  choose the smaller Optuna trial number.

This tie rule is deterministic and prospective.

If an outer study has zero eligible COMPLETE 100-epoch trials:

  STOP for HITL.

Do not select a pruned trial.

Do not increase the trial budget without explicit HITL approval.

SELECTED CONFIGURATION

For each outer study persist:

- outer study
- selected trial number
- depth
- width
- learning rate
- weight decay
- batch size
- J_100
- five individual fold objectives at epoch 100
- across-fold standard deviation
- minimum
- maximum
- model parameter count
- trial runtime
- W&B run ID
- Optuna study identity
- provenance hashes

The selected hyperparameters are specific to that outer study.

Do not average hyperparameters across outer folds.

Do not choose a single global configuration in 6F.

That is a separate later decision if needed for a production full-data
refit.

REPORTING

Create:

  reports/hpo/06f/
    README.md
    environment.json
    execution_manifest.json
    preprocessing_cache_manifest.json
    study_summary.csv
    trial_summary.csv
    trial_intermediate_objectives.csv
    selected_hyperparameters.csv
    selected_hyperparameters.json
    selected_trial_fold_metrics.csv
    selected_trial_epoch_curves.csv
    metric_reconciliation.json
    wandb_runs.csv
    figures_manifest.json
    figures/

COMMITTED TABULAR RECORDS

It is acceptable and desirable to commit compact summaries for all 150
trials, including:

  trial_summary.csv

with fields such as:

- outer study
- trial number
- state
- hyperparameters
- final/common resource reached
- J at recorded ASHA resources
- final J_100 if COMPLETE
- runtime
- fold-epoch resource consumed
- W&B run ID

Also commit:

  trial_intermediate_objectives.csv

containing the compact J_e trajectory at common resources.

Do NOT commit exhaustive raw per-fold/per-epoch logs for all trials.

Those remain ignored artifacts.

SELECTED-TRIAL RECORD

For the five selected trials, commit enough detailed epoch-level metrics
to reproduce the selected training/validation curves without depending
on W&B.

This may include:

  selected_trial_epoch_curves.csv

containing the five selected trials × their five inner folds × executed
epochs.

VISUALIZATIONS

Use the frozen 6E artifact policy.

EXHAUSTIVE INTERACTIVE:
  W&B

EXHAUSTIVE STATIC:
  ignored artifacts

COMMITTED FIGURES:
  curated scientific/model-selection summaries only

Generate at minimum useful 6F report figures for each outer study:

1. Trial objective history
   - COMPLETE trials at J_100
   - clearly distinguish PRUNED trials

2. Best-so-far COMPLETE J_100 by trial number

3. Trial resource/pruning history
   - resource reached
   - COMPLETE / PRUNED state

4. J_e trajectories across ASHA resource levels

5. Selected-trial five-fold validation objective curves

6. Selected-trial BP training/validation curves

7. Selected-trial FLP training/validation curves

8. Selected-trial equal-task training/validation curves

Also produce cross-study summaries:

9. Selected hyperparameters across the five outer studies

10. Selected J_100 and geographic fold dispersion across outer studies

11. Trial-state counts by outer study

12. Actual GPU/runtime/resource consumption by outer study

HYPERPARAMETER DIAGNOSTICS

Generate parameter-vs-objective/slice visualizations from COMPLETE
trials where valid.

Parallel-coordinate plots may be generated as HPO diagnostics.

Hyperparameter importance may be generated only if the implementation
can compute it meaningfully from the available COMPLETE trials.

If the number/diversity of COMPLETE trials is inadequate:

- omit the importance figure
- record why it was omitted

Do not force an importance result from sparse data.

Any importance analysis is an HPO diagnostic, not the paper's SHAP
explainability analysis.

METRIC RECONCILIATION

For EVERY selected trial independently recompute from persisted raw
records:

For every fold/epoch:
  BP mean =
    BP loss sum / BP count

  FLP mean =
    FLP loss sum / FLP-valid count

  equal-task =
    0.5 * BP mean + 0.5 * FLP mean

At resources:
  1, 4, 16, 64, 100

recompute:

  J_e =
    arithmetic mean of five fold validation equal-task objectives

Verify selected-trial values agree with:

- local aggregate records
- Optuna intermediate/final values
- W&B history

within the accepted serialization tolerance.

Also sample-check nonselected trials across:

- COMPLETE
- PRUNED

states.

If any disagreement affects a scientific objective:

  STOP for HITL.

ACTUAL COMPUTE ACCOUNTING

Record realized rather than estimated HPO cost.

For each study and overall report:

- number COMPLETE
- number PRUNED
- number FAILED
- total trials
- total fold-epochs executed
- training rows processed
- validation rows processed
- total training wall time
- total validation wall time
- orchestration overhead
- preprocessing-cache time
- total elapsed study wall time
- GPU-hours / elapsed GPU execution time where defined
- pruning resource distribution

Compare actual realized resource use with the prospective 6B planning
bounds and the 6E one-epoch smoke behavior.

Do not retrofit or change the HPO budget based on this comparison.

The comparison is retrospective reporting only.

METHODOLOGY DOCUMENTATION

Create:

  docs/methodology/checkpoint_06f_hpo_execution.md

Update:

  docs/methodology/methodology_evidence_ledger.csv

Do not introduce new literature claims unless a genuinely new scientific
decision unexpectedly arises.

6F should primarily document execution of the already frozen 6A–6E
methodology.

Clearly distinguish:

FROZEN BEFORE RESULTS:
- search space
- trial budget
- sampler
- pruner
- resource schedule
- spatial folds
- objective
- preprocessing
- training implementation
- selection rule

OBSERVED RESULTS:
- trial outcomes
- pruning behavior
- selected configurations
- selected J_100 values
- runtimes
- realized compute

No observed result may be described as prospectively chosen.

TESTS

Add/extend tests as necessary for:

- exact 30-trial study accounting
- selection uses COMPLETE J_100 only
- pruned trials cannot be selected
- failed trials trigger expected guard
- deterministic tie-break by trial number
- no outer TEST access
- cache provenance verification
- active-trial resume validation
- compact report generation
- selected-trial metric reconciliation
- W&B/local/Optuna identifiers
- study metadata compatibility checks

Run:

  full pytest suite
  Ruff
  git diff --check

Do not modify frozen training mathematics merely to satisfy a test.

HITL STOP CONDITIONS DURING EXECUTION

STOP if:

- W&B authentication is unavailable before launch
- active-trial restart cannot be validated
- preprocessing cache provenance disagrees
- outer TEST / DEAD_ZONE enters any calculation
- any real trial unexpectedly FAILS
- a loss/logit/gradient/parameter becomes nonfinite
- a study's frozen metadata/search-space hash differs
- Optuna/local/W&B scientific metrics disagree
- pruning occurs before all five folds reach a common resource
- a study cannot reach exactly 30 valid realized trials under the frozen
  protocol
- zero COMPLETE 100-epoch trials exist for an outer study
- execution requires changing any frozen 6A–6E methodology

Do not change methodology silently.

USAGE-LIMIT / SESSION INTERRUPTION IS NOT A SCIENTIFIC STOP CONDITION

If Codex usage limits or terminal/session interruption occur while valid
persistent HPO execution state exists:

- do not change methodology
- do not restart from scratch
- re-enter the same container/workspace
- verify repository and study state
- resume the same persisted study/trial

Only request HITL if state cannot be reconciled safely.

FINAL 6F HITL REPORT

After all five studies finish, report:

1. Base commit SHA

2. W&B authentication/project status without credentials

3. Preprocessing-cache validation:
   - all 25 states
   - row counts
   - hashes/status

4. Active-trial restart validation result

5. For each outer study:
   - total 30 trials
   - COMPLETE count
   - PRUNED count
   - FAILED count
   - pruning-resource distribution
   - total runtime/resource consumption

6. For every COMPLETE trial:
   compact trial/config/final-objective record in report artifacts

7. For each outer study's SELECTED trial:
   - trial number
   - depth
   - width
   - learning rate
   - weight decay
   - batch size
   - J_100
   - five fold-specific epoch-100 objectives
   - fold SD/min/max
   - model parameter count
   - runtime
   - W&B run identifier

8. Cross-study selected-hyperparameter table

9. Actual total:
   - trials
   - fold-epochs
   - training rows
   - validation rows
   - wall time
   - realized compute

10. Comparison against prospective 6B compute planning

11. Selected-trial metric reconciliation results

12. Sampled nonselected COMPLETE/PRUNED reconciliation results

13. Confirmation Optuna/local/W&B values agree

14. Static curated figures generated

15. Confirmation exhaustive detailed figures/logs remain ignored

16. Full pytest status

17. Ruff status

18. git diff --check status

19. Files changed

20. Confirmation no OUTER TEST evaluation occurred

21. Confirmation no OOF predictions were generated

22. Confirmation no final outer models were trained

Then STOP for HITL review.

Do not commit.
Do not push.
Do not begin the outer-fold scientific evaluation checkpoint.