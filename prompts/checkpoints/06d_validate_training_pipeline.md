Checkpoint 6D — Validate the end-to-end training and checkpoint/resume pipeline

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

Required base commit:
  4dcd92acefa7c5bb27fd33c01190d5836a4b886e

PREFLIGHT

Before changing anything:

1. Verify branch == dev.
2. Verify HEAD == required base commit.
3. Verify origin/dev == required base commit.
4. Verify working tree is clean.
5. Verify frozen Checkpoints 5B, 6A, 6B, and 6C are present.
6. Verify local ignored 5B and 6C spatial artifacts are present.
7. Verify canonical dataset is present.
8. Verify the pinned PyTorch CUDA environment still sees the RTX 5080.
9. Do not modify main.
10. Save this exact prompt verbatim as:

    prompts/checkpoints/06d_validate_training_pipeline.md

PURPOSE

Validate the complete real neural-network training path before any
hyperparameter optimization begins.

This checkpoint must demonstrate that the frozen methodology can be
implemented correctly end to end:

  canonical Parquet
      -> frozen outer/inner eligibility
      -> INNER-TRAIN-only preprocessing
      -> deterministic bounded-memory training stream
      -> frozen 33 inputs
      -> BF16 shared MLP
      -> BP + FLP losses
      -> AdamW optimization
      -> checkpoint/resume
      -> INNER VALIDATION
      -> reproducible reporting

This is an ENGINEERING VALIDATION checkpoint.

It is NOT:
- hyperparameter optimization
- model selection
- a baseline-performance experiment
- outer-fold evaluation
- uncertainty estimation
- an experiment from which architecture choices may be made

DO NOT:

- run Optuna
- run ASHA pruning
- run a W&B sweep
- compare architectures
- compare learning rates
- compare batch sizes
- inspect OUTER TEST predictions or performance
- train a production model
- train an ensemble
- run SHAP
- change any frozen 6A/6B/6C scientific decision
- commit or push before HITL review

PROSPECTIVELY FIXED VALIDATION CASE

Use exactly:

  outer study = 3
  inner validation fold = 1

Reason:

- outer study 3 was prospectively selected in 6B as the median-sized
  outer TRAIN population for engineering calibration
- inner fold 1 is the deterministic first fold label
- neither is selected from model performance

Read all counts from frozen artifacts rather than hard-coding them.

Expected approximate populations:

  OUTER TRAIN_3:
    79,519,698

  INNER VALIDATION_3,1:
    18,476,910

  INNER TRAIN_3,1:
    61,042,788

Verify exact counts before training.

Eligibility must be:

  INNER TRAIN:
    train_eligible_fold_3 == True
    AND inner_fold_outer_3 != 1

  INNER VALIDATION:
    train_eligible_fold_3 == True
    AND inner_fold_outer_3 == 1

No OUTER TEST or OUTER DEAD_ZONE row may enter either population.

FROZEN MODEL INPUTS

Use exactly the frozen 33 inputs from Checkpoint 6B:

- all 25 canonical schema-role predictor columns
- wui_covered_fraction
- six intensive __valid_area_fraction columns
- wui_failure

Coordinates, targets, masks, outer/inner labels, and other diagnostics
must not become predictors.

Persist the exact ordered 33-column list in the training run manifest.

PREPROCESSING

Fit preprocessing using INNER TRAIN_3,1 ONLY.

Apply frozen 6A rules:

- log1p only:
    HUDEN2020
    POPDEN2020
    OCCHUDEN2020
    VACHUDEN2020

- training-only mean imputation where required

- training-only z-standardization of continuous inputs

- wui_failure remains binary 0/1 and is not z-standardized

INNER VALIDATION contributes no learned preprocessing statistic.

Persist:

- preprocessing parameters
- feature order
- preprocessing hash
- fitted row counts
- missing-value counts
- means
- standard deviations

under an ignored training artifact directory.

Validate that the preprocessing state is identical when refitted from
the same INNER TRAIN population with the same deterministic procedure.

FIXED 6D MODEL CONFIGURATION

This configuration exists only to validate the training machinery.

Use:

  hidden depth = 2
  shared hidden width = 128
  activation = ReLU
  dropout = 0

  BP head:
    1 logit

  FLP head:
    6 logits

  optimizer:
    AdamW

  learning rate:
    1e-3

  weight decay:
    1e-4

  batch size:
    65536

  execution:
    BF16 autocast

  seed:
    42

Batch size 65,536 is the middle frozen 6B candidate and is chosen here
solely as an integration-test constant.

Do not infer that it is preferable to the other HPO candidates.

TRAINING OBJECTIVE

Implement the frozen partition-level objective exactly:

  L =
      0.5 * mean_BP(L_BP)
    + 0.5 * mean_FLP(L_FLP)

where:

  N_BP =
    number of INNER TRAIN rows

  N_FLP =
    number of INNER TRAIN rows with flp_target_valid == True

The streaming/minibatch implementation must not accidentally change task
weight because BP and FLP have different support sizes.

For a uniformly traversed minibatch of B INNER TRAIN rows, an acceptable
mathematically equivalent stochastic estimator is:

  batch_BP =
    mean over all B BP losses

  batch_FLP =
    (N_BP / N_FLP)
    * mean over all B of:
        flp_target_valid * per-row FLP loss

  batch_loss =
      0.5 * batch_BP
    + 0.5 * batch_FLP

This uses B as the denominator of the masked minibatch FLP sum and
corrects by N_BP/N_FLP.

Its partition expectation equals:

  0.5 * global mean BP loss
  +
  0.5 * global mean valid-FLP loss

An algebraically equivalent implementation is acceptable if demonstrated
in tests.

Do NOT simply compute:

  0.5 * batch BP mean
  + 0.5 * mean of valid FLP rows in that batch

without demonstrating that it preserves the frozen partition-level task
weighting.

VALIDATION METRICS

For INNER VALIDATION, compute the objective exactly by streaming global:

- BP loss sum
- BP valid count
- FLP loss sum
- FLP valid count

Then calculate:

  validation_L =
      0.5 * (BP sum / BP count)
    + 0.5 * (FLP sum / FLP-valid count)

Do not average already-averaged minibatch losses.

Also report the two component losses separately.

These values are SANITY DIAGNOSTICS ONLY.

They may not be used to select:
- architecture
- batch size
- learning rate
- epoch count
- optimizer settings
- any later methodology

EPOCH DEFINITION

One epoch means:

  every eligible INNER TRAIN row is processed exactly once.

No eligible row may be duplicated or omitted within an epoch.

Record exact row counts for every epoch.

TRAINING DATA ORDER

Use a deterministic bounded-memory shuffled training stream.

Prefer reuse of the canonical 3,060 partition organization:

1. derive an epoch-specific deterministic ordering of eligible
   partitions from seed 42 and epoch index;
2. deterministically shuffle eligible rows within each loaded partition;
3. stream rows through a bounded batch buffer;
4. guarantee every eligible INNER TRAIN row appears exactly once per
   epoch.

Do not globally load all 61M rows merely to shuffle them.

The exact bounded-memory implementation is an engineering choice.

Persist enough configuration to reproduce the row order.

If implementing a deterministic bounded-memory shuffle would require a
new scientific choice or materially redesign the data representation,
STOP for HITL rather than improvising.

CHECKPOINT / RESUME CONTRACT

Training checkpoints must contain enough state to resume at an epoch
boundary without changing the training trajectory.

Persist at minimum:

- model state
- AdamW optimizer state
- completed epoch
- global optimizer step
- seed
- Python RNG state
- NumPy RNG state
- torch CPU RNG state
- torch CUDA RNG state
- model configuration
- feature-order hash
- preprocessing-state hash
- outer study
- inner fold
- batch size
- precision mode
- source git SHA
- data-order configuration/version

Write scientific-integration checkpoints only under an ignored path such
as:

  artifacts/training/06d/

They are NOT accepted scientific models.

REAL TRAINING VALIDATION

Run exactly TWO epochs for the real integration test.

No early stopping.

No model-quality decision.

Perform:

A. UNINTERRUPTED REFERENCE

Starting from seed 42 and identical initialization:

  epoch 1
  epoch 2

Record validation diagnostics after each epoch.

B. CHECKPOINT / RESUME RUN

Starting again from the identical seed and initialization:

  epoch 1
  save checkpoint
  terminate training process/state
  reload checkpoint
  run epoch 2

Use the same preprocessing state and deterministic epoch data order.

Compare the final uninterrupted and resumed runs.

Check:

- model parameters
- optimizer state
- global step
- row counts
- epoch-2 training loss accumulation
- epoch-2 validation BP loss
- epoch-2 validation FLP loss
- epoch-2 combined validation loss

First test for exact equality where the runtime permits.

If CUDA arithmetic prevents bitwise equality, quantify:

- maximum absolute parameter difference
- maximum relative parameter difference
- validation-metric differences

Do not change model methodology merely to force bitwise identity.

If resume changes results beyond small floating-point numerical
differences, STOP for HITL.

Do not continue to additional epochs.

INITIALIZATION / SANITY METRICS

Before training, compute one full INNER VALIDATION pass from the
identically initialized model.

Then record validation diagnostics after:

  initialization
  epoch 1
  epoch 2

These are used only to confirm finite, operational behavior.

Do NOT require validation loss to improve monotonically.

Do NOT choose anything based on which value is lowest.

NUMERICAL VALIDATION

Require throughout:

- finite model inputs after preprocessing
- finite logits
- finite BP loss
- finite FLP loss
- finite combined loss
- finite gradients
- finite parameters after optimizer steps

Validate:

- BP sigmoid outputs lie in [0,1]
- FLP softmax rows sum to approximately 1
- invalid FLP rows contribute zero FLP loss
- valid FLP rows use the complete soft target distribution
- BP targets remain floating-point probabilities
- no BP thresholding occurs
- no FLP argmax target conversion occurs

SYNTHETIC CORRECTNESS TESTS

Before the full real-data run, add small deterministic tests that prove:

1. soft BP BCE matches a hand-computed reference
2. soft FLP CE matches a hand-computed reference
3. partition-weighted minibatch estimator matches the frozen objective
   in expectation / exact constructed fixture
4. invalid FLP rows contribute no FLP term
5. gradients reach:
   - shared backbone
   - BP head
   - FLP head
6. a tiny synthetic dataset can be overfit sufficiently to show that the
   training loop actually learns
7. preprocessing is fit only on training rows
8. validation rows cannot influence preprocessing
9. outer TEST / DEAD_ZONE rows cannot enter the loader
10. epoch traversal has no duplicates or omissions
11. checkpoint round-trip restores model/optimizer/RNG metadata correctly
12. resumed synthetic training matches uninterrupted training

The synthetic overfit test is an engineering unit test, not scientific
evidence.

REPORTS

Create:

  reports/training/06d/
    README.md
    environment.json
    run_manifest.json
    preprocessing_summary.json
    epoch_metrics.csv
    checkpoint_resume_validation.json
    data_traversal_validation.json

Do not commit large model/checkpoint artifacts.

Report:

- exact outer/inner populations
- exact 33 inputs
- model parameter count
- preprocessing scan runtime
- training runtime per epoch
- validation runtime
- rows/second
- optimizer steps per epoch
- BP/FLP support counts
- initialization / epoch-1 / epoch-2 sanity losses
- uninterrupted-vs-resumed comparison
- peak VRAM
- peak RSS

Do not frame the losses as model performance results.

SOURCE CODE

Build on the 6B training components rather than creating a parallel
training stack.

Expected package organization may include:

  src/wildfire_risk_uncertainty/training/
    model.py
    losses.py
    preprocessing.py
    data.py
    checkpoint.py
    trainer.py
    validation.py
    cli.py

Refactor existing 6B code when useful.

There should be one reusable training implementation that later HPO can
call.

Do not duplicate benchmark-only and HPO-only model/loss logic.

CLI

Expose an appropriate explicit integration-validation command, e.g.:

  uv run wildfire-train-validate

It must be impossible to accidentally run outer TEST evaluation through
this command.

TESTS / QUALITY

Add focused tests for all correctness properties above.

Run:

  full pytest suite
  Ruff
  git diff --check

REPRODUCIBILITY / PROVENANCE

Record:

- seed 42
- git SHA
- CUDA/PyTorch environment
- outer study 3
- inner fold 1
- exact eligibility counts
- feature-order hash
- preprocessing hash
- model configuration
- data-order algorithm
- epoch row counts
- checkpoint metadata

Do not call the 6D weights a trained scientific model.

HITL STOP CONDITIONS

STOP before changing methodology if:

- outer/inner eligibility counts do not match frozen reports
- any outer TEST / DEAD_ZONE row enters training or validation
- the 33-input contract conflicts with actual schema
- preprocessing requires using validation information
- the frozen partition-level loss cannot be implemented correctly
- deterministic epoch traversal duplicates or omits eligible rows
- BF16 gives nonfinite values
- either task head receives no gradients
- checkpoint/resume materially changes the training trajectory
- training requires a new scientific hyperparameter
- the real pipeline requires changing frozen 6A/6B/6C methodology

Do not solve a stop condition by tuning the model.

FINAL HITL REPORT

When complete, report:

1. Base commit SHA
2. Exact INNER TRAIN and INNER VALIDATION populations
3. Exact BP / FLP-valid support counts
4. Frozen 33-input confirmation
5. Preprocessing reproducibility result
6. Model parameter count
7. Training configuration
8. Epoch traversal validation
9. Initialization / epoch-1 / epoch-2 sanity losses
10. Training and validation throughput
11. Peak VRAM / RSS
12. Numerical finite-value checks
13. Gradient-flow checks
14. Uninterrupted-vs-resumed equivalence result
15. Checkpoint contents/provenance
16. Synthetic correctness-test results
17. Full pytest status
18. Ruff status
19. git diff --check status
20. Files changed
21. Any implementation caveat

Then STOP for HITL review.

Do not run Optuna.
Do not run ASHA.
Do not run a W&B sweep.
Do not inspect OUTER TEST performance.
Do not train beyond the fixed two-epoch engineering validation.
Do not commit.
Do not push.
