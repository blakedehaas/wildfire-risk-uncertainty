Checkpoint 6B — Local GPU training throughput calibration and HPO compute planning

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

Required base commit:
  f0cfa1753988e221d456c161aa8650b1de69a014

PREFLIGHT

Before changing anything:

1. Verify branch == dev.
2. Verify HEAD == required base commit.
3. Verify origin/dev == required base commit.
4. Verify working tree is clean.
5. Verify accepted Checkpoints 5B and 6A are present.
6. Verify the frozen 5B outer-fold artifact is available locally.
7. Verify CUDA/PyTorch can access the local NVIDIA GPU.
8. Record:
   - GPU model
   - VRAM
   - NVIDIA driver
   - CUDA runtime
   - PyTorch version
   - Python version
9. Do not modify main.
10. Save this exact prompt verbatim as:

    prompts/checkpoints/06b_local_gpu_throughput_calibration.md

PURPOSE

Measure the computational cost of the frozen Checkpoint 6A training
formulation on the local GPU before any hyperparameter optimization is
performed.

This checkpoint answers engineering questions only:

- How fast can the real training pipeline process canonical rows?
- What batch sizes are technically feasible?
- Where does throughput saturate?
- What limits performance: GPU compute, VRAM, CPU, I/O, or preprocessing?
- Approximately how expensive will five-fold spatial HPO be?
- Is local execution practical, or should HPO be parallelized on a
  compute cluster?

This checkpoint MUST NOT answer:

- which hyperparameters predict best
- which model is scientifically best
- which outer fold performs best
- whether a particular batch size gives lower validation loss
- any question requiring outer TEST performance

DO NOT:

- run Optuna
- run ASHA
- run W&B sweeps
- tune hyperparameters from predictive performance
- examine outer TEST loss or predictions
- train a production model
- train a deep ensemble
- run SHAP
- modify the frozen outer folds
- change the 6A scientific formulation
- commit or push before HITL review

FROZEN CHECKPOINT 6A FORMULATION

Use the accepted 6A methodology exactly.

Model:

  shared feed-forward MLP
  hidden depth = 2
  shared width = 128
  activation = ReLU
  dropout = 0

Heads:

  BP:
    one logit
    sigmoid interpretation

  FLP:
    six logits
    softmax interpretation

Losses:

  soft Bernoulli BCE-with-logits on canonical floating-point BP

  soft categorical cross-entropy on the complete six-element FLP
  probability vector only where flp_target_valid == True

Training objective:

  L =
      0.5 * mean_BP_loss
    + 0.5 * mean_FLP_loss

Do not BP-weight FLP loss.

Fixed benchmark optimizer:

  AdamW

Fixed benchmark-only parameters:

  learning rate = 1e-3
  weight decay = 1e-4

These values are workload calibration constants only.

They are not selected model hyperparameters and their predictive
performance must not be compared.

PREDICTORS AND PREPROCESSING

Read the accepted 6A methodology and canonical schema to identify the
exact frozen model-input predictor set.

Do not invent, drop, or substitute predictors.

Apply the accepted preprocessing rules:

- log1p:
    HUDEN2020
    POPDEN2020
    OCCHUDEN2020
    VACHUDEN2020

- training-partition-only mean imputation where required
- retain applicable canonical missingness / coverage information
- training-partition-only z-standardization of continuous inputs

If the accepted methodology and actual canonical schema disagree about
which columns are model inputs or how null/coverage columns are treated,
STOP for HITL rather than improvising.

REPRESENTATIVE TRAINING POPULATION

Use exactly one frozen outer TRAIN population for benchmarking.

Select the outer fold whose usable TRAIN row count is the median of the
five accepted 5B TRAIN populations.

Determine this programmatically from the committed 5B report.

Expected representative fold:
  outer fold 3

Expected usable TRAIN size:
  approximately 79.5 million rows

Do not hard-code fold 3 without verifying that it remains the median.

Use only that fold's usable TRAIN rows.

Do not read its outer TEST rows for model evaluation.

PREPROCESSING CALIBRATION

Fit the real preprocessing state once on the representative outer TRAIN
population using bounded-memory streaming.

Record:

- rows processed
- elapsed time
- effective rows/second
- peak process RSS
- resulting input feature count
- missing-value counts per modeled feature
- fitted preprocessing-state size

Persist the benchmark preprocessing state under an ignored benchmark
artifact directory, for example:

  artifacts/benchmark/06b/

This preprocessing artifact is for calibration only and is not a final
trained-model preprocessing artifact.

BATCH-SIZE THROUGHPUT BENCHMARK

Benchmark technically feasible powers-of-two batch sizes beginning at:

  1024

Then double:

  2048
  4096
  8192
  ...

until either:

- CUDA OOM occurs, or
- batch size reaches 131072

Do not reduce the search based on model loss.

For each batch size:

1. Reinitialize the same model weights deterministically.
2. Reinitialize the same AdamW optimizer.
3. Use the same deterministic canonical TRAIN row stream.
4. Run enough warm-up work to avoid startup/JIT/cache effects.
5. Measure a sufficiently large fixed amount of real training work to
   obtain stable timings.
6. Include:
     - data loading
     - preprocessing application
     - CPU-to-GPU transfer
     - forward pass
     - BP/FLP losses
     - backward pass
     - optimizer step
7. Synchronize CUDA correctly around timing boundaries.

The exact timing-window implementation is an engineering choice, but it
must be documented and identical across batch sizes insofar as practical.

For each batch size report:

- success / OOM
- rows processed
- optimizer steps
- wall time
- rows/second
- batches/second
- median step time
- p95 step time
- peak allocated VRAM
- peak reserved VRAM
- peak host RSS
- GPU utilization if reliably measurable
- GPU memory utilization if reliably measurable

Do not rank batch sizes by predictive loss.

MIXED-PRECISION CALIBRATION

If the installed GPU and PyTorch stack support BF16 autocast reliably,
benchmark:

  FP32
  BF16 autocast

using the same representative configuration.

Treat numerical precision as an execution-mode benchmark, not an HPO
dimension.

Require:

- finite loss
- finite gradients
- no runtime errors

Do not compare predictive quality.

If BF16 is unavailable or unstable, report that fact and continue with
FP32.

Do not silently use FP16 instead.

END-TO-END DATA PIPELINE

The benchmark must use real canonical Parquet data and the accepted
preprocessing path.

Also separately measure, where practical:

1. raw Parquet read throughput
2. preprocessing-only throughput
3. GPU training throughput with already prepared tensors
4. complete end-to-end training throughput

The purpose is to identify the bottleneck rather than merely time CUDA
kernels.

Do not redesign the storage format in this checkpoint.

EVALUATION / INFERENCE THROUGHPUT

Measure forward-only loss-computation throughput using real training
rows and the same model.

No outer TEST rows are required.

Report:

- rows/second
- peak VRAM
- wall time

This is needed only to estimate the later cost of inner-fold validation.

FULL-PASS SANITY BENCHMARK

After the batch-size sweep:

1. Identify the technically fastest stable execution configuration based
   solely on measured end-to-end throughput and absence of OOM/numerical
   failure.
2. Do NOT declare it the final HPO batch size.
3. Use it for one full streaming pass through the representative outer
   TRAIN population if computationally reasonable.

This full pass is a throughput calibration, not model training for
scientific use.

Record:

- exact rows processed
- wall time
- rows/second
- peak VRAM
- peak RSS
- data-read throughput

Do not save the resulting model as a scientific model.

If the sampled benchmark already makes a full pass clearly unnecessary
or disproportionately expensive, STOP and explain before skipping it.

HPO COST MODEL

Using measured throughput, construct a compute-cost model for the frozen
6A nested spatial HPO procedure.

Report costs in:

- GPU-seconds
- GPU-hours
- local wall-clock time on one GPU

At minimum calculate:

A. Cost per 1 million training rows.

B. Cost per 1 million validation rows.

C. Approximate cost of one training epoch over:
   - the representative full outer TRAIN population
   - 80% of that population as a clearly labeled planning approximation
     for a typical inner-training population

D. Approximate validation cost over:
   - 20% of that population as a clearly labeled planning approximation

E. Cost of one five-inner-fold resource milestone at:
   - 1 epoch
   - 4 epochs
   - 16 epochs
   - 64 epochs
   - 100 epochs

F. Worst-case five-fold cost of one completely unpruned 100-epoch trial.

G. Local one-GPU cost scenarios for:
   - 20 trials
   - 30 trials
   - 50 trials
   - 100 trials

For each trial-count scenario report:

- worst case: no useful pruning
- an explicitly labeled idealized ASHA resource-allocation estimate
  using reduction factor 4
- assumptions used in the estimate

Do not present the idealized ASHA calculation as guaranteed runtime.

The actual inner folds have not yet been instantiated here, so clearly
label 80/20 calculations as planning approximations.

CLUSTER PLANNING

Do not assume distributed data-parallel training is necessary.

The intended cluster parallelism is across independent:

- HPO trials
- inner-fold training jobs

rather than splitting this small MLP across multiple GPUs.

Provide theoretical wall-clock lower bounds for representative GPU
counts such as:

  1
  2
  4
  5
  8

assuming independent jobs and perfect scheduling.

Clearly state that these are lower bounds and do not include:

- shared-filesystem contention
- scheduler latency
- host differences
- GPU-generation differences
- synchronization overhead

Do not choose a cluster solely because the dataset is large.

BATCH-SIZE DECISION SUPPORT

At the end, provide:

- all technically feasible batch sizes
- throughput curve
- VRAM curve
- location of any clear throughput plateau
- technically fastest stable batch size
- 2–4 reasonable batch-size candidates for later HPO, based only on:
    - feasibility
    - throughput
    - memory headroom

Do not freeze the final candidate set without HITL approval.

Likewise, do not freeze:

- final maximum epoch budget
- final Optuna trial count
- final ASHA maximum resource
- local versus cluster execution
- number of parallel workers

Those decisions require HITL review of the measured compute costs.

SOURCE CODE

Implement the benchmark cleanly under the package, for example:

  src/wildfire_risk_uncertainty/training/
    model.py
    losses.py
    preprocessing.py
    data.py

  src/wildfire_risk_uncertainty/benchmark/
    training_throughput.py
    cli.py

Reuse code intended for later training where practical.

Do not build an autonomous agent harness or unrelated infrastructure.

Expose an appropriate CLI, for example:

  uv run wildfire-training-benchmark

REPORTS

Create:

  reports/benchmark/06b/
    README.md
    environment.json
    preprocessing.json
    batch_throughput.csv
    precision_comparison.csv
    pipeline_breakdown.csv
    compute_budget.csv
    cluster_scaling.csv
    figures/

Useful figures include:

1. rows_per_second_vs_batch_size.png
2. peak_vram_vs_batch_size.png
3. estimated_hpo_wall_time_vs_trial_count.png

Do not include model-quality plots.

TESTS

Add focused tests for:

- BP soft BCE loss
- soft FLP cross-entropy
- FLP validity masking
- equal-task loss aggregation with separate denominators
- no BP weighting of FLP loss
- sigmoid / six-softmax output shapes
- log1p application to only the four frozen density variables
- training-only imputation behavior
- training-only standardization behavior
- deterministic preprocessing
- representative-fold selection by median TRAIN count
- benchmark never loads outer TEST rows as evaluation data
- stable benchmark configuration serialization
- batch-size OOM handling if testable without requiring actual OOM

Run:

  full pytest suite
  Ruff
  git diff --check

REPRODUCIBILITY

Use seed 42.

Persist:

- benchmark configuration
- preprocessing-state hash
- exact representative outer fold
- software versions
- git SHA
- timing procedure
- candidate batch sizes
- precision mode
- hardware information

Exact wall-clock timings need not reproduce byte-for-byte, but all
benchmark configuration and deterministic data/model setup must.

HITL STOP CONDITIONS

STOP before changing methodology if:

- CUDA/PyTorch cannot use the RTX 5080 correctly
- actual canonical predictors conflict with frozen 6A
- preprocessing semantics are ambiguous
- benchmark would require outer TEST performance
- real training requires a scientific decision not frozen in 6A
- BF16 produces nonfinite losses/gradients
- memory limitations require changing the model formulation
- required data artifacts are missing or inconsistent
- a storage redesign appears necessary
- a new hyperparameter is required merely to make the benchmark run

FINAL HITL REPORT

When complete, report:

1. Base commit SHA
2. GPU / CUDA / PyTorch environment
3. Representative outer fold and TRAIN row count
4. Predictor count
5. Preprocessing scan time and throughput
6. Batch-size benchmark table
7. FP32 vs BF16 results, if applicable
8. Technically fastest stable execution configuration
9. Full-pass epoch-equivalent throughput/time
10. Primary bottleneck:
    - disk/I/O
    - preprocessing/CPU
    - transfer
    - GPU compute
    - mixed
11. Training and validation rows/second
12. Estimated one-five-fold-trial cost
13. Estimated HPO costs for 20/30/50/100 trials
14. Idealized ASHA versus no-pruning estimates
15. Theoretical 1/2/4/5/8-GPU wall-clock lower bounds
16. Feasible/recommended batch-size candidates
17. Recommendation:
    - clearly practical locally
    - practical locally but slow
    - cluster strongly preferable
   Explain quantitatively.
18. Quantities still requiring HITL freeze:
    - batch-size candidate set
    - max epochs
    - trial count
    - ASHA max resource
    - local vs cluster
    - worker count
19. pytest / Ruff / git diff --check status
20. Files changed
21. Any caveat in the cost model

Then STOP for HITL review.

Do not run Optuna.
Do not run ASHA.
Do not run W&B.
Do not inspect outer TEST performance.
Do not commit.
Do not push.
