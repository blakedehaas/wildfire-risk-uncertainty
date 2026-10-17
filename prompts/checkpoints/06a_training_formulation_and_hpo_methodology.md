Checkpoint 6A — Freeze training formulation, preprocessing, and HPO methodology

Repository:
  /workspace/wildfire-risk-uncertainty

Branch:
  dev

BASELINE / PREFLIGHT

Checkpoint 5B has been completed, accepted, committed, and pushed.

Before changing anything:

1. Verify branch == dev.
2. Verify HEAD == origin/dev.
3. Verify working tree is clean.
4. Record the current HEAD SHA as the Checkpoint 6A base commit.
5. Verify the accepted Checkpoint 5B methodology, reports, and frozen
   outer spatial evaluation design are present.
6. Do not modify main.
7. Save this exact prompt verbatim as:

   prompts/checkpoints/06a_training_formulation_and_hpo_methodology.md

PURPOSE

Freeze the scientific training formulation, predictor preprocessing
rules, inner spatial model-selection procedure, and hyperparameter
optimization strategy before implementing training.

This is an evidence/documentation checkpoint.

DO NOT:

- train any neural network
- run a throughput benchmark
- generate predictions
- inspect outer TEST performance
- implement Optuna studies
- implement W&B logging
- construct final inner-fold artifacts unless needed only for a tiny
  synthetic/unit validation
- tune hyperparameters
- choose hyperparameters based on project validation results
- modify the frozen Checkpoint 5B outer folds
- modify the accepted 211.151 km outer exclusion radius
- commit or push before HITL review

EVIDENCE STANDARD

Every consequential scientific decision must be traceable to a
peer-reviewed journal paper that materially implemented or established
the relevant method.

For each decision below:

1. Verify the cited primary paper.
2. Confirm what the paper actually implemented or established.
3. Do not overstate the precedent.
4. Clearly distinguish:
   - literature-supported method
   - project-specific adaptation
   - engineering implementation choice
5. If a cited paper does not support the stated decision, STOP for HITL
   rather than silently replacing it with another rationale.

Update:

  docs/methodology/methodology_evidence_ledger.csv

Create:

  docs/methodology/checkpoint_06a_training_formulation.md

SCIENTIFIC CONTEXT

The model reconstructs the Dillon/FSim probabilistic wildfire-risk
surface from SILVIS-derived human-environment predictors.

It does not predict observed future-fire truth.

The frozen principal targets are:

  BP:
    burn probability

  FLP1...FLP6:
    conditional flame-length probability distribution given burning

Derived quantities include:

  conditional FLEP8:
    FLP5 + FLP6

  unconditional FLEP8:
    BP * (FLP5 + FLP6)

The model should support later uncertainty decomposition and SHAP
analysis, but neither uncertainty modeling nor SHAP is implemented here.

DECISION 6A-D1 — OCCURRENCE + CONDITIONAL SEVERITY FORMULATION

Primary wildfire precedent:

  Preisler et al. (2004)
  Probability-based models for estimation of wildfire risk
  International Journal of Wildland Fire
  DOI: 10.1071/WF02061

Verify the paper before recording this decision.

Freeze the conceptual structure:

  shared predictor representation
         |
         +--> BP head
         |
         +--> conditional FLP head

The BP head estimates:

  P(burn)

The FLP head estimates:

  P(flame-length class | burn)

Do not train a separate FLEP8 head.

Derive:

  conditional_FLEP8 = q5 + q6

  unconditional_FLEP8 =
      p_burn * (q5 + q6)

Document that the wildfire-literature precedent motivates separating
occurrence probability from conditional fire severity, while our
six-class FLP implementation is a project-specific adaptation to the
Dillon/FSim targets.

DECISION 6A-D2 — PROBABILISTIC OUTPUT HEADS

Freeze:

BP head:
  one logit
  sigmoid probability output

FLP head:
  six logits
  six-way softmax output

The FLP output must remain a normalized conditional probability
distribution.

Do not convert FLP targets to argmax classes.

Do not renormalize canonical FLP targets.

Use FLP loss only where the canonical:

  flp_target_valid == True

DECISION 6A-D3 — PROPER PROBABILISTIC LOSSES

Primary statistical source:

  Gneiting & Raftery (2007)
  Strictly Proper Scoring Rules, Prediction, and Estimation
  Journal of the American Statistical Association
  DOI: 10.1198/016214506000001437

Verify that the logarithmic score is strictly proper.

Freeze the BP training loss as soft Bernoulli cross-entropy:

  L_BP =
    -[p * log(p_hat) +
      (1-p) * log(1-p_hat)]

where the canonical Dillon BP probability is used directly as the
floating-point target.

Implementation may use numerically stable BCE-with-logits.

Do not threshold BP into a binary label.

Freeze the FLP loss as soft categorical cross-entropy:

  L_FLP =
    -sum_k q_k * log(q_hat_k)

where q is the complete six-element canonical FLP probability vector.

Do not replace q with its dominant class.

Document these as logarithmic-score objectives applied to supplied
probability targets.

DECISION 6A-D4 — CONDITIONAL FLP SUPPORT

The FLP head represents:

  P(flame-length class | burn)

Therefore:

- train FLP only on canonical strict FLP-valid support
- do not multiply FLP loss by BP
- do not make low-BP cells contribute proportionally less to the
  conditional distribution objective

Document that BP weighting would change the objective toward
unconditional intensity risk rather than estimation of the supplied
conditional FLP distribution.

If a stronger literature citation is required for this exact
implementation detail and cannot be verified, mark it explicitly as a
mathematical consequence of the chosen conditional-probability
factorization rather than inventing a citation.

DECISION 6A-D5 — SHARED MULTI-OUTPUT MLP

Freeze the model family as:

  predictors
      |
  shared feed-forward MLP backbone
      |
      +--> BP head
      |
      +--> FLP head

This checkpoint does not freeze final depth or width.

The model is intentionally simple because the scientific contribution is
wildfire-risk uncertainty, not neural architecture novelty.

Use peer-reviewed multi-output / multitask neural-network precedent if
available and verify it before adding it to the ledger.

Do not claim that the shared MLP is itself novel.

DECISION 6A-D6 — FIXED MULTITASK MODEL-SELECTION OBJECTIVE

For hyperparameter selection, freeze the scalar objective:

  L_tune =
      0.5 * mean(L_BP)
    + 0.5 * mean(L_FLP)

Important:

- compute mean BP loss over its own valid support
- compute mean FLP loss over its own valid support
- combine the two task means with equal task weight
- do not let differing row counts implicitly weight the two tasks
- do not tune the BP-vs-FLP task weight

The purpose is to keep every HPO trial evaluated against the same
objective.

Find and verify an appropriate peer-reviewed implementation or
methodological precedent for fixed equal weighting of multiple task
losses.

If suitable evidence cannot be established, STOP for HITL rather than
silently treating 0.5 / 0.5 as literature-supported.

Do not introduce dynamic task weighting, uncertainty weighting,
GradNorm, or another multitask-weighting research problem without HITL.

DECISION 6A-D7 — DENSITY TRANSFORMS

The initial retained SILVIS intensive-density predictors are:

  HUDEN2020
  POPDEN2020
  OCCHUDEN2020
  VACHUDEN2020

Use a fixed:

  log1p(x)

transform for these nonnegative, strongly right-skewed density
variables before learned scaling.

Find and verify peer-reviewed wildfire/environmental precedent for log
transformation of population/housing-density predictors.

Document log1p as the zero-preserving project adaptation.

Do not log-transform bounded shares/fractions merely for symmetry.

DECISION 6A-D8 — MISSING VALUES

Do not delete canonical rows solely because an intensive predictor is
missing.

For a numeric intensive predictor:

1. apply any fixed transformation where appropriate;
2. estimate the imputation mean from the CURRENT TRAINING PARTITION only;
3. replace missing values with that training-partition mean;
4. retain the already canonicalized source-null / coverage / valid-area
   information as predictors where applicable.

The canonical coverage/null variables allow the model to distinguish an
imputed value from an actually observed value near the mean.

Find and verify peer-reviewed supervised-prediction precedent for simple
constant/mean imputation with retained missingness information.

Do not introduce:
- spatial-neighbor imputation
- interpolation from nearby test cells
- target-informed imputation

DECISION 6A-D9 — TRAINING-ONLY STANDARDIZATION

Freeze the leakage rule:

Every learned preprocessing statistic must be estimated only from the
training partition for the model being fit.

This includes:

- imputation means
- feature means
- feature standard deviations

After fixed transforms and imputation, z-standardize continuous numeric
model inputs using training-partition statistics.

For an inner-CV fit:

  INNER TRAIN
      -> fit preprocessing
      -> transform INNER TRAIN
      -> transform INNER VALIDATION

Never fit preprocessing on:

  INNER TRAIN + INNER VALIDATION

For a final outer-fold model:

  complete usable OUTER TRAIN
      -> refit preprocessing
      -> transform OUTER TRAIN
      -> transform OUTER TEST

The frozen OUTER TEST population must never contribute preprocessing
statistics.

Document this as a leakage-prevention requirement.

DECISION 6A-D10 — INNER SPATIAL MODEL SELECTION

Primary source:

  Schratz et al. (2019)
  Hyperparameter tuning and performance assessment of statistical and
  machine-learning algorithms using spatial data
  Ecological Modelling
  DOI: 10.1016/j.ecolmodel.2019.06.002

Verify that the paper implements spatial hyperparameter tuning and
supports consistency between spatial tuning and spatial performance
assessment.

Freeze:

  five inner spatial folds

inside each usable OUTER TRAIN population.

Construct inner folds from geographic coordinates only.

Do not use:
- BP
- FLP
- WUI variables
- predictor balancing
- target balancing

The exact scalable implementation of the inner coordinate clustering
may be specified during the implementation checkpoint, but it must
remain deterministic coordinate-only spatial grouping.

Do NOT apply the outer 211.151 km dead zone recursively to the inner
folds.

Interpretation:

  INNER spatial CV:
    model/hyperparameter selection while respecting spatial structure

  OUTER frozen 211 km dead-zone CV:
    final geographic-transferability evaluation

Do not use outer TEST performance for any model-selection decision.

DECISION 6A-D11 — HPO SAMPLER

Primary Earth-system implementation:

  Lu & Ricciuto (2019)
  Efficient surrogate modeling methods for large-scale Earth system
  models based on machine-learning techniques
  Geoscientific Model Development
  DOI: 10.5194/gmd-12-1791-2019

Verify that the paper actually uses a tree-structured Parzen estimator
(TPE) to optimize neural-network hyperparameters.

Freeze:

  Optuna TPESampler
  seed = 42

TPE chooses candidate hyperparameter configurations.

Document Optuna as the engineering framework; the scientific method is
TPE-based sequential model-based hyperparameter search.

Do not use a second competing Bayesian optimizer in W&B.

DECISION 6A-D12 — RESOURCE-AWARE EARLY PRUNING

Find and verify a peer-reviewed journal implementation of ASHA for
neural-network hyperparameter optimization, preferably in an
environmental/weather/GeoAI application.

A candidate already identified is:

  Roy et al. (2024)
  Automated Model Selection Using Bayesian Optimization and the
  Asynchronous Successive Halving Algorithm for Predicting Daily
  Minimum and Maximum Temperatures
  Agriculture
  DOI: 10.3390/agriculture14020278

Verify the actual implementation before using it as primary evidence.

Freeze the intended method as:

  Optuna SuccessiveHalvingPruner

used as an ASHA-style early-pruning/resource-allocation mechanism.

Important conceptual distinction:

  TPE:
    chooses which configuration to try

  ASHA:
    decides whether a partially trained configuration receives more
    epoch resources

Weights & Biases:
    records experiments only

Do not allow W&B Sweeps to independently choose configurations.

DECISION 6A-D13 — ASHA + SPATIAL INNER-CV OBJECTIVE

A trial must not be pruned based on one geographic inner fold alone.

At each ASHA resource milestone e:

1. train/evaluate the SAME hyperparameter configuration on all five
   inner spatial folds to resource level e;
2. compute each fold's validation L_tune;
3. average across the five folds;
4. report that single five-fold mean to Optuna;
5. allow ASHA to prune or continue.

Formally:

  J_e(theta) =
    (1 / 5) * sum_f L_tune(f, e)

This prevents survival from depending on which geographic fold happens
to be evaluated first.

Treat this as a project-specific integration of:
- spatial HPO from Schratz et al.
- resource-aware pruning from the verified ASHA precedent

Do not claim that the cited papers implemented this exact combination
unless they actually did.

DECISION 6A-D14 — ROLE OF WEIGHTS & BIASES

Weights & Biases is an experiment-tracking and provenance system only.

Planned logging should include:

- study / trial ID
- hyperparameters
- inner fold
- epoch / resource level
- training losses
- validation losses
- pruning state
- runtime
- GPU utilization where available
- selected configuration
- software / git provenance

W&B does not:
- choose configurations
- prune trials independently
- alter the scientific objective

Treat this as an engineering decision, not a scientific claim.

DECISION 6A-D15 — MINIMAL HPO SEARCH FAMILY

Freeze the candidate model family sufficiently to prevent architecture
search from becoming a second research project.

Proposed search dimensions:

  hidden depth:
    {2, 3}

  shared hidden width:
    {64, 128, 256}

  learning rate:
    log-distributed in [1e-4, 1e-2]

  weight decay:
    log-distributed in [1e-5, 1e-2]

Fix:

  activation = ReLU
  optimizer family = AdamW
  dropout = 0
  BP output = sigmoid
  FLP output = six-way softmax

Do not tune:
- activation family
- optimizer family
- number of prediction heads
- loss family
- FLEP8 definition
- multitask task weight
- preprocessing family

Literature requirement:

Use Lu & Ricciuto as precedent for TPE tuning of NN depth, width, and
learning rate, but explicitly label our larger width values and
weight-decay dimension as project-specific adaptations.

Find peer-reviewed environmental/climate ML precedent for AdamW /
weight-decay tuning before marking that part LITERATURE_SUPPORTED.

If the exact proposed weight-decay decision lacks suitable evidence,
flag it for HITL rather than inventing support.

UNRESOLVED UNTIL CHECKPOINT 6B

Do NOT freeze these quantities in 6A:

- batch-size candidate set
- maximum epoch/resource budget
- number of Optuna trials
- ASHA exact maximum-resource setting
- local versus compute-cluster execution
- number of parallel workers

These will be determined prospectively using a local RTX 5080 throughput
benchmark in Checkpoint 6B.

Checkpoint 6B must evaluate computational throughput only, not select
models from validation performance.

The HPO budget must be frozen after the benchmark and before HPO results
are observed.

METHODOLOGY DOCUMENT

Create:

  docs/methodology/checkpoint_06a_training_formulation.md

It should contain:

1. Scientific purpose
2. Target factorization
3. Output heads
4. BP loss
5. FLP loss and mask
6. Derived FLEP8 quantities
7. Fixed multitask tuning objective
8. Predictor transforms
9. Missing-value treatment
10. training-only preprocessing rule
11. five-fold inner spatial model selection
12. TPE sampler
13. ASHA pruning
14. interaction between ASHA and five inner spatial folds
15. W&B role
16. provisional/frozen search dimensions
17. quantities explicitly deferred to 6B
18. relationship between inner tuning and frozen outer evaluation
19. evidence table
20. limitations / adaptations

EVIDENCE LEDGER

Update:

  docs/methodology/methodology_evidence_ledger.csv

Add one row per consequential decision.

Use statuses such as:

  LITERATURE_SUPPORTED
  HITL_PENDING
  DEFERRED_TO_6B

Do not mark any decision HITL_APPROVED until this checkpoint is reviewed.

For each row include:
- checkpoint
- decision ID
- decision
- paper
- journal
- year
- DOI
- what the paper actually implemented/established
- our implementation
- adaptation
- status

HITL STOP CONDITIONS

STOP and ask for review if:

- a cited primary source does not support the stated method
- equal BP/FLP task weighting cannot be supported or clearly justified
- a suitable peer-reviewed ASHA implementation cannot be verified
- AdamW / weight-decay tuning lacks adequate support
- the proposed inner spatial tuning design requires a new scientific
  decision
- the current canonical predictor schema conflicts with the proposed
  preprocessing rules
- any preprocessing step risks outer-test leakage
- implementation details would require changing the frozen outer 5B
  design
- a consequential scientific choice not listed here is required

FINAL HITL REPORT

When complete, report:

1. Base commit SHA
2. Every 6A decision and final status
3. Verified primary citation for each literature-supported decision
4. Any project-specific adaptation
5. Exact training objective now frozen
6. Exact preprocessing rules now frozen
7. Exact inner spatial-CV rule now frozen
8. Exact HPO sampler/pruner design now frozen
9. Exact hyperparameter dimensions frozen
10. Every quantity deferred to 6B
11. Any unsupported or ambiguous decision
12. Files changed
13. git diff --check status

Then STOP for HITL review.

Do not train a model.
Do not benchmark the GPU.
Do not implement HPO.
Do not commit.
Do not push.
