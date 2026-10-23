# Phase 7B handoff: model selection and predictive adequacy

**Starting point:** the `dev` commit containing this handoff (`git rev-parse HEAD`);
the preceding implementation/recovery commit is
`d08a42b99400a6e29f1e30b0966eff3c9d0c56ba`. Phase 7A is complete for probabilistic
losses, fixed literature-standard GradNorm, and production training validation.
Its [100-epoch evidence](../../reports/training/07a/gradnorm_full_fold_review.md)
documents a recovered, complete single inner-fold run. No HPO study or outer
TEST evaluation was launched in Phase 7A.

## Frozen decisions and evidence

- The canonical 33 SILVIS WUI inputs, probabilistic BP and conditional FLP1–6
  targets, FLP-validity mask, training-only mean imputation, coverage indicators,
  spatial dead zones, five outer TRAIN/TEST partitions, and five inner folds per
  outer TRAIN remain frozen. Outer TEST is excluded from diagnostics and model
  selection.
- BP is one Bernoulli prediction task and FLP is one six-class categorical
  prediction task. Training uses masked cross entropy; independent BP and FLP
  validation excess log losses are reported on their respective valid supports.
- [Fixed GradNorm](checkpoint_07a_gradnorm.md) uses `gradnorm_alpha=1.5`, a
  separate weight Adam optimizer at learning rate 0.025, first-epoch equal
  weights and separate-support CE references, shared gradients at the final
  backbone Linear weight matrix, positive task weights summing to two, and the
  existing model AdamW and BF16 CUDA pipeline. Do not search GradNorm settings
  or revive the unexecuted static-alpha Pareto sweep.
- The 06F epoch-4 pruning test rejected early pruning. The prepared GradNorm
  HPO study uses `NopPruner` and a 100-epoch cap. Existing two-worker resource
  gates, seeds, architecture/optimizer search spaces, and nested fold boundaries
  are preserved. The existing 30-trial-per-outer-study ceiling is a planning
  bound, not authorization to launch HPO.
- The completed run is
  `artifacts/benchmark/07a_detached/07a-20261022T054112Z-5270cc63/`.
  `config.json`, `source_epochs_001_to_003.json`, `fold_1_epoch_004.json` through
  `fold_1_epoch_100.json`, corresponding checkpoints and SHA-256 sidecars,
  `fold_1.json`, resource samples, logs, and recovered status are retained.
  Its original finalizer exit code remains visible; recovery verified 100
  checkpoint hashes and restored the final checkpoint without retraining.

## Ordered Phase 7B work

1. **Existing-checkpoint TRAIN versus geographic VALIDATION diagnostic.** Use
   the already saved checkpoints from this one representative inner fold.
   Calculate BP and FLP predictive scores on their frozen valid supports on
   inner TRAIN and inner VALIDATION, applying the same preprocessing state and
   reductions. For BP, compare calibration, errors, and discrimination overall
   and within the predeclared target-risk strata, including BP ≥ 0.02. Include
   the training-derived constant BP climatology and partition-level spatial
   discrimination. Record train/validation support counts and evaluate selected
   existing epochs across the trajectory. Do not fit, retune, or retrain models
   during this diagnostic. Diagnose whether the elevated-risk gap is already
   present on TRAIN or emerges primarily under geographic transfer. Treat
   target-defined strata as descriptive, not a tuning objective.
2. **Predefine an HPO ranking criterion.** Review published probabilistic
   forecast scoring and multi-output model-selection methods. Choose and
   document a literature-supported rule that gives the equally important BP
   and FLP tasks defensible influence, respects their separate valid supports,
   and does not silently favor FLP because its CE is numerically larger.
   Distinguish this model-ranking rule from GradNorm's training weights.
   Record its formula, scaling/reference choices, aggregation across the five
   inner folds, tie handling, and limitations in the evidence ledger **before**
   any HPO launch. The existing `0.5·BP CE + 0.5·FLP CE` ranking remains under
   review; no replacement is selected by this handoff.
3. **Bounded nested HPO, after the diagnostic and criterion review.** Use only
   each outer TRAIN's five inner folds for trials and ranking, the fixed GradNorm
   method, the existing architecture/AdamW search space and resource gates,
   full 100-epoch fidelity without epoch-4 pruning, the established trial
   ceiling, detached execution, complete provenance, and checkpoint resume.
   Decide and record a finite initial launch scope and stopping conditions
   before spending the full five-study budget. Do not inspect outer TEST for
   trial selection.

Later uncertainty evaluation must stratify performance by missing-feature
pattern and predictor coverage. Final ensembles and outer TEST evaluation are
outside this handoff and require their own approved protocol.

## Open scientific question

The epoch-100 geographic VALIDATION BP ≥ 0.02 subset has mean soft target
0.03643 versus mean prediction 0.00343 and weak within-stratum association
(cell Pearson 0.0636; equal-partition Pearson −0.099). Full-support BP excess
log loss still beats the inner-TRAIN climatology (0.008574 versus 0.011518).
This combination needs a TRAIN-versus-VALIDATION adequacy analysis. It is not
classified as a Phase 7A loss, GradNorm, numerical, or checkpoint defect. One
geographic validation fold does not establish performance on the other folds
or on outer TEST.

## Working paths and restrictions

| Purpose | Path |
|---|---|
| Phase 7A method | [`checkpoint_07a_gradnorm.md`](checkpoint_07a_gradnorm.md) |
| 100-epoch review | [`gradnorm_full_fold_review.md`](../../reports/training/07a/gradnorm_full_fold_review.md) |
| Three-epoch source | [`gradnorm_ce_reference_fold1.json`](../../reports/training/07a/gradnorm_ce_reference_fold1.json) |
| Training and GradNorm | `src/wildfire_risk_uncertainty/training/{trainer,gradnorm,cache_checkpoint}.py` |
| Detached run and recovery | `scripts/{benchmark_07a_detached,gradnorm_full_fold}.py` |
| HPO orchestration | `src/wildfire_risk_uncertainty/hpo/{real,storage}.py` |
| Decision ledger | [`methodology_evidence_ledger.csv`](methodology_evidence_ledger.csv) |

No retraining, GradNorm changes, outer TEST access, HPO launch, Pareto sweep,
or final ensemble training was authorized by this closeout. The first Phase 7B
action is read-only evaluation of existing checkpoints on inner TRAIN and
geographic VALIDATION.

## Phase 7B progress after this handoff

The [first predictive-adequacy diagnostic](../../reports/training/07b/README.md)
was completed from saved checkpoints at commit `7d72ccd`; its elevated-risk BP
underprediction appears on inner TRAIN as well as geographic VALIDATION. The
subsequent user direction prespecified the [reference-relative HPO ranking
objective](checkpoint_07b_hpo_objective.md), with equal BP and FLP relative
logarithmic skill averaged over the five inner folds. This update records the
new decision without rewriting the authoritative Phase 7A handoff above.
HPO launch, outer TEST evaluation, and additional methodological experiments
remain outside the present work.

The subsequent [production-readiness audit](../../reports/hpo/07b/production_readiness.md)
proposes a first bounded 20-trial outer study, with explicit trial-cap controls,
resource estimates, status commands, and preserved no-pruning execution.
It has not launched HPO or authorized the remaining outer studies.

The subsequent persistence audit keeps the scientific objective and search
space frozen. It records the initially approved 20-trial lifetime cap separately
from immutable study provenance, with explicit controlled extensions and
hash-checked fold checkpoint recovery. The
[operator guide](../../reports/hpo/07b/production_readiness.md#operator-commands)
documents the commands and reproducibility limits. No HPO study was launched
or extended during this audit.
