# Phase 7B: prespecified HPO ranking objective

## Decision and scientific reason

The wildfire study asks for useful probabilistic predictions of both burn
probability (BP) and conditional flame-length class (FLP), including geographic
generalization. We value these **two tasks equally** in model selection. Their
raw cross-entropies have different numerical scales, so equal coefficients on
raw BP and FLP losses would give unequal influence to a comparable fractional
improvement. We rank candidate models by their mean improvement over a
task-specific, inner-TRAIN-derived constant forecast on geographically held-out
inner VALIDATION. This fixes the rule before any GradNorm HPO trial starts.

[Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437)
supports the strictly proper logarithmic score: Bernoulli cross-entropy for BP
and categorical cross-entropy for FLP. This is the existing frozen training and
validation score, not a newly selected alternative. [Roulston and Smith
(2002)](https://www.lse.ac.uk/CATS/Assets/PDFs/Publications/Papers/2002/47-EvalProbFCsUsingInfoTheory-2002.pdf)
motivates comparison of logarithmic forecast quality with reference predictions.
[Vandenhende et al. (2022), Eq. 10](https://doi.org/10.1109/TPAMI.2021.3054719)
provides multi-task evaluation precedent for averaging task performance relative
to task baselines; their baselines are single-task models, whereas ours are
inner-TRAIN constant forecasts. **The two reference definitions, equal BP/FLP arithmetic mean,
and equal fold weighting below are project decisions**, not rules attributed to
those papers. Prior citations for frozen spatial folds, preprocessing, target
masks, and GradNorm remain in the [methodology ledger](methodology_evidence_ledger.csv).

## Exact rule

For outer study `o`, inner fold `k`, and task `t ∈ {BP, FLP}`, let
`T_ok` be that fold's inner TRAIN and `V_ok` its inner VALIDATION. Define the
constant BP forecast as the mean soft BP target over **all BP rows in `T_ok`**.
Define the six-class FLP forecast as the componentwise mean soft FLP target
over **FLP-valid rows in `T_ok`**. These probabilities are held fixed across
models and epochs. Score each constant on the matching task support in `V_ok`,
including the existing FLP-validity mask. With natural-log cross-entropy,

```text
skill_t,o,k(e) = 1 - CE_model,t,o,k(e) / CE_reference,t,o,k
J_o,k(e)       = 0.5 * (skill_BP,o,k(e) + skill_FLP,o,k(e))
J_o(e)         = (1/5) * Σ_(k=1..5) J_o,k(e)
```

**Maximize** `J_o(100)` for the separate outer study's HPO ranking. Optuna's
new GradNorm study uses maximize direction. Each epoch record retains BP and
FLP raw CE, reference CE, component skills, and fold `J`; the common-epoch
record retains the five fold scores and their mean. References and their
support counts enter immutable study provenance. A nonfinite or nonpositive
reference CE, empty task support, or mismatched support fails explicitly. No
VALIDATION targets are used to estimate the constant probabilities. VALIDATION
targets necessarily enter their reference *scores*, just as they enter model
scores.

Equal weighting of the five frozen inner geographic folds keeps each held-out
region represented once, irrespective of support size. The positive mean skill
is the scientific objective; raw CE and excess log loss remain available for
interpretation. GradNorm's adaptive **training** weights do not enter this
**model-selection** equation. We retain the existing 100-epoch cap and
`NopPruner`; no early selection rule or loss-weight search is added. Exact
objective ties are co-best; if one trial identifier is needed for a reproducible
artifact, use the smallest trial number without claiming scientific superiority.

## Assumptions and limits

Relative skill is zero for the constant TRAIN reference when evaluated on the
same VALIDATION support; positive values indicate improvement, and negative
values indicate degradation. A common natural-log base cancels in each ratio.
Equal *relative* skill does **not** imply equal absolute predictive accuracy,
calibration, or practical wildfire consequence. A task with a weak or strong
constant reference can have a different absolute CE at the same relative skill.
The objective therefore requires reporting both raw task scores and the
high-risk BP adequacy diagnostics alongside the selected model. The
[first diagnostic](../../reports/training/07b/README.md) found high-risk BP
underprediction on TRAIN and VALIDATION in one fold; the task-balanced ranking
rule does not by itself repair it. Geographic outer TEST remains excluded from
model selection. No HPO search has launched under this objective.

## Engineering verification

The reference calculation reads the frozen cache labels and targets, checks
all five fold support counts against the frozen 6C report, and records the
training probabilities and validation reference CEs in study metadata.
Unit tests verify TRAIN-only references, separate FLP masks, exact skill
arithmetic, fold aggregation, maximize direction, and invalid-reference and
support-mismatch failures. The established 6E/6F raw-CE test paths retain
their historical minimization behavior. The new GradNorm study identity is
`wildfire_outer_<o>_gradnorm_hpo_v3` with a distinct SQLite file, preventing
accidental continuation of an objective with different meaning.
