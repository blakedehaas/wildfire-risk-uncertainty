# Phase 12 — conditional flame-length reconstruction and uncertainty reliability

This analysis asks whether disagreement among five frozen ensemble members identifies poor reconstruction of FSim's **six-class conditional flame-length probability distribution** (FLP1–FLP6). Each eligible cell has six reference probabilities and six predicted probabilities. The task is to reconstruct the whole distribution, **not** to predict one dominant class or an observed wildfire outcome. The five geographic TEST studies and all predictions come from Phase 9; no models or folds were changed.

## 1. How well are conditional flame-length distributions reconstructed?

**Total variation (TV)** is half the sum of the six absolute predicted-minus-FSim probability differences. It ranges from 0 for identical distributions to 1 for disjoint distributions. The comparison baseline always predicts the mean FLP distribution among that study's **FLP-valid, eligible TRAIN cells**. The baseline never uses TEST targets to set its six probabilities.

| Study | FLP-valid TEST cells | Ensemble mean TV | TRAIN-constant mean TV | Ensemble reduction |
|---:|---:|---:|---:|---:|
| 1 | 16,698,069 | 0.4578 | 0.4634 | 1.2% |
| 2 | 16,787,874 | 0.4565 | 0.4789 | 4.7% |
| 3 | 16,889,664 | 0.4051 | 0.4414 | 8.2% |
| 4 | 12,197,428 | 0.4186 | 0.4643 | 9.8% |
| 5 | 18,495,738 | 0.5053 | 0.5831 | 13.3% |
| Equal-study mean | — | **0.4487** | **0.4862** | — |
| Pooled cell-weighted | 81,068,773 | **0.4515** | **0.4895** | **7.8%** |

The ensemble improves on this simple TRAIN-only baseline in every study, though TV remains substantial and the gain is small in study 1. [Figure 1: ensemble and baseline mean TV](reconstruction_performance.png) ([PDF](reconstruction_performance.pdf)) shows the geographic variation. Phase 9's saved per-class MAE and signed bias are reproduced in [study_summary.csv](study_summary.csv). Signed bias is predicted minus FSim probability; for example, FLP1 bias is +0.193 in study 1 and −0.264 in study 5. A single modal-class accuracy would hide these distributional errors.

## 2. Does ensemble disagreement increase with reconstruction error?

**Disagreement** is the sum of the six Phase 9 population variances across the five members, one variance per FLP class. It is a proxy for epistemic uncertainty among these fitted models, not a calibrated uncertainty estimate. Error is each cell's TV. The full-support within-study Spearman rank correlations between disagreement and TV are **−0.048, +0.055, −0.031, +0.083, and −0.123** for studies 1–5. Three are negative and all are weak. Equal-count disagreement quintiles likewise have no consistent rise in mean TV: study 5 falls from **0.562** in the lowest quintile to **0.480** in the highest, while study 4 rises from **0.413** to **0.451**.

[Figure 2: mean TV by disagreement quintile](disagreement_vs_error.png) ([PDF](disagreement_vs_error.pdf)) shows all five study profiles; exact values and counts are in [disagreement_quintiles.csv](disagreement_quintiles.csv). Higher disagreement therefore does **not** consistently indicate larger FLP reconstruction errors on these held-out studies. The rank correlations are descriptive across spatially dependent cells; they are not independent-pixel inference.

## 3. Can disagreement identify especially large errors?

Within each study, **high-error cells** are the largest `round(0.10 × valid TEST count)` TV errors. We reject the largest `round(fraction × valid TEST count)` ranking scores at 5%, 10%, 20%, and 50%. A score tie goes to the smaller `cell_id`; the same rule fixes the high-error group. **Recall** is the share of high-error cells rejected. **Remaining mean TV** is mean TV among cells kept. Random rejection in expectation leaves mean TV unchanged and has recall equal to the rejected fraction.

At the principal **20% rejection** point:

| Study | Disagreement recall | Predicted FLP entropy recall | Random expected recall | Disagreement remaining TV | Entropy remaining TV |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.8% | 15.9% | 20% | 0.4581 | 0.4705 |
| 2 | 18.8% | 19.0% | 20% | 0.4540 | 0.4564 |
| 3 | 21.3% | 34.0% | 20% | 0.4020 | 0.3912 |
| 4 | 20.5% | 24.5% | 20% | 0.4105 | 0.4155 |
| 5 | 8.9% | 63.2% | 20% | 0.5117 | 0.4757 |
| **Equal-study mean** | **18.2%** | **31.4%** | **20%** | **0.4473** | **0.4418** |
| **Pooled cell-weighted** | **17.9%** | **32.4%** | **20%** | **0.4506** | **0.4440** |

The equal-study random expected remaining TV is **0.4487**. At 20% rejection, disagreement finds fewer high-error cells than random in three studies and on the equal-study mean; it also raises remaining TV in studies 1 and 5. Entropy performs better on the equal-study summary, with pronounced variation, especially between studies 1 and 5. Entropy here is the Shannon entropy of the **ensemble-mean six-class distribution**. It measures predictive diffuseness, not epistemic uncertainty. [Figure 3: high-error recall at 20% rejection](high_error_recall_20pct.png) ([PDF](high_error_recall_20pct.pdf)) displays study variation and the equal-study summary. The full rejection fractions, remaining TV, and recall are in [rejection_performance.csv](rejection_performance.csv) and [rejection_aggregates.csv](rejection_aggregates.csv).

## Relation to burn-probability findings and limits

Phase 10A found **positive BP disagreement–absolute-error Spearman correlations in all five studies** (0.275–0.696). At 20% rejection, BP disagreement found **25.0%** of high BP errors on the equal-study mean, versus **57.7%** for predicted BP/Bernoulli entropy and **20%** for random expectation. FLP disagreement is weaker: its corresponding recall is **18.2%**, versus **31.4%** for FLP entropy and **20%** for random. These comparisons concern different targets and error scales: **TV compares six conditional probabilities; absolute BP error compares two burn probabilities.** The numbers should not be treated as directly interchangeable error magnitudes.

All FLP calculations exclude invalid or inapplicable conditional targets. Phase 12 verified each prediction-file SHA-256 sidecar, cell-ID order and TEST membership, Phase 9 partition ID hash, the canonical FLP-valid mask, finite normalized target and predicted distributions, and the saved FLP support and mean TV (tolerance 2 × 10⁻⁷). It also checked the frozen manifest and a deterministic two-tile pilot. The five studies, rather than individual cells, are the meaningful geographic reporting units; they have overlapping training sets and spatially dependent cells. This is descriptive reliability against FSim reference distributions, not calibration of epistemic uncertainty, a causal test, or validation against realized fire outcomes.

Reproduce with `.venv/bin/python -m wildfire_risk_uncertainty.analysis.phase12` from the repository root. Run `.venv/bin/pytest -q tests/analysis/test_phase12.py` and `.venv/bin/ruff check src/wildfire_risk_uncertainty/analysis/phase12.py tests/analysis/test_phase12.py`. [Provenance](provenance.json) identifies the reused artifacts, support, pilot, and exact definitions.
