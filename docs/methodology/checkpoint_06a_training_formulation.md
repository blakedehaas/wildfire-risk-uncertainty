# Checkpoint 6A: training formulation, preprocessing, and HPO methodology

**Status: HITL approved and frozen.** No model, HPO study, benchmark, or prediction was run in this checkpoint. Base commit: `704070193d325903cbcc325f04ff218a3ddb8def` (`dev`, equal to `origin/dev`, clean at preflight).

## 1. Scientific purpose and estimand

The model reconstructs the Dillon/FSim probabilistic wildfire-risk surface from SILVIS-derived human–environment predictors. Its targets are supplied simulation products, **not observed future-fire truth**. The principal targets are burn probability (`BP`) and the six conditional flame-length probabilities (`FLP1`–`FLP6`). The model family is deliberately a simple shared feed-forward MLP; the scientific aim is later uncertainty decomposition and SHAP analysis, neither of which is implemented in 6A.

## 2. Factorization, heads, and support

For predictors `x`, the shared MLP representation feeds a one-logit BP head and a six-logit FLP head:

```text
SILVIS predictors -> shared MLP -> BP logit -> sigmoid: p_hat = P(burn | x)
                              \-> six FLP logits -> softmax: q_hat_k = P(class k | burn, x)
```

The softmax must sum to one. The canonical BP target `p` remains a floating-point probability. The canonical six-element `q` remains a probability vector: no BP thresholding, FLP argmax, or FLP target renormalization. The canonical `flp_target_valid` flag is the sole FLP loss mask. It already requires BP > 0, all six finite/in-range FLPs, and a float64 sum within `1e-6` of one. BP-valid canonical rows remain eligible for BP loss even when FLP is masked. The current full-support audit reports 110,817,231 BP-valid rows and 81,068,773 strict FLP-valid rows; these are dataset properties, not task weights.

[Marullo et al. (2023)](https://doi.org/10.3390/jpm13030413) implemented a shared multitask network with sigmoid binary and softmax multiclass outputs. Their tasks were surgical image tasks, so the two-head probability parameterization is precedent for implementation, not wildfire evidence. [Bakker and Heskes (2003)](https://jmlr.org/papers/v4/bakker03a.html) modeled tasks with shared network parameters and task-specific output weights; our two-head MLP is a simpler project adaptation.

[Preisler et al. (2004)](https://doi.org/10.1071/WF02061) modeled fire occurrence, conditional probability of a large fire given ignition, and their unconditional product. It did **not** model FSim BP or a six-class conditional flame-length distribution; those are project adaptations. The conditional factorization implies that FLP loss is averaged over strict FLP-valid support without multiplying by BP. BP-weighting would target a burn-frequency-weighted conditional distribution, closer to unconditional intensity risk, instead of estimating the supplied conditional distribution with equal weight per eligible cell. This exact mask/weight rule is a mathematical consequence of the chosen estimand, not a claim that Preisler implemented this loss.

There is no FLEP8 head. Derive `conditional_FLEP8 = q_hat_5 + q_hat_6` and `unconditional_FLEP8 = p_hat * (q_hat_5 + q_hat_6)`. For canonical targets, the 5B definition remains: conditional FLEP8 is defined only on strict FLP-valid support; unconditional FLEP8 is zero when BP is zero, equals `BP*(FLP5+FLP6)` on BP-positive strict-valid support, and is undefined on BP-positive invalid FLPs. This preserves the distinction between a model's always-normalized outputs and target support.

## 3. Probabilistic losses and fixed objective

Use the negative logarithmic score on the supplied probability targets. For one BP-valid row,

`L_BP = -[p log(p_hat) + (1-p) log(1-p_hat)]`.

For one strict FLP-valid row,

`L_FLP = -sum_{k=1}^6 q_k log(q_hat_k)`.

Numerically stable BCE-with-logits and log-softmax implementations are permitted. [Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437) establish strict propriety of the logarithmic score for probability distributions. Applying its expected-score form to the supplied soft BP and FLP targets is our project-specific use; it does not turn simulation probabilities into observed outcomes.

For **both training and inner model selection**, the scalar objective is

`L_tune = 0.5 * mean_{BP-valid}(L_BP) + 0.5 * mean_{strict-FLP-valid}(L_FLP)`.

Each task mean uses its own valid denominator before combination. A minibatch or streaming implementation must preserve these task weights across the complete training partition rather than letting the more numerous BP rows dominate by accident. The same definition is used for every HPO trial, fold, and resource level; BP-vs-FLP weight is not tuned. This is a deliberately simple, fixed multitask baseline that keeps the optimization criterion constant across all HPO trials. [Hervella et al. (2024)](https://doi.org/10.1016/j.neunet.2023.11.038) implemented equal task weights as a baseline alongside adaptive alternatives, showing that task balancing is itself a substantive optimization problem. The paper does **not** establish equal BP/FLP weighting as optimal. The accepted `0.5/0.5` choice is a project-specific policy for methodological scope and transparency: task-weight optimization is not a research question of this wildfire uncertainty study.

## 4. Predictor preprocessing

The four retained nonnegative intensive densities `HUDEN2020`, `POPDEN2020`, `OCCHUDEN2020`, and `VACHUDEN2020` receive fixed `log1p(x)` before any learned scaling. A negative finite value in any of these fields is a data-contract failure, not a value to clip or silently transform. Other intensive inputs stay on their canonical scale before standardization; bounded categorical shares and fractions are not log-transformed merely for symmetry. [Joseph et al. (2019)](https://doi.org/10.1002/eap.1898) modeled wildfire outcomes using log housing density; [Sze et al. (2019)](https://doi.org/10.1088/1748-9326/aaee1d) log-transformed population density and standardized predictors. Neither prescribes `log1p` for these four SILVIS fields. The zero-preserving `log1p` and choice of all four densities are project adaptations informed by their skew and nonnegative support.

Do not delete a canonical row only because an intensive predictor is missing. After any fixed transform, fit the observed-value mean for each numeric intensive predictor on the **current training partition** and fill missing values with that mean. Retain relevant canonical source-null, coverage, and valid-area information as model inputs where applicable, so a missing/imputed value can be distinguished from an observed value near the mean. The schema stores density coverage/null columns with `diagnostic` roles; selecting these existing columns as model inputs does not alter canonical data or the frozen 5B folds. Categorical share and coverage columns can also be null on a WUI geometry/topology failure. Apply the same training-partition mean-imputation rule to any nullable numeric share/coverage column selected as an input and retain the canonical, never-null `wui_failure` flag as an input; this extension is a project-specific treatment of the existing schema. A column with no observed value in the current training partition is a data-contract failure requiring review, not an invitation to use validation/test values. No spatial-neighbor interpolation or target-informed imputation is allowed. [Josse et al. (2024)](https://doi.org/10.1007/s00362-024-01550-4) evaluate constant imputation with missingness indicators, discuss when missingness information helps prediction, and explicitly use training-set imputation values on test data. They do not claim mean imputation is universally best.

Every learned preprocessing statistic is fitted only on the training partition for the model being fitted: imputation means, feature means, and feature standard deviations. After fixed transforms and imputation, z-standardize continuous numeric model inputs with those training statistics. A zero-variance feature requires a deterministic no-division convention (e.g., scale of one); this is an implementation safeguard, not an HPO choice. For an inner fit, fit preprocessing on INNER TRAIN and transform INNER TRAIN and INNER VALIDATION. For a final outer-fold fit, refit on the complete usable OUTER TRAIN and transform OUTER TRAIN and OUTER TEST. The frozen OUTER TEST never contributes a learned statistic. [Moscovich and Rosset (2022)](https://doi.org/10.1111/rssb.12537) demonstrate that even unsupervised preprocessing, including feature rescaling, before cross-validation can bias validation and model selection; [Josse et al. (2024)](https://doi.org/10.1007/s00362-024-01550-4) specifically use training-set imputation values on test data.

The existing canonical schema includes raw intensive values, area fractions, source-null fractions, WUI coverage and category shares, the `wui_failure` flag, and strict target masks. `x_5070` and `y_5070` are spatial-splitting coordinates and provenance, not initial MLP predictors. A final explicit predictor column list remains an implementation detail subject to this leakage and missingness contract; it must not silently introduce target or outer-test information.

## 5. Inner spatial selection and frozen outer assessment

Inside each usable OUTER TRAIN population, use exactly **five deterministic, coordinate-only inner spatial folds** for model/hyperparameter selection. Inner groups may be constructed with a scalable deterministic coordinate-clustering implementation specified at implementation time. Only `x_5070` and `y_5070` may determine inner grouping; BP, FLP, WUI variables, predictor balancing, and target balancing are excluded. No 211.151 km dead zone is applied recursively to inner folds. [Schratz et al. (2019)](https://doi.org/10.1016/j.ecolmodel.2019.06.002) implemented spatial hyperparameter tuning and spatial performance assessment and recommended spatial partitioning for tuning spatial data. The fixed five-inner-fold count and no-inner-dead-zone choice are project-specific choices, not details prescribed by that paper.

The accepted 5B outer design remains untouched: five coordinate-defined outer test memberships, a strict Euclidean training exclusion at the exact BP-derived radius `211150.99027309995 m` (displayed as 211.151 km), and one outer TEST occurrence per canonical cell. Inner spatial CV chooses a configuration using only each outer training population. The selected configuration is then refit on that outer fold's complete usable TRAIN population with freshly fitted preprocessing. Outer TEST is used only for final geographic-transferability evaluation and never for model selection or preprocessing fitting.

## 6. HPO, pruning, and experiment records

Use Optuna `TPESampler(seed=42)` to choose configurations. [Lu and Ricciuto (2019)](https://doi.org/10.5194/gmd-12-1791-2019) used TPE to tune neural-network hidden-layer count, units per layer, and learning rate for an Earth-system surrogate. Optuna and seed 42 are engineering choices; TPE-based sequential model-based search is the scientific precedent. No second Bayesian optimizer runs through W&B.

Use Optuna `SuccessiveHalvingPruner` as an ASHA-style epoch-resource allocation mechanism; [Optuna's own documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html) identifies it as an asynchronous successive-halving implementation and notes that the objective must enforce maximum resource. [Roy et al. (2024)](https://doi.org/10.3390/agriculture14020278) compared Bayesian and ASHA hyperparameter optimization for daily temperature forecasting; their candidate set included an ANN with tunable layer sizes and other ANN parameters. Their selected best models were mostly GPR or ensemble models, and they did **not** implement this TPE-plus-ASHA-plus-spatial-CV combination. They also discuss use of test errors in selection, which this project explicitly prohibits. The source supports environmental ANN candidate tuning under ASHA, not our exact selection protocol.

For a trial configuration `theta` and every ASHA resource milestone `e`, fit/evaluate **the same configuration on all five inner spatial folds** through resource level `e`, calculate each fold's validation `L_tune(f,e)`, then report exactly one value to Optuna:

`J_e(theta) = (1/5) * sum_{f=1}^5 L_tune(f,e)`.

Pruning is allowed only after this complete five-fold mean is available. One geographic fold alone must never determine survival. This is a project-specific integration of spatial HPO and ASHA; neither cited paper implemented this exact combination. The implementation must make epoch/resource comparisons fair across trials and preserve each fold's training trajectory between milestones. Exact ASHA maximum resource and other compute settings remain open until 6B.

W&B records study/trial ID, hyperparameters, inner fold, epoch/resource level, training and validation losses, pruning state, runtime, GPU utilization where available, selected configuration, software versions, and git provenance. W&B neither selects configurations nor prunes nor changes the objective. This is an engineering provenance decision.

## 7. Search family and 6B deferrals

The accepted HPO family is deliberately bounded:

| Dimension | Frozen 6A choice |
| --- | --- |
| Shared hidden depth | `{2, 3}` |
| Shared hidden width | `{64, 128, 256}` |
| Learning rate | log-distributed `[1e-4, 1e-2]` |
| AdamW weight decay | log-distributed `[1e-5, 1e-2]` |
| Activation / optimizer / dropout | ReLU / AdamW / 0, fixed |
| Outputs / losses / task weight | one sigmoid BP and one six-way softmax FLP / soft logarithmic scores / `0.5, 0.5`, fixed |

Do not tune activation family, optimizer family, head count, loss family, FLEP8 definition, task weight, or preprocessing family. Lu and Ricciuto support searching NN depth, width, and learning rate, but their tested widths were smaller and their optimizer was Adam. [Laloyaux et al. (2022)](https://doi.org/10.1029/2022MS003016) used AdamW and explicitly tuned weight decay for a weather-model bias neural network. The frozen width values, AdamW weight-decay interval, fixed ReLU, and zero dropout are project-specific bounded choices; the papers do not validate these exact ranges as optimal here.

**Deferred to 6B:** batch-size candidates; maximum epoch/resource budget; number of Optuna trials; ASHA exact maximum-resource setting; local versus compute-cluster execution; and number of parallel workers. The local RTX 5080 benchmark in 6B must measure throughput only, without choosing models from validation performance. Freeze the HPO budget after that benchmark and before HPO results are observed.

## 8. Evidence and limits

| Decision | Verified primary precedent | Evidence type | HITL status | Project-specific boundary |
| --- | --- | --- | --- | --- |
| D1 | Preisler et al. 2004, `10.1071/WF02061` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Six FSim FLP classes and derived FLEP8 |
| D2 | Marullo et al. 2023, `10.3390/jpm13030413` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Sigmoid BP plus six-way conditional FLP |
| D3 | Gneiting and Raftery 2007, `10.1198/016214506000001437` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Expected logarithmic score on simulator probabilities |
| D4 | Preisler et al. 2004, `10.1071/WF02061` | `MATHEMATICALLY_DERIVED` | `HITL_APPROVED` | Strict FLP mask and no BP multiplier follow from our conditional estimand |
| D5 | Bakker and Heskes 2003, [JMLR article](https://jmlr.org/papers/v4/bakker03a.html) | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Shared MLP plus two task heads |
| D6 | Hervella et al. 2024, `10.1016/j.neunet.2023.11.038` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Fixed baseline for scope and transparency; not an empirical optimum |
| D7 | Joseph et al. 2019, `10.1002/eap.1898`; [Sze et al. 2019](https://doi.org/10.1088/1748-9326/aaee1d) | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | `log1p` on four SILVIS densities |
| D8 | Josse et al. 2024, `10.1007/s00362-024-01550-4` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Canonical coverage/null diagnostics used as inputs |
| D9 | Moscovich and Rosset 2022, `10.1111/rssb.12537`; Josse et al. 2024, `10.1007/s00362-024-01550-4` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Fold-specific z-scaling and imputation rule |
| D10 | Schratz et al. 2019, `10.1016/j.ecolmodel.2019.06.002` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Five inner coordinate-only groups; no inner dead zone |
| D11 | Lu and Ricciuto 2019, `10.5194/gmd-12-1791-2019` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Optuna implementation and seed 42 |
| D12 | Roy et al. 2024, `10.3390/agriculture14020278` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Optuna pruning rather than their MATLAB comparison |
| D13 | Schratz et al. 2019; Roy et al. 2024 | `PROJECT_SPECIFIC_ADAPTATION` | `HITL_APPROVED` | Five-fold mean reported at every ASHA milestone |
| D14 | Engineering provenance choice | `ENGINEERING_DECISION` | `HITL_APPROVED` | W&B records only |
| D15 | Lu and Ricciuto 2019; Laloyaux et al. 2022, `10.1029/2022MS003016` | `LITERATURE_SUPPORTED` | `HITL_APPROVED` | Exact search ranges, wider widths, ReLU, zero dropout |

The 6B compute quantities have ledger status `DEFERRED_TO_6B`.

The [evidence ledger](methodology_evidence_ledger.csv) contains one `06A-D1`–`06A-D15` row per decision, plus a `06A-6B` deferral row. Every D1–D15 decision is `HITL_APPROVED`; the separate evidence type preserves whether a precedent is literature-supported, a rule is mathematically derived, an integration is project-specific, or a framework choice is engineering. `LITERATURE_SUPPORTED` means the cited paper establishes or implements only the **precise precedent described in that row**, not every project adaptation. In particular, the equal-weight objective, five-fold ASHA aggregation, no recursive inner dead zone, exact search ranges, and Optuna/W&B package choices remain project-specific or engineering choices. The source-null and coverage columns are canonical diagnostics that can be used as model inputs without changing source data; their final inclusion list must be explicit before training.

HITL approval freezes this methodology. It does not supply fitted models, empirical validation performance, uncertainty estimates, SHAP values, or a compute budget. Checkpoint 6B is next and will prospectively determine the deferred compute quantities from throughput alone.
