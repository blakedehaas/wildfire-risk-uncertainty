# Phase 7D: compute-efficient HPO retrospective

## Scope and reproduction

This is a read-only analysis of the completed `wildfire_outer_3_gradnorm_hpo_v3` pilot at analysis commit `725d88b24a0c147f920a41da7754b4de3c4f4d46`. Run `uv run python scripts/analyze_07d_compute_reduction.py` from the repository root. The script reads the Optuna SQLite database in read-only mode through the Phase 7C loader, verifies every completed configuration and epoch-100 objective against Optuna, recomputes task skills from saved CE and frozen reference CE for all five folds at epochs 10, 25, 50, 75, and 100, and checks saved aggregate objectives. It writes only this directory. [The script](../../../scripts/analyze_07d_compute_reduction.py) and [arithmetic tests](../../../tests/hpo/test_07d_analysis.py) specify tie breaking and regret. The [audit](trial_audit.csv) retains failures 1 and 17; the 20 successful trials alone enter performance calculations. No outer-test rows, targets, or outcomes are read. No HPO, model fit, or protocol change occurred.

The original objective remains the equal-fold average of the equal-task BP/FLP validation log skills, each `1 − model CE / fold-specific constant-reference CE`. Scores are inner-validation skills, not outer geographic performance or a proof of calibration. All regrets below are differences in this frozen objective. The five geographic folds are fixed regions, not independent random replicates.

## 1. Budget sensitivity

[The original completion trajectory](budget_trajectory.csv) and [figure](figures/budget.png) separate the ten successful random startup suggestions from ten subsequent adaptive TPE suggestions. Failed trial 1 did not count toward TPE startup. Original-order results:

| Completed successes | Last Optuna trial | Phase | Incumbent trial | Skill | Gap to final best |
|---:|---:|---|---:|---:|---:|
| 3 | 3 | startup | 0 | 0.104496 | 0.003717 |
| 5 | 5 | startup | 0 | 0.104496 | 0.003717 |
| 8 | 8 | startup | 0 | 0.104496 | 0.003717 |
| 10 | 10 | startup | 0 | 0.104496 | 0.003717 |
| 15 | 15 | TPE | 15 | 0.105222 | 0.002991 |
| 20 | 21 | TPE | 19 | 0.108213 | 0 |

Thus three to ten successes in this realized sequence found a configuration within 0.003717 skill of the final best, about 3.44% of the final score. That gap has no established practical equivalence threshold; the early incumbent could still matter for BP, FLP, or geographic transfer. The last improvement arrived at successful completion 18, so the record does not establish convergence.

[All unordered candidate subsets](candidate_subset_budgets.csv) of sizes 3, 5, 8, 10, 15, and 20 were enumerated from the same completed 20 points. Their median regrets were 0.002991, 0.002014, 0.001643, 0.000821, 0, and 0; 90th percentiles were 0.006237, 0.003572, 0.003053, 0.002991, 0.001643, and 0. [Figure](figures/candidate_subsets.png). These numbers describe coverage of a finite, adaptively generated candidate pool. They are **not** simulations of fresh Optuna runs: under a 3–5 completion limit, the frozen `n_startup_trials=10` sampler would still be in random startup, and TPE suggestions 11–21 would not exist in that execution. The fractions containing the pilot winner are combinatorial (3/20 or 5/20 for budgets 3 or 5), not a success probability for a new study.

The observed maximum is selected from 20 correlated validation estimates and is subject to selection optimism. Neither its 0.003717 advantage over trial 0 nor a near-neighbor gap estimates future outer-test gain. There is one adaptive study, one seed per trial, fixed geographic folds, and no independent replication of small-budget searches. The evidence supports **scientific adequacy as a plausible cost choice**, not a claim that five trials recover the optimum or that their expected regret is known.

## 2. Geographic fold subsets

[All 25 subsets](fold_subsets.csv) of 2, 3, and 4 folds were evaluated using the epoch-100 equal-task mean on only those folds. Spearman and Kendall compare the ranking of all 20 candidates to their complete five-fold ranking. Regret is always the selected candidate's **full-five-fold** loss relative to trial 19. [Figure](figures/fold_subsets.png).

| Subset size | Subsets | Same top trial | Median full-five-fold regret | Worst regret |
|---:|---:|---:|---:|---:|
| 2 | 10 | 4 | 0.001643 | 0.010543 |
| 3 | 10 | 6 | 0 | 0.003558 |
| 4 | 5 | 4 | 0 | 0.001643 |

All 14 tested subsets containing fold 5 select trial 19; none of the 11 subsets omitting fold 5 does. Mean Spearman is 0.976 with fold 5 and 0.565 without it. The strong eastern fold 5 skill (mean 0.2903) and its candidate spread dominate the arithmetic mean even though fold 2 remains near zero (mean 0.00469). Including fold 2 alone does not restore stable selection: subsets 1–2 and 1–2–4 have poor rank agreement, and 1–2 selects trial 9 with 0.010543 regret. This pattern warns against picking a retrospective “representative” set merely because it reproduces trial 19. A defensible subset would need to be fixed by geography and scientific coverage *before* seeing candidate scores and verified across independent studies. The present pilot supplies no such verification. Keep five folds for every candidate.

## 3. Epoch budget

All 100 candidate-fold fits have comparable saved records at the requested epochs. The [epoch summary](epoch_summary.csv), [candidate and fold detail](epoch_candidate_details.csv), [rank figure](figures/epoch_stability.png), and [candidate curves](figures/candidate_epochs.png) use the frozen skill calculation; no training-loss proxy was substituted.

| Epoch | Spearman vs 100 | Kendall vs 100 | Top-five overlap | Early-selected trial | Full-epoch regret |
|---:|---:|---:|---:|---:|---:|
| 10 | −0.511 | −0.337 | 0/5 | 5 | 0.013161 |
| 25 | 0.167 | 0.116 | 1/5 | 18 | 0.005417 |
| 50 | 0.731 | 0.537 | 2/5 | 13 | 0.003558 |
| 75 | 0.750 | 0.537 | 3/5 | 8 | 0.004119 |
| 100 | 1 | 1 | 5/5 | 19 | 0 |

Trial 19 is not the top-ranked trial at any tested earlier epoch. At epoch 10, the ranking is reversed enough that the top five and final top five do not overlap. Epoch 75 still changes the winner and has 0.004119 regret. Candidate curves are nonmonotone: across all trials, mean final-minus-epoch skill is −0.00476 at epoch 10, +0.01164 at 25, +0.01082 at 50, and −0.00494 at 75. At epoch 25, mean fold-2 skill is −0.01033 while fold 5 is +0.25391; by epoch 100 those means are +0.00469 and +0.29030. Depth-2 and depth-3 groups both improve by about 0.0116 from epoch 25 to 100, but their individual trajectories and fold contributions differ; the small confounded groups cannot establish architecture-specific stopping rules. These rank reversals support the earlier project concern about pruning. Staged allocation and successive halving are **not empirically supportable** from this record. No staged runtime saving is presented as an actionable forecast.

## 4. Promising configurations and targeted sensitivity sweep

[The proposed five-configuration table](proposed_pilot_informed_sweep.csv) contains **existing tested outer-3 configurations**, with exact values, observed local trial runtimes, task skills, worst-fold scores, rationale, and question. It is a candidate set for **exploratory design B only**, not a restriction for design A. No interpolation is included; therefore no untested configuration is represented as measured. The five are trial 19 (best five-fold score, BP-led), 20 (near-neighbor with higher FLP mean), 6 (best worst-fold, depth 2, large batch), 8 (strong depth-2 startup comparator), and 4 (width-128 comparator). This gives diversity in depth, width, batch, and learning-rate scale without simply taking the top five objective scores. Their observed local outer-3 trial runtimes are respectively 2.23, 2.25, 2.09, 2.52, and 2.49 hours; future study costs may differ. Trial 19 and 20 have depth 3, width 64, batch 65,536 and learning rates 0.0001127191 and 0.0002694637. Trial 6 has the best worst-fold skill 0.008231 versus trial 19's 0.003005, while its mean objective is 0.006237 lower. Trial 8 has objective 0.104094. Trial 4 provides width 128 with objective 0.101358.

All eight highest-ranked pilot trials used depth 3 and width 64; the final three near-neighbors used batch 65,536. This is a **descriptive region**, not an independent estimate of hyperparameter importance. TPE concentrated later suggestions, factors vary jointly, and the small categories are confounded. The winner's learning rate lies near the lower bound 1e-4, so this sample does not establish a local optimum. The task-specific and weak-fold differences matter more scientifically than tiny mean-score differences.

## 5. Scientific validity and leakage audit

The five original outer TEST partitions remain untouched. For any outer study, choosing configurations solely from that study's outer TRAIN and its five inner folds, then refitting on its complete outer TRAIN, preserves the core nested separation. Yet the outer-3 TRAIN pool can contain rows belonging to another study's outer TEST. The pilot-informed candidate identities therefore encode information from those other test populations **before** their purported test evaluation. No outer-test outcomes were inspected here, but that cross-partition training-label dependence limits claims if the shared set is used for those studies.

| Design | Selection and cost | Valid claim | Qualification |
|---|---|---|---|
| **A: independent reduced nested search** | For each remaining outer study, use five new successful suggestions from the original unrestricted space and sampler, all five folds and 100 epochs. About 68.22 h new HPO at pilot Optuna wall rate. | Each study's selected model is chosen without its own outer TEST, so its held-out evaluation is valid conditional on the fixed reduced algorithm. | With `n_startup_trials=10`, five successes are random startup, not TPE. The *global decision to reduce budget* followed viewing outer-3 TRAIN validation, which can overlap other TEST populations. Thus a pooled five-study estimate is not a wholly prospectively specified, conventional unbiased nested-CV estimate; report protocol revision and pilot dependence. Strict independence of a protocol-level choice would require an external pilot or a budget set without outcome-dependent tuning. |
| **B: pilot-informed fixed sweep** | Reuse the five exact candidates in the sweep table across remaining partitions; optionally no per-study HPO if one fixed candidate is used. About 68.22 h for five-by-five inner selection, or zero new HPO for a single fixed candidate. | Geographic sensitivity and candidate transfer comparisons are useful exploratory results. Outer-3 TEST can still assess its internally selected winner if untouched. | Candidate identities were selected using outer-3 TRAIN, potentially including another study's TEST rows. Remaining-study TEST results are selection-dependent; they are not conventional unbiased nested-CV performance estimates for the chosen shared set. A fixed candidate eliminates new within-study HPO compute but not this dependence. |
| **C: outer 3 as principal confirmatory evaluation** | Preserve trial 19 selection by original outer-3 inner HPO; after approval, refit and evaluate only against untouched outer-3 TEST. Remaining partitions may be transfer/sensitivity analyses under clearly labeled protocols. | Outer-3 held-out geographic performance is confirmatory for the original outer-3 selection, conditional on the prespecified objective and untouched TEST. | It is one geographic test partition, not an across-geography performance distribution; other partitions analyzed with pilot-informed choices cannot be pooled as an unbiased confirmatory five-fold estimate. |

The recommended path is **A for remaining studies, with C's outer-3 result explicitly identified before TEST evaluation as the primary pilot geographic test**. This preserves within-study TEST isolation and supplies more geographically diverse held-out evaluations at much lower cost. Report the aggregate as a revised-protocol, selection-dependent sensitivity summary, not an unqualified five-fold nested-CV estimate. Design B's table is retained for a separately labeled transfer experiment only. No held-out outcome may refine this protocol or select a row of the table.

[The machine-readable design comparison](design_comparison.csv) records each design's new HPO cost, test-isolation status, dependence, and permissible claim.

## 6. Compute forecast and accounting

[Runtime scenarios](runtime_scenarios.csv) use the pilot's **68.2193 hours of completed Optuna-trial start-to-complete wall time**, or 3.4110 hours per successful trial, for prospective scheduling. This includes time recorded inside each completed trial interval, including pauses, worker waits, and orchestration gaps. Phase 7C's **63.8080 hours** is the sum of local `state.json` `total_runtime_seconds` (3.1904 h per successful trial). The difference is **4.4113 hours** across 20 completions. Neither basis includes the two failed trials' roughly 8.63 wall hours, preparation, outer-study changes, or future retries. These are different clocks, not a discrepancy in the number of fits. The Optuna interval is the more conservative planning basis; the local-clock extrapolation remains in the table.

| Four remaining studies | HPO fits | New HPO wall forecast | Savings versus original |
|---|---:|---:|---:|
| 20 × 5 × 4 | 400 | 272.88 h | 0 |
| 5 × 5 × 4 | 100 | 68.22 h | 204.66 h (75%) |
| 3 × 5 × 4 | 60 | 40.93 h | 231.95 h (85%) |
| One pilot-informed fixed configuration; no HPO | 0 | 0 | 272.88 h (100% HPO only) |

These are **serial trial-wall equivalents**, not measured elapsed times for the remaining studies; parallel execution, outer TRAIN sizes, architecture mix, interruptions, and failures can alter calendar time. A five-success random-startup search might have a different configuration mix from the 20-trial pilot. Trial-specific local runtimes in the sweep table show 2.09–2.52 h, but using them for design A would illegitimately assume a pilot-restricted proposal set.

The later five-member deep ensemble is a **separate final refit**, not five more HPO folds. [Refit estimates](ensemble_refit_estimates.csv) scale saved epoch-100 training seconds per row by outer-3 full TRAIN support (79,519,698 rows) and 100 epochs. For trial 19 this gives about **0.824 h per full-TRAIN member**, **4.12 h for five members per study**, **16.48 h for four studies**, or **20.60 h for all five**. Across the five proposed pilot configurations, four-study five-member refits project **15.41–18.83 h**. These are rough training-throughput projections, not measured ensemble runs; independent seeds, preprocessing, checkpoints, outer-TEST inference, and other outer TRAIN sizes are excluded. For comparison only, adding trial-19 refits gives 289.35 h (original), 84.70 h (five-trial), 57.41 h (three-trial), and 16.48 h (fixed), before excluded work. Fixed pilot-informed HPO savings come with the validity qualification above.

## Interpretation

Five unrestricted, full-epoch, five-fold successes per remaining study are the recommended compute limit. It cuts projected new HPO hours by 75% while keeping the frozen validation definition and within-study geographic separation. The pilot cannot guarantee that five random suggestions match its best score; that is the accepted tradeoff for a research program focused on probabilistic surfaces, aleatoric and epistemic uncertainty, and geographic transfer. The known weak fold 2, BP adequacy concerns, selection optimism, and uncertain cross-geography hyperparameter stability remain material scientific limitations. No training should begin until scientific review approves the revised protocol.
