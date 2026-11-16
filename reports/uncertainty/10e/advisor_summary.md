# Phase 10E — advisor discussion brief

## Research question

When can we trust the neural surrogate's reconstruction of FSim's probabilistic wildfire-risk surface? We tested whether spread among five independently trained models warns of **BP reconstruction error** beyond the predicted BP itself. The target is FSim, not observed wildfire outcomes.

## Five-minute account

1. **Reconstruction is geographically uneven.** On 110.8 million held-out BP cells in five geographic studies, BP MAE ranges from **.002162 to .006486**; cell-pooled MAE is **.003701**. Conditional FLP mean TV is **.405–.505** on 81.1 million valid cells. The model underpredicts target BP ≥ .01 on average in all five studies. [Phase 9 evaluation](phase9_source/README.md).
2. **Disagreement has an association, but limited decision value.** Its rank correlation with absolute BP error is positive in all five studies (**.274–.696**). Yet at 20% rejection, disagreement finds **25.0%** of each study's largest error decile on average, compared with **57.7%** for predicted BP and **20%** for random rejection. Predicted BP yields lower remaining MAE in all studies at all tested nonzero rejection fractions. Within predicted-BP deciles, disagreement offers no consistent additional gain. [Phase 10A](../10a/README.md) and [risk–coverage figure](../10a/risk_coverage.png).
3. **Agreement failures are real.** The full TEST intersection of high error and low disagreement contains **614,188 cells** (**.55%** of support; **2.75%** of pooled absolute-error mass). These are diagnostic outcome-defined groups, not a deployable detector. [Phase 10A failures](../10a/low_disagreement_high_error.csv).
4. **Geography does not rescue the warning.** Beyond the 211 km exclusion radius, distance–disagreement trends change sign or size across studies. In the approximately 1% buffer-zone sample, DEAD_ZONE error is lower than TEST error in three studies and higher in two. At 20% rejection, disagreement's high-error recall is **.240** in DEAD_ZONE and **.253** in sampled TEST, versus **.577/.576** for predicted BP. These are descriptive geographic comparisons, not causal distance effects. [Phase 10B](../10b/README.md), [Phase 10C](../10c/README.md).
5. **SHAP suggests shared model behavior, not a cause of error.** `VEG2019PC` ranks first in all five studies on a 550-cell representative diagnostic sample. Member importance-rank correlations are high (**median pairwise .902–.941** by study), including related behavior on selected agreement failures. Matched failure/control differences are small and not consistently directional. [Phase 10D](../10d/README.md) and [summary plot](../10d/shap_summary.png).

## Interpretation for discussion

The most defensible contribution is the gap between **statistical association** and **useful selective prediction**. Ensemble spread contains some error information, yet a simpler inference-time quantity, predicted BP, is much better at ranking high-error cells in this evaluation. The five models can agree on a shared inaccurate reconstruction. Their agreement is not calibrated confidence and does not prove little epistemic uncertainty.

The largest limitation is scope: the reference is FSim rather than fire outcomes, the five spatial studies are related, and later diagnostics use sampled or outcome-selected cases. We have not calibrated an error interval or tested a new warning policy on a fresh geographic holdout. These results are sufficient for an independent-study report about this surrogate and its observed reliability limits. I recommend **final report preparation first**. If a further experiment is required, prospectively test calibrated BP reconstruction-error intervals against a predicted-BP-only baseline on new geographic assessment data; this is the clearest way to test an actionable trust decision without reusing the examined TEST outcomes.

## Suggested discussion prompts

- Is the intended claim limited to FSim-surrogate reconstruction, with real-world wildfire calibration explicitly outside scope?
- Is a report centered on the association-versus-utility gap sufficient, or is a prospective interval-coverage decision needed?
