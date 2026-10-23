# Phase 7B first diagnostic: BP predictive adequacy

## Scope and reproduction

This is a read-only evaluation of outer study 3, inner fold 1 from the completed
Phase 7A run `07a-20261022T054112Z-5270cc63`. The script
[`scripts/diagnose_07b_bp_adequacy.py`](../../../scripts/diagnose_07b_bp_adequacy.py)
loads hash-checked existing checkpoints at epochs 1, 3, 25, 50, and 100. These
epochs sample the start, early adaptation, middle, and end of the saved trajectory;
they were not chosen by score. It uses the existing frozen mmap cache, inner
TRAIN-derived preprocessing state, inner fold labels, and BP climatology
`0.001848224540038233`. Run with:

```bash
uv run python scripts/diagnose_07b_bp_adequacy.py
```

The script writes [full machine-readable scores](scores.json), [a flat score
table](scores.csv), and [a trajectory figure](bp_adequacy.png). It never accesses
outer TEST, updates a checkpoint, fits preprocessing, or trains a model. Inference
uses the production BF16 CUDA autocast; scores are accumulated on each task's
frozen support. BP uses Bernoulli cross entropy, excess log loss (cross entropy
minus soft-target entropy), and squared error to the soft target; FLP uses
categorical cross entropy and excess log loss on FLP-valid rows. The training
constant BP forecast is the same baseline for both roles. `log_skill` is
climatology cross entropy minus model cross entropy, positive when the model
improves on that constant forecast. Brier here is squared error to the soft
target, equivalent to the Bernoulli Brier score up to target-intrinsic variance
that is fixed across forecasts. Pearson coefficients measure association, not
probabilistic calibration. A partition coefficient gives equal weight to each
frozen cache partition with at least 100 role/stratum cells.

## Findings

| Epoch | Role | BP rows | BP excess log loss | BP log skill | FLP valid rows | FLP excess log loss | BP ≥ 0.02 target / prediction | BP ≥ 0.02 cell / partition Pearson |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | TRAIN | 61,042,788 | 0.002761 | 0.000165 | 41,732,573 | 0.7155 | 0.03494 / 0.00580 | −0.004 / 0.056 |
| 1 | VALIDATION | 18,476,910 | 0.007296 | 0.004222 | 15,909,640 | 0.6754 | 0.03643 / 0.00584 | 0.089 / 0.011 |
| 3 | TRAIN | 61,042,788 | 0.002684 | 0.000242 | 41,732,573 | 0.7145 | 0.03494 / 0.00361 | −0.074 / −0.083 |
| 3 | VALIDATION | 18,476,910 | 0.009013 | 0.002506 | 15,909,640 | 0.6072 | 0.03643 / 0.00330 | 0.016 / 0.048 |
| 25 | TRAIN | 61,042,788 | 0.002172 | 0.000754 | 41,732,573 | 0.6973 | 0.03494 / 0.00342 | 0.086 / 0.057 |
| 25 | VALIDATION | 18,476,910 | 0.008728 | 0.002790 | 15,909,640 | 0.6362 | 0.03643 / 0.00336 | 0.052 / −0.054 |
| 50 | TRAIN | 61,042,788 | 0.002327 | 0.000599 | 41,732,573 | 0.7051 | 0.03494 / 0.00237 | −0.036 / −0.105 |
| 50 | VALIDATION | 18,476,910 | 0.010876 | 0.000642 | 15,909,640 | 0.6767 | 0.03643 / 0.00228 | 0.025 / −0.206 |
| 100 | TRAIN | 61,042,788 | 0.002113 | 0.000813 | 41,732,573 | 0.6867 | 0.03494 / 0.00363 | 0.009 / −0.068 |
| 100 | VALIDATION | 18,476,910 | 0.008574 | 0.002944 | 15,909,640 | 0.6133 | 0.03643 / 0.00343 | 0.064 / −0.099 |

At epoch 100, full-support TRAIN mean BP prediction is 0.001778 against a mean
target of 0.001848 (bias −0.000071). VALIDATION mean prediction is 0.002949
against 0.007587 (bias −0.004638). TRAIN BP excess log loss is 0.002113 versus
its climatology's 0.002926; VALIDATION is 0.008574 versus climatology's
0.011518. Thus the model has positive full-support log skill on both roles,
despite poorer geographic VALIDATION fit. The VALIDATION target distribution is
also riskier: 1,841,578 of 18,476,910 rows (9.97%) have BP ≥ 0.02, compared
with 1,028,783 of 61,042,788 TRAIN rows (1.69%). This prevalence shift helps
explain the larger overall VALIDATION loss and bias; the single fold cannot
quantify a separate causal transfer effect.

The elevated-risk failure is already present on TRAIN. At epoch 100, TRAIN
BP ≥ 0.02 mean prediction is **0.00363** against **0.03494** target (10.4% of
target), and VALIDATION is **0.00343** against **0.03643** (9.4%). At all five
saved epochs, both roles predict only 6–17% of the target-stratum mean. Within
that stratum, cell correlations and equal-partition correlations are near zero
or negative in both roles. The epoch-100 model is almost unbiased overall on
TRAIN because overprediction in the abundant BP < 0.001 stratum offsets
underprediction at higher risk. The pattern supports **risk-dependent BP
prediction failure with additional geographic full-support degradation**. It
does not support attributing the elevated-risk gap primarily to transfer.

FLP excess log loss is reported to retain both task supports. At epoch 100,
its VALIDATION score (0.6133) is below its TRAIN score (0.6867); this does not
explain BP calibration. Selected VALIDATION scores agree with the
original Phase 7A epoch records to at most `6.4e-8` for BP excess log loss
and `7.9e-9` for FLP excess log loss. All BP/FLP row counts and BP stratum sums
match the frozen 6C support report.

## Interpretation and boundary

The BP strata are defined by the **observed target**, so they are descriptive
diagnostics, unavailable when making a new forecast, and unsuitable as an
implicit HPO ranking objective. The constant climatology is derived from the
same inner TRAIN population and is deliberately not re-estimated on VALIDATION.
Partition correlations are descriptive, depend on the frozen tile geometry and
100-row threshold, and cannot establish out-of-fold uncertainty. Only one
architecture and one inner geographic fold were examined. No confidence
interval, cross-fold generalization claim, or outer TEST result follows.

The next review should decide whether this level of high-risk BP underprediction
is acceptable for the paper's wildfire-risk question and specify a response
within the frozen study design. Separately, the literature-supported HPO ranking
objective remains undecided and must be finalized before any search. No area of
applicability analysis is justified by this diagnostic alone.
