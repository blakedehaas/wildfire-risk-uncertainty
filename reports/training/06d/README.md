# Checkpoint 6D: fixed training-path integration validation

**Status:** HITL approved and frozen engineering validation. These calibration weights and loss values are not scientific model results. No model selection, outer TEST evaluation, Optuna, ASHA, W&B, prediction generation, or training beyond the two fixed epochs per trajectory occurred.

## Frozen population and input contract

The case was fixed prospectively: outer study **3**, inner validation fold **1**. The accepted 5B/6C masks provide 79,519,698 outer TRAIN rows, partitioned into **61,042,788 INNER TRAIN** and **18,476,910 INNER VALIDATION** rows. INNER TRAIN contains 19,299,365 BP-zero, 41,743,423 BP-positive, and 41,732,573 strict FLP-valid rows. INNER VALIDATION contains 2,567,269 BP-zero, 15,909,641 BP-positive, and 15,909,640 strict FLP-valid rows. The loader checks aligned canonical/5B/6C cell IDs and requires each non-null inner label to match 5B TRAIN eligibility; outer TEST and DEAD_ZONE rows are never exposed as training or validation data.

The [run manifest](run_manifest.json) lists the **exact ordered 33 model inputs**: 25 schema-role predictors and the eight accepted coverage/valid-area/failure diagnostics. Coordinates, targets, masks, and fold labels are excluded from the model input vector. The MLP has **21,767 trainable parameters**.

## Preprocessing and fixed execution

Two independent bounded-memory scans of INNER TRAIN yielded byte-identical preprocessing state, SHA-256 `2dcb473ea81e00c1108b24fa1d39912379c78ed88cbc73f6e2c32d8a996d476f`. Scan times were **29.44 s** and **24.40 s**. The [summary](preprocessing_summary.json) records every feature's missing count, post-transform mean, standard deviation, feature order, and training support. Validation rows contributed no fitted statistics. Four density inputs receive `log1p`; continuous inputs are mean-imputed and z-standardized from INNER TRAIN; `wui_failure` remains binary.

The fixed integration constants were seed 42, depth 2, shared width 128, ReLU, no dropout, sigmoid BP interpretation, six-way softmax FLP interpretation, AdamW learning rate `1e-3`, weight decay `1e-4`, batch size **65,536**, and **BF16 autocast**. The batch size is the middle accepted 6B candidate and is not a selected winner. The BP/FLP objective retains the separate global supports. Each minibatch divides both task sums by its complete row count, and multiplies the masked FLP sum by `N_BP/N_FLP`, preserving the frozen equal-task objective in partition expectation. Validation accumulates global BP and FLP sums and divides by their own support counts.

The training reader and epoch routines accept any of the five frozen outer studies and five inner folds for later reuse. The `wildfire-train-validate` command itself fixes study 3 and fold 1 and has no outer TEST evaluation mode.

Each epoch permutes the 3,060 partition paths using NumPy `SeedSequence([42, epoch])`, then independently permutes eligible rows within each loaded partition. The stream packs these rows into 65,536-row batches with one final short batch. It never materializes the full INNER TRAIN population. The accepted unique-cell partition contract, one visit per partition, permutation within each partition, and exact measured epoch support provide the no-duplication/no-omission traversal check. Every epoch processed **61,042,788 rows**, **41,732,573 FLP-valid rows**, and **932 optimizer steps**; each validation pass processed **18,476,910 rows** and **15,909,640 FLP-valid rows**. See [traversal validation](data_traversal_validation.json).

This is a deterministic, partition-based bounded-memory shuffle, not a mathematically uniform global row permutation. The accepted traversal implementation is frozen for subsequent HPO and outer-model training unless a later HITL decision changes it.

## Operational sanity diagnostics

The figures below are finite-value diagnostics from the uninterrupted reference, **not** scientific model performance estimates or inputs to model selection. The apparent validation-loss improvement between epochs does not justify choosing an epoch count or any other hyperparameter. Training losses aggregate batches while weights change within the epoch.

| Stage | BP loss | FLP loss | Equal-task loss | Wall time | Rows/s |
|---|---:|---:|---:|---:|---:|
| Initialization validation | 0.678137 | 1.830096 | 1.254116 | 12.53 s | 1,475,031 |
| Epoch 1 training | 0.027593 | 1.463052 | 0.745322 | 31.38 s | 1,945,322 |
| Epoch 1 validation | 0.045156 | 1.657732 | 0.851444 | 11.18 s | 1,652,924 |
| Epoch 2 training | 0.013485 | 1.443270 | 0.728378 | 31.01 s | 1,968,535 |
| Epoch 2 validation | 0.044521 | 1.610620 | 0.827570 | 11.11 s | 1,663,544 |

The resumed run's epoch-1 and epoch-2 component and combined diagnostics match those of the reference exactly; all individual values and timing measurements are in [epoch_metrics.csv](epoch_metrics.csv). Training peak allocated VRAM was **201.34 MB**; validation peak allocated VRAM was **112.53 MB**. The highest recorded process peak RSS was **2.13 GB**. GPU and software provenance are in [environment.json](environment.json).

Every real-data minibatch passed finite-input, finite-target, finite-logit, finite-loss, finite-gradient, and finite-post-step-parameter checks. Synthetic tests confirm BP sigmoid bounds, six-way softmax normalization, complete soft probability targets, strict FLP masking, and gradient flow through both heads and the backbone.

## Checkpoint/resume validation

The uninterrupted reference ran epochs 1 and 2 in one process. An independent process repeated epoch 1 from seed 42, saved an epoch-boundary checkpoint, terminated, and a third process restored it for epoch 2. The checkpoint includes model and AdamW states; completed epoch and global optimizer step; Python, NumPy, torch CPU, and torch CUDA RNG states; seed, feature-order and preprocessing hashes; fixed model configuration; outer/inner identifiers; batch size; precision; source Git SHA; and data-order version. Files live only in ignored `artifacts/training/06d/` and are **not accepted scientific models**.

The [comparison](checkpoint_resume_validation.json) found exact parameter and optimizer-state equality; maximum absolute and relative differences were **0**. Global step was **1,864** in both runs. RNG states, row counts, metadata, epoch-2 training component losses, and epoch-2 validation component and combined losses matched exactly. The checkpoint process boundary did not change the training trajectory.

## Scope and caveats

The deterministic order is a bounded-memory partition shuffle, not a global uniform row permutation. It preserves one traversal of every eligible row. This fixed engineering run makes no claim about optimal hyperparameters or predictive quality. The frozen 5B outer TEST sets and 211.151 km exclusion design were not used for evaluation. The fixed 6D values must not be promoted into HPO decisions.
