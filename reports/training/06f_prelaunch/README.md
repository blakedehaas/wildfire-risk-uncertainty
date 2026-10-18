# Checkpoint 6F prelaunch: HPO input cache and restart gate

**Status:** engineering validation for HITL review. No real Optuna HPO trial, outer TEST evaluation, hyperparameter selection, or OOF prediction was run.

## Frozen input representation

The ignored `artifacts/training/hpo_cache_v1/` contains one uncompressed canonical-row-aligned mmap representation: 110,817,231 rows, 33 float32 inputs, BP and six FLP float32 targets, strict FLP mask, five nullable-as-zero inner-fold labels, cell IDs, and 3,060 partition offsets/names. It occupies **19,282,345,980 bytes (17.96 GiB)** and took **167.23 s** to build. The 1,214 zero-row canonical partitions have equal start/end offsets; all 3,060 partition positions remain in the exact frozen ordering. The cache was built in a sibling temporary directory, with per-row canonical/5B/6C alignment checks, eligibility checks, unique IDs, and file hashes before being renamed into place. The [cache summary](cache_manifest_summary.json) and [validation record](cache_validation.json) give source/file hashes and exact counts. The cache contains no imputation, scaling, or transformed features.

## Scientific equivalence and speed

The [full stream comparison](equivalence_validation.json) compared every selected feature, target, mask, and cell ID for outer study 3 / inner fold 1. The complete epoch-1 TRAIN order (61,042,788 rows) and sorted validation order (18,476,910 rows) matched exactly. Independently fitted preprocessing states had identical inputs, counts, missing counts, means, standard deviations, and digest. The [one-epoch regression](training_regression.json) found exact equality for BP/FLP raw training and validation sums, supports, optimizer steps, derived objectives, final model parameters, and AdamW optimizer state. This is an engineering regression, not a model-quality comparison.

Two paired fixed-configuration measurements in [the throughput table](throughput_comparison.csv) gave these medians:

| Phase | Parquet | mmap | Speedup |
|---|---:|---:|---:|
| Preprocessing scan (one independent fit) | 25.19 s | 11.83 s | 2.13× |
| Raw TRAIN traversal/host preparation | 23.47 s | 9.37 s | 2.50× |
| Raw validation traversal/host preparation | 9.56 s | 1.68 s | 5.69× |
| Full training epoch, BF16, batch 65,536 | 31.90 s | 19.15 s | 1.67× |
| Full validation pass | 11.30 s | 3.49 s | 3.24× |
| Training plus validation | 43.20 s | 22.63 s | **1.91×** |

The two paired training-plus-validation speedups were 1.82× and 2.01×. Training throughput was about 1.91 million versus 3.19 million rows/s; validation throughput was about 1.64 million versus 5.30 million rows/s. VRAM remained small (training peak allocated about 201–202 MB). Peak RSS in a single process includes mmap residency and accumulates across runs, so it should not be read as an isolated per-path memory cost. GPU utilization was not measured reliably. The repeated end-to-end gain is clear, so the **mmap path is accepted for subsequent 6F execution**, subject to this HITL review. The HPO runner requires an explicit cache directory and refuses a manifest/provenance mismatch; it has no silent Parquet fallback.

The [25-state table](preprocessing_states.csv) records all five inner fits in each of five outer studies, including TRAIN/FLP supports, state hashes, fit times, and provenance. All 25 were fitted/validated from their own INNER TRAIN rows once; the five previously existing 6E states were independently checked against the newly fitted values. The 25 scans took 263.27 s in total. They are ignored fold-specific artifacts and will be reused across trials only on exact provenance match.

## Prospective HPO compute planning

The [revised plan](projected_hpo_runtime.json) uses the accepted reader's measured training and validation row rates, the exact five outer TRAIN counts, and the frozen 30-trial-per-study schedule. On one local GPU, the no-pruning planning case is **about 483 GPU-hours**. The previously accepted fractional-survival idealized ASHA calculation becomes **about 16 GPU-hours**, an optimistic lower planning figure rather than expected runtime. Requiring at least one 100-epoch COMPLETE trial in each study gives a minimum-resource case of **about 21 GPU-hours**. A practical planning bracket is therefore roughly **20–485 GPU-hours**, with actual runtime unknown before observing pruning. The older 6B bounds are not divided by the 1.91× 6D path speedup because 6B used a different benchmark workload. Model width, batch size, OS page-cache behavior, and trial survival will affect realized time. The five-study trial budget and ASHA rules remain unchanged.

## Pause and resume

The future real-HPO path keeps one Optuna RUNNING trial identity during an epoch-boundary pause. Each fold writes an atomic, SHA-256-validated model/optimizer/RNG checkpoint bound to the exact cache manifest, partition names/offsets, feature order, preprocessing digest, and data-order version. A new process reopens the same trial and W&B run ID; completed fold/epoch records are read, never appended again. If a process stops during an epoch, that fold replays from its last durable epoch checkpoint. The [synthetic process validation](pause_resume_validation.json) covered graceful pause, hard termination during fold 3 of epoch 2, new-process continuation, and equality with an uninterrupted trajectory. It produced one COMPLETE trial numbered 0, the same mock W&B run ID, 20 unique fold/epoch records, matching intermediate objectives, and identical final synthetic states. The cache-binding unit test rejects changed partition and preprocessing hashes and a corrupted checkpoint. This synthetic check is infrastructure evidence, not real HPO.

The [live W&B resume check](live_wandb_resume_validation.json) used one synthetic online run in two separate processes. The second process reopened the persisted ID with `resume="must"`; the W&B API found exactly one matching run and exactly two ordered metric records, one from each process, without duplicates. The run ID and URL are provenance only. No credential is recorded.

For a future explicitly authorized outer-study-1 run, the manual command is `uv run wildfire-hpo real-hpo --outer-study 1 --cache-dir artifacts/training/hpo_cache_v1 --checkpoint-6f-authorized`. Ctrl+C requests a graceful stop after the current common five-fold epoch; the active Optuna trial stays RUNNING for the same command to resume. If pruning finishes the trial at that epoch, the command stops without starting another trial. `uv run python -m wildfire_risk_uncertainty.hpo.status --outer-study 1` gives a read-only local progress summary.

Only a validated cache is eligible for the future 6F runner. The frozen 6A–6E scientific method, model, targets, objective, spatial folds, seed, trial budget, and pruning schedule remain unchanged.
