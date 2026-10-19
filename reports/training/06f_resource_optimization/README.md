# Checkpoint 6F resource optimization benchmark

Status: **ENGINEERING_DECISION / VALIDATED — HITL accepted and frozen.** Concurrency 2 is implemented in the production HPO path; concurrency 3 remains rejected after crossing the recorded VRAM safety gate. See the [production integration report](PRODUCTION_INTEGRATION.md) and the earlier [candidate validation](final_validation/README.md).

Concurrency 2 completed all five folds with exact agreement against the sequential reference and an observed **1.64× wall-time speedup (38.94% reduction)**. This is the initial single-epoch engineering benchmark; the later four-epoch production regression established exact trajectory equivalence and a **1.453× end-to-end speedup** before adoption.

## Results

| Concurrent folds | Completed folds | Five-fold wall time | Rows/s | Speedup | Outcome |
|---|---:|---:|---:|---:|---|
| 1 | 5 | 193.819 s | 2,179,852 | 1.00× | Sequential reference |
| 2 | 5 | 118.344 s | 3,570,088 | 1.64× | Exact numerical agreement |
| 3 | 0 | — | — | — | Safety stop after 18.690 s |
| 4 | — | — | — | — | Not run after safety stop |
| 5 | — | — | — | — | Not run after safety stop |

The completed work at each of levels 1 and 2 is 337,997,588 training rows plus 84,499,397 validation rows. Wall time includes worker startup, process scheduling, training, validation, state hashing, and coordinator polling. Workers invoke the unchanged frozen trainer with separate CUDA processes; each starts with seed 42. Each fold executes one complete training epoch and one complete validation pass. Configuration and implementation identity are recorded in [metadata.json](metadata.json); they match outer-study-1 trial 0 at commit `8db4238c5f9b141ea67a5a8a746488d1ff2cc0ed`.

## Numerical audit

[audit.json](audit.json) records successful checks for all five c1/c2 fold pairs:

- Exact training and validation rows and FLP support, also checked against frozen fold support.
- Exact training optimizer-step counts and expected batch counts.
- Exact raw BP/FLP loss sums, task means, and equal-task objectives.
- Matching deterministic SHA-256 digests of every final model tensor and the complete AdamW state dictionary, including optimizer parameter groups.
- Matching provenance-validated fold preprocessing states.
- Exact agreement of the sequential reference's raw losses, supports, derived objectives, and training steps with the historical trial-0 epoch-1 records.

Both levels yield the historical five-fold objective **J₁ = 0.7165150485260844**. The final model and optimizer tensors were not retained; their equality is evidenced by the stored deterministic state digests. No new traversal-ID digest was collected. The frozen reader, support checks, exact losses, and final-state digests provide the available trajectory evidence. Validation's `optimizer_steps` field is the trainer's batch counter; validation performs no optimizer updates.

## Resources and safety stop

| Concurrent folds | CPU cores, lower bound¹ | Peak summed worker RSS² | Minimum available RAM | Peak swap used | Mean GPU utilization³ | Mean GPU power³ | Peak device VRAM³ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.930 | 14.82 GiB | 26.31 GiB | 8 KiB | 8.15% | 58.80 W | 3,087 MiB |
| 2 | 1.560 | 28.07 GiB | 25.07 GiB | 8 KiB | 19.60% | 70.34 W | 11,177 MiB |
| 3 (partial) | 3.047 | 13.58 GiB | 23.94 GiB | 8 KiB | 53.33% | 128.21 W | 15,638 MiB |

¹ The original `summary.json` CPU field is the last sample's sum of process counters, **not total CPU time**: counters disappear when workers exit. The audit conservatively sums positive changes in sampled counters, giving lower bounds of 180.19, 184.65, and 56.95 CPU seconds. Dividing by wall time gives the table's core equivalents. Exact total CPU usage cannot be recovered from these samples.

² Summed RSS counts shared mmap pages more than once. It does not measure unique RAM consumption. Available RAM remained above 23.94 GiB, and recorded swap usage stayed at 8 KiB.

³ GPU measurements are device-wide samples at approximately two-second intervals, including process startup and teardown. They are not per-worker allocations and cannot establish the source of all VRAM use or exclude interference from other GPU workloads. Sampling can miss transient peaks. There were 95, 58, and 9 samples respectively.

At 18.372 seconds into c3, device VRAM reached **15,638 MiB**, exceeding the script's **15,500 MiB** gate on the 16,303 MiB GPU. The coordinator terminated the workers and persisted `STOPPED` / `VRAM pressure` in [summary.json](summary.json). The three worker stderr files are empty, and no c3 fold result exists. This evidence identifies a deliberate safety stop, not a model-capacity interruption. It does not prove a CUDA OOM occurred or that c3 intrinsically requires this much VRAM.

The original script also stops on nonzero worker exit (including OOM), exact-equivalence mismatch, swap growth above 128 MiB, available RAM below 2 GiB, elapsed time above 1.5× the sequential reference for later levels, or a 600-second timeout. C1/c2 triggered none of these gates. C3 has no completed numerical comparison. Resuming c3 or advancing to c4/c5 under the same conditions would bypass the recorded safety stop, so no new training was launched during the resume audit.

## Interpretation and handoff

Concurrency 2 is the fastest completed level and reproduced the one-epoch sequential numerical result exactly. Timing is based on one measurement per level, with c1 preceding c2; there are no repeated warm runs or confidence intervals. Process startup, cache warming, and device-wide activity may affect the observed ratio. These results do not establish a best concurrency level, sustained multi-epoch safety, or equivalence beyond epoch 1.

The original outputs are preserved, with checksums in `audit.json`. The benchmark script at [scripts/benchmark_06f_resources.py](../../../scripts/benchmark_06f_resources.py) has no resume handling and would overwrite outputs if invoked as coordinator; it was inspected but not rerun. The audit can be reproduced from the repository root with `.venv/bin/python reports/training/06f_resource_optimization/audit_results.py`.

**HITL decision:** concurrency 2 was accepted for production execution and concurrency 3 or higher was rejected. The integration changes execution only; the frozen HPO methodology is unchanged. No real HPO or outer-TEST evaluation was launched during benchmarking or integration validation.

Raw evidence: [summary](summary.json), [metadata](metadata.json), [resource samples](resource_samples.csv), and per-fold `c1_fold*.json` / `c2_fold*.json`. The original summary's CPU field remains unchanged for provenance; use the qualified audit figures above.
