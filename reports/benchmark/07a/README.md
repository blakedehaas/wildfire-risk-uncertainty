# Phase 7A training throughput initialization

Status: benchmark and analysis only. No loss-balancing sweep, HPO trial, outer TEST access, or production training change was made. Measurements were taken on 2026-10-20 in this WSL2/Docker environment. Raw machine-readable records are in `hardware.json`, `io_memory.json`, `refined/results.json`, `sustained/results.json`, `no_affinity/results.json`, and `budget.json`.

## Accessible hardware

The Linux guest reports an AMD Ryzen 9 9950X, one socket, 16 cores / 32 threads, one NUMA node, 32 MiB L3, AVX2, AVX-512, AVX-512 BF16, and FMA. The process can use logical CPUs 0–31. `cpu.max=max`, `cpuset.cpus.effective=0-31`, and `memory.max=max`; these cgroup values are bounded in practice by the 30.2 GiB WSL2 guest RAM. Host memory size, actual DRAM channels/speed, and host CPU contention cannot be read from the container. A 128 MiB NumPy copy after warmup gave 35.9–38.0 GiB/s **effective read-plus-write copy traffic**; it is not a measured sustained DRAM bandwidth specification.

One accessible NVIDIA GeForce RTX 5080 has 16,303 MiB VRAM, compute capability 12.0, 84 multiprocessors, BF16 support, and a PCIe 5.0 x16 link reported by `nvidia-smi`. It uses the WDDM driver path under WSL2. Free VRAM at discovery was about 13 GiB and can change with host applications. PyTorch 2.13.0+cu130 uses CUDA runtime 13.0, cuDNN 9.2, MKL 2024.2, OpenMP, and AVX-512 CPU kernels; NumPy is 2.5.3 and PyArrow is 23.0.1. Default PyTorch intra/inter-op threads were 16/32; workers set intra-op explicitly and inter-op to one.

The workspace is ext4 on a virtual `/dev/sdd` block device with 891 GiB available at discovery. The guest reports the virtual device as rotational, which does not reveal the host storage medium. A 512 MiB read took 0.150 s on its first pass and 0.019 s on its second pass (the latter is page-cache dominated). A 128 MiB temporary write plus `fsync` took 0.106 s. These are short local probes, not sustained storage ratings. The 18 GiB HPO cache exceeds a large fraction of guest RAM, so cold versus warm cache state matters.

## Reproducible benchmark

Run from the repository root with the existing `.venv`:

```bash
.venv/bin/python scripts/benchmark_07a_training.py --output reports/benchmark/07a/sustained --steps 64 --warmup 8 --repeats 2 --cases 1,3 --max-processes 2
```

The CLI uses the validated, read-only production mmap cache, the frozen outer-study-3 inner TRAIN partitions, fold-specific preprocessing states, production batch packing and device preparation, the 33→128→128→128 shared MLP, BF16 autocast, exact production BP/FLP loss functions and global-support normalization, and AdamW with the requested learning rate and weight decay. Evaluation timing reuses inner TRAIN rows in evaluation mode. The pipeline has no PyTorch `DataLoader`; zero loading workers is its production configuration. A separate training worker uses each inner fold when processes run concurrently. `--max-processes` defaults to two because of the pre-existing 06F three-process VRAM safety stop. Every worker writes stage timings, startup, checkpoint time, CPU time, peak RSS/VRAM, hashes, and transfer microbenchmarks. Device-wide `nvidia-smi` samples are included, but WDDM utilization samples of roughly 0–4% in these short windows are not reliable evidence of GPU idle fraction.

The `refined` short run used 4 warmup and 16 measured batches (1.05 million measured rows per worker), two repeats, partitioned CPU affinity, and one, two, or three processes. The `sustained` run used 8 warmup and 64 measured batches (4.19 million rows per worker), two repeats, one or two processes. `no_affinity` used the short run's one- and two-process settings without affinity partitioning. Runs were serial in that order. The first sustained repeat incurred cache warming. Throughput excludes Python process startup in its steady window; startup-inclusive rates are in the JSON and are inappropriate for 100-epoch projections.

## Results and bottlenecks

| Configuration | Short-window aggregate rows/s, two repeats | 64-step aggregate rows/s, cold / warm | Operational status |
|---|---:|---:|---|
| 1 process × 1 CPU thread | 3.39M / 3.17M | — | Measured |
| 1 process × 4 CPU threads | 3.20M / 3.17M | 1.72M / 3.25M | Reference |
| 2 processes × 4 CPU threads | 5.70M / 6.08M | 3.92M / 5.39M | Recommended |
| 3 processes × 4 CPU threads | 6.81M / 6.81M | — | Short exploratory window only; 06F full-epoch safety stop still applies |

The warm 64-step two-process aggregate rate is 1.66× the one-process rate. The two-process short-window median is 1.85× the corresponding four-thread one-process median. The older [06F complete-epoch audit](../../training/06f_resource_optimization/README.md) found 1.64× on one epoch and 1.453× over a four-epoch production regression. Those stronger, longer measurements support the two-process allocation. Three-process short-window throughput does not override the 06F 15,638 MiB VRAM safety stop.

On the warm 64-step one-process run, read/pack took 0.39 s, preprocessing plus transfer 0.50 s, and forward/backward/optimizer 0.40 s. The first repeat's read/pack time was 1.57 s. Thus the **measured serial host path** accounts for about 69% of warm step time and is strongly sensitive to cache state. This does not prove CPU arithmetic saturation: worker CPU time was roughly one core-equivalent per worker, well below the 32 accessible logical CPUs. At two warm workers, median per-worker read/pack, preparation/transfer, and model-step times were about 0.46, 0.57, and 0.50 s. Contention increases stage latency while aggregate throughput still improves. Peak summed worker RSS in the short two-process run was about 4.0 GiB, but summed RSS double-counts shared mapped pages. Device-wide peak VRAM was about 3.47 GiB during these windows and cannot predict a full epoch's peak.

The transformed-feature transfer microbenchmark found pageable copies around 0.46 ms, reused pinned copies around 0.35 ms, and pinning plus copying around 0.47–0.57 ms per 65,536-row batch. Exact feature tensors matched. This shows only a small copy-level benefit from already-pinned memory; it is **not** an end-to-end speedup. Evaluation on inner TRAIN rows took about 0.66 s for 64 batches in the warm one-process run and 0.78 s per worker in the warm two-process run. Checkpoint serialization averaged about 4–5 ms per worker for this small model; checkpoint file bytes are in the raw records. Identical fold-1 final model SHA-256 digests across repeats and process/thread settings establish exact parameter equivalence for these windows. The accepted cache path already has complete one-epoch data and trajectory equivalence evidence in [06F](../../training/06f_prelaunch/README.md).

## Runtime and Phase 7A budget

Frozen support across all 25 inner folds is 1,605,613,836 TRAIN rows plus 401,403,459 VALIDATION rows per common epoch. If every fold trains and validates for all 100 epochs, the warm two-process 64-step rates project **9.33 hours**; its cold repeat projects **13.20 hours**. The warm one-process projection is **15.48 hours**. The cold one-process repeat yields **28.26 hours**, illustrating cache sensitivity rather than a confidence bound. A working planning range is **10–20 hours of wall time per complete 25-fold, 100-epoch configuration** with two concurrent workers, leaving room for longer-run cache effects and orchestration. One outer study (five inner folds) is roughly one fifth of that work, about **2–4 hours** per full configuration. Multiply by the number of configurations actually selected for Phase 7A; the number of weight and Pareto candidates has not yet been specified. The JSON budget lists exact arithmetic and assumptions. These projections assume per-window throughput persists across all tiles/epochs and every fold is similarly costly. They require a complete-epoch calibration before committing a large experiment budget.

## Recommendations

1. Use **two independent training processes on the one GPU**, four intra-op CPU threads each, one inter-op thread, zero `DataLoader` workers, and about 16 logical CPUs of affinity per worker when practical. The short unpinned-affinity comparison gave 5.50–5.61M rows/s for two workers versus 5.70–6.08M with partitions; this is a modest, noisy difference, so affinity is secondary. One versus four versus eight intra-op threads showed little single-worker throughput difference in the short windows.
2. Keep the accepted mmap cache and frozen row order. It is the proven path; the earlier cache regression showed exact streams, objectives, optimizer state, and parameters. Prioritize warm-cache scheduling and avoiding concurrent scans that evict its pages.
3. Prototype overlapping CPU preparation with GPU work and reducing repeated NumPy transforms/copies, with exact row/order, preprocessing, loss, gradient, and final-state regression checks before adoption. These are hypotheses: no overlapping pipeline was measured here.
4. Test persistent pinned staging and fewer CPU/GPU synchronizations as separate scientific-equivalence experiments. The copy-only pinning result is too small to justify a production change by itself. Finite-value and support checks in the trainer must remain effective.
5. Test PyTorch compilation or CUDA graphs only after profiling longer runs and proving equal trajectories within the project’s required tolerance. Their startup cost, dynamic final batches, and numerical changes are unresolved. Do not alter the objective, spatial partitions, or sample population for speed.

No substantial production pipeline change was made. The benchmark's 64-step window is still a small fraction of an epoch; GPU utilization counters are coarse under WDDM; and physical host RAM bandwidth and storage hardware are not observable inside this guest.
