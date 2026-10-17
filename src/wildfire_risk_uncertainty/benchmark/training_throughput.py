"""Checkpoint 6B real-data throughput measurement and cost arithmetic."""

import csv
import hashlib
import math
import platform
import resource
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow
import torch

from wildfire_risk_uncertainty.training.data import (
    median_train_fold,
    model_inputs,
    training_batches,
)
from wildfire_risk_uncertainty.training.losses import loss_sums, streaming_batch_loss
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    fit_preprocessing,
    stable_json_bytes,
    transform,
)

SEED = 42
BATCH_SIZES = tuple(2**power for power in range(10, 18))
WARMUP_ROWS = 1_048_576
MEASURE_ROWS = 8_388_608


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def packed_batches(
    chunks: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    pieces: list[list[np.ndarray]] = [[], [], [], []]
    buffered = 0
    for chunk in chunks:
        n = len(chunk[0])
        offset = 0
        while offset < n:
            take = min(batch_size - buffered, n - offset)
            for field, column in zip(pieces, chunk, strict=True):
                field.append(column[offset : offset + take])
            buffered += take
            offset += take
            if buffered == batch_size:
                yield tuple(np.concatenate(part, axis=0) for part in pieces)  # type: ignore[misc]
                pieces = [[], [], [], []]
                buffered = 0
    if buffered:
        yield tuple(np.concatenate(part, axis=0) for part in pieces)  # type: ignore[misc]


def model_and_optimizer(inputs: int) -> tuple[RiskMLP, torch.optim.Optimizer]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = RiskMLP(inputs, depth=2, width=128).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer


def cpu_to_device(batch: tuple[np.ndarray, ...], state: PreprocessingState) -> tuple[torch.Tensor, ...]:
    features = torch.from_numpy(transform(batch[0], state)).to("cuda")
    bp = torch.from_numpy(np.ascontiguousarray(batch[1])).to("cuda")
    flp = torch.from_numpy(np.ascontiguousarray(batch[2])).to("cuda")
    valid = torch.from_numpy(np.ascontiguousarray(batch[3])).to("cuda")
    return features, bp, flp, valid


def training_step(
    model: RiskMLP,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...],
    state: PreprocessingState,
    precision: str,
) -> None:
    features, bp, flp, valid = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "BF16"):
        bp_logits, flp_logits = model(features)
        bp_sum, flp_sum, _, _ = loss_sums(bp_logits, flp_logits, bp, flp, valid)
        loss = streaming_batch_loss(bp_sum, flp_sum, len(bp), state.rows, state.flp_valid_rows)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    if not all(bool(torch.isfinite(param.grad).all().item()) for param in model.parameters() if param.grad is not None):
        raise FloatingPointError("Nonfinite training gradient")
    optimizer.step()


def batch_stream(canonical: Path, folds: Path, fold: int, state: PreprocessingState, size: int):
    return packed_batches(training_batches(canonical, folds, fold, state.inputs), size)


def run_training_window(
    canonical: Path,
    folds: Path,
    fold: int,
    state: PreprocessingState,
    size: int,
    precision: str,
    warmup_rows: int = WARMUP_ROWS,
    measure_rows: int = MEASURE_ROWS,
) -> dict:
    model, optimizer = model_and_optimizer(len(state.inputs))
    iterator = batch_stream(canonical, folds, fold, state, size)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(math.ceil(warmup_rows / size)):
        batch = next(iterator)
        training_step(model, optimizer, cpu_to_device(batch, state), state, precision)
    torch.cuda.synchronize()
    steps = math.ceil(measure_rows / size)
    step_times: list[float] = []
    io_seconds = preprocessing_transfer_seconds = gpu_seconds = 0.0
    measured_rows = 0
    start = time.perf_counter()
    for _ in range(steps):
        step_start = time.perf_counter()
        batch = next(iterator)
        after_read = time.perf_counter()
        prepared = cpu_to_device(batch, state)
        torch.cuda.synchronize()
        after_prepare = time.perf_counter()
        training_step(model, optimizer, prepared, state, precision)
        torch.cuda.synchronize()
        after_gpu = time.perf_counter()
        measured_rows += len(batch[0])
        io_seconds += after_read - step_start
        preprocessing_transfer_seconds += after_prepare - after_read
        gpu_seconds += after_gpu - after_prepare
        step_times.append(after_gpu - step_start)
    wall = time.perf_counter() - start
    return {
        "batch_size": size, "precision": precision, "status": "SUCCESS",
        "rows_processed": measured_rows, "optimizer_steps": steps, "wall_seconds": wall,
        "rows_per_second": measured_rows / wall, "batches_per_second": steps / wall,
        "median_step_seconds": float(np.median(step_times)),
        "p95_step_seconds": float(np.percentile(step_times, 95)),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
        "peak_host_rss_bytes": rss_bytes(),
        "read_seconds": io_seconds, "preprocess_and_transfer_seconds": preprocessing_transfer_seconds,
        "gpu_step_seconds": gpu_seconds,
        "gpu_utilization_percent": "unavailable", "gpu_memory_utilization_percent": "unavailable",
    }


def run_forward_window(canonical: Path, folds: Path, fold: int, state: PreprocessingState, size: int, precision: str) -> dict:
    model, _ = model_and_optimizer(len(state.inputs))
    model.eval()
    iterator = batch_stream(canonical, folds, fold, state, size)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(math.ceil(WARMUP_ROWS / size)):
        batch = cpu_to_device(next(iterator), state)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "BF16"):
            a, b = model(batch[0]); loss_sums(a, b, batch[1], batch[2], batch[3])
    torch.cuda.synchronize()
    start = time.perf_counter()
    rows = 0
    with torch.no_grad():
        for _ in range(math.ceil(MEASURE_ROWS / size)):
            batch = cpu_to_device(next(iterator), state)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "BF16"):
                a, b = model(batch[0]); loss_sums(a, b, batch[1], batch[2], batch[3])
            rows += len(batch[0])
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    return {"rows": rows, "wall_seconds": wall, "rows_per_second": rows / wall, "precision": precision,
            "peak_vram_bytes": torch.cuda.max_memory_allocated()}


def run_full_pass(canonical: Path, folds: Path, fold: int, state: PreprocessingState, size: int, precision: str) -> dict:
    model, optimizer = model_and_optimizer(len(state.inputs))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = steps = 0
    read_seconds = 0.0
    iterator = batch_stream(canonical, folds, fold, state, size)
    while True:
        read_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        read_seconds += time.perf_counter() - read_start
        training_step(model, optimizer, cpu_to_device(batch, state), state, precision)
        rows += len(batch[0]); steps += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    if rows != state.rows:
        raise ValueError(f"Full pass had {rows} rows; expected {state.rows}")
    return {"rows": rows, "steps": steps, "wall_seconds": wall, "rows_per_second": rows / wall,
            "data_read_seconds": read_seconds, "data_read_rows_per_second": rows / read_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(), "peak_rss_bytes": rss_bytes()}


def raw_read_and_preprocessing_breakdown(canonical: Path, folds: Path, fold: int, state: PreprocessingState) -> list[dict]:
    source = training_batches(canonical, folds, fold, state.inputs)
    chunks = []; rows = 0
    start = time.perf_counter()
    while rows < MEASURE_ROWS:
        chunk = next(source); chunks.append(chunk); rows += len(chunk[0])
    raw_wall = time.perf_counter() - start
    start = time.perf_counter()
    for chunk in chunks:
        transform(chunk[0], state)
    prep_wall = time.perf_counter() - start
    return [
        {"component": "parquet_read_and_train_filter", "rows": rows, "seconds": raw_wall, "rows_per_second": rows / raw_wall},
        {"component": "preprocessing_only_preloaded", "rows": rows, "seconds": prep_wall, "rows_per_second": rows / prep_wall},
    ]


def prepared_gpu_breakdown(canonical: Path, folds: Path, fold: int, state: PreprocessingState, size: int, precision: str) -> dict:
    model, optimizer = model_and_optimizer(len(state.inputs))
    batch = next(batch_stream(canonical, folds, fold, state, size))
    prepared = cpu_to_device(batch, state)
    for _ in range(10):
        training_step(model, optimizer, prepared, state, precision)
    torch.cuda.synchronize()
    steps = max(20, math.ceil(MEASURE_ROWS / size))
    start = time.perf_counter()
    for _ in range(steps):
        training_step(model, optimizer, prepared, state, precision)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    rows = size * steps
    return {"component": "gpu_training_prepared_tensor", "rows": rows,
            "seconds": wall, "rows_per_second": rows / wall}


def environment() -> dict:
    properties = torch.cuda.get_device_properties(0)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"gpu_query": smi, "gpu_name": properties.name,
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "vram_bytes": properties.total_memory, "bf16_supported": torch.cuda.is_bf16_supported(),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "python_version": platform.python_version(), "pyarrow_version": pyarrow.__version__,
            "numpy_version": np.__version__, "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            ).stdout.strip()}


def cost_tables(rows: int, training_rate: float, validation_rate: float) -> tuple[list[dict], list[dict]]:
    train_million = 1_000_000 / training_rate
    valid_million = 1_000_000 / validation_rate
    per_fold_epoch = 0.8 * rows / training_rate + 0.2 * rows / validation_rate
    five_epoch = 5 * per_fold_epoch
    budgets = []
    for epochs in (1, 4, 16, 64, 100):
        seconds = five_epoch * epochs
        budgets.append({"scenario": f"five_fold_{epochs}_epoch_milestone", "trials": 1,
                        "gpu_seconds": seconds, "gpu_hours": seconds / 3600,
                        "local_one_gpu_hours": seconds / 3600, "assumption": "80/20 planning approximation; cumulative resource"})
    for label, seconds in (
        ("train_1m_rows", train_million), ("validate_1m_rows", valid_million),
        ("full_outer_train_epoch", rows / training_rate),
        ("inner_train_80pct_epoch", 0.8 * rows / training_rate),
        ("inner_validate_20pct", 0.2 * rows / validation_rate),
    ):
        budgets.append({"scenario": label, "trials": 1, "gpu_seconds": seconds,
                        "gpu_hours": seconds / 3600, "local_one_gpu_hours": seconds / 3600,
                        "assumption": "80/20 planning approximation" if "inner" in label else "measured rate extrapolation"})
    # Pruning comparisons occur at epochs 1, 4, 16, and 64. Survivors can
    # continue to the 100-epoch cap; 100 is not another pruning rung.
    increments = (1, 3, 12, 48, 36)
    scaling = []
    for trials in (20, 30, 50, 100):
        worst_seconds = trials * 100 * five_epoch
        ideal_epoch_units = sum(increment * trials / 4**i for i, increment in enumerate(increments))
        ideal_seconds = ideal_epoch_units * five_epoch
        for case, per_study_seconds in (("no_pruning", worst_seconds), ("idealized_ASHA_rf4", ideal_seconds)):
            for scope, outer_studies in (("per_outer_study", 1), ("complete_nested_hpo", 5)):
                seconds = per_study_seconds * outer_studies
                budgets.append({"scenario": case, "scope": scope, "trials_per_outer_study": trials,
                                "outer_studies": outer_studies, "total_planned_trials": trials * outer_studies,
                                "trials": trials * outer_studies, "gpu_seconds": seconds,
                                "gpu_hours": seconds / 3600, "local_one_gpu_hours": seconds / 3600,
                                "assumption": "five inner folds per trial; ASHA expected fractions at 1/4/16/64, cap 100; 80/20 approximation"})
                for gpus in (1, 2, 4, 5, 8):
                    scaling.append({"scenario": case, "scope": scope,
                                    "trials_per_outer_study": trials, "outer_studies": outer_studies,
                                    "total_planned_trials": trials * outer_studies, "gpus": gpus,
                                    "ideal_work_lower_bound_wall_hours": seconds / (3600 * gpus),
                                    "whole_study_schedule_wall_hours":
                                    math.ceil(outer_studies / min(gpus, outer_studies)) * per_study_seconds / 3600,
                                    "assumption": "identical GPUs and study durations; one sequential worker per outer study; perfect scheduling"})
    return budgets, scaling


def make_figures(report: Path, batch_rows: list[dict], budgets: list[dict]) -> None:
    figure_dir = report / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    good = [r for r in batch_rows if r["status"] == "SUCCESS"]
    for field, ylabel, filename in (
        ("rows_per_second", "Rows/s", "rows_per_second_vs_batch_size.png"),
        ("peak_allocated_vram_bytes", "Allocated VRAM (GiB)", "peak_vram_vs_batch_size.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for precision in sorted({r["precision"] for r in good}):
            records = [r for r in good if r["precision"] == precision]
            y = [r[field] / (2**30 if field.endswith("bytes") else 1) for r in records]
            ax.plot([r["batch_size"] for r in records], y, marker="o", label=precision)
        ax.set_xscale("log", base=2); ax.set_xlabel("Batch size"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for case in ("no_pruning", "idealized_ASHA_rf4"):
        for scope in ("per_outer_study", "complete_nested_hpo"):
            records = sorted((r for r in budgets if r["scenario"] == case and r.get("scope") == scope),
                             key=lambda r: r["trials_per_outer_study"])
            ax.plot([r["trials_per_outer_study"] for r in records],
                    [r["local_one_gpu_hours"] for r in records], marker="o", label=f"{case}, {scope}")
    ax.set_xlabel("Trials per outer study"); ax.set_ylabel("One-GPU wall hours (planning)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "estimated_hpo_wall_time_vs_trial_count.png", dpi=160); plt.close(fig)


def run(root: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    schema_path = root / "reports/dataset/canonical/schema.json"
    schema_bytes = schema_path.read_bytes()
    inputs = model_inputs(schema_path)
    fold, expected_rows = median_train_fold(root / "reports/spatial_validation/05b/tables/fold_summary.csv")
    if fold != 3 or expected_rows != 79_519_698:
        raise ValueError(f"Frozen median fold mismatch: {fold}, {expected_rows}")
    canonical = root / "artifacts/dataset/canonical/partitions"
    folds = root / "artifacts/spatial_validation/05b/partitions"
    if not canonical.exists() or not folds.exists():
        raise FileNotFoundError("Canonical or frozen fold partitions missing")
    report = root / "reports/benchmark/06b"
    artifacts = root / "artifacts/benchmark/06b"
    report.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    write_json(report / "environment.json", environment())
    config = {"seed": SEED, "fold": fold, "expected_train_rows": expected_rows,
              "inputs": list(inputs), "batch_sizes": list(BATCH_SIZES),
              "warmup_rows": WARMUP_ROWS, "measured_rows_per_configuration": MEASURE_ROWS,
              "depth": 2, "width": 128, "activation": "ReLU", "dropout": 0,
              "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
              "precision_modes": ["FP32", "BF16"],
              "timing": "serial Parquet read, preprocess, CPU-to-GPU, forward, losses, backward, optimizer; CUDA sync each step"}
    write_json(report / "configuration.json", config)
    print(f"Scanning preprocessing state: outer TRAIN fold {fold}, {expected_rows:,} rows", flush=True)
    start = time.perf_counter()
    state = fit_preprocessing(training_batches(canonical, folds, fold, inputs), inputs,
                              hashlib.sha256(schema_bytes).hexdigest())
    scan_seconds = time.perf_counter() - start
    if state.rows != expected_rows:
        raise ValueError(f"Preprocessing scan read {state.rows}; expected {expected_rows}")
    state_bytes = stable_json_bytes(state.serializable())
    (artifacts / "preprocessing_state.json").write_bytes(state_bytes)
    write_json(report / "preprocessing.json", {**state.serializable(),
               "scan_seconds": scan_seconds, "rows_per_second": state.rows / scan_seconds,
               "peak_host_rss_bytes": rss_bytes(), "state_bytes": len(state_bytes),
               "state_sha256": state.digest(), "artifact_path": str(artifacts / "preprocessing_state.json")})
    print(f"Preprocessing: {scan_seconds:.1f}s, {state.rows / scan_seconds:,.0f} rows/s", flush=True)
    breakdown = raw_read_and_preprocessing_breakdown(canonical, folds, fold, state)
    rows = []
    for size in BATCH_SIZES:
        print(f"FP32 batch {size}", flush=True)
        try:
            row = run_training_window(canonical, folds, fold, state, size, "FP32")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"batch_size": size, "precision": "FP32", "status": "OOM"})
            break
        rows.append(row)
        write_csv(report / "batch_throughput.csv", rows)
        print(f"  {row['status']}: {row['rows_per_second']:,.0f} rows/s", flush=True)
    good_fp32 = [r for r in rows if r["status"] == "SUCCESS"]
    if not good_fp32:
        raise RuntimeError("No feasible batch sizes")
    fp32_best = max(good_fp32, key=lambda r: r["rows_per_second"])
    precision_rows = [fp32_best]
    if torch.cuda.is_bf16_supported():
        # Nonfinite BF16 errors deliberately stop the checkpoint.
        print(f"BF16 batch {fp32_best['batch_size']}", flush=True)
        bf16_row = run_training_window(canonical, folds, fold, state, fp32_best["batch_size"], "BF16")
        precision_rows.append(bf16_row); rows.append(bf16_row)
    write_csv(report / "batch_throughput.csv", rows)
    write_csv(report / "precision_comparison.csv", precision_rows)
    fastest = max((r for r in rows if r["status"] == "SUCCESS"), key=lambda r: r["rows_per_second"])
    breakdown.append(prepared_gpu_breakdown(canonical, folds, fold, state,
                                            fastest["batch_size"], fastest["precision"]))
    breakdown.append({"component": "end_to_end_training_sample", "rows": fastest["rows_processed"],
                      "seconds": fastest["wall_seconds"], "rows_per_second": fastest["rows_per_second"]})
    write_csv(report / "pipeline_breakdown.csv", breakdown)
    forward = run_forward_window(canonical, folds, fold, state, fastest["batch_size"], fastest["precision"])
    write_json(report / "forward_only.json", forward)
    print(f"Full pass: {state.rows:,} TRAIN rows, batch {fastest['batch_size']} {fastest['precision']}", flush=True)
    full = run_full_pass(canonical, folds, fold, state, fastest["batch_size"], fastest["precision"])
    write_json(report / "full_pass.json", full)
    budgets, scaling = cost_tables(state.rows, full["rows_per_second"], forward["rows_per_second"])
    write_csv(report / "compute_budget.csv", budgets)
    write_csv(report / "cluster_scaling.csv", scaling)
    make_figures(report, rows, budgets)
    write_json(report / "summary.json", {"fastest": fastest, "full_pass": full, "forward_only": forward,
               "preprocessing_seconds": scan_seconds, "input_count": len(inputs),
               "representative_fold": fold, "train_rows": state.rows,
               "preprocessing_sha256": state.digest()})
    print(f"Complete: {full['wall_seconds']:.1f}s, {full['rows_per_second']:,.0f} rows/s", flush=True)
