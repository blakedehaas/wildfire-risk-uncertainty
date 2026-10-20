"""Short, inner-TRAIN-only production-path throughput benchmark for Phase 7A."""

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.losses import loss_sums, streaming_batch_loss
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    _batches,
    device_batch,
    make_model,
    seed_everything,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"inputs": 33, "depth": 3, "width": 128, "activation": "ReLU", "dropout": 0,
          "learning_rate": 0.001, "weight_decay": 0.000316227766, "batch_size": 65536,
          "precision": "BF16", "seed": 42, "outer_study": 3, "inner_fold": 1}


def available_cpus() -> list[int]:
    return sorted(os.sched_getaffinity(0))


def recommended_cases(max_requested: int = 2) -> list[tuple[int, int, int]]:
    """Bound process count by accessible CPUs and free GPU memory."""
    cores = len(available_cpus())
    free = torch.cuda.mem_get_info()[0]
    max_processes = min(max_requested, max(1, cores // 4), max(1, int(free // (2 * 1024**3))))
    cases = [(1, 1, 0), (1, min(4, cores), 0), (1, min(8, cores), 0)]
    if max_processes >= 2:
        cases.append((2, min(4, max(1, cores // 2)), 0))
    if max_processes >= 3:
        cases.append((3, min(4, max(1, cores // 3)), 0))
    return list(dict.fromkeys(cases))


def step(model, optimizer, batch, state):
    """Same math and finite checks as production train_epoch, on one batch."""
    features, bp, flp, valid = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bp_logits, flp_logits = model(features)
        if not torch.isfinite(bp_logits).all() or not torch.isfinite(flp_logits).all():
            raise FloatingPointError("Nonfinite logits")
        bp_sum, flp_sum, n_bp, _ = loss_sums(bp_logits, flp_logits, bp, flp, valid)
        loss = streaming_batch_loss(bp_sum, flp_sum, n_bp, state.rows, state.flp_valid_rows)
    if not all(bool(torch.isfinite(x).item()) for x in (bp_sum, flp_sum, loss)):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None or not bool(torch.isfinite(param.grad).all().item()):
            raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all().item()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameter")
    return float(bp_sum.item()), float(flp_sum.item())


def worker(args):
    cpu_set = available_cpus()
    if args.affinity == "partition":
        cpu_set = cpu_set[args.worker_index::args.processes]
        os.sched_setaffinity(0, cpu_set)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_everything(CONFIG["seed"])
    started = time.perf_counter()
    cache = HpoCache(ROOT, args.cache, check_files=False)
    fold = CONFIG["inner_fold"] + args.worker_index
    state = load_states(ROOT, cache, CONFIG["outer_study"])[fold]
    model, optimizer = make_model(33, 3, 128, CONFIG["learning_rate"], CONFIG["weight_decay"])
    model.train()
    torch.cuda.synchronize()
    startup = time.perf_counter() - started
    iterator = iter(_batches(ROOT, state, "train", 1, 3, fold, CONFIG["batch_size"], cache))
    stage = {"read_pack": [], "preprocess_transfer": [], "train_step": []}
    sums = []
    for i in range(args.warmup + args.steps):
        t0 = time.perf_counter()
        raw = next(iterator)
        t1 = time.perf_counter()
        batch = device_batch(raw, state)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sums.append(step(model, optimizer, batch, state))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= args.warmup:
            stage["read_pack"].append(t1 - t0)
            stage["preprocess_transfer"].append(t2 - t1)
            stage["train_step"].append(t3 - t2)
    # Evaluation uses only inner TRAIN rows. It measures validation kernels and
    # host preparation without ever touching inner VALIDATION or outer TEST.
    model.eval()
    eval_stage = {"read_pack": 0.0, "preprocess_transfer": 0.0, "forward_loss": 0.0}
    eval_iterator = iter(_batches(ROOT, state, "train", None, 3, fold,
                                  CONFIG["batch_size"], cache))
    with torch.no_grad():
        for _ in range(args.steps):
            t0 = time.perf_counter()
            raw = next(eval_iterator)
            t1 = time.perf_counter()
            batch = device_batch(raw, state)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bp_logits, flp_logits = model(batch[0])
                loss_sums(bp_logits, flp_logits, batch[1], batch[2], batch[3])
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            for key, seconds in zip(eval_stage, (t1 - t0, t2 - t1, t3 - t2), strict=True):
                eval_stage[key] += seconds
    from wildfire_risk_uncertainty.training.checkpoint import save_checkpoint
    checkpoint_path = args.output / f"checkpoint_{args.case}_{args.repeat}_{args.worker_index}.pt"
    t0 = time.perf_counter()
    save_checkpoint(checkpoint_path, model, optimizer, completed_epoch=0,
                    global_step=args.warmup + args.steps, metadata={"benchmark": True})
    checkpoint_seconds = time.perf_counter() - t0
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_path.unlink()
    model_hash = hashlib.sha256()
    for tensor in model.state_dict().values():
        model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    # Transfer microbenchmark uses the same transformed features, including
    # the cost of making pinned memory. Check exact tensor equality.
    host = batch[0].cpu()
    transfer = {}
    for strategy in ("pageable", "pinned_with_allocation", "pinned_reused"):
        pinned = host.pin_memory() if strategy == "pinned_reused" else None
        times = []
        for _ in range(max(4, args.steps)):
            t0 = time.perf_counter()
            source = host.pin_memory() if strategy == "pinned_with_allocation" else (pinned if pinned is not None else host)
            copied = source.to("cuda", non_blocking=strategy != "pageable")
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        if not torch.equal(copied, batch[0]):
            raise ValueError("Transfer strategy changed feature values")
        transfer[strategy] = statistics.median(times)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"case": args.case, "repeat": args.repeat, "worker": args.worker_index,
              "inner_fold": fold,
              "rows": args.steps * CONFIG["batch_size"], "steps": args.steps,
              "startup_seconds": startup, "checkpoint_seconds": checkpoint_seconds,
              "checkpoint_bytes": checkpoint_bytes, "stage_seconds": {k: sum(v) for k, v in stage.items()},
              "model_sha256": model_hash.hexdigest(), "transfer_median_seconds": transfer,
              "cpu_seconds": usage.ru_utime + usage.ru_stime,
              "stage_median_seconds": {k: statistics.median(v) for k, v in stage.items()},
              "train_only_eval_stage_seconds": eval_stage,
              "total_seconds": sum(map(sum, stage.values())),
              "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "vram_peak_bytes": torch.cuda.max_memory_allocated(),
              "cpu_affinity": cpu_set, "threads": args.threads, "workers": 0,
              "preprocessing_sha256": state.digest(), "cache_source_hashes": cache.manifest["source_hashes"],
              "finite_losses": all(np.isfinite(x).all() for x in sums)}
    path = args.output / f"worker_{args.case}_{args.repeat}_{args.worker_index}.json"
    path.write_bytes(stable_json_bytes(result))


def gpu_sample():
    try:
        s = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                                     "--format=csv,noheader,nounits"], text=True, timeout=5)
        return [float(x.strip()) for x in s.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Production training requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = recommended_cases(args.max_processes)
    if args.cases:
        cases = [cases[int(i)] for i in args.cases.split(",")]
    records = []
    for case, (processes, threads, workers) in enumerate(cases):
        for repeat in range(args.repeats):
            active = []
            launched = time.perf_counter()
            for index in range(processes):
                cmd = [sys.executable, __file__, "--worker", "--output", str(args.output),
                       "--cache", str(args.cache), "--case", str(case), "--repeat", str(repeat),
                       "--worker-index", str(index), "--processes", str(processes),
                       "--threads", str(threads), "--warmup", str(args.warmup),
                       "--steps", str(args.steps), "--affinity", args.affinity]
                active.append(subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                                               stderr=(args.output / f"stderr_{case}_{repeat}_{index}.txt").open("w")))
            samples = []
            while any(p.poll() is None for p in active):
                sample = gpu_sample()
                if sample:
                    samples.append(sample)
                    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
                    if sample[2] > total_mib - 1024:
                        for p in active:
                            p.kill()
                        raise MemoryError(f"case {case} exceeded GPU reserve")
                time.sleep(0.5)
                if time.perf_counter() - launched > args.timeout:
                    for p in active:
                        p.kill()
                    raise TimeoutError(f"case {case} exceeded {args.timeout}s")
            wall = time.perf_counter() - launched
            if any(p.returncode != 0 for p in active):
                raise RuntimeError(f"case {case} worker failed; inspect stderr files")
            workers_out = [json.loads((args.output / f"worker_{case}_{repeat}_{i}.json").read_text())
                           for i in range(processes)]
            record = {"case": case, "repeat": repeat, "processes": processes, "threads_each": threads,
                      "data_loader_workers": workers, "affinity": args.affinity, "wall_seconds": wall,
                      "aggregate_rows_per_second_including_startup": sum(w["rows"] for w in workers_out) / wall,
                      "aggregate_rows_per_second_steady": sum(w["rows"] for w in workers_out) /
                      max(w["total_seconds"] for w in workers_out),
                      "gpu_utilization_samples": samples, "workers": workers_out}
            records.append(record)
            (args.output / "results.json").write_bytes(stable_json_bytes({"config": CONFIG,
                "warmup_steps": args.warmup, "measured_steps": args.steps,
                "repeats": args.repeats, "cases": cases, "records": records}))
            print(f"case {case} repeat {repeat}: {record['aggregate_rows_per_second_steady']:,.0f} aggregate rows/s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/benchmark/07a")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/training/hpo_cache_v1")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--affinity", choices=["none", "partition"], default="partition")
    parser.add_argument("--cases", help="Comma-separated zero-based case indices")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-processes", type=int, choices=[1, 2, 3], default=2,
                        help="Default caps at the previously validated two-process level")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--processes", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("steps, warmup, and repeats must be positive")
    if args.worker:
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
