"""Isolated one-epoch, five-fold CUDA concurrency benchmark for Checkpoint 6F."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from wildfire_risk_uncertainty.hpo.config import Configuration
from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/training/06f_resource_optimization"
CACHE = ROOT / "artifacts/training/hpo_cache_v1"
CONFIG = Configuration(3, 64, 0.00020511104188433984, 1.493656855461762e-05, 32768)


def digest(value):
    h = hashlib.sha256()

    def add(obj):
        if isinstance(obj, torch.Tensor):
            array = obj.detach().cpu().contiguous().numpy()
            h.update(
                stable_json_bytes({"dtype": str(array.dtype), "shape": array.shape})
            )
            h.update(array.tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                h.update(str(key).encode())
                add(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                add(item)
        else:
            h.update(stable_json_bytes(obj))

    add(value)
    return h.hexdigest()


def worker(concurrency, fold):
    cache = HpoCache(ROOT, CACHE, check_files=False)
    state = load_states(ROOT, cache, 1)[fold]
    counts = support_counts(ROOT, 1)[fold]
    seed_everything(42)
    model, optimizer = make_model(
        len(state.inputs),
        CONFIG.depth,
        CONFIG.width,
        CONFIG.learning_rate,
        CONFIG.weight_decay,
    )
    started = time.perf_counter()
    training = train_epoch(
        ROOT,
        state,
        model,
        optimizer,
        1,
        outer_study=1,
        inner_fold=fold,
        batch_size=CONFIG.batch_size,
        cache=cache,
    )
    validation = validate(
        ROOT,
        state,
        model,
        counts["validation"]["rows"],
        counts["validation"]["flp_valid"],
        outer_study=1,
        inner_fold=fold,
        batch_size=CONFIG.batch_size,
        cache=cache,
    )
    result = {
        "concurrency": concurrency,
        "fold": fold,
        "training": training,
        "validation": validation,
        "model_sha256": digest(model.state_dict()),
        "optimizer_sha256": digest(optimizer.state_dict()),
        "elapsed_seconds": time.perf_counter() - started,
        "state_sha256": state.digest(),
    }
    (OUT / f"c{concurrency}_fold{fold}.json").write_bytes(stable_json_bytes(result))


def proc_cpu_rss(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        ticks = os.sysconf("SC_CLK_TCK")
        cpu = (int(fields[11]) + int(fields[12])) / ticks
        rss = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf(
            "SC_PAGE_SIZE"
        )
        return cpu, rss
    except (FileNotFoundError, ProcessLookupError):
        return 0.0, 0


def memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(rest.split()[0]) * 1024
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


def gpu():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        check=True,
    )
    return [float(x.strip()) for x in result.stdout.strip().splitlines()[0].split(",")]


def compare(candidate, reference):
    keys = (
        "bp_loss",
        "flp_loss",
        "combined_loss",
        "bp_loss_sum",
        "flp_loss_sum",
        "bp_rows",
        "flp_valid_rows",
        "optimizer_steps",
    )
    mismatches = []
    for role in ("training", "validation"):
        for key in keys:
            if candidate[role][key] != reference[role][key]:
                mismatches.append(f"{role}.{key}")
    for key in ("model_sha256", "optimizer_sha256", "state_sha256"):
        if candidate[key] != reference[key]:
            mismatches.append(key)
    return mismatches


def terminate(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = HpoCache(ROOT, CACHE, check_files=False)
    metadata = {
        "source_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "outer_study": 1,
        "trial": 0,
        "epochs": 1,
        "configuration": CONFIG.serializable(),
        "cache_manifest_sha256": hashlib.sha256(
            stable_json_bytes(cache.manifest)
        ).hexdigest(),
        "method": "independent CUDA processes; each invokes frozen train_epoch and validate",
    }
    (OUT / "metadata.json").write_bytes(stable_json_bytes(metadata))
    reference = {}
    summary = []
    sample_file = OUT / "resource_samples.csv"
    with sample_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "concurrency",
                "elapsed_seconds",
                "cpu_seconds",
                "worker_rss_bytes",
                "mem_available_bytes",
                "swap_used_bytes",
                "gpu_util_percent",
                "gpu_power_w",
                "gpu_vram_mib",
            ],
        )
        writer.writeheader()
        for concurrency in range(1, 6):
            print(f"Starting concurrency {concurrency}", flush=True)
            started = time.monotonic()
            pending = list(range(1, 6))
            active = {}
            processes = []
            swap_start = memory()[1]
            samples = []
            failure = None
            while pending or active:
                while pending and len(active) < concurrency:
                    fold = pending.pop(0)
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            str(concurrency),
                            str(fold),
                        ],
                        cwd=ROOT,
                        stdout=OUT.open("a") if False else subprocess.DEVNULL,
                        stderr=(OUT / f"c{concurrency}_fold{fold}.stderr").open("w"),
                    )
                    active[fold] = proc
                    processes.append(proc)
                time.sleep(2)
                elapsed = time.monotonic() - started
                cpu_rss = [proc_cpu_rss(p.pid) for p in processes]
                available, swap = memory()
                try:
                    utilization, power, vram = gpu()
                except (subprocess.SubprocessError, ValueError):
                    utilization = power = vram = float("nan")
                row = {
                    "concurrency": concurrency,
                    "elapsed_seconds": elapsed,
                    "cpu_seconds": sum(v[0] for v in cpu_rss),
                    "worker_rss_bytes": sum(v[1] for v in cpu_rss),
                    "mem_available_bytes": available,
                    "swap_used_bytes": swap,
                    "gpu_util_percent": utilization,
                    "gpu_power_w": power,
                    "gpu_vram_mib": vram,
                }
                writer.writerow(row)
                handle.flush()
                samples.append(row)
                if swap - swap_start > 128 * 1024**2 or available < 2 * 1024**3:
                    failure = "swap or RAM pressure"
                if vram > 15500:
                    failure = "VRAM pressure"
                if (
                    concurrency > 1
                    and summary
                    and elapsed > summary[0]["wall_seconds"] * 1.5
                ):
                    failure = "severe wall-time regression"
                if elapsed > 600:
                    failure = "possible deadlock (600-second limit)"
                for fold, proc in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    del active[fold]
                    if code != 0:
                        failure = f"fold {fold} exited {code}; see stderr"
                    else:
                        result = json.loads(
                            (OUT / f"c{concurrency}_fold{fold}.json").read_text()
                        )
                        if concurrency == 1:
                            reference[fold] = result
                        else:
                            mismatches = compare(result, reference[fold])
                            if mismatches:
                                failure = f"fold {fold} mismatch: {mismatches}"
                    print(
                        f"c{concurrency} fold {fold}: {'OK' if failure is None else failure}",
                        flush=True,
                    )
                if failure:
                    terminate(processes)
                    break
            wall = time.monotonic() - started
            results = [
                json.loads((OUT / f"c{concurrency}_fold{fold}.json").read_text())
                for fold in range(1, 6)
                if (OUT / f"c{concurrency}_fold{fold}.json").exists()
            ]
            row = {
                "concurrency": concurrency,
                "status": "STOPPED" if failure else "PASS",
                "stop_reason": failure,
                "folds_completed": len(results),
                "wall_seconds": wall,
                "rows_per_second": sum(
                    r["training"]["bp_rows"] + r["validation"]["bp_rows"]
                    for r in results
                )
                / wall,
                "peak_worker_rss_bytes": max(x["worker_rss_bytes"] for x in samples),
                "min_available_ram_bytes": min(
                    x["mem_available_bytes"] for x in samples
                ),
                "peak_swap_used_bytes": max(x["swap_used_bytes"] for x in samples),
                "mean_gpu_util_percent": sum(x["gpu_util_percent"] for x in samples)
                / len(samples),
                "mean_gpu_power_w": sum(x["gpu_power_w"] for x in samples)
                / len(samples),
                "peak_gpu_vram_mib": max(x["gpu_vram_mib"] for x in samples),
                "cpu_seconds": samples[-1]["cpu_seconds"],
            }
            summary.append(row)
            (OUT / "summary.json").write_bytes(stable_json_bytes(summary))
            print(json.dumps(row), flush=True)
            if failure:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, type=int)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        main()
