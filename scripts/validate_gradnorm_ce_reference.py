"""Three-epoch, one-inner-fold CUDA validation of CE-rate GradNorm."""

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import threading
import time
from pathlib import Path

import torch

from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    _batches,
    device_batch,
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
STRATA = ((0.0, 0.001), (0.001, 0.01), (0.01, 0.02), (0.02, 1.000001))


def digest_state(value: object) -> str:
    digest = hashlib.sha256()

    def add(item: object) -> None:
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous().numpy()
            digest.update(stable_json_bytes({"dtype": str(data.dtype), "shape": data.shape}))
            digest.update(data.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                add(child)
        else:
            digest.update(stable_json_bytes(item))

    add(value)
    return digest.hexdigest()


def resources(stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        status = Path("/proc/self/status").read_text().splitlines()
        memory = Path("/proc/meminfo").read_text().splitlines()
        values = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in memory
                  if line.startswith(("MemAvailable:", "SwapTotal:", "SwapFree:"))}
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
        samples.append({"elapsed_seconds": time.monotonic(), "rss_bytes": rss,
                        "available_ram_bytes": values["MemAvailable"],
                        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"]})
        stop.wait(0.5)


@torch.no_grad()
def bp_strata(root: Path, state, model, cache, outer: int, fold: int) -> list[dict]:
    """Descriptive soft-target BP accuracy on fixed, declared target ranges."""
    sums = [{key: 0.0 for key in ("n", "target", "prediction", "target2", "prediction2",
                                   "cross", "absolute_error", "squared_error")}
            for _ in STRATA]
    model.eval()
    for raw in _batches(root, state, "validation", None, outer, fold, 65536, cache):
        features, bp, _, _ = device_batch(raw, state)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(features)
        pred = logits.float().sigmoid()
        for (low, high), bucket in zip(STRATA, sums, strict=True):
            mask = (bp >= low) & (bp < high)
            target = bp[mask].double()
            forecast = pred[mask].double()
            if not target.numel():
                continue
            difference = forecast - target
            bucket["n"] += target.numel()
            bucket["target"] += float(target.sum())
            bucket["prediction"] += float(forecast.sum())
            bucket["target2"] += float(target.square().sum())
            bucket["prediction2"] += float(forecast.square().sum())
            bucket["cross"] += float((target * forecast).sum())
            bucket["absolute_error"] += float(difference.abs().sum())
            bucket["squared_error"] += float(difference.square().sum())
    output = []
    for (low, high), bucket in zip(STRATA, sums, strict=True):
        n = bucket["n"]
        if n == 0:
            output.append({"target_interval": [low, high], "rows": 0})
            continue
        target_var = bucket["target2"] - bucket["target"]**2 / n
        pred_var = bucket["prediction2"] - bucket["prediction"]**2 / n
        covariance = bucket["cross"] - bucket["target"] * bucket["prediction"] / n
        correlation = covariance / math.sqrt(target_var * pred_var) if target_var > 0 and pred_var > 0 else None
        output.append({"target_interval": [low, high], "rows": int(n),
                       "target_mean": bucket["target"] / n,
                       "prediction_mean": bucket["prediction"] / n,
                       "prediction_sd": math.sqrt(max(0, pred_var / n)),
                       "mae": bucket["absolute_error"] / n,
                       "rmse": math.sqrt(bucket["squared_error"] / n),
                       "prediction_target_pearson": correlation})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint_dir.exists():
        raise FileExistsError("Validation output or checkpoint directory already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        raise RuntimeError("Four accessible CPUs are required")
    os.sched_setaffinity(0, cpus[:4])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.checkpoint_dir.mkdir(parents=True)
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    outer, fold = 3, 1
    state = load_states(ROOT, cache, outer)[fold]
    counts = support_counts(ROOT, outer)[fold]
    seed_everything(42)
    model, optimizer = make_model(33, 3, 128, 0.001, 0.000316227766)
    gradnorm = GradNorm("cuda")
    metadata = {"diagnostic": "gradnorm-ce-first-epoch-v2", "outer_study": outer,
                "inner_fold": fold, "depth": 3, "width": 128, "seed": 42,
                "batch_size": 65536, "learning_rate": 0.001,
                "weight_decay": 0.000316227766, "precision": "BF16"}
    trace, samples, epochs = [], [], []
    original_step = gradnorm.step

    def traced_step(*step_args, **step_kwargs):
        result = original_step(*step_args, **step_kwargs)
        trace.append(result)
        return result

    gradnorm.step = traced_step
    stop = threading.Event()
    monitor = threading.Thread(target=resources, args=(stop, samples), daemon=True)
    started = time.perf_counter()
    monitor.start()
    try:
        global_step = 0
        for epoch in range(1, 4):
            training = train_epoch(ROOT, state, model, optimizer, epoch,
                                   outer_study=outer, inner_fold=fold, batch_size=65536,
                                   cache=cache, gradnorm=gradnorm)
            global_step += training["optimizer_steps"]
            validation = validate(ROOT, state, model,
                                  counts["validation"]["rows"],
                                  counts["validation"]["flp_valid"],
                                  outer_study=outer, inner_fold=fold,
                                  batch_size=65536, cache=cache, report_excess=True)
            strata = bp_strata(ROOT, state, model, cache, outer, fold)
            checkpoint = args.checkpoint_dir / f"fold_1_epoch_{epoch:03d}.pt"
            before = [digest_state(item) for item in (model.state_dict(), optimizer.state_dict(),
                                                       gradnorm.state_dict(), gradnorm.optimizer.state_dict())]
            save_cache_checkpoint(checkpoint, model, optimizer,
                                  completed_epoch=epoch, global_step=global_step,
                                  cache=cache, state=state, metadata=metadata,
                                  gradnorm=gradnorm)
            restored = load_cache_checkpoint(checkpoint, model, optimizer,
                                             cache=cache, state=state,
                                             expected_metadata=metadata,
                                             gradnorm=gradnorm)
            after = [digest_state(item) for item in (model.state_dict(), optimizer.state_dict(),
                                                      gradnorm.state_dict(), gradnorm.optimizer.state_dict())]
            if restored != (epoch, global_step) or before != after:
                raise ValueError("Checkpoint restoration changed trajectory state")
            epoch_trace = trace[-training["optimizer_steps"]:]
            weights = [step["bp_weight_after"] for step in epoch_trace]
            gradient_shares = [step["bp_shared_gradient_norm"] /
                               (step["bp_shared_gradient_norm"] + step["flp_shared_gradient_norm"])
                               for step in epoch_trace if step["gradnorm_weight_updated"] and
                               step["bp_shared_gradient_norm"] + step["flp_shared_gradient_norm"] > 0]
            epochs.append({"epoch": epoch, "training": training, "validation": validation,
                           "bp_strata": strata, "trace_end": len(trace),
                           "bp_weight_min": min(weights), "bp_weight_max": max(weights),
                           "bp_weight_floor_contacts": sum(weight <= 1.1e-8 for weight in weights),
                           "bp_weighted_shared_gradient_share_median": (
                               sorted(gradient_shares)[len(gradient_shares)//2] if gradient_shares else None),
                           "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
                           "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           "checkpoint_restored_exactly": True,
                           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                           "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
            progress = args.output.with_suffix(".progress.json")
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_bytes(stable_json_bytes({"completed_epochs": epochs, "trace": trace}))
    finally:
        stop.set()
        monitor.join()
    output = {"status": "complete", "role": "INNER TRAIN and VALIDATION only",
              "source_git_sha": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "preprocessing_sha256": state.digest(), "outer_study": outer, "inner_fold": fold,
              "strata": [list(x) for x in STRATA], "epochs": epochs, "trace": trace,
              "resource_samples": samples, "wall_seconds": time.perf_counter() - started,
              "peak_rss_bytes": max(x["rss_bytes"] for x in samples),
              "min_available_ram_bytes": min(x["available_ram_bytes"] for x in samples),
              "max_swap_used_bytes": max(x["swap_used_bytes"] for x in samples),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stable_json_bytes(output))
    print(json.dumps({key: value for key, value in output.items()
                      if key not in {"trace", "resource_samples"}}, indent=2))


if __name__ == "__main__":
    main()
