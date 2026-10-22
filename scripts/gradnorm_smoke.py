"""Short, real INNER TRAIN CUDA smoke for the fixed Phase 7A GradNorm path."""

import argparse
import hashlib
import json
import resource
import statistics
import subprocess
import time
from pathlib import Path

import torch

from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.gradnorm import GradNorm, soft_target_kl
from wildfire_risk_uncertainty.training.losses import loss_sums
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    _batches,
    device_batch,
    make_model,
    seed_everything,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/training/07a/gradnorm_smoke.json")
    args = parser.parse_args()
    if not 1 <= args.steps <= 32:
        raise ValueError("Smoke test is limited to 1–32 minibatches")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for production BF16 smoke")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    state = load_states(ROOT, cache, 3)[1]
    seed_everything(42)
    model, optimizer = make_model(33, 3, 128, 0.001, 0.000316227766)
    gradnorm = GradNorm("cuda")
    rss_before_batches = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    times = []
    diagnostics = []
    rows = 0
    for index, raw in enumerate(_batches(ROOT, state, "train", 1, 3, 1, 65536, cache)):
        if index >= args.steps:
            break
        prep_start = time.perf_counter()
        features, bp, flp, valid = device_batch(raw, state)
        torch.cuda.synchronize()
        prep_seconds = time.perf_counter() - prep_start
        step_start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bp_logits, flp_logits = model(features)
            bp_sum, flp_sum, n_bp, n_flp = loss_sums(bp_logits, flp_logits, bp, flp, valid)
        flp_loss = flp_sum * (state.rows / state.flp_valid_rows) / n_bp if n_flp else None
        result = gradnorm.step(model, optimizer, bp_sum / n_bp, flp_loss,
                               soft_target_kl(bp_logits, flp_logits, bp, flp, valid))
        torch.cuda.synchronize()
        times.append({"preprocess_transfer_seconds": prep_seconds,
                      "gradnorm_network_step_seconds": time.perf_counter() - step_start})
        diagnostics.append({"bp_ce": float(bp_sum.detach()) / n_bp,
                            "flp_ce": float(flp_sum.detach()) / n_flp if n_flp else None,
                            **result})
        rows += n_bp
    baseline = json.loads((ROOT / "reports/benchmark/07a/sustained/results.json").read_text())
    warm = next(worker for case in baseline["records"] if case["processes"] == 1 and
                case["threads_each"] == 4 and case["repeat"] == 1
                for worker in case["workers"])
    result = {"status": "complete", "outer_study": 3, "inner_fold": 1,
              "role": "INNER TRAIN only", "source_git_sha": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (Path(__file__),
                                                   ROOT / "src/wildfire_risk_uncertainty/training/gradnorm.py",
                                                   ROOT / "src/wildfire_risk_uncertainty/training/trainer.py")},
              "model": {"inputs": 33, "depth": 3, "width": 128, "batch_size": 65536,
                        "learning_rate": 0.001, "weight_decay": 0.000316227766,
                        "precision": "BF16", "seed": 42},
              "state_sha256": state.digest(), "steps": len(times), "rows": rows,
              "wall_seconds": time.perf_counter() - started,
              "median_preprocess_transfer_seconds": statistics.median(
                  item["preprocess_transfer_seconds"] for item in times),
              "median_gradnorm_network_step_seconds": statistics.median(
                  item["gradnorm_network_step_seconds"] for item in times),
              "baseline_warm_64_step_train_step_median_seconds": warm["stage_median_seconds"]["train_step"],
              "baseline_warm_64_step_peak_cuda_allocated_bytes": warm["vram_peak_bytes"],
              "baseline_warm_64_step_peak_rss_bytes": warm["rss_peak_bytes"],
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "prebatch_process_rss_bytes": rss_before_batches,
              "diagnostics": diagnostics,
              "comparison_limit": "Historical 64-step baseline differs in cache state and measured stage boundary; ratio is descriptive, not a paired speedup."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stable_json_bytes(result))
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
