"""Trace a short continuation of a completed GradNorm inner-TRAIN calibration."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import torch

from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import load_cache_checkpoint
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import make_model, train_epoch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--additional-epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fold not in range(1, 6) or args.additional_epochs not in range(1, 3):
        raise ValueError("Diagnostic is limited to one fold and at most two extra epochs")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        raise RuntimeError("Four accessible CPUs are required")
    os.sched_setaffinity(0, cpus[:4])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    source = ROOT / "artifacts/benchmark/07a_detached" / args.experiment_id
    config = json.loads((source / "config.json").read_text())
    if config["kind"] != "gradnorm_calibration" or config["epochs"] != 1:
        raise ValueError("Expected a completed one-epoch GradNorm calibration")
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    state = load_states(ROOT, cache, config["outer_study"])[args.fold]
    model, optimizer = make_model(33, config["depth"], config["width"],
                                  config["learning_rate"], config["weight_decay"])
    gradnorm = GradNorm("cuda")
    checkpoint = source / f"fold_{args.fold}_epoch_001.pt"
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = {key: value for key, value in raw["metadata"].items() if key != "cache_binding"}
    completed, global_step = load_cache_checkpoint(
        checkpoint, model, optimizer, cache=cache, state=state,
        expected_metadata=metadata, gradnorm=gradnorm)
    if completed != 1 or global_step != json.loads(
            (source / f"fold_{args.fold}_epoch_001.json").read_text())["global_step"]:
        raise ValueError("Checkpoint step mismatch")

    trace = []
    original_step = gradnorm.step

    def traced_step(*step_args, **step_kwargs):
        result = original_step(*step_args, **step_kwargs)
        trace.append({"global_step": global_step + len(trace) + 1, **result})
        return result

    gradnorm.step = traced_step
    started = time.perf_counter()
    epochs = []
    for epoch in range(2, 2 + args.additional_epochs):
        result = train_epoch(ROOT, state, model, optimizer, epoch,
                             outer_study=config["outer_study"], inner_fold=args.fold,
                             batch_size=config["batch_size"], cache=cache, gradnorm=gradnorm)
        epochs.append({"epoch": epoch, "training": result,
                       "trace_end": len(trace), "rss_bytes": resource.getrusage(
                           resource.RUSAGE_SELF).ru_maxrss * 1024})
    output = {"source_experiment_id": args.experiment_id, "source_fold": args.fold,
              "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "source_git_sha": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "role": "INNER TRAIN only", "precision": "BF16", "threads": 4,
              "initial_weight": [float(x) for x in raw["gradnorm_state"]["weights"]],
              "initial_kl": [float(x) for x in raw["gradnorm_state"]["initial_kl"]],
              "epochs": epochs, "trace": trace, "wall_seconds": time.perf_counter() - started,
              "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stable_json_bytes(output))
    print(json.dumps({key: value for key, value in output.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
