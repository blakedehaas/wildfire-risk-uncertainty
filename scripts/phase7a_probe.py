"""TRAIN-only shared-backbone gradient probe and bounded alpha proposal."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.phase7a import (
    balanced_alpha,
    candidate_alphas,
    gradient_diagnostics,
)
from wildfire_risk_uncertainty.training.preprocessing import (
    stable_json_bytes,
    transform,
)
from wildfire_risk_uncertainty.training.trainer import (
    _batches,
    device_batch,
    make_model,
    seed_everything,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/training/07a/gradient_probe.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA driver is unavailable")
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    results = []
    proposals = {}
    for outer in range(1, 6):
        states = load_states(ROOT, cache, outer)
        diagnostics = []
        for fold in range(1, 6):
            state = states[fold]
            seed_everything(42)
            if device == "cuda":
                model, _ = make_model(33, 3, 128, 0.001, 0.000316227766)
            else:
                model = RiskMLP(33, 3, 128)
            batches = _batches(ROOT, state, "train", 1, outer, fold, 65536, cache)
            for raw in batches:
                if raw[3].any():
                    break
            else:
                raise ValueError(f"No common-valid TRAIN batch in outer {outer}, fold {fold}")
            if device == "cuda":
                batch = device_batch(raw, state)
            else:
                batch = (torch.from_numpy(transform(raw[0], state)),
                         torch.from_numpy(raw[1].copy()),
                         torch.from_numpy(raw[2].copy()),
                         torch.from_numpy(raw[3].copy()))
            result = gradient_diagnostics(model, batch,
                                          state.rows, state.flp_valid_rows)
            result.update(outer_study=outer, inner_fold=fold, epoch=1,
                          training_state_sha256=state.digest())
            diagnostics.append(result)
            results.append(result)
            print(f"outer {outer} fold {fold}: ratio {result['actual_support']['bp_to_flp_norm_ratio']:.4f}", flush=True)
        balance = balanced_alpha(diagnostics)
        proposals[str(outer)] = {"balanced_alpha": balance,
                                 "candidates": candidate_alphas(balance)}
    output = {"source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                           cwd=ROOT, text=True).strip(),
              "source_files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (Path(__file__),
                                                   ROOT / "src/wildfire_risk_uncertainty/training/phase7a.py",
                                                   ROOT / "src/wildfire_risk_uncertainty/training/losses.py",
                                                   ROOT / "src/wildfire_risk_uncertainty/training/trainer.py")},
              "cache_manifest_sha256": hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest(),
              "model": {"inputs": 33, "depth": 3, "width": 128, "seed": 42,
                        "precision": "BF16" if device == "cuda" else "FP32",
                        "device": device, "batch_size": 65536},
              "sample": "first shuffled epoch-1 inner TRAIN batch with FLP-valid rows per fold",
              "diagnostics": results, "proposals": proposals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stable_json_bytes(output))
    print(json.dumps(proposals, indent=2))


if __name__ == "__main__":
    main()
