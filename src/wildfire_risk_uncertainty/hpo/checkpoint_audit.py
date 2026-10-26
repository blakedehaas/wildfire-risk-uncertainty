"""Read-only identification of a common, structurally valid checkpoint epoch."""

import hashlib
import pickle
from pathlib import Path

import torch

from wildfire_risk_uncertainty.hpo.records import read_json

REQUIRED_STATE = frozenset({
    "model_state", "optimizer_state", "gradnorm_state", "gradnorm_optimizer_state",
    "python_rng_state", "numpy_rng_state", "torch_cpu_rng_state",
    "torch_cuda_rng_state", "metadata", "completed_epoch", "global_step",
})


def valid_common_epoch(local: Path, expected_metadata: dict[int, dict]) -> int:
    """Find a fallback without modifying any checkpoint, record, or trial state."""
    counts = [len(list(local.glob(f"fold_{fold}_epoch_*.json"))) for fold in range(1, 6)]
    for epoch in range(min(counts), 0, -1):
        for fold in range(1, 6):
            record_path = local / f"fold_{fold}_epoch_{epoch:03d}.json"
            path = local / "checkpoints" / f"fold_{fold}_epoch_{epoch:03d}.pt"
            sidecar = path.with_suffix(path.suffix + ".sha256.json")
            try:
                record = read_json(record_path)
                digest = read_json(sidecar)["sha256"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    break
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if (not REQUIRED_STATE <= saved.keys() or
                        saved["completed_epoch"] != epoch or
                        saved["global_step"] != record["global_optimizer_step"] or
                        saved["metadata"] != expected_metadata[fold] or
                        not saved["model_state"] or not saved["optimizer_state"]["state"] or
                        saved["gradnorm_state"] is None or
                        not saved["gradnorm_optimizer_state"]["state"]):
                    break
            except (OSError, ValueError, KeyError, RuntimeError, EOFError,
                    pickle.UnpicklingError):
                break
        else:
            return epoch
    return 0
