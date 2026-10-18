"""Epoch-boundary trajectory checkpoints for the 6D engineering validation."""

import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    *, completed_epoch: int, global_step: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "completed_epoch": completed_epoch, "global_step": global_step,
                "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state_all(), "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    *, expected_metadata: dict) -> tuple[int, int]:
    saved = torch.load(path, map_location="cuda", weights_only=False)
    if saved["metadata"] != expected_metadata:
        raise ValueError("Checkpoint provenance does not match this training run")
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    random.setstate(saved["python_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    torch.set_rng_state(saved["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in saved["torch_cuda_rng_state"]])
    return int(saved["completed_epoch"]), int(saved["global_step"])
