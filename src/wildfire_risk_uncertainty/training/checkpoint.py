"""Epoch-boundary trajectory checkpoints for the 6D engineering validation."""

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from wildfire_risk_uncertainty.training.gradnorm import GradNorm


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    *, completed_epoch: int, global_step: int, metadata: dict,
                    gradnorm: "GradNorm | None" = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "gradnorm_state": gradnorm.state_dict() if gradnorm is not None else None,
                "gradnorm_optimizer_state": gradnorm.optimizer.state_dict() if gradnorm is not None else None,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "completed_epoch": completed_epoch, "global_step": global_step,
                "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state_all(), "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    *, expected_metadata: dict, gradnorm: "GradNorm | None" = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> tuple[int, int]:
    saved = torch.load(path, map_location="cuda", weights_only=False)
    if saved["metadata"] != expected_metadata:
        raise ValueError("Checkpoint provenance does not match this training run")
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    if (saved.get("gradnorm_state") is None) != (gradnorm is None):
        raise ValueError("GradNorm checkpoint mode mismatch")
    if gradnorm is not None:
        gradnorm.load_state_dict(saved["gradnorm_state"])
        gradnorm.optimizer.load_state_dict(saved["gradnorm_optimizer_state"])
    if (saved.get("scheduler_state") is None) != (scheduler is None):
        raise ValueError("Scheduler checkpoint mode mismatch")
    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler_state"])
    random.setstate(saved["python_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    torch.set_rng_state(saved["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in saved["torch_cuda_rng_state"]])
    return int(saved["completed_epoch"]), int(saved["global_step"])
