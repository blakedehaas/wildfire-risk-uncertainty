"""Reusable frozen BP/FLP training and global-support validation loops."""

import random
import resource
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from wildfire_risk_uncertainty.training.data import inner_chunks, packed_batches
from wildfire_risk_uncertainty.training.losses import loss_sums, streaming_batch_loss
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    transform,
)

if TYPE_CHECKING:
    from wildfire_risk_uncertainty.training.cache import HpoCache

BATCH_SIZE = 65536
SEED = 42
DATA_ORDER_VERSION = "seed-sequence-42-epoch-tile-permutation-and-within-tile-permutation-v1"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(inputs: int = 33, depth: int = 2, width: int = 128,
               learning_rate: float = 1e-3, weight_decay: float = 1e-4) -> tuple[RiskMLP, torch.optim.AdamW]:
    model = RiskMLP(inputs, depth=depth, width=width).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, optimizer


def device_batch(batch: tuple[np.ndarray, ...], state: PreprocessingState) -> tuple[torch.Tensor, ...]:
    features = transform(batch[0], state)
    if not np.isfinite(features).all():
        raise FloatingPointError("Nonfinite model input")
    if not np.isfinite(batch[1]).all() or np.any((batch[1] < 0) | (batch[1] > 1)):
        raise ValueError("BP targets must be finite probabilities")
    if not np.isfinite(batch[2][batch[3]]).all():
        raise FloatingPointError("Nonfinite valid FLP targets")
    return (torch.from_numpy(features).cuda(),
            torch.from_numpy(np.ascontiguousarray(batch[1])).cuda(),
            torch.from_numpy(np.ascontiguousarray(batch[2])).cuda(),
            torch.from_numpy(np.ascontiguousarray(batch[3])).cuda())


def _batches(root: Path, state: PreprocessingState, role: str, epoch: int | None,
             outer_study: int, inner_fold: int, batch_size: int,
             cache: "HpoCache | None" = None):
    return packed_batches(inner_chunks(
        root / "artifacts/dataset/canonical/partitions",
        root / "artifacts/spatial_validation/05b/partitions",
        root / "artifacts/spatial_validation/06c/partitions",
        state.inputs, role=role, epoch=epoch, seed=SEED,
        outer_study=outer_study, inner_fold=inner_fold, cache=cache,
    ), batch_size)


def _metrics(bp_sum: float, flp_sum: float, bp_count: int, flp_count: int,
             seconds: float, steps: int) -> dict:
    if bp_count <= 0 or flp_count <= 0:
        raise ValueError("Both task supports must be nonempty")
    return {"bp_loss": bp_sum / bp_count, "flp_loss": flp_sum / flp_count,
            "combined_loss": 0.5 * (bp_sum / bp_count + flp_sum / flp_count),
            "bp_loss_sum": bp_sum, "flp_loss_sum": flp_sum,
            "bp_rows": bp_count, "flp_valid_rows": flp_count, "optimizer_steps": steps,
            "wall_seconds": seconds, "rows_per_second": bp_count / seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def train_epoch(root: Path, state: PreprocessingState, model: RiskMLP,
                optimizer: torch.optim.AdamW, epoch: int, *, outer_study: int = 3,
                inner_fold: int = 1, batch_size: int = BATCH_SIZE,
                cache: "HpoCache | None" = None) -> dict:
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    bp_total = flp_total = 0.0
    bp_rows = flp_rows = steps = 0
    for raw in _batches(root, state, "train", epoch, outer_study, inner_fold, batch_size, cache):
        features, bp, flp, valid = device_batch(raw, state)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bp_logits, flp_logits = model(features)
            if not torch.isfinite(bp_logits).all() or not torch.isfinite(flp_logits).all():
                raise FloatingPointError("Nonfinite logits")
            bp_sum, flp_sum, n_bp, n_flp = loss_sums(bp_logits, flp_logits, bp, flp, valid)
            loss = streaming_batch_loss(bp_sum, flp_sum, n_bp, state.rows, state.flp_valid_rows)
        if not all(bool(torch.isfinite(x).item()) for x in (bp_sum, flp_sum, loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None or not bool(torch.isfinite(param.grad).all().item()):
                raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all().item()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite parameter after optimizer step")
        bp_total += float(bp_sum.item())
        flp_total += float(flp_sum.item())
        bp_rows += n_bp
        flp_rows += n_flp
        steps += 1
    torch.cuda.synchronize()
    if bp_rows != state.rows or flp_rows != state.flp_valid_rows:
        raise ValueError(f"Epoch {epoch} traversal/support mismatch: {bp_rows}/{flp_rows}")
    return _metrics(bp_total, flp_total, bp_rows, flp_rows, time.perf_counter() - start, steps)


@torch.no_grad()
def validate(root: Path, state: PreprocessingState, model: RiskMLP,
             expected_rows: int, expected_flp: int, *, outer_study: int = 3,
             inner_fold: int = 1, batch_size: int = BATCH_SIZE,
             cache: "HpoCache | None" = None) -> dict:
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    bp_total = flp_total = 0.0
    bp_rows = flp_rows = steps = 0
    for raw in _batches(root, state, "validation", None, outer_study, inner_fold, batch_size, cache):
        features, bp, flp, valid = device_batch(raw, state)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bp_logits, flp_logits = model(features)
            if not torch.isfinite(bp_logits).all() or not torch.isfinite(flp_logits).all():
                raise FloatingPointError("Nonfinite validation logits")
            bp_sum, flp_sum, n_bp, n_flp = loss_sums(bp_logits, flp_logits, bp, flp, valid)
        if not bool(torch.isfinite(bp_sum).item()) or not bool(torch.isfinite(flp_sum).item()):
            raise FloatingPointError("Nonfinite validation loss")
        bp_total += float(bp_sum.item())
        flp_total += float(flp_sum.item())
        bp_rows += n_bp
        flp_rows += n_flp
        steps += 1
    torch.cuda.synchronize()
    if bp_rows != expected_rows or flp_rows != expected_flp:
        raise ValueError(f"Validation traversal/support mismatch: {bp_rows}/{flp_rows}")
    return _metrics(bp_total, flp_total, bp_rows, flp_rows, time.perf_counter() - start, steps)
