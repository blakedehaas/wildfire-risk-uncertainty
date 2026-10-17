"""Proper soft probability losses with separate task supports."""

import torch
from torch.nn import functional as F


def loss_sums(
    bp_logits: torch.Tensor,
    flp_logits: torch.Tensor,
    bp_target: torch.Tensor,
    flp_target: torch.Tensor,
    flp_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    bp_sum = F.binary_cross_entropy_with_logits(bp_logits.float(), bp_target.float(), reduction="sum")
    log_probs = F.log_softmax(flp_logits.float(), dim=-1)
    safe_target = torch.where(flp_valid[:, None], flp_target.float(), 0.0)
    flp_sum = -(safe_target * log_probs).sum()
    return bp_sum, flp_sum, bp_target.numel(), int(flp_valid.sum().item())


def equal_task_loss(
    bp_sum: torch.Tensor, flp_sum: torch.Tensor, bp_count: int, flp_count: int
) -> torch.Tensor:
    if bp_count <= 0 or flp_count <= 0:
        raise ValueError("Both task supports must be nonempty")
    return 0.5 * (bp_sum / bp_count + flp_sum / flp_count)


def streaming_batch_loss(
    bp_sum: torch.Tensor, flp_sum: torch.Tensor, batch_rows: int, total_bp: int, total_flp: int
) -> torch.Tensor:
    """Gradient contribution whose row-weighted epoch mean equals the exact objective.

    This uses global training support counts, so all-BP-zero minibatches are
    handled without changing the FLP task weight across the epoch.
    """
    if batch_rows <= 0 or total_bp <= 0 or total_flp <= 0:
        raise ValueError("Training supports must be nonempty")
    return 0.5 * (bp_sum / batch_rows + flp_sum * (total_bp / total_flp) / batch_rows)
