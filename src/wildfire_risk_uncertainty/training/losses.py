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


def weighted_task_loss(bp_sum: torch.Tensor, flp_sum: torch.Tensor,
                       bp_count: int, flp_count: int, alpha: float) -> torch.Tensor:
    """Weighted means on the separate, frozen BP and conditional-FLP supports."""
    if not 0 <= alpha <= 1 or bp_count <= 0 or flp_count <= 0:
        raise ValueError("Valid alpha and positive task supports are required")
    if alpha == 0.5:
        return equal_task_loss(bp_sum, flp_sum, bp_count, flp_count)
    return alpha * bp_sum / bp_count + (1 - alpha) * flp_sum / flp_count


def streaming_weighted_batch_loss(bp_sum: torch.Tensor, flp_sum: torch.Tensor,
                                  batch_rows: int, total_bp: int, total_flp: int,
                                  alpha: float) -> torch.Tensor:
    """Production minibatch objective with static alpha and global FLP support."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0, 1]")
    if alpha == 0.5:
        return streaming_batch_loss(bp_sum, flp_sum, batch_rows, total_bp, total_flp)
    if batch_rows <= 0 or total_bp <= 0 or total_flp <= 0:
        raise ValueError("Training supports must be nonempty")
    return (alpha * bp_sum + (1 - alpha) * flp_sum * (total_bp / total_flp)) / batch_rows


def target_entropy_sums(bp_target: torch.Tensor, flp_target: torch.Tensor,
                        flp_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable Bernoulli/categorical target entropy; invalid FLP rows contribute zero."""
    p = bp_target.float()
    bp_entropy = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p)).sum()
    q = torch.where(flp_valid[:, None], flp_target.float(), 0.0)
    flp_entropy = -torch.special.xlogy(q, q).sum()
    return bp_entropy, flp_entropy


def excess_log_losses(bp_ce_sum: float, flp_ce_sum: float,
                      bp_entropy_sum: float, flp_entropy_sum: float,
                      bp_count: int, flp_count: int) -> tuple[float, float]:
    """Per-task KL = cross entropy minus target entropy, on each task's support."""
    if bp_count <= 0 or flp_count <= 0:
        raise ValueError("Task supports must be positive")
    bp = (bp_ce_sum - bp_entropy_sum) / bp_count
    flp = (flp_ce_sum - flp_entropy_sum) / flp_count
    if bp < -1e-6 or flp < -1e-6:
        raise FloatingPointError("Excess log loss is negative beyond numerical tolerance")
    return max(0.0, bp), max(0.0, flp)
