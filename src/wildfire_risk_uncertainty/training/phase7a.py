"""Phase 7A gradient geometry and empirical two-task Pareto selection."""

import math
import statistics

import torch
from torch.nn import functional as F

from wildfire_risk_uncertainty.training.losses import loss_sums
from wildfire_risk_uncertainty.training.model import RiskMLP


def _geometry(bp_gradient: torch.Tensor, flp_gradient: torch.Tensor) -> dict:
    bp_norm = float(torch.linalg.vector_norm(bp_gradient).item())
    flp_norm = float(torch.linalg.vector_norm(flp_gradient).item())
    dot = float(torch.dot(bp_gradient, flp_gradient).item())
    return {"bp_norm": bp_norm, "flp_norm": flp_norm,
            "bp_to_flp_norm_ratio": bp_norm / flp_norm if flp_norm else None,
            "cosine_similarity": dot / (bp_norm * flp_norm) if bp_norm and flp_norm else None,
            "dot_product": dot}


def gradient_diagnostics(model: RiskMLP, batch: tuple[torch.Tensor, ...],
                         total_bp: int, total_flp: int) -> dict:
    """Unweighted task gradients on shared parameters under two support rules.

    Actual support uses production global-support minibatch scaling. Common
    support restricts BP to FLP-valid rows and normalizes both by that count.
    No optimizer step or parameter gradient is written.
    """
    features, bp, flp, valid = batch
    n_bp = len(bp)
    n_flp = int(valid.sum().item())
    if n_bp <= 0 or n_flp <= 0 or total_bp <= 0 or total_flp <= 0:
        raise ValueError("Positive batch and global task supports are required")
    backbone = tuple(model.backbone.parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=features.is_cuda):
        bp_logits, flp_logits = model(features)
        bp_sum, flp_sum, _, _ = loss_sums(bp_logits, flp_logits, bp, flp, valid)
        bp_valid_sum = F.binary_cross_entropy_with_logits(
            bp_logits[valid].float(), bp[valid].float(), reduction="sum")
        losses = {
            "actual_bp": bp_sum / n_bp,
            "actual_flp": flp_sum * (total_bp / total_flp) / n_bp,
            "common_bp": bp_valid_sum / n_flp,
            "common_flp": flp_sum / n_flp,
        }
    gradients = {}
    for name, loss in losses.items():
        parts = torch.autograd.grad(loss, backbone, retain_graph=True)
        gradients[name] = torch.cat([part.detach().float().flatten() for part in parts])
    return {"batch_bp_rows": n_bp, "batch_flp_valid_rows": n_flp,
            "total_bp_rows": total_bp, "total_flp_valid_rows": total_flp,
            "actual_support": _geometry(gradients["actual_bp"], gradients["actual_flp"]),
            "common_valid_support": _geometry(gradients["common_bp"], gradients["common_flp"])}


def balanced_alpha(diagnostics: list[dict]) -> float:
    """Median per-fold alpha that equalizes weighted task gradient norms."""
    values = []
    for item in diagnostics:
        geometry = item["actual_support"]
        bp = geometry["bp_norm"]
        flp = geometry["flp_norm"]
        if bp <= 0 or flp <= 0 or not math.isfinite(bp + flp):
            raise ValueError("Cannot balance zero or nonfinite task gradients")
        values.append(flp / (bp + flp))
    if not values:
        raise ValueError("Gradient diagnostics are required")
    return statistics.median(values)


def candidate_alphas(balance: float) -> tuple[float, ...]:
    """Four fixed anchors, historical baseline, and one TRAIN-derived candidate."""
    if not math.isfinite(balance):
        raise ValueError("Balance must be finite")
    anchors = (0.1, 0.3, 0.5, 0.7, 0.9)
    target = round(min(0.95, max(0.05, balance)), 4)
    if min(abs(target - value) for value in anchors) < 0.02:
        alternatives = [round(target + direction * step, 4)
                        for step in (0.02, 0.04, 0.06, 0.08, 0.1)
                        for direction in (1, -1)]
        target = next(value for value in alternatives if 0.05 <= value <= 0.95 and
                      min(abs(value - anchor) for anchor in anchors) >= 0.02)
    return tuple(sorted((*anchors, target)))


def pareto_dominates(a: dict, b: dict, tolerance: float = 1e-12) -> bool:
    """Both excess losses no worse and one materially lower."""
    keys = ("bp_excess_log_loss", "flp_excess_log_loss")
    return (all(a[key] <= b[key] + tolerance for key in keys) and
            any(a[key] < b[key] - tolerance for key in keys))


def aggregate_folds(records: list[dict], candidates: tuple[float, ...]) -> list[dict]:
    """Equal-fold means; require identical validation support across alphas."""
    expected = {(alpha, fold) for alpha in candidates for fold in range(1, 6)}
    actual = [(float(r["alpha"]), int(r["fold"])) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Require exactly five inner validation folds per candidate")
    for fold in range(1, 6):
        support = {(r["bp_rows"], r["flp_valid_rows"]) for r in records if r["fold"] == fold}
        if len(support) != 1:
            raise ValueError(f"Validation support changed across alphas in fold {fold}")
    output = []
    for alpha in candidates:
        selected = [r for r in records if r["alpha"] == alpha]
        output.append({"alpha": alpha,
                       "bp_excess_log_loss": statistics.fmean(r["bp_excess_log_loss"] for r in selected),
                       "flp_excess_log_loss": statistics.fmean(r["flp_excess_log_loss"] for r in selected),
                       "bp_ce": statistics.fmean(r["bp_loss"] for r in selected),
                       "flp_ce": statistics.fmean(r["flp_loss"] for r in selected)})
    return output


def select_pareto_knee(points: list[dict]) -> dict:
    """Empirical nondominated set and normalized maximum-bend compromise."""
    if not points or len({p["alpha"] for p in points}) != len(points):
        raise ValueError("Unique nonempty alpha results are required")
    keys = ("bp_excess_log_loss", "flp_excess_log_loss")
    if any(not all(math.isfinite(p[key]) and p[key] >= 0 for key in keys) for p in points):
        raise ValueError("Pareto objectives must be finite and nonnegative")
    frontier = [p for p in points if not any(pareto_dominates(q, p) for q in points if q is not p)]
    lows = {key: min(p[key] for p in frontier) for key in keys}
    spans = {key: max(p[key] for p in frontier) - lows[key] for key in keys}

    def normalized(point: dict) -> tuple[float, float]:
        return tuple((point[key] - lows[key]) / spans[key] if spans[key] else 0.0 for key in keys)

    def tie_key(point: dict) -> tuple[float, float, float]:
        x, y = normalized(point)
        return (math.hypot(x, y), abs(point["alpha"] - 0.5), point["alpha"])

    bends = {p["alpha"]: 1 - sum(normalized(p)) for p in frontier}
    best_bend = max(bends.values())
    if len(frontier) >= 3 and all(spans.values()) and best_bend > 1e-12:
        tied = [p for p in frontier if bends[p["alpha"]] >= best_bend - 1e-12]
        selected = min(tied, key=tie_key)
        rule = "maximum_normalized_bend"
    else:
        selected = min(frontier, key=tie_key)
        rule = "closest_normalized_ideal_fallback"
    return {"selected_alpha": selected["alpha"], "selection_rule": rule,
            "frontier_alphas": sorted(p["alpha"] for p in frontier),
            "points": [{**p, "pareto": p in frontier,
                        "normalized_bp": normalized(p)[0],
                        "normalized_flp": normalized(p)[1],
                        "normalized_bend": bends.get(p["alpha"])} for p in points],
            "normalization_min": lows, "normalization_span": spans,
            "tie_break": "smaller normalized ideal distance, then closer to 0.5, then smaller alpha"}
