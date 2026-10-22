"""Two-task GradNorm (Chen et al., ICML 2018) for the shared RiskMLP layer."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from wildfire_risk_uncertainty.training.model import RiskMLP

GRADNORM_ALPHA = 1.5
GRADNORM_LEARNING_RATE = 0.025  # Chen et al. (2018), NYUv2 experiments.


@torch.no_grad()
def soft_target_kl(
    bp_logits: torch.Tensor,
    flp_logits: torch.Tensor,
    bp_target: torch.Tensor,
    flp_target: torch.Tensor,
    flp_valid: torch.Tensor,
) -> tuple[float, float | None]:
    """Stable KL rates in float64; prediction gradients still use cross entropy."""
    p = bp_target.double()
    z = bp_logits.double()
    bp_ce = -(p * F.logsigmoid(z) + (1 - p) * F.logsigmoid(-z)).mean()
    bp_h = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p)).mean()
    bp_kl = float((bp_ce - bp_h).item())
    flp_kl = None
    if bool(flp_valid.any()):
        q = flp_target[flp_valid].double()
        logits = flp_logits[flp_valid].double()
        flp_ce = -(q * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        flp_h = -torch.special.xlogy(q, q).sum(dim=-1).mean()
        flp_kl = float((flp_ce - flp_h).item())
    if not math.isfinite(bp_kl) or (flp_kl is not None and not math.isfinite(flp_kl)):
        raise FloatingPointError("Nonfinite GradNorm KL rate")
    if bp_kl < -1e-10 or (flp_kl is not None and flp_kl < -1e-10):
        raise FloatingPointError("Negative GradNorm KL rate beyond float64 roundoff")
    return max(0.0, bp_kl), None if flp_kl is None else max(0.0, flp_kl)


def gradnorm_equations(weights: torch.Tensor, gradient_norms: torch.Tensor,
                       rates: torch.Tensor, alpha: float = GRADNORM_ALPHA,
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Published G_i, relative inverse rates r_i, and detached-target L_grad."""
    if weights.shape != (2,) or gradient_norms.shape != (2,) or rates.shape != (2,):
        raise ValueError("GradNorm requires exactly BP and conditional FLP")
    if (not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()) or
            not bool(torch.isfinite(rates).all()) or not bool(torch.isfinite(gradient_norms).all()) or
            bool((rates < 0).any()) or bool((gradient_norms < 0).any()) or
            not bool(rates.mean() > 0)):
        raise ValueError("GradNorm rates and gradient norms must be finite and nonnegative")
    relative = rates / rates.mean()
    norms = weights * gradient_norms.detach()
    targets = (norms.mean() * relative.pow(alpha)).detach()
    return norms, relative, torch.abs(norms - targets).sum()


class GradNorm(nn.Module):
    """Paper Algorithm 1 with separate Adam updates and positive simplex projection.

    The selected W is the weight matrix of the final shared backbone Linear.
    First references are pre-update KL values of the first batch with FLP support.
    """

    def __init__(self, device: torch.device | str, *, alpha: float = GRADNORM_ALPHA,
                 learning_rate: float = GRADNORM_LEARNING_RATE) -> None:
        super().__init__()
        if alpha <= 0 or learning_rate <= 0:
            raise ValueError("GradNorm alpha and learning rate must be positive")
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(2, device=device, dtype=torch.float32))
        self.register_buffer("initial_kl", torch.zeros(2, device=device, dtype=torch.float64))
        self.register_buffer("initialized", torch.tensor(False, device=device))
        self.optimizer = torch.optim.Adam([self.weights], lr=learning_rate)

    @torch.no_grad()
    def _project(self) -> None:
        # Enforce the paper's sum-to-T constraint and the requested positivity.
        self.weights.clamp_(min=1e-8)
        self.weights.mul_(2.0 / self.weights.sum())

    def step(self, model: RiskMLP, model_optimizer: torch.optim.Optimizer,
             bp_loss: torch.Tensor, flp_loss: torch.Tensor | None,
             kl: tuple[float, float | None]) -> dict:
        """Update network with old weights and weight parameters with L_grad."""
        model_optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        before = self.weights.detach().clone()
        diagnostics = {"bp_weight": float(before[0]), "flp_weight": float(before[1]),
                       "bp_kl_rate": kl[0], "flp_kl_rate": kl[1],
                       "bp_shared_gradient_norm": None, "flp_shared_gradient_norm": None,
                       "bp_relative_rate": None, "flp_relative_rate": None,
                       "gradnorm_objective": None, "gradnorm_weight_updated": False}
        if flp_loss is not None:
            if kl[1] is None:
                raise ValueError("FLP loss and KL support disagree")
            observed = torch.tensor(kl, device=self.weights.device, dtype=torch.float64)
            if not bool(torch.isfinite(observed).all()) or bool((observed < 0).any()):
                raise FloatingPointError("Invalid GradNorm KL rates")
            if not bool(self.initialized):
                if not bool((observed > 0).all()):
                    raise ValueError("First-step KL references must be positive for GradNorm")
                self.initial_kl.copy_(observed)
                self.initialized.fill_(True)
            if bool(observed.sum() > 0):
                rates = (observed / self.initial_kl).float()
                shared_weight = model.backbone[-2].weight
                bp_grad = torch.autograd.grad(bp_loss, shared_weight, retain_graph=True)[0]
                flp_grad = torch.autograd.grad(flp_loss, shared_weight, retain_graph=True)[0]
                gradient_norms = torch.stack((bp_grad.detach().float().norm(),
                                              flp_grad.detach().float().norm()))
                norms, relative, objective = gradnorm_equations(
                    self.weights, gradient_norms, rates, self.alpha)
                weight_gradient = torch.autograd.grad(objective, self.weights)[0]
                self.weights.grad = weight_gradient
                diagnostics.update(bp_shared_gradient_norm=float(norms[0].detach()),
                                   flp_shared_gradient_norm=float(norms[1].detach()),
                                   bp_relative_rate=float(relative[0]),
                                   flp_relative_rate=float(relative[1]),
                                   gradnorm_objective=float(objective.detach()),
                                   gradnorm_weight_updated=True)
            network_loss = before[0] * bp_loss + before[1] * flp_loss
        else:
            # No FLP target: keep both weights and its optimizer state unchanged.
            network_loss = before[0] * bp_loss
        if not bool(torch.isfinite(network_loss)):
            raise FloatingPointError("Nonfinite weighted network loss")
        network_loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None and flp_loss is None and name.startswith("flp_head"):
                continue
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Missing/nonfinite network gradient: {name}")
        if self.weights.grad is not None and not bool(torch.isfinite(self.weights.grad).all()):
            raise FloatingPointError("Nonfinite GradNorm weight gradient")
        model_optimizer.step()
        if diagnostics["gradnorm_weight_updated"]:
            self.optimizer.step()
            self._project()
        if not bool(torch.isfinite(self.weights).all()) or not bool((self.weights > 0).all()):
            raise FloatingPointError("Invalid GradNorm weights")
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite network parameter")
        diagnostics.update(bp_weight_after=float(self.weights[0].detach()),
                           flp_weight_after=float(self.weights[1].detach()))
        return diagnostics
