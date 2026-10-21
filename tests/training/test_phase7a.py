"""Scientific and deterministic Phase 7A loss/selection checks."""

import math

import pytest
import torch

from wildfire_risk_uncertainty.training.losses import (
    excess_log_losses,
    loss_sums,
    streaming_batch_loss,
    streaming_weighted_batch_loss,
    target_entropy_sums,
    weighted_task_loss,
)
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.phase7a import (
    aggregate_folds,
    balanced_alpha,
    candidate_alphas,
    gradient_diagnostics,
    select_pareto_knee,
)


def test_weighted_streaming_preserves_exact_baseline_and_global_support():
    bp = torch.tensor(3.0, requires_grad=True)
    flp = torch.tensor(5.0, requires_grad=True)
    old = streaming_batch_loss(bp, flp, 4, 20, 8)
    new = streaming_weighted_batch_loss(bp, flp, 4, 20, 8, 0.5)
    assert torch.equal(old, new)
    assert torch.autograd.grad(old, (bp, flp), retain_graph=True) == torch.autograd.grad(new, (bp, flp))
    weighted = streaming_weighted_batch_loss(bp, flp, 4, 20, 8, 0.2)
    assert weighted.item() == pytest.approx((0.2 * 3 + 0.8 * 5 * 20 / 8) / 4)
    with pytest.raises(ValueError):
        streaming_weighted_batch_loss(bp, flp, 4, 20, 8, -0.01)
    assert torch.equal(weighted_task_loss(bp, flp, 20, 8, 0.5),
                       0.5 * (bp / 20 + flp / 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_baseline_optimizer_step_is_identical():
    torch.manual_seed(42)
    old_model = RiskMLP(33, depth=3, width=128).cuda()
    new_model = RiskMLP(33, depth=3, width=128).cuda()
    new_model.load_state_dict(old_model.state_dict())
    old_opt = torch.optim.AdamW(old_model.parameters(), lr=0.001, weight_decay=0.000316227766)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=0.001, weight_decay=0.000316227766)
    x = torch.randn(96, 33, device="cuda")
    bp = torch.rand(96, device="cuda")
    flp = torch.softmax(torch.randn(96, 6, device="cuda"), dim=-1)
    valid = torch.arange(96, device="cuda") % 3 != 0
    for model, optimizer, weighted in ((old_model, old_opt, False),
                                       (new_model, new_opt, True)):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bp_logits, flp_logits = model(x)
            bs, fs, nb, _ = loss_sums(bp_logits, flp_logits, bp, flp, valid)
            loss = (streaming_weighted_batch_loss(bs, fs, nb, 960, 640, 0.5)
                    if weighted else streaming_batch_loss(bs, fs, nb, 960, 640))
        loss.backward()
        optimizer.step()
    for old, new in zip(old_model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(old, new)


def test_excess_log_loss_handles_zero_probabilities_and_masked_nan():
    bp = torch.tensor([0.0, 1.0, 0.25, 0.75])
    q = torch.tensor([[1.0, 0, 0, 0, 0, 0], [float("nan")] * 6,
                      [0.0, 0.5, 0.5, 0, 0, 0], [float("nan")] * 6])
    valid = torch.tensor([True, False, True, False])
    bp_logits = torch.logit(bp.clamp(0.001, 0.999))
    flp_logits = torch.zeros(4, 6)
    bs, fs, nb, nf = loss_sums(bp_logits, flp_logits, bp, q, valid)
    be, fe = target_entropy_sums(bp, q, valid)
    assert torch.isfinite(be) and torch.isfinite(fe)
    assert be.item() == pytest.approx(2 * (-(0.25 * math.log(0.25) + 0.75 * math.log(0.75))))
    assert fe.item() == pytest.approx(math.log(2))
    bp_kl, flp_kl = excess_log_losses(bs.item(), fs.item(), be.item(), fe.item(), nb, nf)
    assert bp_kl >= 0 and flp_kl == pytest.approx(math.log(6) - math.log(2) / 2)
    with pytest.raises(FloatingPointError):
        excess_log_losses(0, 0, 2, 2, 1, 1)


def test_gradient_diagnostics_supports_and_no_parameter_side_effects():
    torch.manual_seed(42)
    model = RiskMLP(2, depth=2, width=8)
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    bp = torch.tensor([0.1, 0.8, 0.3, 0.2])
    flp = torch.tensor([[1., 0, 0, 0, 0, 0], [float("nan")] * 6,
                        [0., 1, 0, 0, 0, 0], [float("nan")] * 6])
    valid = torch.tensor([True, False, True, False])
    before = [p.detach().clone() for p in model.parameters()]
    result = gradient_diagnostics(model, (x, bp, flp, valid), 100, 25)
    assert result["batch_bp_rows"] == 4 and result["batch_flp_valid_rows"] == 2
    assert result["actual_support"]["bp_norm"] > 0
    assert result["common_valid_support"]["flp_norm"] > 0
    assert -1 <= result["actual_support"]["cosine_similarity"] <= 1
    assert all(p.grad is None and torch.equal(p, old) for p, old in zip(model.parameters(), before, strict=True))
    a, b = model(x)
    bs, fs, _, _ = loss_sums(a, b, bp, flp, valid)
    params = tuple(model.backbone.parameters())
    b_grad = torch.cat([g.flatten() for g in torch.autograd.grad(bs / 4, params, retain_graph=True)])
    f_grad = torch.cat([g.flatten() for g in torch.autograd.grad(fs * 100 / 25 / 4, params)])
    assert result["actual_support"]["bp_norm"] == pytest.approx(b_grad.norm().item())
    assert result["actual_support"]["flp_norm"] == pytest.approx(f_grad.norm().item())
    cosine = torch.dot(b_grad, f_grad) / (b_grad.norm() * f_grad.norm())
    assert result["actual_support"]["cosine_similarity"] == pytest.approx(cosine.item())


def test_alpha_proposal_and_pareto_fallbacks():
    diagnostics = [{"actual_support": {"bp_norm": 3., "flp_norm": 1.}} for _ in range(5)]
    assert balanced_alpha(diagnostics) == 0.25
    candidates = candidate_alphas(0.5)
    assert len(candidates) == len(set(candidates)) == 6 and 0.5 in candidates
    assert candidate_alphas(0.25)[2] == 0.3
    points = [{"alpha": a, "bp_excess_log_loss": b, "flp_excess_log_loss": f}
              for a, b, f in [(0.1, 0.1, 0.9), (0.3, 0.3, 0.3), (0.5, 0.9, 0.1),
                              (0.7, 0.8, 0.8)]]
    selected = select_pareto_knee(points)
    assert selected["selected_alpha"] == 0.3
    assert selected["selection_rule"] == "maximum_normalized_bend"
    assert selected["frontier_alphas"] == [0.1, 0.3, 0.5]
    equal = [{"alpha": a, "bp_excess_log_loss": 0.1, "flp_excess_log_loss": 0.2}
             for a in (0.1, 0.5, 0.9)]
    assert select_pareto_knee(equal)["selected_alpha"] == 0.5


def test_fold_aggregation_requires_paired_validation_support():
    records = [{"alpha": alpha, "fold": fold, "bp_rows": 10 + fold,
                "flp_valid_rows": 5 + fold, "bp_excess_log_loss": alpha + fold / 10,
                "flp_excess_log_loss": 1 - alpha, "bp_loss": 0.7,
                "flp_loss": 1.4}
               for alpha in (0.3, 0.5) for fold in range(1, 6)]
    result = aggregate_folds(records, (0.3, 0.5))
    assert result[0]["bp_excess_log_loss"] == pytest.approx(0.6)
    records[-1]["bp_rows"] += 1
    with pytest.raises(ValueError, match="support changed"):
        aggregate_folds(records, (0.3, 0.5))
