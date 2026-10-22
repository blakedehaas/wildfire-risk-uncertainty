"""Independent GradNorm equations, target-support, and trajectory checks."""

import math
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import torch
from torch.nn import functional as F

from wildfire_risk_uncertainty.hpo.real import run_real_hpo
from wildfire_risk_uncertainty.hpo.storage import open_study, study_name
from wildfire_risk_uncertainty.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from wildfire_risk_uncertainty.training.gradnorm import (
    GRADNORM_ALPHA,
    GradNorm,
    gradnorm_equations,
    soft_target_kl,
)
from wildfire_risk_uncertainty.training.losses import loss_sums, target_entropy_sums
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.trainer import train_epoch


def test_published_equations_with_detached_targets():
    weights = torch.tensor([1.2, 0.8], requires_grad=True)
    base = torch.tensor([0.4, 1.2], requires_grad=True)
    rates = torch.tensor([0.5, 2.0], requires_grad=True)
    norms, relative, objective = gradnorm_equations(weights, base, rates, 1.5)
    expected_norms = [0.48, 0.96]
    expected_rates = [0.4, 1.6]
    expected_targets = [0.72 * r**1.5 for r in expected_rates]
    assert norms.tolist() == pytest.approx(expected_norms)
    assert relative.tolist() == pytest.approx(expected_rates)
    assert objective.item() == pytest.approx(sum(abs(g - t) for g, t in
                                                 zip(expected_norms, expected_targets, strict=True)))
    objective.backward()
    assert weights.grad.tolist() == pytest.approx([0.4, -1.2])
    assert base.grad is None and rates.grad is None
    assert study_name(3) == "wildfire_outer_3_hpo_v1"
    assert study_name(3, gradnorm=True) == "wildfire_outer_3_gradnorm_hpo_v1"
    with pytest.raises(ValueError, match="rates"):
        gradnorm_equations(torch.ones(2), torch.ones(2), torch.zeros(2))


def test_hpo_launch_is_gated_before_artifacts(tmp_path):
    with pytest.raises(RuntimeError, match="Review equal-task CE validation scale"):
        run_real_hpo(tmp_path, 1, cache_dir=tmp_path / "missing-cache")
    assert not (tmp_path / "artifacts").exists()
    study = open_study(tmp_path / "gradnorm.sqlite3", study_name(1, gradnorm=True),
                       {"kind": "gradnorm_hpo"})
    assert isinstance(study.pruner, optuna.pruners.NopPruner)


def test_soft_targets_kl_and_ce_gradients_with_masked_nan():
    bp_logits = torch.tensor([-12.0, 0.4, 12.0], requires_grad=True)
    flp_logits = torch.tensor([[2., 0, -2, -3, -4, -5], [1., 2, 3, 4, 5, 6],
                               [-2., 2, 0, 0, 0, 0]], requires_grad=True)
    bp = torch.tensor([0., 0.3, 1.])
    flp = torch.tensor([[0., 0.5, 0.5, 0, 0, 0], [float("nan")] * 6,
                        [1., 0, 0, 0, 0, 0]])
    valid = torch.tensor([True, False, True])
    bp_sum, flp_sum, _, n_flp = loss_sums(bp_logits, flp_logits, bp, flp, valid)
    bp_h, flp_h = target_entropy_sums(bp, flp, valid)
    bp_grad = torch.autograd.grad(bp_sum, bp_logits, retain_graph=True)[0]
    flp_grad = torch.autograd.grad(flp_sum, flp_logits, retain_graph=True)[0]
    kl_bp_grad = torch.autograd.grad(bp_sum - bp_h, bp_logits, retain_graph=True)[0]
    kl_flp_grad = torch.autograd.grad(flp_sum - flp_h, flp_logits)[0]
    assert torch.equal(bp_grad, kl_bp_grad)
    assert torch.equal(flp_grad, kl_flp_grad)
    assert torch.equal(flp_grad[1], torch.zeros(6))
    kl_bp, kl_flp = soft_target_kl(bp_logits, flp_logits, bp, flp, valid)
    assert kl_bp == pytest.approx((bp_sum.item() - bp_h.item()) / 3, abs=1e-6)
    assert kl_flp == pytest.approx((flp_sum.item() - flp_h.item()) / n_flp, abs=1e-6)
    assert math.isfinite(kl_bp) and math.isfinite(kl_flp)
    assert GRADNORM_ALPHA == 1.5


def test_no_flp_batch_keeps_weights_and_weight_optimizer_unchanged():
    torch.manual_seed(9)
    model = RiskMLP(3, depth=2, width=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    gradnorm = GradNorm("cpu")
    x = torch.randn(4, 3)
    bp_logits, flp_logits = model(x)
    bp = torch.tensor([0.1, 0.4, 0.9, 0.2])
    flp = torch.full((4, 6), float("nan"))
    valid = torch.zeros(4, dtype=torch.bool)
    bs, fs, _, nf = loss_sums(bp_logits, flp_logits, bp, flp, valid)
    assert nf == 0 and fs.item() == 0
    kl = soft_target_kl(bp_logits, flp_logits, bp, flp, valid)
    assert kl[1] is None
    before = [p.detach().clone() for p in model.parameters()]
    result = gradnorm.step(model, optimizer, bs / 4, None, kl)
    assert result["gradnorm_weight_updated"] is False
    assert gradnorm.weights.tolist() == [1.0, 1.0]
    assert not gradnorm.optimizer.state
    assert not bool(gradnorm.initialized)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    next_bp, next_flp = model(x)
    valid_next = torch.ones(4, dtype=torch.bool)
    flp_next = F.softmax(torch.randn(4, 6), dim=-1)
    bs_next, fs_next, _, nf_next = loss_sums(next_bp, next_flp, bp, flp_next, valid_next)
    gradnorm.step(model, optimizer, bs_next / 4, fs_next / nf_next,
                  soft_target_kl(next_bp, next_flp, bp, flp_next, valid_next))
    assert bool(gradnorm.initialized) and bool((gradnorm.initial_kl > 0).all())


def test_weight_objective_cannot_update_network_parameters():
    torch.manual_seed(12)
    model = RiskMLP(3, depth=2, width=8)
    reference = RiskMLP(3, depth=2, width=8)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
    gradnorm = GradNorm("cpu")
    x = torch.randn(10, 3)
    bp = torch.rand(10)
    flp = F.softmax(torch.randn(10, 6), dim=-1)
    valid = torch.ones(10, dtype=torch.bool)
    bp_logits, flp_logits = model(x)
    bs, fs, nb, nf = loss_sums(bp_logits, flp_logits, bp, flp, valid)
    gradnorm.step(model, optimizer, bs / nb, fs / nf,
                  soft_target_kl(bp_logits, flp_logits, bp, flp, valid))
    ref_bp, ref_flp = reference(x)
    ref_bs, ref_fs, _, _ = loss_sums(ref_bp, ref_flp, bp, flp, valid)
    reference_optimizer.zero_grad(set_to_none=True)
    (ref_bs / nb + ref_fs / nf).backward()
    reference_optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(actual, expected)
    assert gradnorm.weights.grad is not None


def _batch_step(model, optimizer, gradnorm, x, bp, flp):
    logits_bp, logits_flp = model(x)
    valid = torch.ones(len(bp), device=x.device, dtype=torch.bool)
    bp_sum, flp_sum, n_bp, n_flp = loss_sums(logits_bp, logits_flp, bp, flp, valid)
    return gradnorm.step(model, optimizer, bp_sum / n_bp, flp_sum / n_flp,
                         soft_target_kl(logits_bp, logits_flp, bp, flp, valid))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bf16_cuda_weights_and_checkpoint_resume_exact(tmp_path):
    torch.manual_seed(42)
    model = RiskMLP(33, depth=3, width=128).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.000316227766)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    gradnorm = GradNorm("cuda")
    x = torch.randn(64, 33, device="cuda")
    bp = torch.rand(64, device="cuda")
    flp = F.softmax(torch.randn(64, 6, device="cuda"), dim=-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = _batch_step(model, optimizer, gradnorm, x, bp, flp)
    assert first["gradnorm_weight_updated"]
    assert not torch.equal(gradnorm.weights, torch.ones_like(gradnorm.weights))
    assert gradnorm.weights.sum().item() == pytest.approx(2.0, abs=1e-6)
    assert bool((gradnorm.weights > 0).all())
    scheduler.step()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, completed_epoch=1, global_step=1,
                    metadata={"test": "gradnorm"}, gradnorm=gradnorm, scheduler=scheduler)
    restored = RiskMLP(33, depth=3, width=128).cuda()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001,
                                           weight_decay=0.000316227766)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.9)
    restored_gradnorm = GradNorm("cuda")
    torch.rand(1)
    torch.rand(1, device="cuda")
    assert load_checkpoint(path, restored, restored_optimizer,
                           expected_metadata={"test": "gradnorm"},
                           gradnorm=restored_gradnorm, scheduler=restored_scheduler) == (1, 1)
    assert torch.equal(gradnorm.initial_kl, restored_gradnorm.initial_kl)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert gradnorm.optimizer.state_dict()["state"].keys() == restored_gradnorm.optimizer.state_dict()["state"].keys()
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _batch_step(model, optimizer, gradnorm, x, bp, flp)
        _batch_step(restored, restored_optimizer, restored_gradnorm, x, bp, flp)
    assert torch.equal(gradnorm.weights, restored_gradnorm.weights)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_production_trainer_gradnorm_branch_and_empty_flp_batch(monkeypatch):
    torch.manual_seed(8)
    model = RiskMLP(33, depth=3, width=128).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    gradnorm = GradNorm("cuda")
    first = (torch.randn(8, 33, device="cuda"), torch.rand(8, device="cuda"),
             F.softmax(torch.randn(8, 6, device="cuda"), dim=-1),
             torch.ones(8, device="cuda", dtype=torch.bool))
    second = (torch.randn(8, 33, device="cuda"), torch.rand(8, device="cuda"),
              torch.full((8, 6), float("nan"), device="cuda"),
              torch.zeros(8, device="cuda", dtype=torch.bool))
    monkeypatch.setattr("wildfire_risk_uncertainty.training.trainer._batches",
                        lambda *args, **kwargs: iter((first, second)))
    monkeypatch.setattr("wildfire_risk_uncertainty.training.trainer.device_batch",
                        lambda raw, state: raw)
    state = SimpleNamespace(rows=16, flp_valid_rows=8)
    result = train_epoch(Path("."), state, model, optimizer, 1, gradnorm=gradnorm)
    assert result["optimizer_steps"] == 2
    assert result["bp_rows"] == 16 and result["flp_valid_rows"] == 8
    assert result["gradnorm"]["updates"] == 1
    assert result["gradnorm"]["bp_only_steps"] == 1
    assert result["gradnorm"]["bp_weight_final"] > 0
