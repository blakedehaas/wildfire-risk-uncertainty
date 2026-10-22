"""Focused checks for the fixed-fold continuation's validation reporting."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/gradnorm_full_fold.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gradnorm_full_fold", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_risk_strata_use_stored_target_precision():
    module = load_module()
    values = torch.tensor([0.0, 0.001, 0.01, 0.02, 0.5], dtype=torch.float32)
    masks = module.risk_masks(values)
    assert [int(mask.sum()) for mask in masks] == [5, 1, 1, 1, 2]
    assert bool(masks[-1][-2])
    assert torch.equal(torch.stack(masks[1:]).sum(dim=0), torch.ones(5))


def test_climatology_skill_uses_same_soft_target_support_and_entropy():
    module = load_module()
    target = torch.tensor([0.1, 0.3], dtype=torch.float64)
    forecast = torch.tensor([0.2, 0.4], dtype=torch.float64)
    climate = 0.2
    ce = -(target * forecast.log() + (1 - target) * torch.log1p(-forecast))
    entropy = -(torch.special.xlogy(target, target) +
                torch.special.xlogy(1 - target, 1 - target))
    sums = {"n": 2, "target": float(target.sum()), "predicted": float(forecast.sum()),
            "target2": float(target.square().sum()),
            "predicted2": float(forecast.square().sum()),
            "cross": float((target * forecast).sum()),
            "abs_error": float((forecast - target).abs().sum()),
            "squared_error": float((forecast - target).square().sum()),
            "model_ce": float(ce.sum()), "entropy": float(entropy.sum()),
            "climate_abs_error": float((target - climate).abs().sum()),
            "climate_squared_error": float((target - climate).square().sum())}
    result = module._summary(sums, climate)
    expected_base_kl = float((-(target * math.log(climate) +
                               (1 - target) * math.log1p(-climate)) - entropy).mean())
    assert math.isclose(result["climatology_excess_log_loss"], expected_base_kl)
    assert math.isclose(result["excess_log_loss_skill_vs_climatology"],
                        expected_base_kl - result["excess_log_loss"])


def test_epoch_record_discovery_excludes_checkpoint_sidecars_and_sorts_numerically(tmp_path):
    module = load_module()
    for epoch in (9, 100, 99):
        (tmp_path / f"fold_1_epoch_{epoch:03d}.json").write_text("{}")
        (tmp_path / f"fold_1_epoch_{epoch:03d}.pt").write_text("checkpoint")
        (tmp_path / f"fold_1_epoch_{epoch:03d}.pt.sha256.json").write_text("{}")
    assert [epoch for epoch, _ in module.epoch_records(tmp_path)] == [9, 99, 100]
    (tmp_path / "fold_1_epoch_99.json").write_text("{}")
    with pytest.raises(ValueError, match="Duplicate metric records for epoch 99"):
        module.epoch_records(tmp_path)
