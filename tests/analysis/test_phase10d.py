import numpy as np
import pyarrow as pa
import torch
from torch import nn

from wildfire_risk_uncertainty.analysis.phase10a import exact_tail_mask
from wildfire_risk_uncertainty.analysis.phase10d import (
    BPWrapper,
    exact_sample_failures,
    select_background,
    select_test,
)


class TinyMember(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.backbone = nn.Identity()
        self.bp_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.bp_head.weight[:] = torch.tensor([weight])


def test_probability_wrapper_averages_member_probabilities_and_is_differentiable():
    members = [TinyMember([1., 0.]), TinyMember([0., 2.])]
    model = BPWrapper(members)
    x = torch.tensor([[1., 2.], [-1., 0.]], requires_grad=True)
    got = model(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = torch.stack([m.bp_head(m.backbone(x)).sigmoid().float() for m in members]).mean(0)
    assert torch.allclose(got, expected)
    got.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.any(x.grad != 0)


def test_diagnostic_sampling_is_reproducible_and_matches_predicted_bp():
    n = 1_000
    pred = np.linspace(.001, .02, n).astype(np.float32)
    target = pred.copy()
    target[:100] += .02
    variance = np.full(n, .1, dtype=np.float32)
    variance[:100] = 0
    table = pa.table({"bp_prediction": pred, "bp_target": target,
        "bp_between_variance": variance, "outer_study": np.ones(n, dtype=np.uint8)})
    membership = np.arange(n) < 100
    first = select_test(table, 1, membership)
    second = select_test(table, 1, membership)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
    selected, category, failure, controls = first
    assert len(failure) == len(controls) == 40
    assert np.all(failure < 100)
    assert np.all(controls >= 100)
    assert np.median(np.abs(pred[failure] - pred[controls])) < .002
    assert np.count_nonzero(category == "failure") == 40
    assert np.count_nonzero(category == "control") == 40
    assert len(selected) >= 200


def test_background_only_uses_frozen_outer_train():
    class FakeCache:
        def __init__(self):
            self.arrays = {"cell_id": np.arange(1_000),
                "inner_fold": np.column_stack([np.arange(1_000) % 2, np.zeros(1_000)])}

    rows = select_background(FakeCache(), 1)
    assert len(rows) == 128
    assert np.all(rows % 2 == 1)
    assert np.array_equal(rows, select_background(FakeCache(), 1))


def test_exact_failure_mapping_respects_full_support_ties(monkeypatch):
    ids = np.array([90, 10, 70, 30, 50, 20, 80, 40, 60, 0])
    target = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    pred = np.zeros(10, dtype=np.float32)
    variance = np.r_[np.zeros(2), np.ones(8)].astype(np.float32)
    mask = exact_tail_mask(abs(pred - target), .1, largest=True) & exact_tail_mask(
        variance, .2, largest=False)
    assert mask.sum() == 1
    monkeypatch.setattr("wildfire_risk_uncertainty.analysis.phase10d.load_study",
        lambda _study: (ids, target, pred, variance))
    sample_ids = np.array([ids[1], ids[0], ids[8]])
    table = pa.table({"cell_id": sample_ids, "bp_prediction": pred[[1, 0, 8]]})
    assert np.array_equal(exact_sample_failures(table, 1, int(mask.sum())), mask[[1, 0, 8]])
