"""Small correctness checks for the fixed 6D integration path."""

import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F

from wildfire_risk_uncertainty.training import data
from wildfire_risk_uncertainty.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from wildfire_risk_uncertainty.training.data import inner_partition, packed_batches
from wildfire_risk_uncertainty.training.losses import loss_sums, streaming_batch_loss
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.preprocessing import (
    fit_preprocessing,
    transform,
)


def test_soft_losses_mask_and_global_support_weight() -> None:
    bp_logits = torch.tensor([0.2, -0.3, 0.8, 0.0])
    flp_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 4)
    bp = torch.tensor([0.1, 0.7, 0.0, 1.0])
    q = torch.tensor([[0.2, 0.1, 0.0, 0.3, 0.2, 0.2],
                      [float("nan")] * 6,
                      [0.0, 0.0, 0.0, 0.0, 0.4, 0.6],
                      [float("nan")] * 6])
    valid = torch.tensor([True, False, True, False])
    bs, fs, nb, nf = loss_sums(bp_logits, flp_logits, bp, q, valid)
    expected_bp = F.binary_cross_entropy_with_logits(bp_logits, bp, reduction="sum")
    expected_flp = -(q[valid] * F.log_softmax(flp_logits[valid], dim=-1)).sum()
    assert torch.allclose(bs, expected_bp) and torch.allclose(fs, expected_flp)
    assert (nb, nf) == (4, 2)
    first = streaming_batch_loss(*loss_sums(bp_logits[:2], flp_logits[:2], bp[:2], q[:2], valid[:2])[:2],
                                 2, 4, 2)
    second = streaming_batch_loss(*loss_sums(bp_logits[2:], flp_logits[2:], bp[2:], q[2:], valid[2:])[:2],
                                  2, 4, 2)
    assert torch.allclose((first + second) / 2, 0.5 * (bs / 4 + fs / 2))
    assert not torch.allclose((first + second) / 2, 0.5 * (bs / 4 + fs / 4))
    one_bp, one_flp, _, _ = loss_sums(bp_logits[:1], flp_logits[:1], bp[:1], q[:1], valid[:1])
    three_bp, three_flp, _, _ = loss_sums(bp_logits[1:], flp_logits[1:], bp[1:], q[1:], valid[1:])
    row_weighted = (
        streaming_batch_loss(one_bp, one_flp, 1, 4, 2) / 4
        + 3 * streaming_batch_loss(three_bp, three_flp, 3, 4, 2) / 4
    )
    assert torch.allclose(row_weighted, 0.5 * (bs / 4 + fs / 2))


def test_soft_outputs_gradients_and_synthetic_learning() -> None:
    torch.manual_seed(42)
    model = RiskMLP(2, 2, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).repeat(16, 1)
    bp = x[:, 0] * 0.8 + 0.1
    q = torch.zeros(len(x), 6)
    q[:, 0] = 1 - x[:, 1] * 0.8 - 0.1
    q[:, 1] = 1 - q[:, 0]
    valid = torch.ones(len(x), dtype=torch.bool)
    initial = None
    for _ in range(100):
        optimizer.zero_grad()
        a, b = model(x)
        bs, fs, _, _ = loss_sums(a, b, bp, q, valid)
        loss = streaming_batch_loss(bs, fs, len(x), len(x), len(x))
        if initial is None:
            initial = loss.item()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(any(p.grad.abs().sum() > 0 for p in module.parameters()) for module in (
            model.backbone, model.bp_head, model.flp_head))
        optimizer.step()
    assert loss.item() < initial * 0.8
    p, classes = model.probabilities(x)
    assert p.shape == (len(x),) and classes.shape == (len(x), 6)
    assert torch.all((p >= 0) & (p <= 1))
    assert torch.allclose(classes.sum(dim=1), torch.ones(len(x)), atol=1e-6)


def test_training_only_preprocessing_and_determinism() -> None:
    names = ("HUDEN2020", "POPDEN2020", "OCCHUDEN2020", "VACHUDEN2020", "other", "wui_failure")
    x = np.array([[0, 1, 2, 3, np.nan, 0], [3, 4, 5, 6, 10, 1]], dtype=np.float32)
    batch = (x, np.array([0.1, 0.8]), np.ones((2, 6)), np.array([True, True]))
    state = fit_preprocessing([batch], names, "schema")
    repeat = fit_preprocessing([batch], names, "schema")
    assert state.digest() == repeat.digest() and state.rows == 2
    before = state.digest()
    transformed = transform(np.array([[1000, 1000, 1000, 1000, 1000, 1]], np.float32), state)
    assert state.digest() == before and transformed[0, -1] == 1
    assert state.means[0] == pytest.approx(np.log1p(3) / 2)
    assert state.means[4] == 10
    assert not np.isclose(transformed[0, 4], 0)


def test_inner_filter_excludes_outer_test_and_dead_zone(tmp_path) -> None:
    canonical = tmp_path / "canonical"; outer = tmp_path / "outer"; inner = tmp_path / "inner"
    for path in (canonical, outer, inner):
        path.mkdir()
    name = "fixture.parquet"
    ids = [1, 2, 3, 4]
    pq.write_table(pa.table({"cell_id": ids, "x": [1., 2., 3., 4.], "BP": [0.1] * 4,
                             **{f"FLP{k}": [1 / 6] * 4 for k in range(1, 7)},
                             "flp_target_valid": [True] * 4}), canonical / name)
    pq.write_table(pa.table({"cell_id": ids, "train_eligible_fold_3": [True, True, False, False]}),
                   outer / name)
    pq.write_table(pa.table({"cell_id": ids, "inner_fold_outer_3": [1, 2, None, None]},
                            schema=pa.schema([("cell_id", pa.int64()), ("inner_fold_outer_3", pa.uint8())])),
                   inner / name)
    train = inner_partition(canonical / name, outer, inner, ("x",), role="train")
    validation = inner_partition(canonical / name, outer, inner, ("x",), role="validation")
    assert train[-1].tolist() == [2] and validation[-1].tolist() == [1]
    assert set(train[-1]).isdisjoint(validation[-1])
    assert set(train[-1]) | set(validation[-1]) == {1, 2}
    chunks = iter([(np.array([[1], [2], [3]]), np.arange(3), np.zeros((3, 6)), np.ones(3, bool)),
                   (np.array([[4], [5]]), np.arange(3, 5), np.zeros((2, 6)), np.ones(2, bool))])
    packed = list(packed_batches(chunks, 2))
    assert [b[1].tolist() for b in packed] == [[0, 1], [2, 3], [4]]


def test_epoch_partition_and_row_shuffle_is_stable_and_exhaustive(tmp_path, monkeypatch) -> None:
    paths = [tmp_path / f"tile_{j}.parquet" for j in range(4)]
    monkeypatch.setattr(data, "inner_partition_paths", lambda *_: paths.copy())

    def fixture_partition(path, _outer, _inner, _inputs, *, role, **_kwargs):
        assert role == "train"
        j = int(path.stem[-1])
        ids = np.arange(j * 5, (j + 1) * 5)
        return ids[:, None].astype(np.float32), ids.astype(np.float32), np.zeros((5, 6)), np.ones(5, bool), ids

    monkeypatch.setattr(data, "inner_partition", fixture_partition)

    def order(epoch):
        return np.concatenate([chunk[1] for chunk in data.inner_chunks(
            tmp_path, tmp_path, tmp_path, ("id",), role="train", epoch=epoch)]).astype(int).tolist()

    first = order(1)
    assert first == order(1)
    assert sorted(first) == sorted(order(2)) == list(range(20))
    assert first != order(2)


def test_checkpoint_round_trip_matches_uninterrupted_synthetic(tmp_path) -> None:
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    x = torch.ones((8, 2), device="cuda")
    target = torch.full((8,), 0.3, device="cuda")
    q = torch.full((8, 6), 1 / 6, device="cuda")
    valid = torch.ones(8, dtype=torch.bool, device="cuda")

    def make():
        m = RiskMLP(2, 2, 8).cuda()
        return m, torch.optim.AdamW(m.parameters(), lr=0.01)

    def step(m, opt):
        opt.zero_grad(); a, b = m(x)
        bs, fs, _, _ = loss_sums(a, b, target, q, valid)
        streaming_batch_loss(bs, fs, 8, 8, 8).backward(); opt.step()

    original, opt = make()
    step(original, opt)
    save_checkpoint(tmp_path / "checkpoint.pt", original, opt, completed_epoch=1,
                    global_step=1, metadata={"seed": 42, "version": "test"})
    stored = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert {"model_state", "optimizer_state", "completed_epoch", "global_step", "python_rng_state",
            "numpy_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_state", "metadata"} <= stored.keys()
    expected_rng = (random.random(), np.random.random(), torch.rand(1).item(),
                    torch.rand(1, device="cuda").item())
    step(original, opt)
    restored, restored_opt = make()
    epoch, global_step = load_checkpoint(tmp_path / "checkpoint.pt", restored, restored_opt,
                                         expected_metadata={"seed": 42, "version": "test"})
    assert (epoch, global_step) == (1, 1)
    actual_rng = (random.random(), np.random.random(), torch.rand(1).item(),
                  torch.rand(1, device="cuda").item())
    assert actual_rng == expected_rng
    step(restored, restored_opt)
    assert all(torch.equal(a, b) for a, b in zip(original.parameters(), restored.parameters(), strict=True))
