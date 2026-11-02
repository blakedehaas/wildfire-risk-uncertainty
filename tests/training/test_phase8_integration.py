"""Small real outer-TRAIN trajectory through the production training path."""

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.phase8 import (
    ROOT,
    checkpoint_metadata,
    chosen_trials,
    model_key,
    outer_preprocessing,
)
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
)


class SmallOuterTrainCache:
    """Bounded real outer-TRAIN sample with the production stream interface."""

    def __init__(self, base: HpoCache, rows: int = 2048):
        first = next(base.outer_train_chunks(outer_study=1, epoch=None))
        self.sample = tuple(np.ascontiguousarray(part[:rows]) for part in first)
        self.manifest = {**base.manifest,
                         "outer_train_counts": [rows, *base.manifest["outer_train_counts"][1:]]}

    def outer_train_chunks(self, *, outer_study: int, epoch: int | None, seed: int = 42):
        assert outer_study == 1
        order = (np.arange(len(self.sample[0])) if epoch is None else
                 np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(self.sample[0])))
        yield tuple(part[order] for part in self.sample)


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production trainer requires CUDA")
def test_real_outer_train_epoch_checkpoint_restore_and_continuation(tmp_path: Path):
    base = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    cache = SmallOuterTrainCache(base)
    schema_sha = "small-outer-train-integration-schema"
    state = outer_preprocessing(tmp_path, 1, cache, schema_sha)
    restored_state = outer_preprocessing(tmp_path, 1, cache, schema_sha)
    assert restored_state == state
    assert state.rows == 2048 and state.flp_valid_rows > 0

    configuration = chosen_trials()["1"]["configuration"]
    seed = 43000
    seed_everything(seed)
    model, optimizer = make_model(inputs=len(state.inputs), depth=configuration["depth"],
                                  width=configuration["width"],
                                  learning_rate=configuration["learning_rate"],
                                  weight_decay=configuration["weight_decay"])
    gradnorm = GradNorm("cuda")
    first = train_epoch(ROOT, state, model, optimizer, 1, outer_study=1, inner_fold=0,
                        batch_size=configuration["batch_size"], cache=cache, gradnorm=gradnorm)
    assert first["bp_rows"] == state.rows
    assert first["flp_valid_rows"] == state.flp_valid_rows
    assert first["optimizer_steps"] == 1
    assert bool(gradnorm.initialized)

    metadata = checkpoint_metadata({"integration": True}, model_key(1, 0), seed, state.digest())
    checkpoint = tmp_path / "epoch_001.pt"
    save_cache_checkpoint(checkpoint, model, optimizer, completed_epoch=1, global_step=1,
                          cache=cache, state=state, metadata=metadata, gradnorm=gradnorm)
    expected = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(gradnorm.state_dict()), copy.deepcopy(gradnorm.optimizer.state_dict()),
                random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                [value.clone() for value in torch.cuda.get_rng_state_all()])

    seed_everything(seed + 1)
    restored_model, restored_optimizer = make_model(inputs=len(state.inputs),
                                                    depth=configuration["depth"],
                                                    width=configuration["width"],
                                                    learning_rate=configuration["learning_rate"],
                                                    weight_decay=configuration["weight_decay"])
    restored_gradnorm = GradNorm("cuda")
    assert load_cache_checkpoint(checkpoint, restored_model, restored_optimizer,
                                 cache=cache, state=restored_state, expected_metadata=metadata,
                                 gradnorm=restored_gradnorm) == (1, 1)
    actual = (restored_model.state_dict(), restored_optimizer.state_dict(),
              restored_gradnorm.state_dict(), restored_gradnorm.optimizer.state_dict(),
              random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all())
    assert_equal(expected, actual)

    second = train_epoch(ROOT, restored_state, restored_model, restored_optimizer, 2,
                         outer_study=1, inner_fold=0, batch_size=configuration["batch_size"],
                         cache=cache, gradnorm=restored_gradnorm, global_step_base=1)
    assert second["bp_rows"] == state.rows and second["optimizer_steps"] == 1
    assert second["gradnorm"]["updates"] == 1
    checkpoint2 = tmp_path / "epoch_002.pt"
    save_cache_checkpoint(checkpoint2, restored_model, restored_optimizer,
                          completed_epoch=2, global_step=2, cache=cache,
                          state=restored_state, metadata=metadata, gradnorm=restored_gradnorm)
    expected_gradnorm_optimizer = copy.deepcopy(restored_gradnorm.optimizer.state_dict())
    assert expected_gradnorm_optimizer["state"]
    third_model, third_optimizer = make_model(inputs=len(state.inputs),
                                             depth=configuration["depth"], width=configuration["width"],
                                             learning_rate=configuration["learning_rate"],
                                             weight_decay=configuration["weight_decay"])
    third_gradnorm = GradNorm("cuda")
    assert load_cache_checkpoint(checkpoint2, third_model, third_optimizer,
                                 cache=cache, state=restored_state, expected_metadata=metadata,
                                 gradnorm=third_gradnorm) == (2, 2)
    assert_equal(expected_gradnorm_optimizer, third_gradnorm.optimizer.state_dict())
    print(f"outer TRAIN epoch 1 BP={first['bp_loss']:.6g} FLP={first['flp_loss']:.6g}; "
          f"restored epoch 2 BP={second['bp_loss']:.6g} FLP={second['flp_loss']:.6g}")
