"""Persistence failures must stop a controller without consuming a trial."""

import hashlib

import optuna
import pytest
import torch

from wildfire_risk_uncertainty.hpo.checkpoint_audit import valid_common_epoch
from wildfire_risk_uncertainty.hpo.config import FIXED_SMOKE
from wildfire_risk_uncertainty.hpo.records import write_json
from wildfire_risk_uncertainty.hpo.resume import (
    CheckpointRestoreError,
    run_resumable_trial,
)
from wildfire_risk_uncertainty.hpo.wandb_tracking import DisabledTracker
from wildfire_risk_uncertainty.training import cache_checkpoint, checkpoint


def test_restore_error_preserves_running_trial_and_records(tmp_path):
    study = optuna.create_study(storage=f"sqlite:///{tmp_path / 'study.sqlite3'}")
    trial = study.ask()

    class BrokenProvider:
        def restore(self, fold, completed_epoch, global_step):
            raise ValueError("Checkpoint file hash mismatch")

    with pytest.raises(CheckpointRestoreError, match="inspect retained artifacts"):
        run_resumable_trial(
            trial=trial, outer_study=3, configuration=FIXED_SMOKE,
            provider=BrokenProvider(), record_root=tmp_path / "records",
            study_name="test", provenance={"fixture": True},
            tracker=DisabledTracker(), max_epochs=2,
        )
    assert study.trials[0].state is optuna.trial.TrialState.RUNNING
    assert not list((tmp_path / "records/test/trial_000").glob("fold_*.json"))


def test_interrupted_checkpoint_persistence_never_publishes_destination(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    destination = tmp_path / "epoch.pt"
    original = checkpoint.os.fsync
    calls = []

    def fail_first_sync(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise OSError("simulated interrupted persistence")
        return original(fd)

    monkeypatch.setattr(checkpoint.os, "fsync", fail_first_sync)
    with pytest.raises(OSError, match="interrupted persistence"):
        checkpoint.save_checkpoint(destination, model, optimizer,
                                   completed_epoch=1, global_step=3, metadata={})
    assert not destination.exists()
    assert (tmp_path / "epoch.tmp").exists()


def test_existing_checkpoint_or_sidecar_cannot_be_silently_rewritten(tmp_path, monkeypatch):
    destination = tmp_path / "epoch.pt"
    sidecar = destination.with_suffix(".pt.sha256.json")

    def unexpected_save(*args, **kwargs):
        raise AssertionError("save was called")

    monkeypatch.setattr(cache_checkpoint, "save_checkpoint", unexpected_save)
    for existing in (destination, sidecar, destination.with_suffix(".tmp"),
                     destination.with_name(destination.name + ".sha256.json.tmp")):
        existing.write_bytes(b"original")
        with pytest.raises(FileExistsError, match="already exists"):
            cache_checkpoint.save_cache_checkpoint(
                destination, None, None, completed_epoch=1, global_step=3,
                cache=None, state=None, metadata={},
            )
        assert existing.read_bytes() == b"original"
        existing.unlink()


def test_common_epoch_falls_back_without_rewriting_any_artifacts(tmp_path):
    local = tmp_path / "trial_000"
    checkpoints = local / "checkpoints"
    checkpoints.mkdir(parents=True)
    metadata = {fold: {"fold": fold} for fold in range(1, 6)}
    for fold in range(1, 6):
        for epoch in (1, 2):
            path = checkpoints / f"fold_{fold}_epoch_{epoch:03d}.pt"
            write_json(local / f"fold_{fold}_epoch_{epoch:03d}.json",
                       {"global_optimizer_step": epoch * 3})
            torch.save({"model_state": {"weight": torch.ones(1)},
                        "optimizer_state": {"state": {1: {"step": epoch}}},
                        "gradnorm_state": {"weights": torch.ones(2)},
                        "gradnorm_optimizer_state": {"state": {1: {"step": epoch}}},
                        "python_rng_state": (1,), "numpy_rng_state": (1,),
                        "torch_cpu_rng_state": torch.get_rng_state(),
                        "torch_cuda_rng_state": [], "metadata": metadata[fold],
                        "completed_epoch": epoch, "global_step": epoch * 3}, path)
            write_json(path.with_suffix(".pt.sha256.json"),
                       {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    broken = checkpoints / "fold_1_epoch_002.pt"
    broken.write_bytes(b"")
    before = {path: path.read_bytes() for path in local.rglob("*") if path.is_file()}
    assert valid_common_epoch(local, metadata) == 1
    assert {path: path.read_bytes() for path in before} == before
