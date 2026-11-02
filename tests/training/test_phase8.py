"""Phase 8 selection and outer-TRAIN isolation checks."""

import os
import subprocess
import sys

import numpy as np
import pytest

from wildfire_risk_uncertainty.hpo.records import write_json
from wildfire_risk_uncertainty.training import phase8
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.phase8 import (
    SELECTED,
    chosen_trials,
    complete,
    model_key,
    quarantine_incomplete,
    retryable_worker_exit,
    worker_command,
)


def test_frozen_hpo_selection():
    trials = chosen_trials()
    assert {int(outer): record["trial_number"] for outer, record in trials.items()} == SELECTED
    assert trials["3"]["full_pilot_sensitivity_trial"] == 19
    assert trials["3"]["full_pilot_sensitivity_value"] > trials["3"]["value"]
    assert len({model_key(outer, member) for outer in SELECTED for member in range(5)}) == 25


def test_outer_train_stream_excludes_test_and_covers_all_inner_folds():
    cache = object.__new__(HpoCache)
    cache.names = ["a", "b"]
    cache.arrays = {
        "partition_offsets": np.array([0, 3, 6]),
        "inner_fold": np.array([[1], [0], [2], [3], [0], [4]], dtype=np.uint8),
        "features": np.arange(6, dtype=np.float32).reshape(-1, 1),
        "bp": np.arange(6, dtype=np.float32),
        "flp": np.zeros((6, 6), dtype=np.float32),
        "flp_valid": np.ones(6, dtype=np.uint8),
    }
    rows = np.concatenate([chunk[1] for chunk in cache.outer_train_chunks(outer_study=1, epoch=None)])
    np.testing.assert_array_equal(rows, [0, 2, 3, 5])
    first = np.concatenate([chunk[1] for chunk in cache.outer_train_chunks(outer_study=1, epoch=2)])
    second = np.concatenate([chunk[1] for chunk in cache.outer_train_chunks(outer_study=1, epoch=2)])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), rows)


def test_incomplete_checkpoint_artifacts_are_retained_for_replay(tmp_path):
    target = tmp_path / "epoch_004.pt"
    target.write_bytes(b"incomplete")
    target.with_name("epoch_004.pt.sha256.json.tmp").write_bytes(b"partial-sidecar")
    quarantine_incomplete(target)
    assert not target.exists()
    assert (tmp_path / "epoch_004.pt.corrupt").read_bytes() == b"incomplete"
    assert (tmp_path / "epoch_004.pt.sha256.json.tmp.corrupt").read_bytes() == b"partial-sidecar"


def test_completed_model_requires_matching_final_checkpoint_and_sidecar(tmp_path):
    key = model_key(1, 0)
    directory = tmp_path / "models" / key
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True)
    final = checkpoints / "epoch_100.pt"
    final.write_bytes(b"checkpoint")
    digest = sha256(final)
    write_json(directory / "complete.json", {"model_key": key, "checkpoint_sha256": digest})
    write_json(final.with_suffix(".pt.sha256.json"), {"sha256": digest})
    assert complete(tmp_path, key)
    final.write_bytes(b"damaged")
    assert not complete(tmp_path, key)


def test_console_entrypoint_spawns_importable_worker_module(tmp_path):
    # The original command used __name__, which equals "__main__" under -m.
    original = subprocess.run([sys.executable, "-m", "__main__"], capture_output=True,
                              text=True, check=False)
    assert "__main__.__spec__ is None" in original.stderr
    command = worker_command(tmp_path / "missing-experiment", 1, 0, os.getpid())
    assert command[2] == "wildfire_risk_uncertainty.training.phase8"
    repaired = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repaired.returncode == 1
    assert "FileNotFoundError" in repaired.stderr
    assert "__main__.__spec__ is None" not in repaired.stderr


def test_programming_failures_are_not_retried():
    assert not retryable_worker_exit(1)
    assert retryable_worker_exit(-9)


def test_supervisor_stops_after_one_deterministic_worker_error(tmp_path, monkeypatch):
    key = model_key(1, 0)
    write_json(tmp_path / "ledger.json", {"state": "starting", "models": {
        key: {"state": "pending", "attempts": 0, "failures": 0, "completed_epoch": 0}}})
    monkeypatch.setattr(phase8, "verify_manifest", lambda _path: {"outer_studies": [1]})
    monkeypatch.setattr(phase8, "MEMBERS", 1)
    monkeypatch.setattr(phase8, "complete", lambda _path, _key: False)
    calls = []

    class FailedWorker:
        pid = os.getpid()

        def poll(self):
            return 1

        def wait(self):
            return 1

    def launch_worker(*args, **kwargs):
        calls.append((args, kwargs))
        return FailedWorker()

    monkeypatch.setattr(phase8.subprocess, "Popen", launch_worker)
    with pytest.raises(RuntimeError, match="deterministic worker error"):
        phase8.supervise(tmp_path)
    ledger = phase8.read_json(tmp_path / "ledger.json")
    assert len(calls) == 1
    assert ledger["state"] == "failed"
    assert ledger["models"][key]["attempts"] == 1
    assert ledger["models"][key]["failures"] == 1
