"""Failure reconciliation for the detached Phase 7A supervisor."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/benchmark_07a_detached.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_07a_detached", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dead_supervisor_marks_run_and_worker_failed(tmp_path):
    module = load_module()
    run = tmp_path / "07a-fixture"
    run.mkdir()
    module.write(run / "status.json", {
        "state": "running", "supervisor_pid": 99999999,
        "supervisor_pid_start": "1", "workers": {"1": {"state": "running"}},
    })
    result = module.status(run)
    assert result["state"] == "failed"
    assert result["workers"]["1"]["state"] == "parent_lost"
    assert result["ended_at"]
    assert module.status(run) == result


def test_full_epoch_configuration_stays_within_validated_limit():
    module = load_module()
    assert module.CONFIG["processes"] == 2
    assert module.CONFIG["threads_each"] == 4
    assert module.CONFIG["data_loader_workers"] == 0
    assert module.CONFIG["gpu_vram_stop_mib"] == 15500
    assert module.CONFIG["folds"] == [1, 2, 3, 4, 5]
    assert module.CONFIG["epochs"] == 3


def test_epoch_one_reference_mismatch_is_persisted(tmp_path):
    module = load_module()
    module.RUNS = tmp_path
    reference = tmp_path / "old"
    reference.mkdir()
    training = {key: 1 for key in ("bp_loss", "flp_loss", "combined_loss",
                                   "bp_loss_sum", "flp_loss_sum", "bp_rows",
                                   "flp_valid_rows", "optimizer_steps")}
    module.write(reference / "fold_1.json", {"state_sha256": "state",
                                              "training": training})
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(ValueError, match="reference mismatch"):
        module.reference_check(candidate, {"reference_run": "old"}, 1,
                               {**training, "bp_loss": 2.0}, "state")
    assert module.read(candidate / "fold_1.reference_check.json")["differences"]["bp_loss"] == [1, 2.0]
