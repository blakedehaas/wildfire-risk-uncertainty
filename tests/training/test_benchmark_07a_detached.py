"""Failure reconciliation for the detached Phase 7A supervisor."""

import importlib.util
from pathlib import Path

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
