"""Real-data regression for the production two-worker HPO integration."""

import json
import statistics
import time
from pathlib import Path

import optuna
import torch

from wildfire_risk_uncertainty.hpo.config import Configuration
from wildfire_risk_uncertainty.hpo.fold_workers import (
    PersistentFoldProvider,
    _state_digest,
)
from wildfire_risk_uncertainty.hpo.real import RealProvider, support_counts
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.resume import run_resumable_trial
from wildfire_risk_uncertainty.hpo.wandb_tracking import MockTracker
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "artifacts/training/hpo_cache_v1"
ARTIFACT = ROOT / "artifacts/hpo/06f_production_concurrency_validation"
REPORT = (
    ROOT
    / "reports/training/06f_resource_optimization/production_integration_validation.json"
)
CONFIG = Configuration(3, 64, 0.00020511104188433984, 1.493656855461762e-05, 32768)
PROVENANCE = {"fixture": "checkpoint-6f-production-concurrency-v1"}


class TimedSequentialProvider(RealProvider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch_wall_seconds = {}
        self._epoch_started = None

    def __call__(self, fold: int, epoch: int) -> tuple[dict, dict]:
        if fold == 1:
            self._epoch_started = time.perf_counter()
        result = super().__call__(fold, epoch)
        if fold == 5:
            self.epoch_wall_seconds[epoch] = time.perf_counter() - self._epoch_started
        return result


class TimedConcurrentProvider(PersistentFoldProvider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch_wall_seconds = {}

    def run_epoch(self, epoch: int, folds: list[int]) -> dict[int, tuple[dict, dict]]:
        started = time.perf_counter()
        result = super().run_epoch(epoch, folds)
        self.epoch_wall_seconds[epoch] = time.perf_counter() - started
        return result


def warm_cache() -> float:
    started = time.perf_counter()
    total = 0
    for name in (
        "features.npy",
        "bp.npy",
        "flp.npy",
        "flp_valid.npy",
        "inner_fold.npy",
        "partition_offsets.npy",
    ):
        with (CACHE_PATH / name).open("rb", buffering=16 * 1024 * 1024) as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                total += len(block)
    if total == 0:
        raise RuntimeError("Cache warming read no data")
    return time.perf_counter() - started


def scientific_fold(record: dict) -> dict:
    return {
        key: record[key]
        for key in (
            "outer_study",
            "trial_number",
            "inner_fold",
            "epoch",
            "resource_milestone",
            "global_optimizer_step",
            "train_rows",
            "validation_rows",
            "training",
            "validation",
            "configuration",
            "precision_mode",
            "activation",
            "optimizer",
            "dropout",
            "optimizer_steps",
            "all_finite",
        )
    }


def checkpoint_digests(root: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(root.glob("fold_*_epoch_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        result[path.name] = {
            "model_sha256": _state_digest(saved["model_state"]),
            "optimizer_sha256": _state_digest(saved["optimizer_state"]),
            "python_rng_sha256": _state_digest(saved["python_rng_state"]),
            "numpy_rng_sha256": _state_digest(saved["numpy_rng_state"]),
            "torch_cpu_rng_sha256": _state_digest(saved["torch_cpu_rng_state"]),
            "torch_cuda_rng_sha256": _state_digest(saved["torch_cuda_rng_state"]),
            "completed_epoch": saved["completed_epoch"],
            "global_step": saved["global_step"],
            "metadata": saved["metadata"],
        }
    return result


def trial_records(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    folds = {
        path.name: scientific_fold(read_json(path))
        for path in sorted(root.glob("fold_*_epoch_*.json"))
    }
    aggregates = {
        path.name: read_json(path)
        for path in sorted(root.glob("aggregate_epoch_*.json"))
    }
    return folds, aggregates


def make_trial() -> tuple[optuna.Study, optuna.Trial]:
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    return study, study.ask()


def run_case(name: str, provider, *, pause_after_epoch: int | None = None) -> dict:
    study, trial = make_trial()
    tracker = MockTracker()
    started = time.perf_counter()
    try:
        result = run_resumable_trial(
            trial=trial,
            outer_study=1,
            configuration=CONFIG,
            provider=provider,
            record_root=ARTIFACT / name / "records",
            study_name="validation",
            provenance=PROVENANCE,
            tracker=tracker,
            max_epochs=4,
            pause_after_epoch=pause_after_epoch,
        )
    finally:
        if hasattr(provider, "close"):
            provider.close()
    wall = time.perf_counter() - started
    if result is not None:
        study.tell(trial, result)
    return {
        "study": study,
        "trial": trial,
        "tracker": tracker,
        "result": result,
        "wall_seconds": wall,
        "epoch_wall_seconds": dict(provider.epoch_wall_seconds),
    }


def main() -> None:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("Production integration validation output already exists")
    cache = HpoCache(ROOT, CACHE_PATH)
    states = load_states(ROOT, cache, 1)
    counts = support_counts(ROOT, 1)

    sequential_root = ARTIFACT / "sequential"
    sequential_warm = warm_cache()
    sequential = run_case(
        "sequential",
        TimedSequentialProvider(
            ROOT,
            1,
            CONFIG,
            states,
            counts,
            cache=cache,
            trial_number=0,
            checkpoint_root=sequential_root / "checkpoints",
            retain_checkpoints=True,
        ),
    )

    concurrent_root = ARTIFACT / "concurrent"
    concurrent_warm = warm_cache()
    concurrent = run_case(
        "concurrent",
        TimedConcurrentProvider(
            ROOT,
            1,
            CONFIG,
            cache_path=CACHE_PATH,
            trial_number=0,
            checkpoint_root=concurrent_root / "checkpoints",
            retain_checkpoints=True,
        ),
    )

    sequential_local = sequential_root / "records/validation/trial_000"
    concurrent_local = concurrent_root / "records/validation/trial_000"
    sequential_folds, sequential_aggregates = trial_records(sequential_local)
    concurrent_folds, concurrent_aggregates = trial_records(concurrent_local)
    if sequential_folds != concurrent_folds:
        raise ValueError("Sequential/concurrent fold-epoch scientific records differ")
    if sequential_aggregates != concurrent_aggregates:
        raise ValueError("Sequential/concurrent aggregate records differ")
    sequential_checkpoints = checkpoint_digests(sequential_root / "checkpoints")
    concurrent_checkpoints = checkpoint_digests(concurrent_root / "checkpoints")
    if sequential_checkpoints != concurrent_checkpoints:
        raise ValueError(
            "Sequential/concurrent model, AdamW, or RNG checkpoints differ"
        )

    resume_root = ARTIFACT / "resume"
    resume_study, resume_trial = make_trial()
    first_tracker = MockTracker()
    first_provider = TimedConcurrentProvider(
        ROOT,
        1,
        CONFIG,
        cache_path=CACHE_PATH,
        trial_number=0,
        checkpoint_root=resume_root / "checkpoints",
        retain_checkpoints=True,
    )
    try:
        paused = run_resumable_trial(
            trial=resume_trial,
            outer_study=1,
            configuration=CONFIG,
            provider=first_provider,
            record_root=resume_root / "records",
            study_name="validation",
            provenance=PROVENANCE,
            tracker=first_tracker,
            max_epochs=4,
            pause_after_epoch=2,
        )
    finally:
        first_provider.close()
    if paused is not None or resume_study.trials[0].state.name != "RUNNING":
        raise ValueError("Concurrent pause did not leave the Optuna trial RUNNING")
    run_id = read_json(resume_root / "records/validation/trial_000/wandb_link.json")[
        "run_id"
    ]
    second_tracker = MockTracker()
    second_provider = TimedConcurrentProvider(
        ROOT,
        1,
        CONFIG,
        cache_path=CACHE_PATH,
        trial_number=0,
        checkpoint_root=resume_root / "checkpoints",
        retain_checkpoints=True,
    )
    try:
        resumed = run_resumable_trial(
            trial=resume_trial,
            outer_study=1,
            configuration=CONFIG,
            provider=second_provider,
            record_root=resume_root / "records",
            study_name="validation",
            provenance=PROVENANCE,
            tracker=second_tracker,
            max_epochs=4,
        )
    finally:
        second_provider.close()
    if resumed is None:
        raise ValueError("Concurrent resumed trial did not complete")
    resume_study.tell(resume_trial, resumed)
    resume_local = resume_root / "records/validation/trial_000"
    resume_folds, resume_aggregates = trial_records(resume_local)
    if resume_folds != concurrent_folds or resume_aggregates != concurrent_aggregates:
        raise ValueError("Interrupted/resumed concurrent scientific trajectory differs")
    if checkpoint_digests(resume_root / "checkpoints") != concurrent_checkpoints:
        raise ValueError("Interrupted/resumed model, AdamW, or RNG checkpoints differ")
    if run_id != first_tracker.run_id or run_id != second_tracker.run_id:
        raise ValueError("Interrupted/resumed W&B run ID differs")
    if len(resume_folds) != 20 or len(resume_aggregates) != 4:
        raise ValueError("Interrupted/resumed records are duplicated or incomplete")

    speedup = sequential["wall_seconds"] / concurrent["wall_seconds"]
    per_epoch = [
        {
            "epoch": epoch,
            "sequential_seconds": sequential["epoch_wall_seconds"][epoch],
            "concurrent_seconds": concurrent["epoch_wall_seconds"][epoch],
            "speedup": sequential["epoch_wall_seconds"][epoch]
            / concurrent["epoch_wall_seconds"][epoch],
        }
        for epoch in range(1, 5)
    ]
    report = {
        "passed": True,
        "configuration": CONFIG.serializable(),
        "outer_study": 1,
        "epochs": 4,
        "fold_epoch_comparisons": len(sequential_folds),
        "exact_scientific_records": True,
        "exact_model_adamw_rng_checkpoints": True,
        "exact_aggregate_J_e": True,
        "sequential_wall_seconds": sequential["wall_seconds"],
        "concurrent_wall_seconds": concurrent["wall_seconds"],
        "speedup": speedup,
        "per_epoch": per_epoch,
        "median_epoch_speedup": statistics.median(row["speedup"] for row in per_epoch),
        "cache_warm_seconds_excluded": {
            "sequential": sequential_warm,
            "concurrent": concurrent_warm,
        },
        "pause_resume": {
            "passed": True,
            "trial_number": resume_study.trials[0].number,
            "final_state": resume_study.trials[0].state.name,
            "wandb_run_id": run_id,
            "fold_epoch_records": len(resume_folds),
            "aggregate_records": len(resume_aggregates),
            "no_duplicates": True,
        },
    }
    write_json(REPORT, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
