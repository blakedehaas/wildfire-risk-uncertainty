"""Compare sequential and batch-provider ASHA reporting and decisions."""

import json
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import pruner, sampler, suggest
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.resume import run_resumable_trial
from wildfire_risk_uncertainty.hpo.synthetic import raw_metrics
from wildfire_risk_uncertainty.hpo.wandb_tracking import MockTracker

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/hpo/06f_production_pruning_validation"
REPORT = ROOT / "reports/training/06f_resource_optimization/pruning_equivalence.json"


class SequentialProvider:
    def __init__(self, penalty: float) -> None:
        self.penalty = penalty
        self.states = {}

    def restore(self, fold: int, completed_epoch: int, global_step: int) -> None:
        self.states[fold] = completed_epoch

    def __call__(self, fold: int, epoch: int) -> tuple[dict, dict]:
        self.states[fold] += 1
        value = 0.2 + self.penalty + fold / 100 + self.states[fold] / 1000
        return raw_metrics(value, 0.8 + value), raw_metrics(value + 0.1, 0.9 + value)

    def save(self, fold: int, epoch: int, global_step: int) -> None:
        pass

    def cleanup(self, fold: int, epoch: int) -> None:
        pass


class ConcurrentProvider:
    def __init__(self, penalty: float) -> None:
        self.penalty = penalty
        self.states = {}

    def restore_many(self, completed: dict[int, int], steps: dict[int, int]) -> None:
        self.states = dict(completed)

    def run_epoch(self, epoch: int, folds: list[int]) -> dict[int, tuple[dict, dict]]:
        result = {}
        for fold in folds:
            self.states[fold] += 1
            value = 0.2 + self.penalty + fold / 100 + self.states[fold] / 1000
            result[fold] = (
                raw_metrics(value, 0.8 + value),
                raw_metrics(value + 0.1, 0.9 + value),
            )
        return result

    def save_many(self, folds: dict[int, tuple[int, int]]) -> None:
        pass

    def cleanup_many(self, folds: dict[int, int]) -> None:
        pass


def run(mode: str) -> list[dict]:
    study = optuna.create_study(
        direction="minimize", sampler=sampler(), pruner=pruner()
    )
    record_root = ARTIFACT / mode

    def objective(trial: optuna.Trial) -> float:
        configuration = suggest(trial)
        provider = (
            SequentialProvider(trial.number * 0.5)
            if mode == "sequential"
            else ConcurrentProvider(trial.number * 0.5)
        )
        return run_resumable_trial(
            trial=trial,
            outer_study=1,
            configuration=configuration,
            provider=provider,
            record_root=record_root,
            study_name="pruning_fixture",
            provenance={"fixture": "production-batch-pruning-v1"},
            tracker=MockTracker(),
            max_epochs=4,
        )

    study.optimize(objective, n_trials=4, n_jobs=1)
    result = []
    for trial in study.trials:
        local = record_root / "pruning_fixture" / f"trial_{trial.number:03d}"
        aggregates = [
            read_json(path) for path in sorted(local.glob("aggregate_epoch_*.json"))
        ]
        result.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "reported_rungs": {
                    str(k): v for k, v in trial.intermediate_values.items()
                },
                "aggregate_J_e": {
                    str(row["resource_epoch"]): row["J_e"] for row in aggregates
                },
                "pruning_decisions": {
                    str(row["resource_epoch"]): row["pruning_decision"]
                    for row in aggregates
                    if row["resource_milestone"]
                },
            }
        )
    return result


def main() -> None:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("Pruning validation output already exists")
    sequential = run("sequential")
    concurrent = run("concurrent")
    if sequential != concurrent:
        raise ValueError("Concurrent rung reporting or pruning decisions differ")
    result = {
        "passed": True,
        "n_jobs": 1,
        "trials": len(sequential),
        "states": [trial["state"] for trial in sequential],
        "sequential": sequential,
        "concurrent": concurrent,
    }
    write_json(REPORT, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
