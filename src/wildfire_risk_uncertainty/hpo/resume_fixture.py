"""Cheap process-level synthetic fixture for RUNNING trial pause and replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import optuna

from wildfire_risk_uncertainty.hpo.config import FIXED_SMOKE, pruner, sampler
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.resume import run_resumable_trial
from wildfire_risk_uncertainty.hpo.synthetic import raw_metrics
from wildfire_risk_uncertainty.hpo.wandb_tracking import MockTracker
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes


class ToyProvider:
    def __init__(self, root: Path, *, interrupt_fold: int | None = None) -> None:
        self.root = root
        self.interrupt_fold = interrupt_fold
        self.states = {}

    def _path(self, fold: int, epoch: int) -> Path:
        return self.root / "checkpoints" / f"fold_{fold}_epoch_{epoch:03d}.json"

    def restore(self, fold: int, completed_epoch: int, global_step: int) -> None:
        if completed_epoch == 0:
            self.states[fold] = 0
            return
        path = self._path(fold, completed_epoch)
        state = read_json(path)
        if hashlib.sha256(stable_json_bytes(state)).hexdigest() != read_json(
            path.with_suffix(".sha256.json"))["sha256"]:
            raise ValueError("Synthetic checkpoint hash mismatch")
        if state["epoch"] != completed_epoch or state["step"] != global_step:
            raise ValueError("Synthetic checkpoint epoch/step mismatch")
        self.states[fold] = state["value"]

    def __call__(self, fold: int, epoch: int) -> tuple[dict, dict]:
        self.states[fold] += 1
        if self.interrupt_fold == fold and epoch == 2:
            os._exit(37)
        value = 0.2 + fold / 100 + self.states[fold] / 1000
        return raw_metrics(value, 0.8 + value), raw_metrics(value + 0.1, 0.9 + value)

    def save(self, fold: int, epoch: int, global_step: int) -> None:
        path = self._path(fold, epoch)
        state = {"fold": fold, "epoch": epoch, "step": global_step,
                 "value": self.states[fold]}
        write_json(path, state)
        write_json(path.with_suffix(".sha256.json"), {
            "sha256": hashlib.sha256(stable_json_bytes(state)).hexdigest()})

    def cleanup(self, fold: int, epoch: int) -> None:
        if epoch > 2:
            path = self._path(fold, epoch - 2)
            path.unlink(missing_ok=True)
            path.with_suffix(".sha256.json").unlink(missing_ok=True)


def phase(root: Path, action: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(root / 'study.sqlite3').resolve()}"
    study = optuna.create_study(study_name="restart_fixture", storage=storage,
                                direction="minimize", sampler=sampler(), pruner=pruner(),
                                load_if_exists=True)
    if not study.trials:
        trial = study.ask()
    else:
        if len(study.trials) != 1 or study.trials[0].state.name != "RUNNING":
            raise ValueError("Synthetic fixture must resume its one RUNNING trial")
        trial = optuna.Trial(study, study.trials[0]._trial_id)
    local = root / "records/restart_fixture/trial_000"
    tracker = MockTracker()
    result = run_resumable_trial(
        trial=trial, outer_study=1, configuration=FIXED_SMOKE,
        provider=ToyProvider(local, interrupt_fold=(3 if action == "interrupt" else
                                                  4 if action == "interrupt_after_three" else None)),
        record_root=root / "records", study_name="restart_fixture",
        provenance={"fixture": "synthetic-restart-v1"}, tracker=tracker,
        max_epochs=4, pause_after_epoch=1 if action == "pause" else None)
    if action == "pause":
        if result is not None or study.trials[0].state.name != "RUNNING":
            raise ValueError("Synthetic pause did not preserve RUNNING trial")
    elif action in {"interrupt", "interrupt_after_three"}:
        raise AssertionError("Synthetic interruption did not terminate the process")
    else:
        if result is None:
            raise ValueError("Synthetic trial did not finish")
        study.tell(trial, result)
        aggregate = [read_json(path) for path in sorted(local.glob("aggregate_epoch_*.json"))]
        write_json(root / "result.json", {
            "trial_number": study.trials[0].number, "state": study.trials[0].state.name,
            "wandb_run_id": read_json(local / "wandb_link.json")["run_id"],
            "J_by_epoch": [row["J_e"] for row in aggregate],
            "fold_epoch_record_count": len(list(local.glob("fold_*_epoch_*.json"))),
            "intermediate_values": {str(k): v for k, v in study.trials[0].intermediate_values.items()},
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("baseline", "pause", "interrupt",
                                           "interrupt_after_three", "resume"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    phase(args.root, args.action)
    print(json.dumps({"action": args.action, "trial_count": 1}))


if __name__ == "__main__":
    main()
