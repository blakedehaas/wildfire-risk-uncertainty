"""Two persistent CUDA workers for epoch-synchronous inner-fold execution."""

import hashlib
import multiprocessing as mp
import os
import random
import signal
import traceback
from pathlib import Path
from typing import Self

import numpy as np
import torch

from wildfire_risk_uncertainty.hpo.config import Configuration
from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

FOLD_ASSIGNMENTS = ((1, 3, 5), (2, 4))
MAX_CUDA_WORKERS = 2


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _state_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(
                stable_json_bytes({"dtype": str(array.dtype), "shape": array.shape})
            )
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(
                stable_json_bytes({"dtype": str(array.dtype), "shape": array.shape})
            )
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            digest.update(stable_json_bytes(item))

    update(value)
    return digest.hexdigest()


def _checkpoint_path(root: Path, fold: int, epoch: int) -> Path:
    return root / f"fold_{fold}_epoch_{epoch:03d}.pt"


def _worker(
    root: Path,
    cache_path: Path,
    checkpoint_root: Path,
    outer_study: int,
    trial_number: int,
    configuration: Configuration,
    folds: tuple[int, ...],
    commands: mp.Queue,
    results: mp.Queue,
) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cache = HpoCache(root, cache_path, check_files=False)
        states = load_states(root, cache, outer_study)
        counts = support_counts(root, outer_study)
        models = {}
        rng = {}
        for fold in folds:
            seed_everything(42)
            models[fold] = make_model(
                inputs=len(states[fold].inputs),
                depth=configuration.depth,
                width=configuration.width,
                learning_rate=configuration.learning_rate,
                weight_decay=configuration.weight_decay,
            )
            rng[fold] = _rng_state()
        results.put({"status": "ready", "folds": folds})
        while True:
            message = commands.get()
            action = message["action"]
            if action == "close":
                results.put({"status": "closed"})
                return
            if action == "restore":
                restored = {}
                for fold in folds:
                    completed_epoch, global_step = message["folds"][fold]
                    _set_rng_state(rng[fold])
                    if completed_epoch:
                        model, optimizer = models[fold]
                        epoch, step = load_cache_checkpoint(
                            _checkpoint_path(checkpoint_root, fold, completed_epoch),
                            model,
                            optimizer,
                            cache=cache,
                            state=states[fold],
                            expected_metadata={
                                "outer_study": outer_study,
                                "trial_number": trial_number,
                                "inner_fold": fold,
                                "configuration": configuration.serializable(),
                                "seed": 42,
                            },
                        )
                        if epoch != completed_epoch or step != global_step:
                            raise ValueError(
                                "Checkpoint epoch/optimizer step differs from local fold record"
                            )
                    rng[fold] = _rng_state()
                    restored[fold] = (completed_epoch, global_step)
                results.put({"status": "restored", "folds": restored})
                continue
            if action == "epoch":
                epoch = message["epoch"]
                selected = set(message["folds"])
                records = {}
                for fold in folds:
                    if fold not in selected:
                        continue
                    _set_rng_state(rng[fold])
                    model, optimizer = models[fold]
                    training = train_epoch(
                        root,
                        states[fold],
                        model,
                        optimizer,
                        epoch,
                        outer_study=outer_study,
                        inner_fold=fold,
                        batch_size=configuration.batch_size,
                        cache=cache,
                    )
                    validation = validate(
                        root,
                        states[fold],
                        model,
                        counts[fold]["validation"]["rows"],
                        counts[fold]["validation"]["flp_valid"],
                        outer_study=outer_study,
                        inner_fold=fold,
                        batch_size=configuration.batch_size,
                        cache=cache,
                    )
                    rng[fold] = _rng_state()
                    records[fold] = (training, validation)
                results.put({"status": "epoch", "epoch": epoch, "records": records})
                continue
            if action == "save":
                saved = []
                for fold, (epoch, global_step) in message["folds"].items():
                    if fold not in models:
                        continue
                    _set_rng_state(rng[fold])
                    model, optimizer = models[fold]
                    save_cache_checkpoint(
                        _checkpoint_path(checkpoint_root, fold, epoch),
                        model,
                        optimizer,
                        completed_epoch=epoch,
                        global_step=global_step,
                        cache=cache,
                        state=states[fold],
                        metadata={
                            "outer_study": outer_study,
                            "trial_number": trial_number,
                            "inner_fold": fold,
                            "configuration": configuration.serializable(),
                            "seed": 42,
                        },
                    )
                    rng[fold] = _rng_state()
                    saved.append(fold)
                results.put({"status": "saved", "folds": saved})
                continue
            if action == "cleanup":
                for fold, epoch in message["folds"].items():
                    if fold not in models or epoch <= 2:
                        continue
                    previous = _checkpoint_path(checkpoint_root, fold, epoch - 2)
                    previous.unlink(missing_ok=True)
                    previous.with_suffix(previous.suffix + ".sha256.json").unlink(
                        missing_ok=True
                    )
                results.put({"status": "cleaned"})
                continue
            if action == "digests":
                values = {}
                for fold in folds:
                    model, optimizer = models[fold]
                    values[fold] = {
                        "model_sha256": _state_digest(model.state_dict()),
                        "optimizer_sha256": _state_digest(optimizer.state_dict()),
                        "rng_sha256": _state_digest(rng[fold]),
                    }
                results.put({"status": "digests", "folds": values})
                continue
            raise ValueError(f"Unknown fold-worker action: {action}")
    except BaseException as error:
        results.put(
            {
                "status": "error",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise


class PersistentFoldProvider:
    """Parent-side epoch barrier; workers own only fold training state."""

    def __init__(
        self,
        root: Path,
        outer_study: int,
        configuration: Configuration,
        *,
        cache_path: Path,
        trial_number: int,
        checkpoint_root: Path,
        retain_checkpoints: bool = False,
    ) -> None:
        self.root = root
        self.outer_study = outer_study
        self.configuration = configuration
        self.cache_path = cache_path
        self.trial_number = trial_number
        self.checkpoint_root = checkpoint_root
        self.retain_checkpoints = retain_checkpoints
        self._context = mp.get_context("spawn")
        self._workers: list[tuple[mp.Process, mp.Queue, mp.Queue]] = []
        self._closed = False

    def _start(self) -> None:
        if self._workers:
            return
        previous = {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        try:
            for name in previous:
                os.environ[name] = "1"
            for folds in FOLD_ASSIGNMENTS:
                commands = self._context.Queue()
                results = self._context.Queue()
                process = self._context.Process(
                    target=_worker,
                    args=(
                        self.root,
                        self.cache_path,
                        self.checkpoint_root,
                        self.outer_study,
                        self.trial_number,
                        self.configuration,
                        folds,
                        commands,
                        results,
                    ),
                )
                process.start()
                self._workers.append((process, commands, results))
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for _, _, results in self._workers:
            self._expect(results, "ready")
        if len(self._workers) != MAX_CUDA_WORKERS:
            raise RuntimeError(
                "Production fold-worker count differs from accepted concurrency"
            )

    @staticmethod
    def _expect(results: mp.Queue, expected: str) -> dict:
        message = results.get(timeout=900)
        if message.get("status") == "error":
            raise RuntimeError(
                f"Fold worker failed: {message['exception_type']}: "
                f"{message['exception_message']}\n{message['traceback']}"
            )
        if message.get("status") != expected:
            raise RuntimeError(f"Unexpected fold-worker response: {message}")
        return message

    def restore_many(self, completed: dict[int, int], steps: dict[int, int]) -> None:
        self._start()
        for _, commands, _ in self._workers:
            commands.put(
                {
                    "action": "restore",
                    "folds": {
                        fold: (completed[fold], steps[fold]) for fold in range(1, 6)
                    },
                }
            )
        for _, _, results in self._workers:
            self._expect(results, "restored")

    def run_epoch(self, epoch: int, folds: list[int]) -> dict[int, tuple[dict, dict]]:
        selected = sorted(folds)
        for _, commands, _ in self._workers:
            commands.put({"action": "epoch", "epoch": epoch, "folds": selected})
        combined = {}
        for _, _, results in self._workers:
            message = self._expect(results, "epoch")
            if message["epoch"] != epoch:
                raise RuntimeError("Fold worker returned a different epoch")
            combined.update(message["records"])
        if sorted(combined) != selected:
            raise RuntimeError(
                "Fold-worker epoch barrier returned incomplete/duplicate folds"
            )
        return combined

    def save_many(self, folds: dict[int, tuple[int, int]]) -> None:
        for _, commands, _ in self._workers:
            commands.put({"action": "save", "folds": folds})
        saved = []
        for _, _, results in self._workers:
            saved.extend(self._expect(results, "saved")["folds"])
        if sorted(saved) != sorted(folds):
            raise RuntimeError("Fold-worker checkpoint barrier was incomplete")

    def cleanup_many(self, folds: dict[int, int]) -> None:
        if self.retain_checkpoints:
            return
        for _, commands, _ in self._workers:
            commands.put({"action": "cleanup", "folds": folds})
        for _, _, results in self._workers:
            self._expect(results, "cleaned")

    def state_digests(self) -> dict[int, dict]:
        for _, commands, _ in self._workers:
            commands.put({"action": "digests"})
        combined = {}
        for _, _, results in self._workers:
            combined.update(self._expect(results, "digests")["folds"])
        return combined

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process, commands, _ in self._workers:
            if process.is_alive():
                commands.put({"action": "close"})
        for process, _, results in self._workers:
            if process.is_alive():
                self._expect(results, "closed")
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            if process.is_alive():
                raise RuntimeError("Fold worker did not stop")
            if process.exitcode not in (0, None):
                raise RuntimeError(f"Fold worker exited with code {process.exitcode}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
