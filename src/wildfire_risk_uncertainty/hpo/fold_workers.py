"""Two persistent CUDA workers for epoch-synchronous inner-fold execution."""

import ctypes
import hashlib
import multiprocessing as mp
import os
import queue
import random
import signal
import time
import traceback
from pathlib import Path
from typing import Self

import numpy as np
import torch

from wildfire_risk_uncertainty.hpo.config import (
    GRADNORM_TRAINING_VERSION,
    Configuration,
)
from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

FOLD_ASSIGNMENTS = ((1, 3, 5), (2, 4))
MAX_CUDA_WORKERS = 2


class FoldWorkerError(RuntimeError):
    def __init__(self, message: dict) -> None:
        self.exception_type = message["exception_type"]
        super().__init__(
            f"Fold worker failed: {self.exception_type}: {message['exception_message']}\n"
            f"{message['traceback']}"
        )


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
    loss_balancing: str,
    parent_pid: int,
    folds: tuple[int, ...],
    commands: mp.Queue,
    results: mp.Queue,
) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)
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
        completed_steps = {fold: 0 for fold in folds}
        for fold in folds:
            seed_everything(42)
            model, optimizer = make_model(
                inputs=len(states[fold].inputs),
                depth=configuration.depth,
                width=configuration.width,
                learning_rate=configuration.learning_rate,
                weight_decay=configuration.weight_decay,
            )
            gradnorm = GradNorm("cuda") if loss_balancing == "gradnorm" else None
            models[fold] = (model, optimizer, gradnorm)
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
                        model, optimizer, gradnorm = models[fold]
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
                                **({"loss_balancing": "gradnorm", "gradnorm_alpha": 1.5,
                                    "gradnorm_training_version": GRADNORM_TRAINING_VERSION}
                                   if gradnorm is not None else {}),
                            },
                            gradnorm=gradnorm,
                        )
                        if epoch != completed_epoch or step != global_step:
                            raise ValueError(
                                "Checkpoint epoch/optimizer step differs from local fold record"
                            )
                    rng[fold] = _rng_state()
                    completed_steps[fold] = global_step
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
                    model, optimizer, gradnorm = models[fold]
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
                        gradnorm=gradnorm,
                        global_step_base=completed_steps[fold],
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
                        report_excess=gradnorm is not None,
                    )
                    rng[fold] = _rng_state()
                    completed_steps[fold] += training["optimizer_steps"]
                    records[fold] = (training, validation)
                results.put({"status": "epoch", "epoch": epoch, "records": records})
                continue
            if action == "save":
                saved = []
                for fold, (epoch, global_step) in message["folds"].items():
                    if fold not in models:
                        continue
                    _set_rng_state(rng[fold])
                    model, optimizer, gradnorm = models[fold]
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
                            **({"loss_balancing": "gradnorm", "gradnorm_alpha": 1.5,
                                "gradnorm_training_version": GRADNORM_TRAINING_VERSION}
                               if gradnorm is not None else {}),
                        },
                        gradnorm=gradnorm,
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
                    model, optimizer, gradnorm = models[fold]
                    values[fold] = {
                        "model_sha256": _state_digest(model.state_dict()),
                        "optimizer_sha256": _state_digest(optimizer.state_dict()),
                        "rng_sha256": _state_digest(rng[fold]),
                        **({"gradnorm_sha256": _state_digest(gradnorm.state_dict()),
                            "gradnorm_optimizer_sha256": _state_digest(gradnorm.optimizer.state_dict())}
                           if gradnorm is not None else {}),
                    }
                results.put({"status": "digests", "folds": values})
                continue
            raise ValueError(f"Unknown fold-worker action: {action}")
    except BaseException as error:
        results.put(
            {
                "status": "error",
                "exception_type": type(error).__name__,
                "exception_message": (f"{error}; trial_number={trial_number} "
                                      f"configuration={configuration.serializable()}"),
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
        loss_balancing: str = "equal",
    ) -> None:
        self.root = root
        self.outer_study = outer_study
        self.configuration = configuration
        self.cache_path = cache_path
        self.trial_number = trial_number
        self.checkpoint_root = checkpoint_root
        self.retain_checkpoints = retain_checkpoints
        if loss_balancing not in {"equal", "gradnorm"}:
            raise ValueError("Unknown loss-balancing method")
        self.loss_balancing = loss_balancing
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
                        self.loss_balancing,
                        os.getpid(),
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
        for process, _, results in self._workers:
            self._expect(results, "ready", process)
        if len(self._workers) != MAX_CUDA_WORKERS:
            raise RuntimeError(
                "Production fold-worker count differs from accepted concurrency"
            )

    @staticmethod
    def _expect(results: mp.Queue, expected: str, process: mp.Process) -> dict:
        deadline = time.monotonic() + 900
        while True:
            try:
                message = results.get(timeout=1)
                break
            except queue.Empty:
                if not process.is_alive():
                    raise RuntimeError(
                        f"Fold worker exited with code {process.exitcode} before {expected} response"
                    ) from None
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Fold worker did not return {expected} within 900 seconds") from None
        if message.get("status") == "error":
            raise FoldWorkerError(message)
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
        for process, _, results in self._workers:
            self._expect(results, "restored", process)

    def run_epoch(self, epoch: int, folds: list[int]) -> dict[int, tuple[dict, dict]]:
        selected = sorted(folds)
        for _, commands, _ in self._workers:
            commands.put({"action": "epoch", "epoch": epoch, "folds": selected})
        combined = {}
        for process, _, results in self._workers:
            message = self._expect(results, "epoch", process)
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
        for process, _, results in self._workers:
            saved.extend(self._expect(results, "saved", process)["folds"])
        if sorted(saved) != sorted(folds):
            raise RuntimeError("Fold-worker checkpoint barrier was incomplete")

    def cleanup_many(self, folds: dict[int, int]) -> None:
        if self.retain_checkpoints:
            return
        for _, commands, _ in self._workers:
            commands.put({"action": "cleanup", "folds": folds})
        for process, _, results in self._workers:
            self._expect(results, "cleaned", process)

    def state_digests(self) -> dict[int, dict]:
        for _, commands, _ in self._workers:
            commands.put({"action": "digests"})
        combined = {}
        for process, _, results in self._workers:
            combined.update(self._expect(results, "digests", process)["folds"])
        return combined

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if abort:
            for process, _, _ in self._workers:
                if process.is_alive():
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            for process, _, _ in self._workers:
                process.join(timeout=5)
            return
        for process, commands, _ in self._workers:
            if process.is_alive():
                commands.put({"action": "close"})
        for process, _, results in self._workers:
            if process.is_alive():
                self._expect(results, "closed", process)
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
