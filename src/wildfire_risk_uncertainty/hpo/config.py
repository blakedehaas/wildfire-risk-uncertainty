"""Frozen HPO configuration and deterministic provenance identifiers."""

import hashlib
from dataclasses import asdict, dataclass

import optuna

from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes

SCHEMA_VERSION = "nested-spatial-hpo-v1"
TRAINING_VERSION = "checkpoint-6d-frozen-v1"
SEED = 42
MILESTONES = (1, 4, 16, 64)
MAX_EPOCHS = 100
SEARCH_SPACE = {
    "depth": {"type": "categorical", "choices": [2, 3]},
    "width": {"type": "categorical", "choices": [64, 128, 256]},
    "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [32768, 65536, 131072]},
}


def search_space_hash() -> str:
    return hashlib.sha256(stable_json_bytes(SEARCH_SPACE)).hexdigest()


@dataclass(frozen=True)
class Configuration:
    depth: int
    width: int
    learning_rate: float
    weight_decay: float
    batch_size: int

    def serializable(self) -> dict:
        return asdict(self)


FIXED_SMOKE = Configuration(2, 128, 1e-3, 1e-4, 65536)


def suggest(trial: optuna.Trial) -> Configuration:
    return Configuration(
        depth=trial.suggest_categorical("depth", SEARCH_SPACE["depth"]["choices"]),
        width=trial.suggest_categorical("width", SEARCH_SPACE["width"]["choices"]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]["choices"]),
    )


def sampler() -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(seed=SEED)


def pruner() -> optuna.pruners.SuccessiveHalvingPruner:
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=4, min_early_stopping_rate=0,
    )
