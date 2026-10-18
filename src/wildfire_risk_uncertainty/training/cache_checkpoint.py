"""Strict mmap-cache identity for future active HPO checkpoint resumes."""

import hashlib
from pathlib import Path

import torch

from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    stable_json_bytes,
)
from wildfire_risk_uncertainty.training.trainer import DATA_ORDER_VERSION


def cache_binding(cache: HpoCache, state: PreprocessingState) -> dict:
    """Bind a trajectory to the exact immutable data and fold preprocessing."""
    if list(state.inputs) != cache.manifest["inputs"]:
        raise ValueError("Checkpoint feature order differs from cache")
    return {"cache_version": cache.manifest["version"],
            "cache_manifest_sha256": hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest(),
            "partition_offsets_sha256": cache.manifest["files_sha256"]["partition_offsets.npy"],
            "partition_names_sha256": cache.manifest["files_sha256"]["partition_names.json"],
            "feature_order_sha256": hashlib.sha256(stable_json_bytes(cache.manifest["inputs"])).hexdigest(),
            "preprocessing_sha256": state.digest(),
            "data_order_version": DATA_ORDER_VERSION}


def save_cache_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                          *, completed_epoch: int, global_step: int, cache: HpoCache,
                          state: PreprocessingState, metadata: dict) -> None:
    save_checkpoint(path, model, optimizer, completed_epoch=completed_epoch,
                    global_step=global_step,
                    metadata={**metadata, "cache_binding": cache_binding(cache, state)})
    write_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": sha256(path)})


def load_cache_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                          *, cache: HpoCache, state: PreprocessingState,
                          expected_metadata: dict) -> tuple[int, int]:
    checksum = read_json(path.with_suffix(path.suffix + ".sha256.json"))["sha256"]
    if sha256(path) != checksum:
        raise ValueError("Checkpoint file hash mismatch")
    return load_checkpoint(path, model, optimizer,
                           expected_metadata={**expected_metadata,
                                              "cache_binding": cache_binding(cache, state)})
