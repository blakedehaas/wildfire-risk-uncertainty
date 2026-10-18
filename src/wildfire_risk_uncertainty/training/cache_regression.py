"""Fixed 6D trajectory equivalence and bounded execution-path timing."""

import copy
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.data import inner_chunks, model_inputs
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    stable_json_bytes,
)
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)


def _state(root: Path, cache: HpoCache) -> PreprocessingState:
    report = json.loads((root / "reports/training/06f_prelaunch/equivalence_validation.json").read_text())
    if not report["preprocessing_identical"]:
        raise ValueError("Preprocessing equivalence has not passed")
    item = report["preprocessing_fits"]["parquet"]
    if report["preprocessing_fits"]["mmap"]["sha256"] != item["sha256"]:
        raise ValueError("Preprocessing hashes differ")
    saved = item["state"]
    if saved["inputs"] != cache.manifest["inputs"]:
        raise ValueError("Frozen cache feature order differs from fitted preprocessing")
    state = PreprocessingState(tuple(saved["inputs"]), tuple(saved["means_after_fixed_transform"]),
                               tuple(saved["standard_deviations_after_imputation"]),
                               tuple(saved["missing_counts"][name] for name in saved["inputs"]),
                               saved["rows"], saved["flp_valid_rows"], saved["schema_sha256"])
    if state.digest() != item["sha256"]:
        raise ValueError("Fitted preprocessing digest mismatch")
    if tuple(cache.manifest["inputs"]) != model_inputs(root / "reports/dataset/canonical/schema.json"):
        raise ValueError("Frozen 33-input contract differs")
    return state


def _equal_nested(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _equal_nested(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _raw_pass(root: Path, inputs: tuple[str, ...], cache: HpoCache | None, role: str) -> dict:
    started = time.perf_counter()
    chunks = inner_chunks(root / "artifacts/dataset/canonical/partitions",
                          root / "artifacts/spatial_validation/05b/partitions",
                          root / "artifacts/spatial_validation/06c/partitions",
                          inputs, role=role, epoch=1 if role == "train" else None,
                          outer_study=3, inner_fold=1, cache=cache)
    rows = flp_rows = 0
    digest = hashlib.sha256()
    for features, bp, flp, valid in chunks:
        rows += len(bp)
        flp_rows += int(valid.sum())
        digest.update(np.ascontiguousarray(features).tobytes())
        digest.update(np.ascontiguousarray(bp).tobytes())
        digest.update(np.ascontiguousarray(flp).tobytes())
        digest.update(np.ascontiguousarray(valid).tobytes())
    elapsed = time.perf_counter() - started
    return {"rows": rows, "flp_valid": flp_rows, "seconds": elapsed,
            "rows_per_second": rows / elapsed, "stream_sha256": digest.hexdigest()}


def _one_epoch(root: Path, state: PreprocessingState, cache: HpoCache | None) -> tuple[dict, dict, dict]:
    seed_everything(42)
    model, optimizer = make_model(inputs=33, depth=2, width=128,
                                  learning_rate=1e-3, weight_decay=1e-4)
    train = train_epoch(root, state, model, optimizer, 1, outer_study=3, inner_fold=1,
                        batch_size=65536, cache=cache)
    valid = validate(root, state, model, 18_476_910, 15_909_640, outer_study=3,
                     inner_fold=1, batch_size=65536, cache=cache)
    snapshot = {"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "optimizer": copy.deepcopy(optimizer.state_dict())}
    del model, optimizer
    return train, valid, snapshot


def run(root: Path) -> dict:
    cache = HpoCache(root, root / "artifacts/training/hpo_cache_v1")
    state = _state(root, cache)
    samples = []
    first = {}
    exact_keys = ("bp_loss_sum", "flp_loss_sum", "bp_rows", "flp_valid_rows",
                  "bp_loss", "flp_loss", "combined_loss", "optimizer_steps")
    for repetition in (1, 2):
        for name, reader in (("parquet", None), ("mmap", cache)):
            raw_train = _raw_pass(root, state.inputs, reader, "train")
            raw_valid = _raw_pass(root, state.inputs, reader, "validation")
            train, valid, snapshot = _one_epoch(root, state, reader)
            samples.append({"repetition": repetition, "source": name,
                            "raw_train": raw_train, "raw_validation": raw_valid,
                            "training": train, "validation": valid})
            if repetition == 1:
                first[name] = (train, valid, snapshot)
    old_train, old_valid, old_snapshot = first["parquet"]
    new_train, new_valid, new_snapshot = first["mmap"]
    for left, right in ((old_train, new_train), (old_valid, new_valid)):
        if any(left[key] != right[key] for key in exact_keys):
            raise ValueError("One-epoch Parquet/mmap loss or support trajectory differs")
    if not _equal_nested(old_snapshot, new_snapshot):
        raise ValueError("One-epoch Parquet/mmap model or optimizer state differs")
    for repetition in (1, 2):
        pair = [x for x in samples if x["repetition"] == repetition]
        if pair[0]["raw_train"]["stream_sha256"] != pair[1]["raw_train"]["stream_sha256"]:
            raise ValueError("Raw train stream differs")
        if pair[0]["raw_validation"]["stream_sha256"] != pair[1]["raw_validation"]["stream_sha256"]:
            raise ValueError("Raw validation stream differs")
    columns = ["repetition", "source", "phase", "rows", "seconds", "rows_per_second",
               "peak_allocated_vram_bytes", "peak_reserved_vram_bytes", "peak_process_rss_bytes"]
    report = root / "reports/training/06f_prelaunch"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "throughput_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            for phase in ("raw_train", "raw_validation", "training", "validation"):
                data = sample[phase]
                writer.writerow({"repetition": sample["repetition"], "source": sample["source"],
                                 "phase": phase, "rows": data.get("rows", data.get("bp_rows")),
                                 "seconds": data.get("seconds", data.get("wall_seconds")),
                                 "rows_per_second": data["rows_per_second"],
                                 "peak_allocated_vram_bytes": data.get("peak_vram_bytes"),
                                 "peak_reserved_vram_bytes": data.get("peak_reserved_vram_bytes"),
                                 "peak_process_rss_bytes": data.get("peak_rss_bytes")})
    medians = {name: {phase: statistics.median(s[phase].get("seconds", s[phase].get("wall_seconds"))
                                   for s in samples if s["source"] == name)
                      for phase in ("raw_train", "raw_validation", "training", "validation")}
               for name in ("parquet", "mmap")}
    old_total = medians["parquet"]["training"] + medians["parquet"]["validation"]
    new_total = medians["mmap"]["training"] + medians["mmap"]["validation"]
    result = {"fixed_case": {"outer_study": 3, "inner_fold": 1, "epoch": 1,
                              "depth": 2, "width": 128, "batch_size": 65536,
                              "learning_rate": 1e-3, "weight_decay": 1e-4,
                              "precision": "BF16", "seed": 42},
              "exact_support_and_loss_equality": True,
              "exact_model_parameter_equality": True,
              "exact_optimizer_state_equality": True,
              "medians_seconds": medians, "end_to_end_speedup": old_total / new_total,
              "repetitions": 2, "samples": samples,
              "cache_manifest_sha256": hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest()}
    (report / "training_regression.json").write_bytes(stable_json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps({key: value for key, value in run(Path.cwd()).items()
                      if key not in {"samples"}}, sort_keys=True))
