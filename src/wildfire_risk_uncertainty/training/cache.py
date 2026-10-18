"""Uncompressed row-aligned mmap cache for the frozen HPO data stream."""

import csv
import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wildfire_risk_uncertainty.training.data import inner_partition_paths, model_inputs
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes

CACHE_VERSION = "hpo-canonical-row-aligned-v1"
ROWS = 110_817_231
PARTITIONS = 3060
ARRAYS = {
    "cell_id": (np.int64, (ROWS,)),
    "features": (np.float32, (ROWS, 33)),
    "bp": (np.float32, (ROWS,)),
    "flp": (np.float32, (ROWS, 6)),
    "flp_valid": (np.uint8, (ROWS,)),
    "inner_fold": (np.uint8, (ROWS, 5)),
    "partition_offsets": (np.int64, (PARTITIONS + 1,)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {key: sha256(root / name) for key, name in {
        "canonical": "reports/dataset/canonical/manifest.json",
        "schema": "reports/dataset/canonical/schema.json",
        "outer_5b": "artifacts/spatial_validation/05b/manifest.json",
        "inner_6c": "artifacts/spatial_validation/06c/manifest.json",
    }.items()}


def expected_bytes(rows: int = ROWS, partitions: int = PARTITIONS) -> int:
    return rows * (8 + 33 * 4 + 4 + 6 * 4 + 1 + 5) + (partitions + 1) * 8 + 65536


def compare_streams(root: Path, cache: "HpoCache", *, outer_study: int = 3,
                    inner_fold: int = 1, role: str, epoch: int | None) -> dict:
    """Independently compare every selected row and its full traversal-order ID digest."""
    from wildfire_risk_uncertainty.training.data import inner_chunks

    args = (root / "artifacts/dataset/canonical/partitions",
            root / "artifacts/spatial_validation/05b/partitions",
            root / "artifacts/spatial_validation/06c/partitions", tuple(cache.manifest["inputs"]))
    old = inner_chunks(*args, role=role, epoch=epoch, outer_study=outer_study,
                       inner_fold=inner_fold, include_ids=True)
    new = inner_chunks(*args, role=role, epoch=epoch, outer_study=outer_study,
                       inner_fold=inner_fold, include_ids=True, cache=cache)
    old_digest, new_digest = hashlib.sha256(), hashlib.sha256()
    rows = chunks = 0
    for previous, optimized in zip(old, new, strict=True):
        for left, right in zip(previous, optimized, strict=True):
            np.testing.assert_array_equal(left, right, strict=True)
        old_digest.update(np.ascontiguousarray(previous[4]).tobytes())
        new_digest.update(np.ascontiguousarray(optimized[4]).tobytes())
        rows += len(previous[4])
        chunks += 1
    if old_digest.digest() != new_digest.digest():
        raise ValueError("Traversal-order cell IDs differ")
    return {"outer_study": outer_study, "inner_fold": inner_fold, "role": role,
            "epoch": epoch, "rows": rows, "chunks": chunks,
            "cell_id_sequence_sha256": old_digest.hexdigest(),
            "all_features_targets_masks_equal": True}


def _columns(table: object, names: tuple[str, ...], selected: np.ndarray | None = None) -> np.ndarray:
    values = [table.column(name).to_numpy(zero_copy_only=False) for name in names]
    if selected is not None:
        values = [value[selected] for value in values]
    return np.column_stack(values).astype(np.float32, copy=False)


def build(root: Path, target: Path) -> dict:
    """Build a complete cache in a sibling temporary directory, then rename."""
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    required = expected_bytes()
    if shutil.disk_usage(target.parent).free < 2 * required:
        raise OSError("Insufficient free storage for cache build and validation headroom")
    temporary = target.with_name(target.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"Incomplete prior build requires inspection: {temporary}")
    temporary.mkdir(parents=True)
    started = time.perf_counter()
    inputs = model_inputs(root / "reports/dataset/canonical/schema.json")
    canonical = root / "artifacts/dataset/canonical/partitions"
    outer = root / "artifacts/spatial_validation/05b/partitions"
    inner = root / "artifacts/spatial_validation/06c/partitions"
    paths = inner_partition_paths(canonical, outer, inner)
    arrays = {name: np.lib.format.open_memmap(temporary / f"{name}.npy", mode="w+",
                                               dtype=dtype, shape=shape)
              for name, (dtype, shape) in ARRAYS.items()}
    names = []
    counts = np.zeros(5, dtype=np.int64)
    offset = 0
    arrays["partition_offsets"][0] = 0
    target_columns = tuple(f"FLP{k}" for k in range(1, 7))
    for number, path in enumerate(paths, start=1):
        name = path.name
        data = pq.read_table(path, columns=["cell_id", *inputs, "BP", *target_columns,
                                                  "flp_target_valid"])
        outer_table = pq.read_table(outer / name, columns=["cell_id", *(
            f"train_eligible_fold_{j}" for j in range(1, 6))])
        inner_table = pq.read_table(inner / name, columns=["cell_id", *(
            f"inner_fold_outer_{j}" for j in range(1, 6))])
        ids = data.column("cell_id").to_numpy(zero_copy_only=False)
        outer_ids = outer_table.column("cell_id").to_numpy(zero_copy_only=False)
        inner_ids = inner_table.column("cell_id").to_numpy(zero_copy_only=False)
        if not np.array_equal(ids, outer_ids) or not np.array_equal(ids, inner_ids):
            raise ValueError(f"Canonical/5B/6C cell IDs disagree in {name}")
        size = len(ids)
        stop = offset + size
        if stop > ROWS:
            raise ValueError("Cache exceeds canonical row count")
        arrays["cell_id"][offset:stop] = ids
        arrays["features"][offset:stop] = _columns(data, inputs)
        arrays["bp"][offset:stop] = data.column("BP").to_numpy(
            zero_copy_only=False).astype(np.float32, copy=False)
        arrays["flp"][offset:stop] = _columns(data, target_columns)
        arrays["flp_valid"][offset:stop] = data.column("flp_target_valid").to_numpy(
            zero_copy_only=False).astype(np.uint8)
        for j in range(1, 6):
            eligibility = outer_table.column(f"train_eligible_fold_{j}").to_numpy(
                zero_copy_only=False).astype(bool, copy=False)
            labels = inner_table.column(f"inner_fold_outer_{j}").to_numpy(zero_copy_only=False)
            if not np.array_equal(eligibility, ~np.isnan(labels)):
                raise ValueError(f"5B/6C eligibility mismatch in {name}, outer {j}")
            numeric = np.nan_to_num(labels, nan=0).astype(np.uint8)
            if np.any(numeric > 5) or np.any((numeric > 0) != eligibility):
                raise ValueError(f"Invalid inner labels in {name}, outer {j}")
            arrays["inner_fold"][offset:stop, j - 1] = numeric
            counts[j - 1] += int(eligibility.sum())
        offset = stop
        arrays["partition_offsets"][number] = stop
        names.append(name)
    if offset != ROWS or len(names) != PARTITIONS or len(set(names)) != PARTITIONS:
        raise ValueError("Cache row or partition count differs from canonical provenance")
    with (root / "reports/spatial_validation/05b/tables/fold_summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    if sorted(int(row["fold"]) for row in summary) != [1, 2, 3, 4, 5]:
        raise ValueError("Frozen 5B report does not contain five outer folds")
    expected_counts = [int(row["train_rows"]) for row in sorted(summary, key=lambda row: int(row["fold"]))]
    if counts.tolist() != expected_counts:
        raise ValueError("Cache eligibility counts differ from frozen 5B")
    ids = np.load(temporary / "cell_id.npy", mmap_mode="r")
    if len(np.unique(ids)) != ROWS:
        raise ValueError("Cache canonical cell IDs are not unique")
    for array in arrays.values():
        array.flush()
    del arrays
    (temporary / "partition_names.json").write_bytes(stable_json_bytes(names))
    files = [*sorted(temporary.glob("*.npy")), temporary / "partition_names.json"]
    manifest = {"version": CACHE_VERSION, "rows": ROWS, "partitions": PARTITIONS,
                "inputs": list(inputs), "outer_train_counts": counts.tolist(),
                "source_hashes": source_hashes(root),
                "files_sha256": {path.name: sha256(path) for path in files},
                "size_bytes": sum(path.stat().st_size for path in files),
                "build_seconds": time.perf_counter() - started}
    (temporary / "manifest.json").write_bytes(stable_json_bytes(manifest))
    os.rename(temporary, target)
    return manifest


class HpoCache:
    """Read-only mmap view; all model inputs retain the frozen Parquet representation."""

    def __init__(self, root: Path, path: Path, *, check_files: bool = True) -> None:
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        if self.manifest["version"] != CACHE_VERSION or self.manifest["source_hashes"] != source_hashes(root):
            raise ValueError("Cache version/source provenance mismatch")
        inputs = model_inputs(root / "reports/dataset/canonical/schema.json")
        if self.manifest["inputs"] != list(inputs):
            raise ValueError("Cache feature order mismatch")
        if check_files:
            for name, digest in self.manifest["files_sha256"].items():
                if sha256(path / name) != digest:
                    raise ValueError(f"Cache file hash mismatch: {name}")
        self.names = json.loads((path / "partition_names.json").read_text())
        self.arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
        if len(self.names) != PARTITIONS or len(self.arrays["cell_id"]) != ROWS:
            raise ValueError("Cache row/partition dimensions mismatch")
        offsets = self.arrays["partition_offsets"]
        if offsets[0] != 0 or offsets[-1] != ROWS or np.any(np.diff(offsets) < 0):
            raise ValueError("Cache partition offsets invalid")

    def chunks(self, *, outer_study: int, inner_fold: int, role: str,
               epoch: int | None, seed: int = 42,
               include_ids: bool = False) -> Iterator[tuple[np.ndarray, ...]]:
        if outer_study not in range(1, 6) or inner_fold not in range(1, 6):
            raise ValueError("Fold ID must be 1..5")
        if role not in {"train", "validation"}:
            raise ValueError("Role must be train or validation")
        order = list(range(len(self.names)))
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch])) if epoch is not None else None
        if rng is not None:
            rng.shuffle(order)
        offsets = self.arrays["partition_offsets"]
        for number in order:
            start, stop = map(int, offsets[number:number + 2])
            labels = self.arrays["inner_fold"][start:stop, outer_study - 1]
            selected = (labels != 0) & (labels != inner_fold) if role == "train" else labels == inner_fold
            if not np.any(selected):
                continue
            rows = np.flatnonzero(selected) + start
            if rng is not None:
                rows = rows[rng.permutation(len(rows))]
            result = tuple(np.asarray(self.arrays[name][rows]) for name in (
                "features", "bp", "flp", "flp_valid"))
            result = (result[0], result[1], result[2], result[3].astype(bool))
            yield (*result, np.asarray(self.arrays["cell_id"][rows])) if include_ids else result
