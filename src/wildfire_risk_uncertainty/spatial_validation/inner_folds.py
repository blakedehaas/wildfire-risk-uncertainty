"""Coordinate-only inner-fold primitives using the frozen 5B fitting machinery."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .folds import (
    N_FOLDS,
    SAMPLE_ROWS,
    SEED,
    fit_coordinate_sample,
    nearest_centroid,
    uniform_positions,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def outer_train_counts(report: Path) -> dict[int, int]:
    with report.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {int(row["fold"]): int(row["train_rows"]) for row in rows}
    if len(rows) != N_FOLDS or set(result) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Accepted 5B summary must have exactly five outer TRAIN counts")
    return result


def aligned_tiles(canonical: Path, outer: Path, expected_partitions: int = 3060) -> list[tuple[Path, Path]]:
    canonical_files = sorted(canonical.glob("*.parquet"))
    outer_files = sorted(outer.glob("*.parquet"))
    if len(canonical_files) != expected_partitions or [p.name for p in canonical_files] != [p.name for p in outer_files]:
        raise ValueError("Canonical and 5B tile names/counts do not align")
    return list(zip(canonical_files, outer_files, strict=True))


def sample_eligible_coordinates(
    tiles: list[tuple[Path, Path]], outer_study: int, eligible_total: int,
    sample_size: int = SAMPLE_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample ranks in the eligible TRAIN stream, then map to IDs/xy.

    This is the 5B ``random.Random(42).sample(range(N), k)`` rule applied to
    the sorted, filtered outer-TRAIN population. Each eligible row has one
    rank, hence exactly equal inclusion probability without replacement.
    """
    positions = uniform_positions(eligible_total, sample_size, SEED)
    ids_parts: list[np.ndarray] = []
    xy_parts: list[np.ndarray] = []
    offset = 0
    for canonical_path, outer_path in tiles:
        flags = pq.read_table(outer_path, columns=[f"train_eligible_fold_{outer_study}"])[0].to_numpy()
        count = int(flags.sum())
        lo = int(np.searchsorted(positions, offset))
        hi = int(np.searchsorted(positions, offset + count))
        if hi > lo:
            local = np.flatnonzero(flags)[positions[lo:hi] - offset]
            table = pq.read_table(canonical_path, columns=["cell_id", "x_5070", "y_5070"])
            ids_parts.append(table["cell_id"].to_numpy()[local])
            xy_parts.append(np.column_stack((table["x_5070"].to_numpy()[local],
                                             table["y_5070"].to_numpy()[local])))
        offset += count
    if offset != eligible_total:
        raise ValueError(f"Outer {outer_study}: eligible count differs from accepted 5B report")
    ids = np.concatenate(ids_parts)
    xy = np.concatenate(xy_parts)
    if len(ids) != sample_size or len(np.unique(ids)) != sample_size or xy.shape != (sample_size, 2):
        raise ValueError("Eligible fitting sample incomplete or duplicated")
    return ids, xy


def persist_sample(ids: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"cell_id": pa.array(ids, type=pa.int64())}), path, compression="zstd")
    return digest(path)


def fit_inner_centroids(xy: np.ndarray) -> np.ndarray:
    centers = fit_coordinate_sample(xy)
    if centers.shape != (N_FOLDS, 2) or not np.isfinite(centers).all():
        raise ValueError("Five finite inner centroids required")
    return centers


def eligible_labels(xy: np.ndarray, eligible: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return 0 for ineligible rows and one-based nearest-centroid labels."""
    labels = np.zeros(len(eligible), dtype=np.uint8)
    labels[eligible] = nearest_centroid(xy[eligible], centers)
    return labels


def inner_assignment_table(ids: np.ndarray, labels_by_study: dict[int, np.ndarray]) -> pa.Table:
    columns: dict[str, pa.Array] = {"cell_id": pa.array(ids, type=pa.int64())}
    for study in range(1, N_FOLDS + 1):
        labels = labels_by_study[study]
        if len(labels) != len(ids) or np.any(labels > N_FOLDS):
            raise ValueError("Inner label length or range invalid")
        columns[f"inner_fold_outer_{study}"] = pa.array(labels, mask=labels == 0, type=pa.uint8())
    return pa.table(columns)
