"""Canonical Parquet reader that exposes only frozen outer TRAIN rows."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EXTRA_INPUTS = (
    "wui_covered_fraction",
    "HUDEN2020__valid_area_fraction",
    "POPDEN2020__valid_area_fraction",
    "OCCHUDEN2020__valid_area_fraction",
    "VACHUDEN2020__valid_area_fraction",
    "VEG2019PC__valid_area_fraction",
    "AWATER20PC__valid_area_fraction",
    "wui_failure",
)


def model_inputs(schema_path: Path) -> tuple[str, ...]:
    schema = json.loads(schema_path.read_text())
    columns = schema["columns"]
    by_name = {column["name"]: column for column in columns}
    predictors = tuple(column["name"] for column in columns if column["role"] == "predictor")
    if len(predictors) != 25 or len(set(predictors)) != 25:
        raise ValueError("Canonical schema must have exactly 25 predictor columns")
    if any(by_name.get(name, {}).get("role") != "diagnostic" for name in EXTRA_INPUTS):
        raise ValueError("Frozen coverage/failure input is absent or has wrong schema role")
    if by_name["wui_failure"]["dtype"] != "bool":
        raise ValueError("wui_failure must remain a boolean")
    inputs = predictors + EXTRA_INPUTS
    if len(inputs) != 33 or len(set(inputs)) != 33:
        raise ValueError("Frozen input set must contain 33 distinct columns")
    return inputs


def median_train_fold(report: Path) -> tuple[int, int]:
    with report.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 5:
        raise ValueError("Expected five frozen outer-fold summaries")
    ordered = sorted((int(row["train_rows"]), int(row["fold"])) for row in records)
    if sorted(row[1] for row in ordered) != [1, 2, 3, 4, 5]:
        raise ValueError("Outer fold IDs must be 1..5")
    return ordered[2][1], ordered[2][0]


def training_batches(
    canonical_dir: Path,
    folds_dir: Path,
    fold: int,
    inputs: tuple[str, ...],
    parquet_batch_rows: int = 65536,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield features, BP, six FLPs, mask from TRAIN rows only in tile order.

    The fold file is read first, and a canonical tile is skipped entirely when
    it has no eligible rows. No outer TEST target or predictor is returned.
    """
    fold_column = f"train_eligible_fold_{fold}"
    for fold_path in sorted(folds_dir.glob("*.parquet")):
        fold_file = pq.ParquetFile(fold_path)
        if fold_file.metadata.num_rows == 0:
            continue
        canonical_path = canonical_dir / fold_path.name
        if not canonical_path.exists():
            raise FileNotFoundError(canonical_path)
        data_file = pq.ParquetFile(canonical_path)
        if data_file.metadata.num_rows != fold_file.metadata.num_rows:
            raise ValueError(f"Tile row mismatch: {fold_path.name}")
        fold_batches = fold_file.iter_batches(batch_size=parquet_batch_rows, columns=[fold_column])
        data_batches = data_file.iter_batches(
            batch_size=parquet_batch_rows, columns=[*inputs, "BP", *(f"FLP{k}" for k in range(1, 7)), "flp_target_valid"]
        )
        for fold_batch, data_batch in zip(fold_batches, data_batches, strict=True):
            eligible = fold_batch.column(0).to_numpy(zero_copy_only=False).astype(bool, copy=False)
            if not eligible.any():
                continue
            features = np.column_stack(
                [data_batch.column(i).to_numpy(zero_copy_only=False)[eligible] for i in range(len(inputs))]
            ).astype(np.float32, copy=False)
            bp = data_batch.column(len(inputs)).to_numpy(zero_copy_only=False)[eligible].astype(np.float32)
            flp = np.column_stack(
                [data_batch.column(len(inputs) + 1 + k).to_numpy(zero_copy_only=False)[eligible] for k in range(6)]
            ).astype(np.float32, copy=False)
            valid = data_batch.column(len(inputs) + 7).to_numpy(zero_copy_only=False)[eligible].astype(bool)
            yield features, bp, flp, valid


def inner_partition_paths(canonical_dir: Path, outer_dir: Path, inner_dir: Path) -> list[Path]:
    """Require exactly aligned canonical, frozen outer, and frozen inner tiles."""
    paths = sorted(canonical_dir.glob("*.parquet"))
    names = {path.name for path in paths}
    if len(paths) != 3060 or names != {p.name for p in outer_dir.glob("*.parquet")} or names != {
        p.name for p in inner_dir.glob("*.parquet")
    }:
        raise ValueError("Canonical, outer, and inner partition sets disagree")
    return paths


def inner_partition(
    canonical_path: Path,
    outer_dir: Path,
    inner_dir: Path,
    inputs: tuple[str, ...],
    *,
    outer_study: int = 3,
    inner_fold: int = 1,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one aligned tile and filter by frozen TRAIN eligibility before exposing targets."""
    if role not in {"train", "validation"} or outer_study not in range(1, 6) or inner_fold not in range(1, 6):
        raise ValueError("Inner reader requires an accepted outer/inner fold and train or validation role")
    outer_column = f"train_eligible_fold_{outer_study}"
    inner_column = f"inner_fold_outer_{outer_study}"
    outer = pq.read_table(outer_dir / canonical_path.name, columns=["cell_id", outer_column])
    inner = pq.read_table(inner_dir / canonical_path.name, columns=["cell_id", inner_column])
    ids = outer.column("cell_id").to_numpy(zero_copy_only=False)
    inner_ids = inner.column("cell_id").to_numpy(zero_copy_only=False)
    if not np.array_equal(ids, inner_ids):
        raise ValueError(f"Outer/inner cell IDs disagree in {canonical_path.name}")
    eligible = outer.column(outer_column).to_numpy(zero_copy_only=False).astype(bool)
    labels = inner.column(inner_column).to_numpy(zero_copy_only=False)
    if not np.array_equal(eligible, ~np.isnan(labels)):
        raise ValueError(f"Inner eligibility disagrees with frozen outer TRAIN in {canonical_path.name}")
    selected = eligible & (labels != inner_fold if role == "train" else labels == inner_fold)
    if not selected.any():
        return (np.empty((0, len(inputs)), np.float32), np.empty(0, np.float32),
                np.empty((0, 6), np.float32), np.empty(0, bool), ids[:0])
    columns = ["cell_id", *inputs, "BP", *(f"FLP{k}" for k in range(1, 7)), "flp_target_valid"]
    table = pq.read_table(canonical_path, columns=columns)
    canonical_ids = table.column("cell_id").to_numpy(zero_copy_only=False)
    if not np.array_equal(ids, canonical_ids):
        raise ValueError(f"Canonical/outer cell IDs disagree in {canonical_path.name}")
    features = np.column_stack([
        table.column(name).to_numpy(zero_copy_only=False)[selected] for name in inputs
    ]).astype(np.float32, copy=False)
    bp = table.column("BP").to_numpy(zero_copy_only=False)[selected].astype(np.float32)
    flp = np.column_stack([
        table.column(f"FLP{k}").to_numpy(zero_copy_only=False)[selected] for k in range(1, 7)
    ]).astype(np.float32, copy=False)
    valid = table.column("flp_target_valid").to_numpy(zero_copy_only=False)[selected].astype(bool)
    return features, bp, flp, valid, ids[selected]


def inner_chunks(
    canonical_dir: Path,
    outer_dir: Path,
    inner_dir: Path,
    inputs: tuple[str, ...],
    *,
    role: str,
    epoch: int | None = None,
    seed: int = 42,
    outer_study: int = 3,
    inner_fold: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Deterministic bounded-memory stream; epoch shuffles tiles and rows exactly once."""
    paths = inner_partition_paths(canonical_dir, outer_dir, inner_dir)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch])) if epoch is not None else None
    if rng is not None:
        rng.shuffle(paths)
    for path in paths:
        features, bp, flp, valid, _ids = inner_partition(
            path, outer_dir, inner_dir, inputs, role=role,
            outer_study=outer_study, inner_fold=inner_fold,
        )
        if not len(bp):
            continue
        if rng is not None:
            order = rng.permutation(len(bp))
            features, bp, flp, valid = features[order], bp[order], flp[order], valid[order]
        yield features, bp, flp, valid


def packed_batches(
    chunks: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pack a complete traversal into fixed batches with one final short batch."""
    pieces: list[list[np.ndarray]] = [[], [], [], []]
    buffered = 0
    for chunk in chunks:
        position = 0
        while position < len(chunk[0]):
            take = min(size - buffered, len(chunk[0]) - position)
            for item, column in zip(pieces, chunk, strict=True):
                item.append(column[position : position + take])
            buffered += take
            position += take
            if buffered == size:
                yield tuple(np.concatenate(part) for part in pieces)  # type: ignore[misc]
                pieces, buffered = [[], [], [], []], 0
    if buffered:
        yield tuple(np.concatenate(part) for part in pieces)  # type: ignore[misc]
