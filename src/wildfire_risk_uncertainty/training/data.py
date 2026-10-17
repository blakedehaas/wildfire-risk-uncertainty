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
