"""Bounded-memory training-only feature statistics and transformations."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

DENSITIES = ("HUDEN2020", "POPDEN2020", "OCCHUDEN2020", "VACHUDEN2020")


def stable_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class PreprocessingState:
    inputs: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    missing_counts: tuple[int, ...]
    rows: int
    flp_valid_rows: int
    schema_sha256: str

    def serializable(self) -> dict:
        return {
            "inputs": list(self.inputs),
            "means_after_fixed_transform": list(self.means),
            "standard_deviations_after_imputation": list(self.standard_deviations),
            "missing_counts": dict(zip(self.inputs, self.missing_counts, strict=True)),
            "rows": self.rows,
            "flp_valid_rows": self.flp_valid_rows,
            "schema_sha256": self.schema_sha256,
            "fixed_log1p_columns": list(DENSITIES),
            "binary_unstandardized_column": "wui_failure",
        }

    def digest(self) -> str:
        return hashlib.sha256(stable_json_bytes(self.serializable())).hexdigest()


def fixed_transform(features: np.ndarray, inputs: tuple[str, ...]) -> np.ndarray:
    result = np.array(features, dtype=np.float32, copy=True)
    for name in DENSITIES:
        index = inputs.index(name)
        observed = np.isfinite(result[:, index])
        if np.any(result[observed, index] < 0):
            raise ValueError(f"Negative value in nonnegative density {name}")
        result[observed, index] = np.log1p(result[observed, index])
    return result


def fit_preprocessing(
    batches: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    inputs: tuple[str, ...],
    schema_sha256: str,
) -> PreprocessingState:
    count = np.zeros(len(inputs), dtype=np.int64)
    mean = np.zeros(len(inputs), dtype=np.float64)
    m2 = np.zeros(len(inputs), dtype=np.float64)
    rows = flp_rows = 0
    for features, _bp, _flp, flp_valid in batches:
        values = fixed_transform(features, inputs)
        rows += len(values)
        flp_rows += int(flp_valid.sum())
        for j in range(len(inputs)):
            observed = values[np.isfinite(values[:, j]), j].astype(np.float64)
            n = len(observed)
            if not n:
                continue
            batch_mean = float(observed.mean())
            batch_m2 = float(((observed - batch_mean) ** 2).sum())
            old = int(count[j])
            new = old + n
            delta = batch_mean - mean[j]
            mean[j] += delta * n / new
            m2[j] += batch_m2 + delta * delta * old * n / new
            count[j] = new
    if rows == 0 or flp_rows == 0 or np.any(count == 0):
        raise ValueError("Empty training support or modeled feature with no observed training values")
    scale = np.sqrt(m2 / rows)  # Mean-imputed entries contribute zero squared deviation.
    scale[scale == 0] = 1.0
    scale[inputs.index("wui_failure")] = 1.0
    mean[inputs.index("wui_failure")] = 0.0
    return PreprocessingState(
        inputs, tuple(map(float, mean)), tuple(map(float, scale)),
        tuple(map(int, rows - count)), rows, flp_rows, schema_sha256,
    )


def transform(features: np.ndarray, state: PreprocessingState) -> np.ndarray:
    values = fixed_transform(features, state.inputs)
    for j, (mean, scale) in enumerate(zip(state.means, state.standard_deviations, strict=True)):
        missing = ~np.isfinite(values[:, j])
        if missing.any():
            values[missing, j] = mean
        if state.inputs[j] != "wui_failure":
            values[:, j] = (values[:, j] - mean) / scale
    return np.ascontiguousarray(values, dtype=np.float32)
