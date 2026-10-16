"""Bounded-memory, uniform cell sampling from frozen canonical partitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TARGET_ALIASES = {"BP": "bp", **{f"FLP{i}": f"flp{i}" for i in range(1, 7)}}
SOURCE_COLUMNS = (
    "cell_id", "x_5070", "y_5070", "BP", "flp_target_valid",
    *(f"FLP{i}" for i in range(1, 7)),
)
POPULATIONS = ("bp", "conditional", "unconditional")
EXPECTED = {"bp": 110_817_231, "conditional": 81_068_773, "unconditional": 110_804_580}


def validate_contract(repo: Path) -> list[Path]:
    """Fail before analysis if the frozen reports or partition schema differ."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"Canonical contract mismatch: {message}")

    report = repo / "reports/dataset/canonical"
    manifest = json.loads((report / "manifest.json").read_text())
    validation = json.loads((report / "validation_summary.json").read_text())
    schema = json.loads((report / "schema.json").read_text())
    counts = validation["target_counts"]
    grid = manifest["grid"]
    require(manifest["canonical_rows"] == EXPECTED["bp"] == counts["rows"], "row count")
    require(manifest["crs"] == "EPSG:5070", "CRS")
    require(grid["width"] == 17372 and grid["height"] == 11283, "grid dimensions")
    require(grid["transform"][:2] == [270.0, 0.0], "x transform")
    require(grid["transform"][3:5] == [0.0, -270.0], "y transform")
    require(grid["cell_area_m2"] == 72900.0, "cell area")
    require(counts["flp_target_valid"] == EXPECTED["conditional"], "FLP support")
    require(counts["bp_zero"] + counts["flp_target_valid"] == EXPECTED["unconditional"], "unconditional support")
    require(counts["bp_zero"] + counts["bp_positive"] == counts["rows"], "BP support partition")
    require(validation["hashes_match_manifest"] is True, "partition hashes")
    require(manifest["target_mask_rule"] ==
            "BP>0 and all FLPs GDAL-valid, finite, in [0,1], and abs(float64 sum-1)<=1e-6",
            "strict FLP mask rule")
    described = {c["name"]: c for c in schema["columns"]}
    require(set(SOURCE_COLUMNS) <= set(described), "required columns")
    require(all(described[name]["dtype"] == "float32" for name in TARGET_ALIASES), "target dtypes")
    require(described["flp_target_valid"]["dtype"] == "bool", "mask dtype")
    require(described["cell_id"]["dtype"] == "int64", "cell ID dtype")
    require(described["x_5070"]["aggregation_rule"] == "native pixel center", "x center")
    require(described["y_5070"]["aggregation_rule"] == "native pixel center", "y center")
    require(described["x_5070"]["dtype"] == described["y_5070"]["dtype"] == "float64", "coordinate dtype")
    files = sorted((repo / "artifacts/dataset/canonical/partitions").glob("*.parquet"))
    require(len(files) == manifest["partition_count"] == 3060, "partition count")
    require([p.name for p in files] == [p["name"] for p in manifest["partitions"]], "partition names")
    first = pq.ParquetFile(files[0]).schema_arrow.names
    require(set(SOURCE_COLUMNS) <= set(first), "Parquet columns")
    return files


def derive_targets(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derived fields preserve the exact raw FLPs and their support."""
    bp = data["bp"]
    valid = data["flp_target_valid"]
    conditional = data["flp5"] + data["flp6"]
    unconditional = np.full(len(bp), np.nan, dtype=np.result_type(bp.dtype, conditional.dtype))
    unconditional[bp == 0] = 0.0
    unconditional[valid] = bp[valid] * conditional[valid]
    return {"conditional_flep8": conditional, "unconditional_flep8": unconditional}


def support_masks(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    bp = data["bp"]
    valid = data["flp_target_valid"]
    return {
        "bp": np.isfinite(bp),
        "conditional": valid,
        "unconditional": (bp == 0) | valid,
    }


class PrioritySampler:
    """Retain the k smallest independent uniform priorities per population."""

    def __init__(self, k: int = 5000, seed: int = 42) -> None:
        self.k = k
        streams = np.random.SeedSequence(seed).spawn(len(POPULATIONS))
        self.rng = {name: np.random.default_rng(stream) for name, stream in zip(POPULATIONS, streams, strict=True)}
        self.selected: dict[str, dict[str, np.ndarray]] = {name: {} for name in POPULATIONS}
        self.counts = dict.fromkeys(POPULATIONS, 0)

    def add(self, data: dict[str, np.ndarray]) -> None:
        masks = support_masks(data)
        for name in POPULATIONS:
            indices = np.flatnonzero(masks[name])
            self.counts[name] += len(indices)
            if not len(indices):
                continue
            priority = self.rng[name].random(len(indices))
            if len(indices) > self.k:
                local = np.argpartition(priority, self.k - 1)[: self.k]
                indices, priority = indices[local], priority[local]
            candidate = {col: value[indices] for col, value in data.items()}
            candidate["priority"] = priority
            old = self.selected[name]
            if old:
                candidate = {col: np.concatenate((old[col], value)) for col, value in candidate.items()}
            keep = np.argsort(candidate["priority"], kind="stable")[: self.k]
            self.selected[name] = {col: value[keep] for col, value in candidate.items()}

    def result(self, name: str) -> dict[str, np.ndarray]:
        result = self.selected[name]
        order = np.argsort(result["cell_id"], kind="stable")
        return {col: value[order] for col, value in result.items() if col != "priority"}


def sample_partitions(files: list[Path], k: int = 5000, seed: int = 42) -> PrioritySampler:
    sampler = PrioritySampler(k, seed)
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65_536, columns=SOURCE_COLUMNS):
            data = {
                TARGET_ALIASES.get(col, col): batch.column(col).to_numpy(zero_copy_only=False)
                for col in SOURCE_COLUMNS
            }
            if not np.all(np.isfinite(data["bp"])):
                raise ValueError(f"Invalid BP in {path}")
            sampler.add(data)
    if k == 5000 and sampler.counts != EXPECTED:
        raise ValueError(f"Canonical support mismatch: {sampler.counts} != {EXPECTED}")
    return sampler
