"""Coordinate-only, deterministic scalable outer-fold clustering."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans

SEED = 42
N_FOLDS = 5
EXPECTED_ROWS = 110_817_231
SAMPLE_ROWS = 1_097_262
EXPECTED_GRID = {"width": 17_372, "height": 11_283}
ACCEPTED_BP_KM = 211.15099027309995


def accepted_bp_radius(repo: Path) -> float:
    """Read the exact accepted 5A BP range; reject unresolved or altered findings."""
    report = repo / "reports/spatial_dependence/05a"
    with (report / "tables/variogram_selected_ranges.csv").open(newline="") as handle:
        bp = [r for r in csv.DictReader(handle) if r["target"] == "bp"]
    metadata = json.loads((report / "run_metadata.json").read_text())
    if len(bp) != 1 or bp[0]["status"] != "RESOLVED" or metadata["selected_status"].get("bp") != "RESOLVED":
        raise ValueError("Accepted Checkpoint 5A BP range is missing or unresolved")
    km = float(bp[0]["effective_range_km"])
    metres = float(bp[0]["effective_range_m"])
    if not np.isfinite(metres) or metres <= 0 or not np.isclose(metres / 1000, km, rtol=0, atol=1e-10) or not np.isclose(km, ACCEPTED_BP_KM, rtol=0, atol=1e-12):
        raise ValueError("BP range does not match accepted Checkpoint 5A result")
    return metres


def canonical_contract(repo: Path) -> tuple[list[Path], dict]:
    manifest = json.loads((repo / "reports/dataset/canonical/manifest.json").read_text())
    validation = json.loads((repo / "reports/dataset/canonical/validation_summary.json").read_text())
    grid = manifest["grid"]
    if (manifest["canonical_rows"] != EXPECTED_ROWS or validation["target_counts"]["rows"] != EXPECTED_ROWS
            or manifest["crs"] != "EPSG:5070" or grid["width"] != EXPECTED_GRID["width"]
            or grid["height"] != EXPECTED_GRID["height"] or grid["transform"][0] != 270
            or grid["transform"][4] != -270 or not validation["hashes_match_manifest"]):
        raise ValueError("Canonical committed report contract changed")
    files = sorted((repo / "artifacts/dataset/canonical/partitions").glob("*.parquet"))
    if len(files) != manifest["partition_count"]:
        raise ValueError("Canonical partition count differs from committed report")
    return files, manifest


def relabel_centroids(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order by x, then y; return ordered centers and old-to-new labels."""
    order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    mapping = np.empty(len(order), dtype=np.uint8)
    mapping[order] = np.arange(1, len(order) + 1, dtype=np.uint8)
    return centroids[order], mapping


def nearest_centroid(coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """One-based nearest Euclidean centroid, ties to lowest ordered fold."""
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Clustering accepts only x_5070 and y_5070")
    squared = ((xy[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(squared, axis=1).astype(np.uint8) + 1


def uniform_coordinate_sample(files: list[Path], sample_size: int = SAMPLE_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Draw exact uniform row positions without replacement, then stream selected coordinates.

    ``random.sample(range(N), k)`` uses bounded O(k) memory and samples the
    integer positions uniformly without replacement. Sorted partition and row
    order maps each position bijectively to one canonical row. Only matching
    partitions are read, one at a time; targets and predictors are never read.
    """
    counts = [pq.ParquetFile(path).metadata.num_rows for path in files]
    total = sum(counts)
    if not 0 < sample_size <= total:
        raise ValueError("Invalid fitting sample size")
    positions = np.sort(np.asarray(random.Random(SEED).sample(range(total), sample_size), dtype=np.int64))
    id_chunks = []
    xy_chunks = []
    offset = 0
    for path, count in zip(files, counts, strict=True):
        lo = int(np.searchsorted(positions, offset, side="left"))
        hi = int(np.searchsorted(positions, offset + count, side="left"))
        if hi > lo:
            table = pq.read_table(path, columns=["cell_id", "x_5070", "y_5070"])
            local = positions[lo:hi] - offset
            id_chunks.append(table["cell_id"].to_numpy()[local])
            xy_chunks.append(np.column_stack((table["x_5070"].to_numpy()[local], table["y_5070"].to_numpy()[local])))
        offset += count
    ids = np.concatenate(id_chunks)
    coords = np.concatenate(xy_chunks)
    if len(ids) != sample_size or len(np.unique(ids)) != sample_size:
        raise ValueError("Uniform fitting sample is incomplete or duplicated")
    return ids, coords


def column_alias_audit(ids: np.ndarray, width: int = 17_372) -> dict:
    """The old modulo-101 alias occupies one residue; corrected sample must span all."""
    residues = np.bincount((ids % width % 101).astype(np.int64), minlength=101)
    if np.count_nonzero(residues) != 101:
        raise ValueError("Fitting sample still aliases native raster columns")
    return {"column_mod_101_nonempty_residues": 101,
            "column_mod_101_min_count": int(residues.min()),
            "column_mod_101_max_count": int(residues.max()),
            "column_mod_101_counts": residues.tolist()}


def fit_centroids(files: list[Path], *, sample_size: int = SAMPLE_ROWS, sample_ids_path: Path | None = None) -> tuple[np.ndarray, int]:
    """Fit scalable Euclidean coordinate k-means on an exact uniform row sample."""
    ids, sample = uniform_coordinate_sample(files, sample_size)
    if sample_ids_path is not None:
        if sample_ids_path.exists():
            stored = pq.read_table(sample_ids_path, columns=["cell_id"])["cell_id"].to_numpy()
            if not np.array_equal(ids, stored):
                raise ValueError("Persisted fitting sample differs from deterministic regeneration")
        else:
            import pyarrow as pa

            sample_ids_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"cell_id": pa.array(ids, type=pa.int64())}), sample_ids_path, compression="zstd")
    rng = np.random.default_rng(SEED)
    sample = sample[rng.permutation(len(sample))]
    model = MiniBatchKMeans(n_clusters=N_FOLDS, random_state=SEED, batch_size=16_384,
                            n_init=10, max_iter=100, reassignment_ratio=0)
    model.fit(sample)
    ordered, _ = relabel_centroids(model.cluster_centers_)
    return ordered, len(sample)
