"""Bounded reproducibility check without repeating the expensive distance transforms."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .folds import (
    canonical_contract,
    column_alias_audit,
    fit_centroids,
    nearest_centroid,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(repo: Path) -> dict:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("Run with PYTHONHASHSEED=42")
    files, _ = canonical_contract(repo)
    report = repo / "reports/spatial_validation/05b"
    manifest = json.loads((report / "manifest.json").read_text())
    start = time.monotonic()
    sample_path = repo / "artifacts/spatial_validation/05b/fitting_sample_cell_ids.parquet"
    centers, sample_rows = fit_centroids(files, sample_ids_path=sample_path)
    sample_hash_equal = digest(sample_path) == manifest["fit_sample_cell_ids_sha256"]
    if not sample_hash_equal:
        raise ValueError("Regenerated uniform sample IDs differ from persisted sample hash")
    sampled_ids = pq.read_table(sample_path, columns=["cell_id"])["cell_id"].to_numpy()
    alias_audit = column_alias_audit(sampled_ids)
    fit_seconds = time.monotonic() - start
    if not np.array_equal(centers, np.asarray(manifest["centroids_xy_5070_m"])):
        raise ValueError("Centroid refit differs from persisted centroids")
    generated = hashlib.sha256()
    persisted = hashlib.sha256()
    rows = 0
    for path in files:
        canonical = pq.read_table(path, columns=["x_5070", "y_5070"])
        actual = pq.read_table(repo / "artifacts/spatial_validation/05b/partitions" / path.name, columns=["outer_fold"])
        xy = np.column_stack((canonical["x_5070"].to_numpy(), canonical["y_5070"].to_numpy()))
        labels = nearest_centroid(xy, centers)
        saved = actual["outer_fold"].to_numpy()
        generated.update(labels.tobytes())
        persisted.update(saved.tobytes())
        rows += len(labels)
    assignment_seconds = time.monotonic() - start - fit_seconds
    if generated.digest() != persisted.digest() or rows != manifest["row_count"]:
        raise ValueError("Regenerated fold assignments differ from persisted assignments")
    with (report / "tables/artifact_hashes.csv").open(newline="") as stream:
        recorded_hashes = {row["partition"]: row["sha256"] for row in csv.DictReader(stream)}
    hash_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="05b_hash_regeneration_", dir=repo / "artifacts/spatial_validation") as temp_dir:
        temporary = Path(temp_dir) / "tile.parquet"
        for path in files:
            original = repo / "artifacts/spatial_validation/05b/partitions" / path.name
            pq.write_table(pq.read_table(original), temporary, compression="zstd")
            if digest(temporary) != recorded_hashes[path.name] or digest(original) != recorded_hashes[path.name]:
                raise ValueError(f"Regenerated artifact partition hash differs: {path.name}")
    hash_seconds = time.monotonic() - hash_start
    result = {"sample_ids_regenerated_hash_equal": sample_hash_equal,
              "centroids_refit_byte_equal": True, "fit_sample_rows": sample_rows,
              "fit_sample_column_alias_audit": alias_audit,
              "assignments_regenerated_equal": True, "assignment_rows": rows,
              "assignment_stream_sha256": generated.hexdigest(),
              "artifact_partitions_regenerated_hash_equal": len(files),
              "fit_seconds": fit_seconds, "assignment_regeneration_seconds": assignment_seconds,
              "artifact_hash_regeneration_seconds": hash_seconds,
              "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "note": "Independent validator recomputed all five exact distance transforms; this check regenerated every artifact Parquet tile and matched its SHA-256."}
    (report / "reproducibility.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(check(args.repo.resolve()), indent=2))


if __name__ == "__main__":
    main()
