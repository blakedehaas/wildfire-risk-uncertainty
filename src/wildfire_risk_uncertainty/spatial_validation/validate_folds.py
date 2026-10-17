"""Independent disk-level validation of Checkpoint 5B artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import distance_transform_edt

from .folds import (
    EXPECTED_ROWS,
    accepted_bp_radius,
    canonical_contract,
    column_alias_audit,
    nearest_centroid,
)

REQUIRED_COLUMNS = ["cell_id", "outer_fold", "oof_nearest_training_distance_m", *(f"train_eligible_fold_{j}" for j in range(1, 6))]
REPORT_TABLES = ("fold_centroids.csv", "fold_summary.csv", "target_summary_by_fold.csv", "nearest_training_distance_summary.csv", "separation_validation.csv", "artifact_hashes.csv")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(repo: Path) -> list[dict]:
    """Reconstruct support from canonical files and geometry from persisted folds."""
    files, source = canonical_contract(repo)
    radius = accepted_bp_radius(repo)
    report = repo / "reports/spatial_validation/05b"
    manifest = json.loads((report / "manifest.json").read_text())
    centroids = np.asarray(manifest["centroids_xy_5070_m"], dtype=np.float64)
    if manifest["radius_m"] != radius or centroids.shape != (5, 2):
        raise ValueError("Manifest radius or centroids invalid")
    with (report / "tables/artifact_hashes.csv").open(newline="") as handle:
        artifact_hashes = {r["partition"]: r["sha256"] for r in csv.DictReader(handle)}
    with (repo / "reports/dataset/canonical/partition_hashes.csv").open(newline="") as handle:
        source_hashes = {r["partition"]: r["sha256"] for r in csv.DictReader(handle)}
    if len(artifact_hashes) != len(files) or len(source_hashes) != len(files):
        raise ValueError("Partition hash manifest incomplete")
    shape = (source["grid"]["height"], source["grid"]["width"])
    support = np.zeros(shape, dtype=bool)
    fold = np.zeros(shape, dtype=np.uint8)
    flags = np.zeros(shape, dtype=np.uint8)
    distances = np.full(shape, np.nan, dtype=np.float64)
    count = 0
    for canonical in files:
        assigned = repo / "artifacts/spatial_validation/05b/partitions" / canonical.name
        if digest(canonical) != source_hashes[canonical.name] or digest(assigned) != artifact_hashes[canonical.name]:
            raise ValueError(f"Partition hash mismatch: {canonical.name}")
        p = pq.read_table(assigned)
        c = pq.read_table(canonical, columns=["cell_id", "x_5070", "y_5070"])
        if p.column_names != REQUIRED_COLUMNS or p.num_rows != c.num_rows:
            raise ValueError(f"Artifact schema/rows incorrect: {assigned}")
        ids = p["cell_id"].to_numpy()
        if not np.array_equal(ids, c["cell_id"].to_numpy()) or (len(ids) and np.any(np.diff(ids) <= 0)):
            raise ValueError(f"Artifact IDs/order incorrect: {assigned}")
        f = p["outer_fold"].to_numpy()
        xy = np.column_stack((c["x_5070"].to_numpy(), c["y_5070"].to_numpy()))
        if not np.array_equal(f, nearest_centroid(xy, centroids)):
            raise ValueError(f"Fold assignment is not nearest coordinate centroid: {assigned}")
        if np.any(support.ravel()[ids]):
            raise ValueError(f"Duplicate canonical IDs: {assigned}")
        support.ravel()[ids] = True
        fold.ravel()[ids] = f
        distances.ravel()[ids] = p["oof_nearest_training_distance_m"].to_numpy()
        for j in range(1, 6):
            flags.ravel()[ids] |= p[f"train_eligible_fold_{j}"].to_numpy().astype(np.uint8) << (j - 1)
        count += len(ids)
    rows = []

    def check(name: str, passed: bool) -> None:
        rows.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(f"Independent validation failed: {name}")

    check("canonical_unique_rows", count == EXPECTED_ROWS and int(support.sum()) == EXPECTED_ROWS)
    sample_path = repo / manifest["fit_sample_cell_ids_path"]
    sample_ids = pq.read_table(sample_path, columns=["cell_id"])["cell_id"].to_numpy()
    check("uniform_sample_ids_hash_match", digest(sample_path) == manifest["fit_sample_cell_ids_sha256"])
    check("uniform_sample_size_unique_canonical", len(sample_ids) == manifest["fit_sample_rows"] == 1_097_262
          and len(np.unique(sample_ids)) == len(sample_ids) and bool(np.all(support.ravel()[sample_ids])))
    alias = column_alias_audit(sample_ids)
    check("uniform_sample_no_column_alias", alias == manifest["fit_sample_column_alias_audit"]
          and alias["column_mod_101_nonempty_residues"] == 101
          and alias["column_mod_101_min_count"] > 0)
    check("one_test_fold_per_row", bool(np.all((fold[support] >= 1) & (fold[support] <= 5))))
    check("test_union_exactly_once", sum(int((fold == j).sum()) for j in range(1, 6)) == EXPECTED_ROWS)
    check("no_noncanonical_model_rows", bool(np.all(fold[~support] == 0) and np.all(flags[~support] == 0)))
    check("coordinate_only_nearest_centroid", True)
    check("all_oof_distances_finite", bool(np.isfinite(distances[support]).all()))
    for j in range(1, 6):
        is_test = support & (fold == j)
        to_test = distance_transform_edt(~is_test, sampling=270.0)
        is_train = support & ((flags & (1 << (j - 1))) != 0)
        is_dead = support & ~is_test & ~is_train
        check(f"fold_{j}_status_exhaustive_disjoint", int(is_test.sum()) + int(is_dead.sum()) + int(is_train.sum()) == EXPECTED_ROWS and not np.any(is_test & is_train))
        check(f"fold_{j}_train_strictly_above_radius", bool(np.all(to_test[is_train] > radius)))
        check(f"fold_{j}_dead_within_radius", bool(np.all(to_test[is_dead] <= radius)))
        check(f"fold_{j}_all_expected_train_included", bool(np.array_equal(is_train, support & ~is_test & (to_test > radius))))
        to_train = distance_transform_edt(~is_train, sampling=270.0)
        check(f"fold_{j}_oof_exact_nearest_train", bool(np.array_equal(to_train[is_test], distances[is_test])))
        check(f"fold_{j}_oof_strictly_above_radius", bool(np.all(distances[is_test] > radius)))
        del to_test, to_train
    check("canonical_sources_unchanged", True)
    check("artifact_hashes_match", True)
    check("report_schema_complete", all((report / "tables" / name).is_file() for name in REPORT_TABLES)
          and all((report / "figures" / name).is_file() for name in ("outer_folds.png", "nearest_training_distance_distribution.png", *(f"fold_{j}_status.png" for j in range(1, 6)))))
    with (report / "tables/separation_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "passed"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    rows = validate(args.repo.resolve())
    print(f"Passed {len(rows)} independent validation checks")


if __name__ == "__main__":
    main()
