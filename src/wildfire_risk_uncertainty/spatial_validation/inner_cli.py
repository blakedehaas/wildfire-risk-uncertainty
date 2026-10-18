"""Build five nested coordinate-only inner spatial designs on frozen 5B TRAIN rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn

from .folds import EXPECTED_ROWS, N_FOLDS, SAMPLE_ROWS, SEED, canonical_contract
from .inner_folds import (
    aligned_tiles,
    digest,
    eligible_labels,
    fit_inner_centroids,
    inner_assignment_table,
    outer_train_counts,
    persist_sample,
    sample_eligible_coordinates,
)
from .outer_cli import write_csv


def new_bucket() -> dict:
    return {"rows": 0, "bp_zero": 0, "bp_positive": 0, "flp_valid": 0, "bp_sum": 0.0,
            "conditional_flep8_sum": 0.0, "unconditional_flep8_defined": 0,
            "unconditional_flep8_sum": 0.0,
            "x_min": math.inf, "x_max": -math.inf, "y_min": math.inf, "y_max": -math.inf}


def add_bucket(bucket: dict, mask: np.ndarray, xy: np.ndarray, bp: np.ndarray,
               flp5: np.ndarray, flp6: np.ndarray, valid: np.ndarray) -> None:
    n = int(mask.sum())
    if not n:
        return
    values = bp[mask]
    is_valid = valid[mask]
    is_zero = values == 0
    q = flp5[mask] + flp6[mask]
    defined = is_zero | is_valid
    positive_valid = (~is_zero) & is_valid
    unconditional = np.zeros(n, dtype=np.float64)
    unconditional[positive_valid] = values[positive_valid] * q[positive_valid]
    coords = xy[mask]
    bucket["rows"] += n
    bucket["bp_zero"] += int(is_zero.sum())
    bucket["bp_positive"] += int((~is_zero).sum())
    bucket["flp_valid"] += int(is_valid.sum())
    bucket["bp_sum"] += float(values.sum(dtype=np.float64))
    bucket["conditional_flep8_sum"] += float(q[is_valid].sum(dtype=np.float64))
    bucket["unconditional_flep8_defined"] += int(defined.sum())
    bucket["unconditional_flep8_sum"] += float(unconditional[defined].sum(dtype=np.float64))
    bucket["x_min"] = min(bucket["x_min"], float(coords[:, 0].min()))
    bucket["x_max"] = max(bucket["x_max"], float(coords[:, 0].max()))
    bucket["y_min"] = min(bucket["y_min"], float(coords[:, 1].min()))
    bucket["y_max"] = max(bucket["y_max"], float(coords[:, 1].max()))


def support_row(study: int, fold: int, role: str, bucket: dict) -> dict:
    rows = bucket["rows"]
    valid = bucket["flp_valid"]
    defined = bucket["unconditional_flep8_defined"]
    return {"outer_study": study, "inner_fold": fold, "role": role,
            "rows": rows, "bp_zero": bucket["bp_zero"], "bp_positive": bucket["bp_positive"],
            "flp_target_valid": valid, "bp_mean": bucket["bp_sum"] / rows if rows else None,
            "conditional_flep8_valid_rows": valid,
            "conditional_flep8_mean": bucket["conditional_flep8_sum"] / valid if valid else None,
            "unconditional_flep8_defined_rows": defined,
            "unconditional_flep8_mean": bucket["unconditional_flep8_sum"] / defined if defined else None}


def training_complement(total: dict, validation: dict) -> dict:
    result = new_bucket()
    for key in ("rows", "bp_zero", "bp_positive", "flp_valid", "bp_sum",
                "conditional_flep8_sum", "unconditional_flep8_defined", "unconditional_flep8_sum"):
        result[key] = total[key] - validation[key]
    return result


def plot_inner_maps(report: Path, maps: dict[int, np.ndarray]) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = ["#eeeeee", "#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#666666", "#bbbbbb"]
    cmap = ListedColormap(colors)
    for study, data in maps.items():
        fig, ax = plt.subplots(figsize=(13, 8))
        im = ax.imshow(data, interpolation="nearest", cmap=cmap,
                       norm=BoundaryNorm(np.arange(-0.5, 8.5), cmap.N))
        ax.set(title=f"Outer study {study}: five inner folds and excluded region (EPSG:5070)",
               xlabel="Raster column / 8", ylabel="Raster row / 8")
        fig.colorbar(im, ax=ax, ticks=range(8),
                     label="0 background; 1–5 inner folds; 6 outer TEST; 7 outer DEAD_ZONE")
        fig.tight_layout()
        fig.savefig(report / "figures" / f"outer_{study}_inner_folds.png", dpi=150)
        plt.close(fig)


def run(repo: Path) -> None:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("Run with PYTHONHASHSEED=42")
    start = time.monotonic()
    canonical_files, canonical_manifest = canonical_contract(repo)
    outer_manifest_path = repo / "reports/spatial_validation/05b/manifest.json"
    outer_manifest = json.loads(outer_manifest_path.read_text())
    if outer_manifest["row_count"] != EXPECTED_ROWS or outer_manifest["partition_count"] != len(canonical_files):
        raise ValueError("Frozen outer manifest does not match canonical support")
    outer_counts = outer_train_counts(repo / "reports/spatial_validation/05b/tables/fold_summary.csv")
    tiles = aligned_tiles(repo / "artifacts/dataset/canonical/partitions",
                          repo / "artifacts/spatial_validation/05b/partitions")
    expected_outer_hashes = {r["partition"]: r["sha256"] for r in outer_manifest["partition_hashes"]}
    report = repo / "reports/spatial_validation/06c"
    artifact = repo / "artifacts/spatial_validation/06c"
    (report / "tables").mkdir(parents=True, exist_ok=True)
    (report / "figures").mkdir(parents=True, exist_ok=True)
    (artifact / "partitions").mkdir(parents=True, exist_ok=True)
    centers_by_study: dict[int, np.ndarray] = {}
    sample_hashes: dict[int, str] = {}
    sample_rows: list[dict] = []
    for study in range(1, N_FOLDS + 1):
        ids, xy = sample_eligible_coordinates(tiles, study, outer_counts[study], SAMPLE_ROWS)
        sample_path = artifact / f"fitting_sample_outer_{study}.parquet"
        sample_hashes[study] = persist_sample(ids, sample_path)
        centers_by_study[study] = fit_inner_centroids(xy)
        sample_rows.append({"outer_study": study, "outer_train_rows": outer_counts[study],
                            "fit_sample_rows": len(ids), "fit_sample_sha256": sample_hashes[study]})
        print(f"outer {study}: sample {len(ids):,}, hash {sample_hashes[study]}", flush=True)
    width = canonical_manifest["grid"]["width"]
    height = canonical_manifest["grid"]["height"]
    factor = 8
    maps = {j: np.zeros((math.ceil(height / factor), math.ceil(width / factor)), dtype=np.uint8)
            for j in range(1, N_FOLDS + 1)}
    buckets = {(j, k): new_bucket() for j in range(1, 6) for k in range(1, 6)}
    artifact_rows: list[dict] = []
    seen_counts = {j: 0 for j in range(1, 6)}
    test_assigned = {j: 0 for j in range(1, 6)}
    dead_assigned = {j: 0 for j in range(1, 6)}
    label_errors = {j: 0 for j in range(1, 6)}
    total_rows = 0
    for canonical_path, outer_path in tiles:
        if digest(outer_path) != expected_outer_hashes[outer_path.name]:
            raise ValueError(f"Frozen 5B partition hash changed: {outer_path.name}")
        outer = pq.read_table(outer_path, columns=["cell_id", "outer_fold", *(f"train_eligible_fold_{j}" for j in range(1, 6))])
        canonical = pq.read_table(canonical_path, columns=["cell_id", "x_5070", "y_5070", "BP", "FLP5", "FLP6", "flp_target_valid"])
        ids = canonical["cell_id"].to_numpy()
        if len(ids) != len(outer) or not np.array_equal(ids, outer["cell_id"].to_numpy()):
            raise ValueError(f"Canonical/5B cell_id mismatch: {canonical_path.name}")
        xy = np.column_stack((canonical["x_5070"].to_numpy(), canonical["y_5070"].to_numpy()))
        bp = canonical["BP"].to_numpy()
        flp5 = canonical["FLP5"].to_numpy()
        flp6 = canonical["FLP6"].to_numpy()
        flp_valid = canonical["flp_target_valid"].to_numpy()
        outer_fold = outer["outer_fold"].to_numpy()
        labels_by_study: dict[int, np.ndarray] = {}
        raster_row, raster_col = ids // width, ids % width
        map_mask = ((raster_row % factor) == 0) & ((raster_col % factor) == 0)
        for study in range(1, N_FOLDS + 1):
            eligible = outer[f"train_eligible_fold_{study}"].to_numpy()
            labels = eligible_labels(xy, eligible, centers_by_study[study])
            labels_by_study[study] = labels
            seen_counts[study] += int(eligible.sum())
            test_assigned[study] += int(np.count_nonzero((outer_fold == study) & (labels != 0)))
            dead_assigned[study] += int(np.count_nonzero((outer_fold != study) & ~eligible & (labels != 0)))
            label_errors[study] += int(np.count_nonzero(eligible & ((labels < 1) | (labels > 5))))
            status = np.where(eligible, labels, np.where(outer_fold == study, 6, 7)).astype(np.uint8)
            maps[study][raster_row[map_mask] // factor, raster_col[map_mask] // factor] = status[map_mask]
            for fold in range(1, N_FOLDS + 1):
                add_bucket(buckets[(study, fold)], labels == fold, xy, bp, flp5, flp6, flp_valid)
        target = artifact / "partitions" / canonical_path.name
        pq.write_table(inner_assignment_table(ids, labels_by_study), target, compression="zstd")
        artifact_rows.append({"partition": target.name, "rows": len(ids),
                              "sha256": digest(target), "bytes": target.stat().st_size})
        total_rows += len(ids)
    if total_rows != EXPECTED_ROWS or len(artifact_rows) != 3060:
        raise ValueError("06C artifact does not cover all canonical cells and partitions")
    checks: list[dict] = []
    summaries: list[dict] = []
    supports: list[dict] = []
    centroid_rows: list[dict] = []
    for study in range(1, N_FOLDS + 1):
        total = new_bucket()
        for fold in range(1, N_FOLDS + 1):
            b = buckets[(study, fold)]
            for key in ("rows", "bp_zero", "bp_positive", "flp_valid", "bp_sum",
                        "conditional_flep8_sum", "unconditional_flep8_defined", "unconditional_flep8_sum"):
                total[key] += b[key]
        for check, passed in (
            ("eligible_count_matches_5b", seen_counts[study] == outer_counts[study]),
            ("five_validation_sets_exhaust_outer_train", total["rows"] == outer_counts[study]),
            ("outer_test_unassigned", test_assigned[study] == 0),
            ("outer_dead_zone_unassigned", dead_assigned[study] == 0),
            ("labels_only_1_to_5", label_errors[study] == 0),
        ):
            checks.append({"outer_study": study, "check": check, "passed": passed})
        for fold in range(1, N_FOLDS + 1):
            b = buckets[(study, fold)]
            train = training_complement(total, b)
            if min(b["rows"], train["rows"], b["bp_positive"], train["bp_positive"],
                   b["flp_valid"], train["flp_valid"]) <= 0:
                raise ValueError(f"Missing required inner support for outer {study}, fold {fold}")
            center = centers_by_study[study][fold - 1]
            centroid_rows.append({"outer_study": study, "inner_fold": fold,
                                  "x_5070": float(center[0]), "y_5070": float(center[1])})
            summaries.append({"outer_study": study, "inner_fold": fold,
                              "validation_rows": b["rows"], "validation_fraction": b["rows"] / total["rows"],
                              "inner_train_rows": train["rows"], "x_min": b["x_min"], "x_max": b["x_max"],
                              "y_min": b["y_min"], "y_max": b["y_max"]})
            supports.extend((support_row(study, fold, "validation", b),
                             support_row(study, fold, "train", train)))
    if not all(row["passed"] for row in checks):
        raise ValueError("Inner eligibility validation failed")
    plot_inner_maps(report, maps)
    tables = {"outer_study_summary.csv": sample_rows, "inner_fold_centroids.csv": centroid_rows,
              "inner_fold_summary.csv": summaries, "target_support_by_inner_fold.csv": supports,
              "eligibility_validation.csv": checks, "artifact_hashes.csv": artifact_rows}
    for name, rows in tables.items():
        write_csv(report / "tables" / name, rows, list(rows[0]))
    manifest = {
        "base_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "canonical_manifest_sha256": digest(repo / "reports/dataset/canonical/manifest.json"),
        "outer_5b_manifest_sha256": digest(outer_manifest_path),
        "outer_5b_source_commit": outer_manifest["source_commit"],
        "seed": SEED, "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "sample_algorithm": "random.Random(42).sample(range(eligible TRAIN row count), 1097262); exact uniform ranks without replacement in sorted tile and row order",
        "fit_sample_rows_per_study": SAMPLE_ROWS, "fit_sample_sha256_by_study": sample_hashes,
        "centroids_xy_5070_m_by_study": {j: centers_by_study[j].tolist() for j in range(1, 6)},
        "fold_label_convention": "ascending centroid x_5070 then y_5070; one-based",
        "clustering": "same 5B MiniBatchKMeans settings; Euclidean x_5070/y_5070 only; nearest-centroid assignment of all eligible rows",
        "outer_train_rows": outer_counts, "row_count": total_rows, "partition_count": len(artifact_rows),
        "schema": {"cell_id": "int64", **{f"inner_fold_outer_{j}": "nullable uint8" for j in range(1, 6)}},
        "partition_hashes": artifact_rows,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pyarrow": pa.__version__},
        "runtime_seconds": time.monotonic() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    for path in (report / "manifest.json", artifact / "manifest.json"):
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": total_rows, "partitions": len(artifact_rows),
                      "runtime_seconds": manifest["runtime_seconds"], "peak_rss_kb": manifest["peak_rss_kb"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.repo.resolve())


if __name__ == "__main__":
    main()
