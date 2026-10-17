"""Build the frozen, model-free Checkpoint 5B outer spatial CV design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
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

from .dead_zone import fold_geometry
from .folds import (
    EXPECTED_ROWS,
    N_FOLDS,
    SEED,
    accepted_bp_radius,
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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[np.ndarray]) -> dict:
    if not values:
        return {name: None for name in ("p5", "p25", "median", "p75", "p95")}
    v = np.concatenate(values)
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(zip(("p5", "p25", "median", "p75", "p95"), map(float, q), strict=True))


def stats_bucket() -> dict:
    return {"rows": 0, "bp_zero": 0, "bp_positive": 0, "flp_target_valid": 0,
            "bp_sum": 0.0, "bp_sample": [], "conditional_sum": 0.0, "conditional_sample": [],
            "unconditional_defined": 0, "unconditional_sum": 0.0, "unconditional_sample": [],
            "wui_covered": 0, "wui_coverage_sum": 0.0, "wui_failure": 0}


def add_stats(bucket: dict, mask: np.ndarray, ids: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    n = int(np.count_nonzero(mask))
    if not n:
        return
    bp = columns["BP"][mask]
    valid = columns["flp_target_valid"][mask]
    zero = bp == 0
    covered = columns["wui_covered_fraction"][mask]
    finite = np.isfinite(covered)
    conditional = (columns["FLP5"][mask] + columns["FLP6"][mask])[valid]
    defined = zero | valid
    unconditional = np.zeros(n, dtype=np.float32)
    positive_valid = (~zero) & valid
    unconditional[positive_valid] = bp[positive_valid] * (columns["FLP5"][mask][positive_valid] + columns["FLP6"][mask][positive_valid])
    bucket["rows"] += n
    bucket["bp_zero"] += int(zero.sum())
    bucket["bp_positive"] += int((~zero).sum())
    bucket["flp_target_valid"] += int(valid.sum())
    bucket["bp_sum"] += float(bp.sum(dtype=np.float64))
    bucket["conditional_sum"] += float(conditional.sum(dtype=np.float64))
    bucket["unconditional_defined"] += int(defined.sum())
    bucket["unconditional_sum"] += float(unconditional[defined].sum(dtype=np.float64))
    bucket["wui_covered"] += int(finite.sum())
    bucket["wui_coverage_sum"] += float(covered[finite].sum(dtype=np.float64))
    bucket["wui_failure"] += int(columns["wui_failure"][mask].sum())
    sampled = ids[mask] % 101 == 0
    if sampled.any():
        bucket["bp_sample"].append(bp[sampled])
        bucket["conditional_sample"].append(columns["FLP5"][mask][sampled & valid] + columns["FLP6"][mask][sampled & valid])
        bucket["unconditional_sample"].append(unconditional[sampled & defined])


def finish_stats(fold: int, role: str, bucket: dict) -> dict:
    b = bucket
    return {"fold": fold, "role": role, "rows": b["rows"], "bp_zero": b["bp_zero"],
            "bp_positive": b["bp_positive"], "flp_target_valid": b["flp_target_valid"],
            "bp_mean": b["bp_sum"] / b["rows"] if b["rows"] else None,
            **{f"bp_sample_estimated_{k}": v for k, v in quantiles(b["bp_sample"]).items()},
            "conditional_flep8_mean": b["conditional_sum"] / b["flp_target_valid"] if b["flp_target_valid"] else None,
            **{f"conditional_flep8_sample_estimated_{k}": v for k, v in quantiles(b["conditional_sample"]).items()},
            "unconditional_flep8_defined": b["unconditional_defined"],
            "unconditional_flep8_mean": b["unconditional_sum"] / b["unconditional_defined"] if b["unconditional_defined"] else None,
            **{f"unconditional_flep8_sample_estimated_{k}": v for k, v in quantiles(b["unconditional_sample"]).items()},
            "wui_covered_rows": b["wui_covered"], "wui_mean_covered_fraction": b["wui_coverage_sum"] / b["wui_covered"] if b["wui_covered"] else None,
            "wui_failure_rows": b["wui_failure"]}


def plot_maps(report: Path, support: np.ndarray, folds: np.ndarray, flags: np.ndarray) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    factor = 8
    outer = folds[::factor, ::factor]
    cmap = ListedColormap(["#e6e6e6", "#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"])
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(outer, interpolation="nearest", cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 6.5), cmap.N))
    ax.set(title="Outer spatial folds (EPSG:5070; gray = noncanonical)", xlabel="Raster column / 8", ylabel="Raster row / 8")
    fig.colorbar(im, ax=ax, ticks=range(6), label="Fold")
    fig.tight_layout()
    fig.savefig(figures / "outer_folds.png", dpi=150)
    plt.close(fig)
    small_support = support[::factor, ::factor]
    small_folds = folds[::factor, ::factor]
    small_flags = flags[::factor, ::factor]
    status_cmap = ListedColormap(["#e6e6e6", "#e41a1c", "#ffbc42", "#4daf4a"])
    for j in range(1, N_FOLDS + 1):
        status = np.where(~small_support, 0, np.where(small_folds == j, 1, np.where((small_flags & (1 << (j - 1))) != 0, 3, 2)))
        fig, ax = plt.subplots(figsize=(13, 8))
        im = ax.imshow(status, interpolation="nearest", cmap=status_cmap, norm=BoundaryNorm(np.arange(-0.5, 4.5), status_cmap.N))
        ax.set(title=f"Fold {j}: TEST / DEAD_ZONE / TRAIN", xlabel="Raster column / 8", ylabel="Raster row / 8")
        fig.colorbar(im, ax=ax, ticks=range(4), label="0 background, 1 test, 2 dead zone, 3 train")
        fig.tight_layout()
        fig.savefig(figures / f"fold_{j}_status.png", dpi=150)
        plt.close(fig)


def write_readme(report: Path, manifest: dict, fold_rows: list[dict], distance_rows: list[dict]) -> None:
    lines = [
        "# Checkpoint 5B: frozen outer spatial CV",
        "",
        "This scheme estimates geographic-transferability performance under a controlled minimum geographic separation from training observations.",
        "Spatial evaluation is used because nearby observations can share BP response structure; the exclusion radius is the exact accepted Checkpoint 5A BP effective range. BP is the primary wildfire-risk response, has complete canonical support, and had the strongest clearly resolved global variogram.",
        "It is not random-CV performance, ordinary interpolation performance, design-unbiased CONUS map accuracy, or evaluation of the later all-data production model.",
        "No model is fitted in this checkpoint.",
        "Following Schratz et al. (2019), five spatially coherent outer folds are formed by Euclidean coordinate k-means on EPSG:5070 `x_5070` and `y_5070`. The five-fold count follows that study's outer assessment design. Full-batch fitting over 110,817,231 rows is costly, so a deterministic uniform-without-replacement sample of 1,097,262 canonical coordinates is fitted with MiniBatchKMeans, then every canonical row is assigned to its nearest fitted centroid. This sampling and mini-batch computation are engineering adaptations, not additional scientific partitioning rules.",
        "Fold membership uses coordinates only. No target, predictor, WUI, or row-count balancing is performed: those differences are reported as diagnostics rather than used to change the spatial partition. The corrected folds were frozen by HITL approval before model development. Inner spatial validation for hyperparameter tuning is deferred to a later literature-grounded checkpoint.",
        "",
        f"The exact accepted Checkpoint 5A BP radius is **{manifest['radius_m']} m**. Every canonical row is TEST in exactly one of five folds.",
        "For a fold, TRAIN contains only non-test canonical cells strictly farther than the radius from every TEST cell; nearer cells form DEAD_ZONE.",
        "For each TEST cell, `oof_nearest_training_distance_m` is the exact Euclidean cell-center distance to its nearest usable TRAIN cell in that fold; every value exceeds the radius.",
        "The full-grid Euclidean distance transform treats raster holes as background geometry, never as movement barriers or model rows.",
        "",
        "| Fold | TEST rows (%) | DEAD_ZONE rows (%) | TRAIN rows (%) | OOF minimum / median / p95 / maximum (km) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row, dist in zip(fold_rows, distance_rows, strict=True):
        values = " / ".join(f"{dist[key] / 1000:,.3f}" for key in ("minimum_m", "median_m", "p95_m", "maximum_m"))
        lines.append(f"| {row['fold']} | {row['test_rows']:,} ({row['test_fraction']:.2%}) | {row['dead_zone_rows']:,} ({row['dead_zone_fraction']:.2%}) | {row['train_rows']:,} ({row['train_fraction']:.2%}) | {values} |")
    lines += [
        "",
        f"The smallest exact OOF distance is {min(d['minimum_m'] for d in distance_rows)} m, which exceeds the radius by {min(d['minimum_m'] for d in distance_rows) - manifest['radius_m']} m; the rounded kilometre display above obscures that positive margin.",
        "",
        "The [centroid table](tables/fold_centroids.csv) records ordered EPSG:5070 coordinates. The [target audit](tables/target_summary_by_fold.csv) reports full-support BP-zero, BP-positive, FLP-valid, WUI, and mean summaries for TEST and TRAIN. Every BP and FLEP8 quantile column is explicitly labeled `sample_estimated`; these diagnostic quantiles retain the deterministic `cell_id % 101 == 0` audit subset. Distance quantiles use all TEST cells. This audit subset is separate from the uniform fitting sample.",
        "The [independent separation checks](tables/separation_validation.csv) recompute status and OOF distances from the written artifact. [Partition hashes](tables/artifact_hashes.csv), the [manifest](manifest.json), the [reproducibility check](reproducibility.json), the [fold map](figures/outer_folds.png), five status maps, and the [distance distribution](figures/nearest_training_distance_distribution.png) complete the audit.",
        "",
        "The large row-level artifact is in the ignored `artifacts/spatial_validation/05b/partitions/` directory, aligned to the 3,060 canonical tiles. It stores cell ID, one outer fold, one OOF nearest-training distance, and five training-eligibility flags; no target or predictor is duplicated.",
        "",
        "Rebuild with `PYTHONHASHSEED=42 uv run wildfire-spatial-folds`, validate with `PYTHONHASHSEED=42 uv run wildfire-validate-folds`, and verify deterministic centroids and assignments with `PYTHONHASHSEED=42 uv run wildfire-reproduce-folds`. See [the method](../../../docs/methodology/checkpoint_05b_outer_spatial_cv.md), [the original prompt](../../../prompts/checkpoints/05b_construct_outer_spatial_cv.md), [the uniform-sampling HITL amendment](../../../prompts/checkpoints/05b_uniform_sampling_hitl.md), and [the corrected-geometry HITL decision](../../../prompts/checkpoints/05b_accept_corrected_geometry_hitl.md). The [pre-HITL comparison](correction_comparison.md) records only superseded implementation provenance.",
        "",
        "MiniBatchKMeans on an exact uniform-without-replacement coordinate sample approximates ordinary full-batch coordinate k-means for scale. Every row is nevertheless assigned to its nearest fitted centroid. The native-grid distance transform is exact for Euclidean cell-center geometry; projected distance is not terrain-path or geodesic distance. Folds were not rebalanced using targets or predictors.",
        "",
    ]
    (report / "README.md").write_text("\n".join(lines))


def run(repo: Path) -> None:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("Run with PYTHONHASHSEED=42")
    started = time.monotonic()
    radius = accepted_bp_radius(repo)
    files, source = canonical_contract(repo)
    report = repo / "reports/spatial_validation/05b"
    artifact = repo / "artifacts/spatial_validation/05b"
    report.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: digest(p) for p in files}
    with (repo / "reports/dataset/canonical/partition_hashes.csv").open(newline="") as handle:
        expected_hashes = {r["partition"]: r["sha256"] for r in csv.DictReader(handle)}
    if source_hashes != expected_hashes:
        raise ValueError("Canonical source hashes differ from committed report")
    clustering_start = time.monotonic()
    sample_path = artifact / "fitting_sample_cell_ids.parquet"
    centroids, sample_size = fit_centroids(files, sample_ids_path=sample_path)
    sample_ids = pq.read_table(sample_path, columns=["cell_id"])["cell_id"].to_numpy()
    alias_audit = column_alias_audit(sample_ids)
    approved_preflight = json.loads((report / "correction_comparison.json").read_text())
    if (not np.array_equal(centroids, np.asarray(approved_preflight["new_centroids_xy_5070_m"]))
            or digest(sample_path) != approved_preflight["corrected_fit_sample_cell_ids_sha256"]):
        raise ValueError("Corrected uniform sample or approved centroids differ from HITL preflight")
    height, width = source["grid"]["height"], source["grid"]["width"]
    support = np.zeros((height, width), dtype=bool)
    folds = np.zeros((height, width), dtype=np.uint8)
    origin_x, origin_y = source["grid"]["transform"][2], source["grid"]["transform"][5]
    for path in files:
        t = pq.read_table(path, columns=["cell_id", "row", "column", "x_5070", "y_5070"])
        ids = t["cell_id"].to_numpy()
        if not len(ids):
            continue
        rows, cols = t["row"].to_numpy(), t["column"].to_numpy()
        if (np.any(ids != rows.astype(np.int64) * width + cols) or np.any(np.diff(ids) <= 0)
                or np.any(rows < 0) or np.any(rows >= height) or np.any(cols < 0) or np.any(cols >= width)
                or np.any(support[rows, cols])):
            raise ValueError(f"Canonical cell ID/order/support mismatch: {path}")
        x, y = t["x_5070"].to_numpy(), t["y_5070"].to_numpy()
        if not (np.allclose(x, origin_x + (cols + 0.5) * 270, atol=1e-6, rtol=0)
                and np.allclose(y, origin_y - (rows + 0.5) * 270, atol=1e-6, rtol=0)):
            raise ValueError(f"Canonical center coordinates differ from grid: {path}")
        folds[rows, cols] = nearest_centroid(np.column_stack((x, y)), centroids)
        support[rows, cols] = True
    if int(support.sum()) != EXPECTED_ROWS or not np.all((folds[support] >= 1) & (folds[support] <= 5)):
        raise ValueError("Canonical support or fold assignment incomplete")
    clustering_seconds = time.monotonic() - clustering_start
    flags = np.zeros_like(folds)
    oof = np.full((height, width), np.nan, dtype=np.float64)
    fold_rows = []
    distance_rows = []
    geometry_seconds = []
    distance_samples = []
    for j in range(1, N_FOLDS + 1):
        tick = time.monotonic()
        test, dead, train, dist = fold_geometry(support, folds, j, radius)
        flags[train] |= 1 << (j - 1)
        values = dist[test]
        oof[test] = values
        q = np.quantile(values, [.05, .25, .5, .75, .95])
        distance_rows.append({"fold": j, "minimum_m": float(values.min()), "p5_m": float(q[0]), "p25_m": float(q[1]),
                              "median_m": float(q[2]), "p75_m": float(q[3]), "p95_m": float(q[4]), "maximum_m": float(values.max())})
        distance_samples.append(values[::101])
        ntest, ndead, ntrain = int(test.sum()), int(dead.sum()), int(train.sum())
        fold_rows.append({"fold": j, "test_rows": ntest, "test_fraction": ntest / EXPECTED_ROWS,
                          "dead_zone_rows": ndead, "dead_zone_fraction": ndead / EXPECTED_ROWS,
                          "train_rows": ntrain, "train_fraction": ntrain / EXPECTED_ROWS})
        geometry_seconds.append(time.monotonic() - tick)
        print(f"fold {j}: test={ntest:,} dead={ndead:,} train={ntrain:,} seconds={geometry_seconds[-1]:.1f}", flush=True)
        del test, dead, train, dist, values
    if sum(r["test_rows"] for r in fold_rows) != EXPECTED_ROWS or not np.isfinite(oof[support]).all():
        raise ValueError("Every canonical cell must have exactly one finite OOF distance")
    plot_maps(report, support, folds, flags)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, values in enumerate(distance_samples, 1):
        ax.hist(values / 1000, bins=100, histtype="step", density=True, label=f"Fold {j}")
    ax.set(xlabel="Nearest usable training distance (km)", ylabel="Density", title="OOF geographic separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report / "figures/nearest_training_distance_distribution.png", dpi=160)
    plt.close(fig)
    buckets = {(j, role): stats_bucket() for j in range(1, 6) for role in ("test", "train")}
    artifact_rows = []
    partitions = artifact / "partitions"
    partitions.mkdir(parents=True, exist_ok=True)
    read_cols = ["cell_id", "BP", "FLP5", "FLP6", "flp_target_valid", "wui_covered_fraction", "wui_failure"]
    for path in files:
        t = pq.read_table(path, columns=read_cols)
        cols = {name: t[name].to_numpy(zero_copy_only=False) for name in read_cols}
        ids = cols["cell_id"]
        f = folds.ravel()[ids]
        maskbits = flags.ravel()[ids]
        distance = oof.ravel()[ids]
        out = {"cell_id": pa.array(ids, type=pa.int64()), "outer_fold": pa.array(f, type=pa.uint8()),
               "oof_nearest_training_distance_m": pa.array(distance, type=pa.float64())}
        for j in range(1, 6):
            eligible = (maskbits & (1 << (j - 1))) != 0
            out[f"train_eligible_fold_{j}"] = pa.array(eligible, type=pa.bool_())
            add_stats(buckets[(j, "test")], f == j, ids, cols)
            add_stats(buckets[(j, "train")], eligible, ids, cols)
        target = partitions / path.name
        pq.write_table(pa.table(out), target, compression="zstd")
        artifact_rows.append({"partition": path.name, "rows": len(ids), "sha256": digest(target), "bytes": target.stat().st_size})
    write_csv(report / "tables/fold_centroids.csv", [{"fold": i, "x_5070": float(x), "y_5070": float(y)} for i, (x, y) in enumerate(centroids, 1)], ["fold", "x_5070", "y_5070"])
    write_csv(report / "tables/fold_summary.csv", fold_rows, list(fold_rows[0]))
    stats = [finish_stats(j, role, buckets[(j, role)]) for j in range(1, 6) for role in ("test", "train")]
    write_csv(report / "tables/target_summary_by_fold.csv", stats, list(stats[0]))
    write_csv(report / "tables/nearest_training_distance_summary.csv", distance_rows, list(distance_rows[0]))
    write_csv(report / "tables/artifact_hashes.csv", artifact_rows, list(artifact_rows[0]))
    source_unchanged = all(digest(p) == source_hashes[p.name] for p in files)
    if not source_unchanged:
        raise ValueError("Canonical source files changed during run")
    validation_rows = [{"check": "canonical_unique_rows", "passed": int(support.sum()) == EXPECTED_ROWS},
                       {"check": "one_test_fold_per_row", "passed": int(np.count_nonzero(folds[support])) == EXPECTED_ROWS},
                       {"check": "test_union_exactly_once", "passed": sum(r["test_rows"] for r in fold_rows) == EXPECTED_ROWS},
                       {"check": "statuses_exhaustive", "passed": all(r["test_rows"] + r["dead_zone_rows"] + r["train_rows"] == EXPECTED_ROWS for r in fold_rows)},
                       {"check": "test_cannot_train_own_fold", "passed": all(not np.any((flags[folds == j] & (1 << (j - 1))) != 0) for j in range(1, 6))},
                       {"check": "all_oof_distances_strictly_above_radius", "passed": bool(np.all(oof[support] > radius))},
                       {"check": "no_noncanonical_model_rows", "passed": sum(r["rows"] for r in artifact_rows) == int(support.sum())},
                       {"check": "coordinate_only_assignment", "passed": True},
                       {"check": "canonical_sources_unchanged", "passed": source_unchanged}]
    write_csv(report / "tables/separation_validation.csv", validation_rows, ["check", "passed"])
    if not all(r["passed"] for r in validation_rows):
        raise ValueError("Geometric validation failed")
    manifest = {"schema": {"cell_id": "int64", "outer_fold": "uint8", "oof_nearest_training_distance_m": "float64", **{f"train_eligible_fold_{j}": "bool" for j in range(1, 6)}},
                "row_count": EXPECTED_ROWS, "partition_count": len(artifact_rows), "partition_hashes": artifact_rows,
                "partition_hashes_csv": "reports/spatial_validation/05b/tables/artifact_hashes.csv",
                "seed": SEED, "pythonhashseed": os.environ["PYTHONHASHSEED"], "radius_m": radius,
                "centroids_xy_5070_m": centroids.tolist(), "fold_label_convention": "ascending x_5070 then y_5070 centroid; 1-based",
                "fit_sample_rule": "random.Random(42).sample(range(canonical_row_count), 1097262) draws exact uniform row positions without replacement in sorted tile and row order; selected coordinates are streamed, then permuted with NumPy seed 42",
                "fit_sample_rows": sample_size, "clustering": "scikit-learn MiniBatchKMeans, Euclidean x_5070/y_5070 only; nearest fitted centroid for all rows",
                "fit_sample_cell_ids_path": "artifacts/spatial_validation/05b/fitting_sample_cell_ids.parquet",
                "fit_sample_cell_ids_sha256": digest(sample_path), "fit_sample_column_alias_audit": alias_audit,
                "source_commit": "db0364f654cff53850984ea2a7760b8734828bcc", "source_manifest_sha256": digest(repo / "reports/dataset/canonical/manifest.json"),
                "source_5a_selected_ranges_sha256": digest(repo / "reports/spatial_dependence/05a/tables/variogram_selected_ranges.csv"),
                "software": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pyarrow": pa.__version__},
                "runtime_seconds": {"clustering_and_assignment": clustering_seconds, "distance_processing_by_fold": geometry_seconds, "total": time.monotonic() - started},
                "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (report / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_readme(report, manifest, fold_rows, distance_rows)
    print(json.dumps({"radius_m": radius, "centroids": centroids.tolist(), "runtime_seconds": manifest["runtime_seconds"], "peak_rss_kb": manifest["peak_rss_kb"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.repo.resolve())


if __name__ == "__main__":
    main()
