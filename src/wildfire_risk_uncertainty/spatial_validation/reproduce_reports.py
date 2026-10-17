"""Regenerate deterministic compact Checkpoint 5B reports from persisted rows."""

from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from .folds import EXPECTED_ROWS, N_FOLDS, canonical_contract
from .outer_cli import (
    add_stats,
    digest,
    finish_stats,
    plot_maps,
    stats_bucket,
    write_csv,
    write_readme,
)


def check(repo: Path) -> dict:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("Run with PYTHONHASHSEED=42")
    start = time.monotonic()
    files, source = canonical_contract(repo)
    report = repo / "reports/spatial_validation/05b"
    manifest = json.loads((report / "manifest.json").read_text())
    height, width = source["grid"]["height"], source["grid"]["width"]
    support = np.zeros((height, width), dtype=bool)
    folds = np.zeros((height, width), dtype=np.uint8)
    flags = np.zeros((height, width), dtype=np.uint8)
    oof = np.full((height, width), np.nan, dtype=np.float64)
    buckets = {(j, role): stats_bucket() for j in range(1, 6) for role in ("test", "train")}
    artifact_rows = []
    read_cols = ["cell_id", "BP", "FLP5", "FLP6", "flp_target_valid", "wui_covered_fraction", "wui_failure"]
    for path in files:
        canonical = pq.read_table(path, columns=read_cols)
        cols = {name: canonical[name].to_numpy(zero_copy_only=False) for name in read_cols}
        ids = cols["cell_id"]
        target = repo / "artifacts/spatial_validation/05b/partitions" / path.name
        artifact = pq.read_table(target)
        if not np.array_equal(ids, artifact["cell_id"].to_numpy()):
            raise ValueError(f"Artifact IDs differ from canonical IDs: {path.name}")
        f = artifact["outer_fold"].to_numpy()
        distance = artifact["oof_nearest_training_distance_m"].to_numpy()
        bits = np.zeros(len(ids), dtype=np.uint8)
        for j in range(1, 6):
            eligible = artifact[f"train_eligible_fold_{j}"].to_numpy()
            bits[eligible] |= 1 << (j - 1)
            add_stats(buckets[(j, "test")], f == j, ids, cols)
            add_stats(buckets[(j, "train")], eligible, ids, cols)
        flat_support = support.ravel()
        if np.any(flat_support[ids]):
            raise ValueError(f"Duplicate model row: {path.name}")
        flat_support[ids] = True
        folds.ravel()[ids] = f
        flags.ravel()[ids] = bits
        oof.ravel()[ids] = distance
        artifact_rows.append({"partition": path.name, "rows": len(ids), "sha256": digest(target), "bytes": target.stat().st_size})
    if int(support.sum()) != EXPECTED_ROWS:
        raise ValueError("Canonical support count differs")

    fold_rows = []
    distance_rows = []
    distance_samples = []
    for j in range(1, N_FOLDS + 1):
        test = support & (folds == j)
        train = support & ((flags & (1 << (j - 1))) != 0)
        ntest, ntrain = int(test.sum()), int(train.sum())
        ndead = EXPECTED_ROWS - ntest - ntrain
        fold_rows.append({"fold": j, "test_rows": ntest, "test_fraction": ntest / EXPECTED_ROWS,
                          "dead_zone_rows": ndead, "dead_zone_fraction": ndead / EXPECTED_ROWS,
                          "train_rows": ntrain, "train_fraction": ntrain / EXPECTED_ROWS})
        values = oof[test]
        q = np.quantile(values, [.05, .25, .5, .75, .95])
        distance_rows.append({"fold": j, "minimum_m": float(values.min()), "p5_m": float(q[0]), "p25_m": float(q[1]),
                              "median_m": float(q[2]), "p75_m": float(q[3]), "p95_m": float(q[4]), "maximum_m": float(values.max())})
        distance_samples.append(values[::101])

    with tempfile.TemporaryDirectory(prefix="05b_report_regeneration_") as temp_dir:
        regenerated = Path(temp_dir)
        plot_maps(regenerated, support, folds, flags)
        fig, ax = plt.subplots(figsize=(9, 5))
        for j, values in enumerate(distance_samples, 1):
            ax.hist(values / 1000, bins=100, histtype="step", density=True, label=f"Fold {j}")
        ax.set(xlabel="Nearest usable training distance (km)", ylabel="Density", title="OOF geographic separation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(regenerated / "figures/nearest_training_distance_distribution.png", dpi=160)
        plt.close(fig)
        centers = np.asarray(manifest["centroids_xy_5070_m"])
        write_csv(regenerated / "tables/fold_centroids.csv", [{"fold": i, "x_5070": float(x), "y_5070": float(y)} for i, (x, y) in enumerate(centers, 1)], ["fold", "x_5070", "y_5070"])
        write_csv(regenerated / "tables/fold_summary.csv", fold_rows, list(fold_rows[0]))
        stats = [finish_stats(j, role, buckets[(j, role)]) for j in range(1, 6) for role in ("test", "train")]
        write_csv(regenerated / "tables/target_summary_by_fold.csv", stats, list(stats[0]))
        write_csv(regenerated / "tables/nearest_training_distance_summary.csv", distance_rows, list(distance_rows[0]))
        write_csv(regenerated / "tables/artifact_hashes.csv", artifact_rows, list(artifact_rows[0]))
        write_readme(regenerated, manifest, fold_rows, distance_rows)
        paths = sorted([p.relative_to(regenerated) for p in regenerated.rglob("*") if p.is_file()])
        differences = [str(path) for path in paths if digest(regenerated / path) != digest(report / path)]
        if differences:
            raise ValueError(f"Compact reports differ on regeneration: {differences}")
    result = {"compact_report_files_regenerated_hash_equal": len(paths),
              "compact_report_paths": [str(path) for path in paths],
              "seconds": time.monotonic() - start,
              "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (report / "report_reproducibility.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(check(args.repo.resolve()), indent=2))


if __name__ == "__main__":
    main()
