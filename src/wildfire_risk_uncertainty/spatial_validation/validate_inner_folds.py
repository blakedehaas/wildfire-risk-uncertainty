"""Independently validate and fully reproduce the Checkpoint 6C artifact."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .folds import (
    EXPECTED_ROWS,
    N_FOLDS,
    SAMPLE_ROWS,
    canonical_contract,
    nearest_centroid,
)
from .inner_folds import (
    aligned_tiles,
    digest,
    fit_inner_centroids,
    inner_assignment_table,
    outer_train_counts,
    persist_sample,
    sample_eligible_coordinates,
)
from .outer_cli import write_csv


def validate(repo: Path) -> dict:
    start = time.monotonic()
    _canonical_files, source = canonical_contract(repo)
    tiles = aligned_tiles(repo / "artifacts/dataset/canonical/partitions",
                          repo / "artifacts/spatial_validation/05b/partitions")
    outer_counts = outer_train_counts(repo / "reports/spatial_validation/05b/tables/fold_summary.csv")
    artifact = repo / "artifacts/spatial_validation/06c"
    report = repo / "reports/spatial_validation/06c"
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest != json.loads((artifact / "manifest.json").read_text()):
        raise ValueError("Report/artifact manifests differ")
    if manifest["row_count"] != EXPECTED_ROWS or manifest["partition_count"] != 3060:
        raise ValueError("06C manifest support incorrect")
    if manifest["canonical_manifest_sha256"] != digest(repo / "reports/dataset/canonical/manifest.json"):
        raise ValueError("Canonical provenance changed")
    if manifest["outer_5b_manifest_sha256"] != digest(repo / "reports/spatial_validation/05b/manifest.json"):
        raise ValueError("5B provenance changed")
    centers = {j: np.asarray(manifest["centroids_xy_5070_m_by_study"][str(j)])
               for j in range(1, N_FOLDS + 1)}
    with tempfile.TemporaryDirectory(dir=artifact, prefix="reproduce_") as temporary:
        temp = Path(temporary)
        for study in range(1, N_FOLDS + 1):
            ids, xy = sample_eligible_coordinates(tiles, study, outer_counts[study], SAMPLE_ROWS)
            generated_hash = persist_sample(ids, temp / f"fitting_sample_outer_{study}.parquet")
            stored_path = artifact / f"fitting_sample_outer_{study}.parquet"
            if (generated_hash != digest(stored_path)
                    or generated_hash != manifest["fit_sample_sha256_by_study"][str(study)]):
                raise ValueError(f"Outer {study}: fitting sample hash not reproducible")
            if not np.array_equal(fit_inner_centroids(xy), centers[study]):
                raise ValueError(f"Outer {study}: centroid refit differs")
        expected_hashes = {r["partition"]: r["sha256"] for r in manifest["partition_hashes"]}
        if len(expected_hashes) != 3060:
            raise ValueError("Incomplete 06C partition hash manifest")
        support = np.zeros(source["grid"]["width"] * source["grid"]["height"], dtype=bool)
        validation_counts = {(j, k): {"rows": 0, "bp_zero": 0, "bp_positive": 0, "flp_valid": 0}
                             for j in range(1, 6) for k in range(1, 6)}
        eligible_counts = {j: 0 for j in range(1, 6)}
        test_assigned = {j: 0 for j in range(1, 6)}
        dead_assigned = {j: 0 for j in range(1, 6)}
        row_count = 0
        for canonical_path, outer_path in tiles:
            inner_path = artifact / "partitions" / canonical_path.name
            if not inner_path.exists() or digest(inner_path) != expected_hashes[inner_path.name]:
                raise ValueError(f"06C partition hash mismatch: {inner_path.name}")
            canonical = pq.read_table(canonical_path, columns=["cell_id", "x_5070", "y_5070", "BP", "flp_target_valid"])
            outer = pq.read_table(outer_path, columns=["cell_id", "outer_fold", *(f"train_eligible_fold_{j}" for j in range(1, 6))])
            inner = pq.read_table(inner_path)
            required = ["cell_id", *(f"inner_fold_outer_{j}" for j in range(1, 6))]
            if inner.column_names != required or len(inner) != len(canonical):
                raise ValueError("06C partition schema/order/row count invalid")
            if (inner.schema.field("cell_id").type != pa.int64()
                    or any(inner.schema.field(name).type != pa.uint8() for name in required[1:])):
                raise ValueError("06C partition column types invalid")
            ids = canonical["cell_id"].to_numpy()
            if not (np.array_equal(ids, outer["cell_id"].to_numpy())
                    and np.array_equal(ids, inner["cell_id"].to_numpy())):
                raise ValueError(f"Canonical/5B/6C cell IDs differ: {inner_path.name}")
            if np.any(support[ids]):
                raise ValueError("Duplicate canonical cell ID")
            support[ids] = True
            xy = np.column_stack((canonical["x_5070"].to_numpy(), canonical["y_5070"].to_numpy()))
            bp = canonical["BP"].to_numpy()
            valid = canonical["flp_target_valid"].to_numpy()
            outer_fold = outer["outer_fold"].to_numpy()
            predicted: dict[int, np.ndarray] = {}
            for study in range(1, N_FOLDS + 1):
                eligible = outer[f"train_eligible_fold_{study}"].to_numpy()
                arr = inner[f"inner_fold_outer_{study}"].combine_chunks()
                nulls = arr.is_null().to_numpy(zero_copy_only=False)
                if not np.array_equal(~nulls, eligible):
                    raise ValueError(f"Outer {study}: inner null mask differs from 5B eligibility")
                labels = arr.fill_null(0).to_numpy().astype(np.uint8)
                if np.any(labels[eligible] != nearest_centroid(xy[eligible], centers[study])):
                    raise ValueError(f"Outer {study}: assignment differs from nearest centroid")
                if np.any((labels[eligible] < 1) | (labels[eligible] > 5)):
                    raise ValueError("Inner labels outside 1..5")
                predicted[study] = labels
                eligible_counts[study] += int(eligible.sum())
                test_assigned[study] += int(np.count_nonzero((outer_fold == study) & (labels != 0)))
                dead_assigned[study] += int(np.count_nonzero((outer_fold != study) & ~eligible & (labels != 0)))
                for fold in range(1, N_FOLDS + 1):
                    mask = labels == fold
                    b = validation_counts[(study, fold)]
                    b["rows"] += int(mask.sum())
                    b["bp_zero"] += int(np.count_nonzero(mask & (bp == 0)))
                    b["bp_positive"] += int(np.count_nonzero(mask & (bp > 0)))
                    b["flp_valid"] += int(np.count_nonzero(mask & valid))
            reproduced = temp / inner_path.name
            pq.write_table(inner_assignment_table(ids, predicted), reproduced, compression="zstd")
            if digest(reproduced) != expected_hashes[inner_path.name]:
                raise ValueError(f"Full partition regeneration differs: {inner_path.name}")
            row_count += len(ids)
        if row_count != EXPECTED_ROWS or int(support.sum()) != EXPECTED_ROWS:
            raise ValueError("Canonical unique support incomplete")
        with (report / "tables/target_support_by_inner_fold.csv").open(newline="") as stream:
            audit = list(csv.DictReader(stream))
        for study in range(1, N_FOLDS + 1):
            if (eligible_counts[study] != outer_counts[study]
                    or test_assigned[study] != 0 or dead_assigned[study] != 0):
                raise ValueError(f"Outer {study}: TRAIN/TEST/DEAD_ZONE eligibility failed")
            for fold in range(1, N_FOLDS + 1):
                b = validation_counts[(study, fold)]
                reported = next(r for r in audit if r["outer_study"] == str(study)
                                and r["inner_fold"] == str(fold) and r["role"] == "validation")
                if (b["rows"] != int(reported["rows"]) or b["bp_zero"] != int(reported["bp_zero"])
                        or b["bp_positive"] != int(reported["bp_positive"])
                        or b["flp_valid"] != int(reported["flp_target_valid"])):
                    raise ValueError("Independent target-support audit differs")
                train = next(r for r in audit if r["outer_study"] == str(study)
                             and r["inner_fold"] == str(fold) and r["role"] == "train")
                for key, column in (("rows", "rows"), ("bp_zero", "bp_zero"),
                                    ("bp_positive", "bp_positive"), ("flp_valid", "flp_target_valid")):
                    total = sum(validation_counts[(study, k)][key] for k in range(1, N_FOLDS + 1))
                    if total - b[key] != int(train[column]):
                        raise ValueError("Independent inner TRAIN complement support differs")
                if min(b["rows"], b["bp_positive"], b["flp_valid"],
                       int(train["rows"]), int(train["bp_positive"]),
                       int(train["flp_target_valid"])) <= 0:
                    raise ValueError("Empty inner TRAIN or validation support")
        checks = [{"check": "all_five_uniform_samples_regenerated", "passed": True},
                  {"check": "all_25_centroids_refit_byte_equal", "passed": True},
                  {"check": "all_3060_partitions_regenerated_hash_equal", "passed": True},
                  {"check": "all_110817231_canonical_ids_unique", "passed": True},
                  {"check": "eligibility_exact_for_five_studies", "passed": True},
                  {"check": "nearest_centroid_exact_for_all_eligible_rows", "passed": True},
                  {"check": "target_support_counts_independently_recomputed", "passed": True}]
    result = {"checks": checks, "row_count": row_count, "partition_count": len(tiles),
              "runtime_seconds": time.monotonic() - start,
              "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "python_version": platform.python_version()}
    (report / "reproducibility.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(report / "tables" / "independent_validation.csv", checks, list(checks[0]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate(args.repo.resolve()), indent=2))


if __name__ == "__main__":
    main()
