"""Explicit build and verification commands for the ignored HPO mmap cache."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from wildfire_risk_uncertainty.training.cache import HpoCache, build, compare_streams
from wildfire_risk_uncertainty.training.data import inner_chunks
from wildfire_risk_uncertainty.training.preprocessing import (
    fit_preprocessing,
    stable_json_bytes,
)


def validate(root: Path, path: Path) -> dict:
    cache = HpoCache(root, path)
    report = root / "reports/training/06f_prelaunch"
    report.mkdir(parents=True, exist_ok=True)
    results = []
    for role, epoch in (("train", 1), ("validation", None)):
        started = time.perf_counter()
        item = compare_streams(root, cache, role=role, epoch=epoch)
        item["seconds"] = time.perf_counter() - started
        results.append(item)
    args = (root / "artifacts/dataset/canonical/partitions",
            root / "artifacts/spatial_validation/05b/partitions",
            root / "artifacts/spatial_validation/06c/partitions",
            tuple(cache.manifest["inputs"]))
    schema_hash = hashlib.sha256((root / "reports/dataset/canonical/schema.json").read_bytes()).hexdigest()
    fits = {}
    for source, kwargs in (("parquet", {}), ("mmap", {"cache": cache})):
        started = time.perf_counter()
        state = fit_preprocessing(inner_chunks(*args, role="train", outer_study=3,
                                               inner_fold=1, **kwargs), args[3], schema_hash)
        fits[source] = {"state": state.serializable(), "sha256": state.digest(),
                        "seconds": time.perf_counter() - started}
    if fits["parquet"]["state"] != fits["mmap"]["state"]:
        raise ValueError("Preprocessing state differs between Parquet and mmap readers")
    result = {"cache_version": cache.manifest["version"],
              "cache_manifest_sha256": hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest(),
              "rows": cache.manifest["rows"], "partitions": cache.manifest["partitions"],
              "source_hashes": cache.manifest["source_hashes"],
              "outer_train_counts": cache.manifest["outer_train_counts"],
              "all_cache_files_hashed": True, "stream_equivalence": results,
              "preprocessing_fits": fits,
              "preprocessing_identical": fits["parquet"]["sha256"] == fits["mmap"]["sha256"]}
    (report / "equivalence_validation.json").write_bytes(stable_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen HPO input cache")
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    root = Path.cwd()
    cache_path = root / "artifacts/training/hpo_cache_v1"
    if args.command == "build":
        manifest = build(root, cache_path)
    else:
        result = validate(root, cache_path)
        manifest = HpoCache(root, cache_path, check_files=False).manifest
        print(json.dumps({"stream_equivalence": result["stream_equivalence"],
                          "preprocessing_identical": result["preprocessing_identical"]}, sort_keys=True))
    print(json.dumps({"version": manifest["version"], "rows": manifest["rows"],
                      "partitions": manifest["partitions"],
                      "size_bytes": manifest["size_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
