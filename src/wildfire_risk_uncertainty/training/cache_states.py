"""Fit and provenance-check the 25 fold-specific preprocessing states from mmap data."""

import csv
import hashlib
import json
import time
from pathlib import Path

from wildfire_risk_uncertainty.hpo.real import state_from_record, support_counts
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.data import inner_chunks
from wildfire_risk_uncertainty.training.preprocessing import (
    fit_preprocessing,
    stable_json_bytes,
)


def build_states(root: Path) -> list[dict]:
    cache = HpoCache(root, root / "artifacts/training/hpo_cache_v1")
    cache_sha = hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest()
    inputs = tuple(cache.manifest["inputs"])
    schema_sha = hashlib.sha256((root / "reports/dataset/canonical/schema.json").read_bytes()).hexdigest()
    implementation_sha = hashlib.sha256((root / "src/wildfire_risk_uncertainty/training/preprocessing.py").read_bytes()).hexdigest()
    feature_sha = hashlib.sha256(stable_json_bytes(list(inputs))).hexdigest()
    report = root / "reports/training/06f_prelaunch"
    destination = root / "artifacts/hpo/preprocessing_cache_v1"
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    args = (root / "artifacts/dataset/canonical/partitions",
            root / "artifacts/spatial_validation/05b/partitions",
            root / "artifacts/spatial_validation/06c/partitions", inputs)
    for outer in range(1, 6):
        counts = support_counts(root, outer)
        for fold in range(1, 6):
            path = destination / f"outer_{outer}_inner_{fold}.json"
            provenance = {"cache_version": cache.manifest["version"],
                          "cache_manifest_sha256": cache_sha,
                          "source_hashes": cache.manifest["source_hashes"],
                          "outer_study": outer, "inner_fold": fold,
                          "feature_order_sha256": feature_sha,
                          "preprocessing_implementation_sha256": implementation_sha,
                          "schema_sha256": schema_sha,
                          "train_rows": counts[fold]["train"]["rows"],
                          "train_flp_valid": counts[fold]["train"]["flp_valid"]}
            started = time.perf_counter()
            if path.exists():
                record = read_json(path)
                if record["provenance"] != provenance:
                    raise ValueError(f"Cache preprocessing provenance mismatch: {outer}/{fold}")
                state = state_from_record(record)
                if state.digest() != record["state_sha256"]:
                    raise ValueError(f"Cache preprocessing state hash mismatch: {outer}/{fold}")
                status = "verified_existing"
            else:
                state = fit_preprocessing(inner_chunks(*args, role="train", outer_study=outer,
                                                       inner_fold=fold, cache=cache), inputs, schema_sha)
                record = {"provenance": provenance, "state": state.serializable(),
                          "state_sha256": state.digest()}
                write_json(path, record)
                status = "fitted_and_validated"
            if state.rows != counts[fold]["train"]["rows"] or state.flp_valid_rows != counts[fold]["train"]["flp_valid"]:
                raise ValueError(f"Cache preprocessing support differs from frozen 6C: {outer}/{fold}")
            old_path = root / "artifacts/hpo/preprocessing" / f"outer_{outer}_inner_{fold}.json"
            old_verified = False
            if old_path.exists():
                old = read_json(old_path)
                old_provenance = old["provenance"]
                if (old_provenance["outer_study"] != outer or
                    old_provenance["inner_fold"] != fold or
                    old_provenance["feature_order_sha256"] != feature_sha or
                    old_provenance["schema_sha256"] != schema_sha or
                    old_provenance["outer_fold_manifest_sha256"] != cache.manifest["source_hashes"]["outer_5b"] or
                    old_provenance["inner_fold_manifest_sha256"] != cache.manifest["source_hashes"]["inner_6c"]):
                    raise ValueError(f"Existing 6E preprocessing provenance differs: {outer}/{fold}")
                previous = state_from_record(old)
                if previous.digest() != old["state_sha256"] or previous.digest() != state.digest():
                    raise ValueError(f"Existing 6E preprocessing state differs: {outer}/{fold}")
                if previous.rows != state.rows or previous.flp_valid_rows != state.flp_valid_rows:
                    raise ValueError(f"Existing 6E preprocessing support differs: {outer}/{fold}")
                old_verified = True
            rows.append({"outer_study": outer, "inner_fold": fold,
                         "train_rows": state.rows, "flp_valid": state.flp_valid_rows,
                         "preprocessing_sha256": state.digest(), "cache_manifest_sha256": cache_sha,
                         "preprocessing_implementation_sha256": implementation_sha,
                         "fit_or_verify_seconds": time.perf_counter() - started,
                         "status": status, "existing_6e_state_verified": old_verified})
    with (report / "preprocessing_states.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(report / "preprocessing_states_summary.json", {
        "states": len(rows), "cache_manifest_sha256": cache_sha,
        "preprocessing_implementation_sha256": implementation_sha,
        "all_validated": len(rows) == 25,
        "previous_6e_states_verified": sum(row["existing_6e_state_verified"] for row in rows)})
    return rows


def load_states(root: Path, cache: HpoCache, outer: int) -> dict:
    """Load only previously validated fold states; never fit during a trial."""
    summary = read_json(root / "reports/training/06f_prelaunch/preprocessing_states_summary.json")
    cache_sha = hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest()
    implementation_sha = hashlib.sha256((root / "src/wildfire_risk_uncertainty/training/preprocessing.py").read_bytes()).hexdigest()
    if (not summary["all_validated"] or summary["states"] != 25 or
        summary["cache_manifest_sha256"] != cache_sha or
        summary["preprocessing_implementation_sha256"] != implementation_sha):
        raise ValueError("25-state preprocessing cache manifest mismatch")
    inputs = tuple(cache.manifest["inputs"])
    feature_sha = hashlib.sha256(stable_json_bytes(list(inputs))).hexdigest()
    schema_sha = hashlib.sha256((root / "reports/dataset/canonical/schema.json").read_bytes()).hexdigest()
    counts = support_counts(root, outer)
    states = {}
    for fold in range(1, 6):
        path = root / "artifacts/hpo/preprocessing_cache_v1" / f"outer_{outer}_inner_{fold}.json"
        record = read_json(path)
        expected = {"cache_version": cache.manifest["version"],
                    "cache_manifest_sha256": cache_sha,
                    "source_hashes": cache.manifest["source_hashes"],
                    "outer_study": outer, "inner_fold": fold,
                    "feature_order_sha256": feature_sha,
                    "preprocessing_implementation_sha256": implementation_sha,
                    "schema_sha256": schema_sha,
                    "train_rows": counts[fold]["train"]["rows"],
                    "train_flp_valid": counts[fold]["train"]["flp_valid"]}
        state = state_from_record(record)
        if record["provenance"] != expected or state.digest() != record["state_sha256"]:
            raise ValueError(f"Fold preprocessing cache mismatch: {outer}/{fold}")
        states[fold] = state
    return states


if __name__ == "__main__":
    print(json.dumps({"validated_states": len(build_states(Path.cwd()))}))
