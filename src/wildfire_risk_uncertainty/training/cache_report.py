"""Compact 6F prelaunch provenance and prospective compute accounting."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value))


def run(root: Path) -> dict:
    report = root / "reports/training/06f_prelaunch"
    cache = root / "artifacts/training/hpo_cache_v1"
    manifest = json.loads((cache / "manifest.json").read_text())
    equality = json.loads((report / "equivalence_validation.json").read_text())
    regression = json.loads((report / "training_regression.json").read_text())
    states = json.loads((report / "preprocessing_states_summary.json").read_text())
    if not all((equality["preprocessing_identical"], regression["exact_support_and_loss_equality"],
                regression["exact_model_parameter_equality"],
                regression["exact_optimizer_state_equality"], states["all_validated"])):
        raise ValueError("Scientific equivalence or 25-state gate has not passed")
    old = regression["medians_seconds"]["parquet"]
    new = regression["medians_seconds"]["mmap"]
    speedups = []
    for repetition in (1, 2):
        samples = {s["source"]: s for s in regression["samples"] if s["repetition"] == repetition}
        before = samples["parquet"]["training"]["wall_seconds"] + samples["parquet"]["validation"]["wall_seconds"]
        after = samples["mmap"]["training"]["wall_seconds"] + samples["mmap"]["validation"]["wall_seconds"]
        speedups.append(before / after)
    accepted = min(speedups) > 1.2 and regression["end_to_end_speedup"] > 1.2
    if not accepted:
        raise ValueError("Mmap did not show a clear repeatable end-to-end speedup")
    cache_sha = hashlib.sha256(stable_json_bytes(manifest)).hexdigest()
    offsets = np.load(cache / "partition_offsets.npy", mmap_mode="r")
    _write(report / "cache_manifest_summary.json", {
        "version": manifest["version"], "manifest_sha256": cache_sha,
        "rows": manifest["rows"], "partitions": manifest["partitions"],
        "empty_partitions": int(np.count_nonzero(np.diff(offsets) == 0)),
        "size_bytes": manifest["size_bytes"], "build_seconds": manifest["build_seconds"],
        "array_dtypes": {"cell_id": "int64", "features": "float32", "bp": "float32",
                         "flp": "float32", "flp_valid": "uint8", "inner_fold": "uint8",
                         "partition_offsets": "int64"},
        "feature_count": len(manifest["inputs"]), "source_hashes": manifest["source_hashes"],
        "files_sha256": manifest["files_sha256"],
    })
    _write(report / "cache_validation.json", {
        "canonical_unique_cell_ids": manifest["rows"],
        "unique_ids_checked_during_build": True,
        "partition_names_and_row_offsets_checked": True,
        "outer_5b_eligibility_equal_for_every_row": True,
        "inner_6c_labels_equal_for_every_row": True,
        "outer_train_counts": manifest["outer_train_counts"],
        "all_file_sha256_verified_on_load": equality["all_cache_files_hashed"],
        "full_train_validation_order_and_values_equal": all(
            x["all_features_targets_masks_equal"] for x in equality["stream_equivalence"]),
    })
    train_rate = 61_042_788 / new["training"]
    validation_rate = 18_476_910 / new["validation"]
    populations = json.loads((root / "reports/spatial_validation/06c/manifest.json").read_text())[
        "outer_train_rows"]
    per_outer_epoch = {study: 4 * count / train_rate + count / validation_rate
                       for study, count in ((study, int(n)) for study, n in populations.items())}
    five_study_epoch_seconds = sum(per_outer_epoch.values())
    idealized_units_per_trial = 1 + 3 / 4 + 12 / 16 + 48 / 64 + 36 / 256
    idealized_hours = five_study_epoch_seconds * 30 * idealized_units_per_trial / 3600
    minimum_complete_hours = five_study_epoch_seconds * (29 + 100) / 3600
    no_pruning_hours = five_study_epoch_seconds * 30 * 100 / 3600
    plan = {"accepted_mmap_for_6f": accepted, "cache_manifest_sha256": cache_sha,
            "fixed_case_median_seconds": {"parquet": old, "mmap": new},
            "paired_end_to_end_speedups": speedups,
            "median_end_to_end_speedup": regression["end_to_end_speedup"],
            "mmap_training_rows_per_second": train_rate,
            "mmap_validation_rows_per_second": validation_rate,
            "per_outer_five_fold_epoch_seconds": per_outer_epoch,
            "all_five_studies_one_resource_epoch_seconds": five_study_epoch_seconds,
            "frozen_trials_per_study": 30, "outer_studies": 5,
            "idealized_asha_gpu_hours_optimistic": idealized_hours,
            "minimum_with_one_complete_per_study_gpu_hours": minimum_complete_hours,
            "no_pruning_gpu_hours": no_pruning_hours,
            "interpretation": "Planning bracket, not a runtime prediction. ASHA survival, model width, batch size, OS page cache, and study-specific throughput remain unknown before HPO."}
    _write(report / "projected_hpo_runtime.json", plan)
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            name, value, _unit = line.split()
            mem[name.removesuffix(":")] = int(value) * 1024
    gpu = torch.cuda.get_device_properties(0)
    environment = {"base_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                          cwd=root, text=True).strip(),
                   "python": platform.python_version(), "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "gpu": gpu.name,
                   "gpu_vram_bytes": gpu.total_memory,
                   "cpu_logical": os.cpu_count(), "cpu_affinity": len(os.sched_getaffinity(0)),
                   "system_memory_bytes": mem,
                   "workspace_free_bytes_after_cache": shutil.disk_usage(root).free}
    _write(report / "environment.json", environment)
    return plan


if __name__ == "__main__":
    result = run(Path.cwd())
    print(json.dumps({"accepted_mmap_for_6f": result["accepted_mmap_for_6f"],
                      "median_end_to_end_speedup": result["median_end_to_end_speedup"],
                      "idealized_asha_gpu_hours_optimistic": result["idealized_asha_gpu_hours_optimistic"],
                      "minimum_with_one_complete_per_study_gpu_hours": result["minimum_with_one_complete_per_study_gpu_hours"],
                      "no_pruning_gpu_hours": result["no_pruning_gpu_hours"]}))
