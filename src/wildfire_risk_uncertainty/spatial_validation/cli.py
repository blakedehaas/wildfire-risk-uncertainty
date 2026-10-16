"""Generate Checkpoint 5A samples, variograms, tables, and figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .sampling import (
    EXPECTED,
    POPULATIONS,
    derive_targets,
    sample_partitions,
    validate_contract,
)
from .variogram import (
    BOUNDARY_FIELDS,
    BOUNDARY_PERCENTAGES,
    EMPIRICAL_FIELDS,
    FIT_FIELDS,
    MIN_PAIRS,
    SELECTED_FIELDS,
    TARGETS,
    automap_boundaries,
    fit_target,
)

REPORT = Path("reports/spatial_dependence/05a")
POPULATION_FOR = {"bp": "bp", **{f"flp{i}": "conditional" for i in range(1, 7)},
                  "conditional_flep8": "conditional", "unconditional_flep8": "unconditional"}
FROZEN_SAMPLE_HASHES = {
    "bp": "e2a3580930ff24122843a5659b71f18b75cbc44294998efdf9c4aef25529ea34",
    "conditional": "697fcde354361d6a04a635e8851f16ddde06410e889fc2e38b117772e843f5eb",
    "unconditional": "f30e4350cbef3c07a40eba8f74727a8c0f7c63158fda8617904f5e1ed1a8387e",
}


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_samples(repo: Path, output: Path) -> None:
    files = validate_contract(repo)
    sampler = sample_partitions(files)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in POPULATIONS:
        data = sampler.result(name)
        data.update(derive_targets(data))
        if len(data["cell_id"]) != 5000:
            raise ValueError(f"Sample shortfall: {name}")
        common = ("cell_id", "x_5070", "y_5070", "bp")
        if name == "conditional":
            columns = (*common, "flp_target_valid", *(f"flp{i}" for i in range(1, 7)), "conditional_flep8")
        elif name == "unconditional":
            columns = (*common, "flp_target_valid", "unconditional_flep8")
        else:
            columns = common
        table = pa.table({key: pa.array(data[key]) for key in columns})
        sample_path = sample_dir / f"{name}.parquet"
        pq.write_table(table, sample_path, compression="zstd")
        manifest.append({"population": name, "support_size": sampler.counts[name], "sample_size": 5000,
                         "seed": 42, "path": f"samples/{name}.parquet",
                         "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest()})
    write_csv(output / "tables/sample_manifest.csv",
              ("population", "support_size", "sample_size", "seed", "path", "sha256"), manifest)


def make_plots(output: Path, target: str, coords: np.ndarray, empirical: list[dict], best: dict | None, curve) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array([r["mean_distance_m"] for r in empirical]) / 1000
    y = np.array([r["semivariance"] for r in empirical])
    ax.plot(x, y, "o", label="Matheron empirical")
    if best is not None and curve is not None:
        ax.plot(curve[0] / 1000, curve[1], label=f"{best['model']} fit")
        if best["range_resolved"]:
            ax.axvline(best["effective_range_km"], color="black", linestyle="--",
                       label=f"effective range {best['effective_range_km']:.1f} km")
    ax.set(xlabel="Distance (km)", ylabel="Semivariance", title=f"{target} (n={len(coords):,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / f"{target}_variogram.png", dpi=160)
    plt.close(fig)


def analyze_samples(output: Path) -> None:
    with (output / "tables/sample_manifest.csv").open(newline="") as stream:
        sample_manifest = list(csv.DictReader(stream))
    if [row["population"] for row in sample_manifest] != list(POPULATIONS):
        raise ValueError("Unexpected sample populations or ordering")
    for row in sample_manifest:
        name = row["population"]
        path = output / row["path"]
        if int(row["support_size"]) != EXPECTED[name] or int(row["sample_size"]) != 5000 or int(row["seed"]) != 42:
            raise ValueError(f"Sample manifest contract mismatch: {name}")
        if row["sha256"] != FROZEN_SAMPLE_HASHES[name]:
            raise ValueError(f"Frozen sample digest mismatch in manifest: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_SAMPLE_HASHES[name]:
            raise ValueError(f"Sample hash mismatch: {name}")
    sample_coords = {}
    population_boundaries = {}
    for name in POPULATIONS:
        location = pq.read_table(output / "samples" / f"{name}.parquet", columns=["x_5070", "y_5070"])
        coords = np.column_stack((location["x_5070"].to_numpy(), location["y_5070"].to_numpy()))
        sample_coords[name] = coords
        population_boundaries[name] = automap_boundaries(coords)
    empirical_all, fits_all, selected_all, boundary_rows = [], [], [], []
    selected = {}
    for target in TARGETS:
        name = POPULATION_FOR[target]
        data = pq.read_table(output / "samples" / f"{name}.parquet", columns=[target])
        coords = sample_coords[name]
        values = data[target].to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite sampled target: {target}")
        diagonal, boundaries = population_boundaries[name]
        empirical, fits, best, curve = fit_target(target, coords, values, boundaries, diagonal)
        empirical_all.extend(empirical)
        fits_all.extend(fits)
        if not any(row["population"] == name for row in boundary_rows):
            retained = {row["upper_boundary_m"] for row in empirical}
            boundary_rows.extend({
                "population": name, "boundary_index": i + 1,
                "percentage": float(percentage), "bbox_diagonal_m": diagonal,
                "boundary_m": float(boundary), "boundary_km": float(boundary / 1000),
                "retained_after_merge": float(boundary) in retained, "minimum_pairs": MIN_PAIRS,
            } for i, (percentage, boundary) in enumerate(zip(BOUNDARY_PERCENTAGES, boundaries, strict=True)))
        status = "RESOLVED" if best is not None and best["range_resolved"] else "RANGE_UNRESOLVED"
        selected[target] = status
        selected_all.append({"target": target, "sample_size": len(values), "population": name,
                             "model": best["model"] if best else "",
                             "status": status,
                             "nugget": best["nugget"] if best else "",
                             "partial_sill": best["partial_sill"] if best else "",
                             "total_sill": best["total_sill"] if best else "",
                             "nugget_to_total_sill": best["nugget_to_total_sill"] if best else "",
                             "effective_range_m": best["effective_range_m"] if status == "RESOLVED" else "",
                             "effective_range_km": best["effective_range_km"] if status == "RESOLVED" else "",
                             "weighted_sse": best["weighted_sse"] if best else "",
                             "fit_rmse": best["fit_rmse"] if best else "",
                             "fit_r2": best["fit_r2"] if best else "",
                             "max_lag_m": best["max_lag_m"] if best else "",
                             "supported_sill": best["supported_sill"] if best else False})
        make_plots(output, target, coords, empirical, best, curve)
        print(f"{target}: {status}; {best['model'] if best else 'none'}", flush=True)
    write_csv(output / "tables/variogram_empirical.csv", EMPIRICAL_FIELDS, empirical_all)
    write_csv(output / "tables/variogram_fits.csv", FIT_FIELDS, fits_all)
    write_csv(output / "tables/variogram_selected_ranges.csv", SELECTED_FIELDS, selected_all)
    write_csv(output / "tables/lag_boundaries.csv", BOUNDARY_FIELDS, boundary_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    resolved_rows = [row for row in selected_all if row["status"] == "RESOLVED"]
    if resolved_rows:
        max_range = max(row["effective_range_km"] for row in resolved_rows)
        for i, row in enumerate(selected_all):
            if row["status"] == "RESOLVED":
                ax.barh(i, row["effective_range_km"])
            else:
                ax.text(0.02 * max_range, i, "RANGE_UNRESOLVED", va="center")
        ax.set_xlabel("Effective range (km)")
    else:
        for i in range(len(selected_all)):
            ax.text(0.05, i, "RANGE_UNRESOLVED", va="center")
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.spines[["bottom", "top", "right"]].set_visible(False)
        ax.set_title("No target has a resolved effective range")
    ax.set(yticks=range(len(TARGETS)), yticklabels=TARGETS)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "figures/effective_range_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, POPULATIONS, strict=True):
        sample = pq.read_table(output / "samples" / f"{name}.parquet", columns=["x_5070", "y_5070"])
        ax.scatter(sample["x_5070"].to_numpy() / 1000, sample["y_5070"].to_numpy() / 1000, s=1, alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel("EPSG:5070 x (km)")
    axes[0].set_ylabel("EPSG:5070 y (km)")
    fig.tight_layout()
    fig.savefig(output / "figures/sample_locations.png", dpi=160)
    plt.close(fig)
    (output / "run_metadata.json").write_text(json.dumps({"seed": 42, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "sample_supports": EXPECTED, "selected_status": selected}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "42":
        parser.error("Set PYTHONHASHSEED=42")
    start = time.perf_counter()
    output = args.repo / REPORT
    (output / "figures").mkdir(parents=True, exist_ok=True)
    validate_contract(args.repo)
    analyze_samples(output)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    telemetry_path = output / "runtime_telemetry.json"
    telemetry_path.write_text(json.dumps({"run2_automap_style": {
        "wall_seconds": elapsed, "peak_resident_kib": peak,
    }}, indent=2, sort_keys=True) + "\n")
    print(f"Elapsed seconds: {elapsed:.3f}; peak RSS KiB: {peak}")


if __name__ == "__main__":
    main()
