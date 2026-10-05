"""Canonical EDA entry point. Source inventory plus Dillon statistical/spatial EDA."""

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pyogrio
import rasterio

from . import dillon, wui


def output_paths(root: Path) -> dict[str, Path]:
    return {name: root / "reports/eda/raw" / name / "tables" for name in ("dillon", "wui")}


def source_state(root: Path, include_wui: bool = True) -> dict:
    """Cheap mutation guard: size and mtime, not a full payload checksum."""
    paths = sorted(p for p in (root / "data/interim/dillon_2023").rglob("*") if p.is_file())
    if include_wui:
        paths += [root / wui.METADATA]
        paths += sorted(p for p in (root / wui.GDB).rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_inventory(root: Path) -> None:
    root = root.resolve()
    before = source_state(root)
    rasters = dillon.inventory(root)
    metadata = wui.metadata(root)
    layers, fields = wui.inventory(root, metadata["attributes"])
    if source_state(root) != before:
        raise RuntimeError("Source size/mtime changed during inspection")
    paths = output_paths(root)
    # Resolve output destinations before writing, rejecting redirects into source data.
    report_root = (root / "reports/eda/raw").absolute()
    for path in paths.values():
        if path.resolve() != path.absolute() or not path.is_relative_to(report_root):
            raise ValueError(f"Output path must not traverse symlinks: {path}")
    outputs = {
        paths["dillon"] / "rasters.json": rasters,
        paths["dillon"] / "alignment.json": dillon.alignment(rasters),
        paths["wui"] / "layers.json": layers,
        paths["wui"] / "metadata.json": metadata,
        paths["wui"] / "fields.csv": None,
    }
    provenance = report_root / "inventory_run.json"
    for path in [*outputs, provenance]:
        if path.is_symlink():
            raise ValueError(f"Output file must not be a symlink: {path}")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for path, data in outputs.items():
        if data is not None:
            write_json(path, data)
    with (paths["wui"] / "fields.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(fields)
    write_json(provenance, {
        "scope": "structural inventory only; no pixel or feature reads",
        "packages": {p: version(p) for p in ("numpy", "rasterio", "pyogrio")},
        "gdal": {"rasterio": rasterio.__gdal_version__, "pyogrio": pyogrio.__gdal_version_string__},
        "metadata_sha256": hashlib.sha256((root / wui.METADATA).read_bytes()).hexdigest(),
        "source_guard": "File sizes and mtimes unchanged during inspection (not content hashes)",
        "sampling": "none; deterministic metadata inventory, no RNG",
    })
    if source_state(root) != before:
        raise RuntimeError("Source size/mtime changed during report generation")
    print("Structural inventory written to reports/eda/raw.")


def run_dillon(root: Path) -> None:
    root = root.resolve()
    before = source_state(root, include_wui=False)
    # Disable persistent GDAL auxiliary metadata writes and bound its block cache.
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=64 * 1024 * 1024):
        result = dillon.analyze(root)
        dillon.write_products(result, root)
    if source_state(root, include_wui=False) != before:
        raise RuntimeError("Dillon source package changed during analysis")
    print("Dillon statistical/spatial products written to reports/eda/raw/dillon.")


def run_raw(root: Path) -> None:
    before = source_state(root)
    run_inventory(root)
    run_dillon(root)
    if source_state(root) != before:
        raise RuntimeError("Source package changed during raw EDA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["raw", "dillon"], help="Source inventory + Dillon EDA, or Dillon EDA only")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: cwd)")
    args = parser.parse_args()
    {"raw": run_raw, "dillon": run_dillon}[args.command](args.root)
