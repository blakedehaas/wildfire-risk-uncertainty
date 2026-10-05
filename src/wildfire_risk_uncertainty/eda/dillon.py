"""Read raster headers only; never load pixel arrays."""

import math
from pathlib import Path

import rasterio

DILLON_DIR = Path("data/interim/dillon_2023/Data/I_FSim_CONUS_LF2020_270m")
FILENAMES = ("CONUS_BP.tif", *(f"CONUS_FLP{i}.tif" for i in range(1, 7)))


def inventory(root: Path) -> list[dict]:
    rows = []
    for name in FILENAMES:
        path = root / DILLON_DIR / name
        with rasterio.open(path, "r") as src:
            rows.append({
                "filename": name, "path": path.relative_to(root).as_posix(),
                "crs": src.crs.to_string() if src.crs else None,
                "crs_wkt": src.crs.to_wkt() if src.crs else None,
                "width": src.width, "height": src.height,
                "transform": list(src.transform)[:6],
                "resolution": list(src.res), "bounds": list(src.bounds),
                "dtypes": list(src.dtypes), "band_count": src.count,
                "nodata": [
                    "NaN" if v is not None and math.isnan(v) else v
                    for v in src.nodatavals
                ],
                "overviews": [src.overviews(i) for i in src.indexes],
            })
    return rows


def alignment(rows: list[dict]) -> dict[str, bool]:
    """Exact comparisons, without rounding; grid means CRS + shape + transform."""
    if not rows:
        raise ValueError("Cannot compare an empty raster inventory")
    groups = {
        "crs": ("crs_wkt",), "shape": ("width", "height"),
        "transform": ("transform",), "bounds": ("bounds",),
        "nodata_convention": ("nodata",),
        "grid": ("crs_wkt", "width", "height", "transform"),
    }
    return {
        name: all(all(row[key] == rows[0][key] for key in keys) for row in rows)
        for name, keys in groups.items()
    }
