"""Automap-aligned empirical lags and weighted SciKit-GStat model fits."""

from __future__ import annotations

import warnings

import numpy as np
from skgstat import Variogram

MODELS = ("spherical", "exponential", "gaussian", "matern")
TARGETS = ("bp", *(f"flp{i}" for i in range(1, 7)), "conditional_flep8", "unconditional_flep8")
BOUNDARY_PERCENTAGES = np.array([2, 4, 6, 9, 12, 15, 25, 35, 50, 65, 80, 100], dtype=np.float64)
MIN_PAIRS = 5  # automap autofitVariogram default min.np.bin
FIT_FIELDS = (
    "target", "sample_size", "model", "nugget", "partial_sill", "total_sill",
    "nugget_to_total_sill", "effective_range_m", "effective_range_km", "smoothness",
    "weighted_sse", "fit_rss", "fit_rmse", "fit_r2", "max_lag_m",
    "range_resolved", "supported_sill", "selected", "fit_status",
)
EMPIRICAL_FIELDS = (
    "target", "lag_index", "lower_boundary_m", "upper_boundary_m", "lower_boundary_km",
    "upper_boundary_km", "mean_distance_m", "mean_distance_km", "semivariance", "pair_count",
    "fit_weight_n_over_h2", "fit_sigma",
)
BOUNDARY_FIELDS = (
    "population", "boundary_index", "percentage", "bbox_diagonal_m", "boundary_m",
    "boundary_km", "retained_after_merge", "minimum_pairs",
)
SELECTED_FIELDS = (
    "target", "sample_size", "population", "model", "status", "nugget", "partial_sill",
    "total_sill", "nugget_to_total_sill", "effective_range_m", "effective_range_km",
    "weighted_sse", "fit_rmse", "fit_r2", "max_lag_m", "supported_sill",
)


def matheron(differences: np.ndarray) -> float:
    """Classical empirical semivariance for a known pair set."""
    return float(np.sum(np.square(differences, dtype=np.float64)) / (2 * len(differences)))


def automap_boundaries(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Projected-coordinate rule in automap's autofitVariogram."""
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        raise ValueError("Expected at least two 2D coordinates")
    diagonal = float(np.linalg.norm(np.max(coords, axis=0) - np.min(coords, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Invalid sampled-location bounding box")
    return diagonal, BOUNDARY_PERCENTAGES * (0.35 * diagonal) / 100.0


def fitting_sigma(pair_counts: np.ndarray, mean_distances: np.ndarray,
                  semivariance: np.ndarray) -> np.ndarray:
    """Curve-fit sigma with N_k / h_k² weights and stable numerical scale."""
    counts = np.asarray(pair_counts, dtype=np.float64)
    distances = np.asarray(mean_distances, dtype=np.float64)
    if (not np.all(np.isfinite(distances)) or np.any(distances <= 0)
            or np.any(counts < MIN_PAIRS)):
        raise ValueError("Unsupported empirical lag for weighted fitting")
    scale = float(np.median(np.abs(semivariance)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Zero or invalid empirical semivariance scale")
    raw = distances / np.sqrt(counts)
    return raw / np.median(raw) * scale


def weighted_sse(residuals: np.ndarray, pair_counts: np.ndarray, mean_distances: np.ndarray) -> float:
    """The unscaled automap/gstat N_k / h_k² residual objective."""
    return float(np.sum(pair_counts * np.square(residuals / mean_distances)))


def resolved(effective_range: float, max_lag: float, partial_sill: float,
             nugget: float = 0, fit_r2: float = 0, supported_bins: int = 2,
             tail_discrepancy: float = 0, fit_rmse: float = 1) -> bool:
    """Require finite parameters, interior range, and supported empirical sill."""
    return bool(
        np.all(np.isfinite([effective_range, max_lag, partial_sill, nugget, fit_r2,
                            tail_discrepancy, fit_rmse]))
        and partial_sill > 0 and nugget >= 0 and effective_range > 0
        and effective_range < 0.9 * max_lag
        and supported_bins >= 2 and fit_r2 >= 0
        and tail_discrepancy <= 2 * fit_rmse
    )


def _unfitted_variogram(coords: np.ndarray, values: np.ndarray, boundaries: np.ndarray) -> Variogram:
    # Manual construction preserves empirical lags even if every fit fails.
    # These placeholder parameters are never reported as a fitted model.
    return Variogram(
        coords, values, estimator="matheron", model="spherical",
        dist_func="euclidean", bin_func=boundaries, normalize=False,
        fit_method="manual", fit_range=1, fit_sill=1, fit_nugget=0,
        use_nugget=True,
    )


def empirical_variogram(coords: np.ndarray, values: np.ndarray,
                        boundaries: np.ndarray, target: str) -> tuple[Variogram, list[dict], np.ndarray, np.ndarray]:
    """Merge small bins by dropping earliest boundaries, as automap does."""
    current = boundaries.copy()
    while True:
        v = _unfitted_variogram(coords, values, current)
        counts = np.asarray(v.bin_count)
        if np.all(counts >= MIN_PAIRS):
            break
        if len(current) == 1:
            raise ValueError(f"No supported empirical bins for {target}")
        current = current[1:]
    groups = v.lag_groups()
    distance = v.distance
    inside = (groups >= 0) & (groups < len(current))
    sums = np.bincount(groups[inside], weights=distance[inside], minlength=len(current))
    means = sums / counts
    sigma = fitting_sigma(counts, means, np.asarray(v.experimental))
    lower = np.r_[0.0, current[:-1]]
    rows = [
        {"target": target, "lag_index": i + 1,
         "lower_boundary_m": float(lo), "upper_boundary_m": float(hi),
         "lower_boundary_km": float(lo / 1000), "upper_boundary_km": float(hi / 1000),
         "mean_distance_m": float(mean), "mean_distance_km": float(mean / 1000),
         "semivariance": float(gamma), "pair_count": int(count),
         "fit_weight_n_over_h2": float(count / mean**2), "fit_sigma": float(sig)}
        for i, (lo, hi, mean, gamma, count, sig) in enumerate(zip(
            lower, current, means, v.experimental, counts, sigma, strict=True))
    ]
    return v, rows, counts, means


def _initial_guess(model: str, diagonal: float, empirical: np.ndarray) -> list[float]:
    """Automap initial nugget/range/sill adapted to SciKit effective ranges."""
    nugget = float(np.min(empirical))
    total_sill = float((np.max(empirical) + np.median(empirical)) / 2)
    partial = max(total_sill - nugget, np.finfo(float).eps * max(total_sill, 1.0))
    initial = [0.1 * diagonal, partial]
    if model == "matern":
        initial.append(1.0)  # continuous SciKit smoothness initial value
    initial.append(nugget)
    return initial


def summarize_fit(target: str, n: int, model: str, v: Variogram,
                  pair_counts: np.ndarray, mean_distances: np.ndarray) -> dict:
    description = v.describe()
    lag = np.asarray(v.bins)
    empirical = np.asarray(v.experimental)
    predicted = np.asarray(v.fitted_model(lag))
    residuals = empirical - predicted
    rss = float(np.sum(np.square(residuals)))
    rmse = float(np.sqrt(rss / len(lag)))
    total_variation = float(np.sum(np.square(empirical - empirical.mean())))
    r2 = float(1 - rss / total_variation) if total_variation > 0 else float("nan")
    effective = float(description["effective_range"])
    partial = float(description["sill"])
    nugget = float(description["nugget"])
    smoothness = float(description.get("smoothness", float("nan")))
    total_sill = partial + nugget
    support = (mean_distances >= effective) & (pair_counts >= MIN_PAIRS)
    tail = float(np.mean(np.abs(empirical[support][-2:] - total_sill))) if np.sum(support) >= 2 else float("inf")
    finite = bool(np.all(np.isfinite([rss, effective, partial, nugget, total_sill]))
                  and (model != "matern" or (np.isfinite(smoothness) and smoothness > 0))
                  and len(lag) >= 3)
    supported_sill = finite and resolved(effective, float(np.max(lag)), partial, nugget, r2,
                                          int(np.sum(support)), tail, rmse)
    return {
        "target": target, "sample_size": n, "model": model, "nugget": nugget,
        "partial_sill": partial, "total_sill": total_sill,
        "nugget_to_total_sill": nugget / total_sill if total_sill > 0 else float("nan"),
        "effective_range_m": effective, "effective_range_km": effective / 1000,
        "smoothness": smoothness,
        "weighted_sse": weighted_sse(residuals, pair_counts, mean_distances),
        "fit_rss": rss, "fit_rmse": rmse, "fit_r2": r2, "max_lag_m": float(np.max(lag)),
        "range_resolved": supported_sill, "supported_sill": supported_sill,
        "selected": False, "fit_status": "OK" if finite else "FIT_NONFINITE",
    }


def choose_fit(rows: list[dict]) -> dict | None:
    viable = [r for r in rows if r["fit_status"] == "OK" and np.isfinite(r["weighted_sse"])]
    if not viable:
        return None
    best = min(viable, key=lambda r: (r["weighted_sse"], MODELS.index(r["model"])))
    best["selected"] = True
    return best


def fit_target(target: str, coords: np.ndarray, values: np.ndarray,
               boundaries: np.ndarray, diagonal: float) -> tuple[list[dict], list[dict], dict | None, tuple[np.ndarray, np.ndarray] | None]:
    """Fit four theoretical families to fixed, weighted empirical lags."""
    v, empirical_rows, counts, means = empirical_variogram(coords, values, boundaries, target)
    sigma = fitting_sigma(counts, means, np.asarray(v.experimental))
    v.fit_method = "trf"
    v.fit_sigma = sigma
    fits = []
    curves = {}
    for model in MODELS:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                v.model = model
                v.fit(p0=_initial_guess(model, diagonal, np.asarray(v.experimental)))
            row = summarize_fit(target, len(values), model, v, counts, means)
            curve_distance = np.linspace(0, float(np.max(v.bins)), 400)
            curves[model] = (curve_distance, np.asarray(v.fitted_model(curve_distance)))
        except (RuntimeError, ValueError, ZeroDivisionError, FloatingPointError, OverflowError) as error:
            row = {field: float("nan") for field in FIT_FIELDS}
            row.update(target=target, sample_size=len(values), model=model, selected=False,
                       range_resolved=False, supported_sill=False,
                       fit_status=f"FIT_FAILED:{type(error).__name__}")
        fits.append(row)
    best = choose_fit(fits)
    return empirical_rows, fits, best, curves.get(best["model"]) if best else None
