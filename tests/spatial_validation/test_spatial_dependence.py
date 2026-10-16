"""Checkpoint 5A support, sampling, and variogram contract tests."""

from __future__ import annotations

import csv

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from skgstat.estimators import matheron as skg_matheron

from wildfire_risk_uncertainty.spatial_validation.cli import write_csv
from wildfire_risk_uncertainty.spatial_validation.sampling import (
    TARGET_ALIASES,
    PrioritySampler,
    derive_targets,
    sample_partitions,
    support_masks,
)
from wildfire_risk_uncertainty.spatial_validation.variogram import (
    BOUNDARY_FIELDS,
    EMPIRICAL_FIELDS,
    FIT_FIELDS,
    MODELS,
    SELECTED_FIELDS,
    automap_boundaries,
    choose_fit,
    empirical_variogram,
    fit_target,
    fitting_sigma,
    matheron,
    resolved,
    weighted_sse,
)


def small_data() -> dict[str, np.ndarray]:
    data = {
        "cell_id": np.arange(8),
        "x_5070": np.arange(8, dtype=np.float64) * 270,
        "y_5070": np.zeros(8),
        "bp": np.array([0, 1, 1, 0, 2, 3, 0, 4], dtype=np.float32),
        "flp_target_valid": np.array([False, True, False, False, True, True, False, False]),
    }
    for i in range(1, 7):
        data[f"flp{i}"] = np.array([np.nan, i / 21, np.nan, np.nan, i / 21,
                                    i / 21, np.nan, np.nan], dtype=np.float32)
    return data


def test_supports_and_exact_derivation_without_renormalization() -> None:
    data = small_data()
    before = {name: values.copy() for name, values in data.items() if name.startswith("flp")}
    masks = support_masks(data)
    assert masks["bp"].tolist() == [True] * 8
    assert masks["conditional"].tolist() == [False, True, False, False, True, True, False, False]
    assert masks["unconditional"].tolist() == [True, True, False, True, True, True, True, False]
    derived = derive_targets(data)
    expected = data["flp5"] + data["flp6"]
    np.testing.assert_array_equal(derived["conditional_flep8"], expected)
    np.testing.assert_array_equal(derived["unconditional_flep8"][[0, 3, 6]], 0)
    np.testing.assert_array_equal(derived["unconditional_flep8"][[1, 4, 5]],
                                  data["bp"][[1, 4, 5]] * expected[[1, 4, 5]])
    assert np.isnan(derived["unconditional_flep8"][[2, 7]]).all()
    for name, value in before.items():
        np.testing.assert_array_equal(data[name], value)


def test_deterministic_uniform_priority_and_order() -> None:
    data = small_data()
    results = []
    for _ in range(2):
        sampler = PrioritySampler(k=2, seed=42)
        sampler.add({name: value[:4] for name, value in data.items()})
        sampler.add({name: value[4:] for name, value in data.items()})
        results.append({name: sampler.result(name)["cell_id"] for name in sampler.selected})
        assert sampler.counts == {"bp": 8, "conditional": 3, "unconditional": 6}
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name])
        assert np.all(np.diff(results[0][name]) > 0)


def test_alias_read_preserves_physical_float_bits(tmp_path) -> None:
    data = small_data()
    physical = {TARGET_ALIASES.get(name, name): data[name] for name in data}
    source = {name: data[alias] for name, alias in TARGET_ALIASES.items()}
    source.update({name: data[name] for name in ("cell_id", "x_5070", "y_5070", "flp_target_valid")})
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.table(source), path)
    sampler = sample_partitions([path], k=8)
    result = sampler.result("bp")
    for name in ("bp", *(f"flp{i}" for i in range(1, 7))):
        assert result[name].dtype == physical[name].dtype
        np.testing.assert_array_equal(result[name].view(np.uint32), physical[name].view(np.uint32))


def test_matheron_known_pairs() -> None:
    differences = np.array([2.0, 4.0, 2.0])
    expected = (4 + 16 + 4) / 6
    assert matheron(differences) == expected
    assert skg_matheron(differences) == expected


def test_automap_boundaries_are_deterministic() -> None:
    coords = np.array([[0.0, 0.0], [3000.0, 4000.0], [1500.0, 1000.0]])
    diagonal, bounds = automap_boundaries(coords)
    assert diagonal == 5000.0
    np.testing.assert_allclose(bounds, [35, 70, 105, 157.5, 210, 262.5,
                                        437.5, 612.5, 875, 1137.5, 1400, 1750])
    np.testing.assert_array_equal(automap_boundaries(coords)[1], bounds)


def test_automap_short_distance_bin_merging() -> None:
    coords = np.column_stack((np.arange(20, dtype=float), np.zeros(20)))
    values = np.sin(coords[:, 0])
    _, rows, counts, _ = empirical_variogram(coords, values, np.array([0.5, 1.5, 3.5, 8.0]), "toy")
    assert rows[0]["lower_boundary_m"] == 0
    assert rows[0]["upper_boundary_m"] == 1.5
    assert len(rows) == 3
    assert np.all(counts >= 5)


def test_weighted_sigma_matches_n_over_h_squared() -> None:
    counts = np.array([5, 20, 20])
    distances = np.array([1.0, 2.0, 4.0])
    sigma = fitting_sigma(counts, distances, np.array([0.5, 1.0, 1.5]))
    np.testing.assert_allclose((1 / sigma**2) / (1 / sigma[0]**2),
                               (counts / distances**2) / (counts[0] / distances[0]**2))
    residuals = np.array([1.0, 2.0, -1.0])
    assert weighted_sse(residuals, counts, distances) == 26.25
    np.testing.assert_allclose(np.sum((residuals / sigma)**2),
                               weighted_sse(residuals, counts, distances) *
                               (np.median(distances / np.sqrt(counts)) / 1.0)**2)


def test_model_families_and_nugget_enabled_fit() -> None:
    assert MODELS == ("spherical", "exponential", "gaussian", "matern")
    coords = np.column_stack((np.arange(50, dtype=float), np.zeros(50)))
    values = np.sin(coords[:, 0] / 8) + 0.02 * np.cos(coords[:, 0])
    diagonal, boundaries = automap_boundaries(coords)
    empirical, fits, best, _ = fit_target("toy", coords, values, boundaries, diagonal)
    assert len(empirical) >= 3
    assert best is not None
    assert [row["model"] for row in fits] == list(MODELS)
    assert all(row["fit_status"] == "OK" for row in fits)
    assert all(np.isfinite(row["nugget"]) and row["nugget"] >= 0 for row in fits)
    assert all(np.isclose(row["total_sill"], row["partial_sill"] + row["nugget"]) for row in fits)


def test_weighted_sigma_is_used_by_synthetic_variogram_fit() -> None:
    coords = np.column_stack((np.arange(50, dtype=float), np.zeros(50)))
    values = np.sin(coords[:, 0] / 8) + 0.02 * np.cos(coords[:, 0])
    diagonal, boundaries = automap_boundaries(coords)
    v, _, counts, means = empirical_variogram(coords, values, boundaries, "toy")
    sigma = fitting_sigma(counts, means, np.asarray(v.experimental))
    v.fit_method = "trf"
    v.fit_sigma = sigma
    v.model = "spherical"
    v.fit(p0=[0.1 * diagonal, 0.5, 0.1])
    np.testing.assert_array_equal(v.fit_sigma, sigma)
    residuals = v.experimental - v.fitted_model(v.bins)
    factor = (np.median(means / np.sqrt(counts)) / np.median(np.abs(v.experimental))) ** 2
    np.testing.assert_allclose(np.sum((residuals / sigma) ** 2),
                               weighted_sse(residuals, counts, means) * factor)


def test_range_boundary_and_model_selection() -> None:
    assert resolved(800.0, 1000.0, 1.0)
    assert not resolved(900.0, 1000.0, 1.0)
    assert not resolved(float("nan"), 1000.0, 1.0)
    assert not resolved(800.0, 1000.0, 0.0)
    assert not resolved(800.0, 1000.0, 1.0, supported_bins=1)
    assert not resolved(800.0, 1000.0, 1.0, fit_r2=-0.1)
    assert not resolved(800.0, 1000.0, 1.0, tail_discrepancy=3, fit_rmse=1)
    rows = [
        {"model": model, "fit_status": "OK", "weighted_sse": float(i + 1), "selected": False}
        for i, model in enumerate(MODELS)
    ]
    assert choose_fit(rows) is rows[0]
    assert sum(row["selected"] for row in rows) == 1
    assert choose_fit([{"model": "spherical", "fit_status": "FIT_FAILED", "weighted_sse": float("nan")}]) is None


def test_report_schemas(tmp_path) -> None:
    for fields in (BOUNDARY_FIELDS, EMPIRICAL_FIELDS, FIT_FIELDS, SELECTED_FIELDS):
        path = tmp_path / f"{fields[0]}_{len(fields)}.csv"
        write_csv(path, fields, [{field: field for field in fields}])
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            assert tuple(reader.fieldnames) == fields
            assert next(reader) == {field: field for field in fields}
