"""Small exact geometry and deterministic clustering contracts for 5B."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wildfire_risk_uncertainty.spatial_validation.dead_zone import fold_geometry
from wildfire_risk_uncertainty.spatial_validation.folds import (
    accepted_bp_radius,
    column_alias_audit,
    fit_centroids,
    nearest_centroid,
    relabel_centroids,
    uniform_coordinate_sample,
)
from wildfire_risk_uncertainty.spatial_validation.outer_cli import (
    finish_stats,
    stats_bucket,
)
from wildfire_risk_uncertainty.spatial_validation.validate_folds import REQUIRED_COLUMNS


def write_bp_report(root: Path, status: str = "RESOLVED") -> None:
    report = root / "reports/spatial_dependence/05a"
    (report / "tables").mkdir(parents=True, exist_ok=True)
    with (report / "tables/variogram_selected_ranges.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target", "status", "effective_range_m", "effective_range_km"])
        writer.writeheader()
        writer.writerow({"target": "bp", "status": status, "effective_range_m": "211150.99027309995", "effective_range_km": "211.15099027309995"})
    (report / "run_metadata.json").write_text(json.dumps({"selected_status": {"bp": status}}))


def test_exact_accepted_bp_radius_and_unresolved_failure(tmp_path: Path) -> None:
    write_bp_report(tmp_path)
    assert accepted_bp_radius(tmp_path) == 211150.99027309995
    write_bp_report(tmp_path, "RANGE_UNRESOLVED")
    with pytest.raises(ValueError, match="unresolved"):
        accepted_bp_radius(tmp_path)


def test_stable_relabeling_and_coordinate_only_assignment() -> None:
    centers = np.array([[9., 3.], [1., 4.], [1., 2.], [5., 0.], [7., 0.]])
    ordered, mapping = relabel_centroids(centers)
    assert ordered.tolist() == [[1., 2.], [1., 4.], [5., 0.], [7., 0.], [9., 3.]]
    assert mapping.tolist() == [5, 2, 1, 3, 4]
    xy = np.array([[1., 2.], [9., 3.], [5., 0.]])
    assert nearest_centroid(xy, ordered).tolist() == [1, 5, 3]
    with pytest.raises(ValueError, match="only"):
        nearest_centroid(np.column_stack((xy, [100., -100., 0.])), ordered)


def test_deterministic_minibatch_clustering_and_output_order(tmp_path: Path) -> None:
    ids = np.arange(10_100, dtype=np.int64)
    xy = np.column_stack(((ids % 101) * 300., (ids // 101) * 300.))
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"cell_id": ids, "x_5070": xy[:, 0], "y_5070": xy[:, 1], "BP": np.arange(len(ids))}), path)
    selected_ids, selected_xy = uniform_coordinate_sample([path], 100)
    expected_positions = sorted(random.Random(42).sample(range(len(ids)), 100))
    assert selected_ids.tolist() == expected_positions
    assert np.array_equal(selected_xy, xy[expected_positions])
    first, count = fit_centroids([path], sample_size=100)
    second, count2 = fit_centroids([path], sample_size=100)
    pq.write_table(pa.table({"cell_id": ids, "x_5070": xy[:, 0], "y_5070": xy[:, 1], "BP": -np.arange(len(ids))}), path)
    changed_target, _ = fit_centroids([path], sample_size=100)
    assert count == count2 == 100
    assert np.array_equal(first, second)
    assert np.array_equal(first, changed_target)
    assert np.all(np.diff(first[:, 0]) >= 0)
    labels = nearest_centroid(xy, first)
    assert len(labels) == len(ids) and np.all((1 <= labels) & (labels <= 5))


def test_column_alias_audit() -> None:
    audit = column_alias_audit(np.arange(101, dtype=np.int64))
    assert audit["column_mod_101_nonempty_residues"] == 101
    assert audit["column_mod_101_min_count"] == 1
    with pytest.raises(ValueError, match="aliases"):
        column_alias_audit(np.arange(101, dtype=np.int64) * 101)


def test_exact_dead_zone_holes_and_nearest_training() -> None:
    support = np.ones((3, 7), dtype=bool)
    support[1, 3] = False  # A raster hole is background, never a distance barrier.
    folds = np.full((3, 7), 2, dtype=np.uint8)
    folds[1, 1] = 1
    test, dead, train, nearest = fold_geometry(support, folds, 1, radius_m=540.)
    assert test[1, 1] and not train[1, 1]
    assert dead[1, 3] == 0 and not train[1, 3]
    assert dead[1, 2] and dead[1, 0]
    assert train[1, 4]
    assert train[1, 5] and nearest[1, 1] == pytest.approx(270 * np.sqrt(5))
    assert nearest[1, 1] > 540.
    assert np.all((test.astype(int) + dead.astype(int) + train.astype(int))[support] == 1)
    # Exactly r lies in the dead zone even if that point is canonical.
    assert not dead[1, 3]


def test_exact_radius_boundary() -> None:
    support = np.ones((1, 5), dtype=bool)
    folds = np.array([[1, 2, 2, 2, 2]], dtype=np.uint8)
    test, dead, train, distance = fold_geometry(support, folds, 1, 540.)
    assert test[0, 0] and dead[0, 2] and not train[0, 2]
    assert train[0, 3] and distance[0, 0] == 810.


def test_artifact_and_report_schema(tmp_path: Path) -> None:
    fields = {"cell_id": pa.array([1], type=pa.int64()), "outer_fold": pa.array([1], type=pa.uint8()),
              "oof_nearest_training_distance_m": pa.array([810.], type=pa.float64())}
    fields.update({f"train_eligible_fold_{j}": pa.array([j != 1], type=pa.bool_()) for j in range(1, 6)})
    path = tmp_path / "artifact.parquet"
    pq.write_table(pa.table(fields), path)
    assert pq.read_schema(path).names == REQUIRED_COLUMNS
    report_row = finish_stats(1, "test", stats_bucket())
    assert {"fold", "role", "rows", "bp_zero", "bp_positive", "flp_target_valid", "bp_mean", "bp_sample_estimated_p5", "conditional_flep8_mean", "unconditional_flep8_sample_estimated_p5", "unconditional_flep8_defined", "wui_mean_covered_fraction", "wui_failure_rows"} <= report_row.keys()
    assert "wui_full_rows" not in report_row
