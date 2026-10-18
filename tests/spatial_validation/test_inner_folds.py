"""Synthetic checks of the frozen nested coordinate-only partition mechanics."""

import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wildfire_risk_uncertainty.spatial_validation.folds import (
    fit_coordinate_sample,
    nearest_centroid,
    relabel_centroids,
    uniform_positions,
)
from wildfire_risk_uncertainty.spatial_validation.inner_cli import (
    add_bucket,
    new_bucket,
    training_complement,
)
from wildfire_risk_uncertainty.spatial_validation.inner_folds import (
    aligned_tiles,
    digest,
    eligible_labels,
    inner_assignment_table,
    persist_sample,
    sample_eligible_coordinates,
)


def fixture_tiles(tmp_path):
    tiles = []
    for tile, flags in enumerate(([True, False, True, True, False, True],
                                  [False, True, True, False, True, False])):
        base = tile * 10
        ids = np.arange(base, base + 6, dtype=np.int64)
        canonical = tmp_path / f"canonical_{tile}.parquet"
        outer = tmp_path / f"outer_{tile}.parquet"
        pq.write_table(pa.table({"cell_id": ids, "x_5070": ids.astype(float),
                                 "y_5070": -ids.astype(float),
                                 "BP": np.arange(6, dtype=float)}), canonical)
        pq.write_table(pa.table({"cell_id": ids, "train_eligible_fold_1": flags}), outer)
        tiles.append((canonical, outer))
    return tiles


def test_aligned_tile_order_is_stable(tmp_path):
    canonical = tmp_path / "canonical"
    outer = tmp_path / "outer"
    canonical.mkdir()
    outer.mkdir()
    for name in ("tile_z.parquet", "tile_a.parquet"):
        pq.write_table(pa.table({"cell_id": [1]}), canonical / name)
        pq.write_table(pa.table({"cell_id": [1]}), outer / name)
    assert [pair[0].name for pair in aligned_tiles(canonical, outer, expected_partitions=2)] == [
        "tile_a.parquet", "tile_z.parquet"
    ]


def test_uniform_eligible_rank_sample_and_reproduction(tmp_path):
    tiles = fixture_tiles(tmp_path)
    eligible_ids = np.array([0, 2, 3, 5, 11, 12, 14])
    expected_positions = sorted(random.Random(42).sample(range(7), 4))
    assert np.array_equal(uniform_positions(7, 4), expected_positions)
    ids, xy = sample_eligible_coordinates(tiles, 1, 7, 4)
    assert np.array_equal(ids, eligible_ids[expected_positions])
    assert np.array_equal(xy[:, 0], ids.astype(float))
    regenerated_ids, regenerated_xy = sample_eligible_coordinates(tiles, 1, 7, 4)
    assert np.array_equal(ids, regenerated_ids) and np.array_equal(xy, regenerated_xy)
    first_hash = persist_sample(ids, tmp_path / "sample1.parquet")
    assert first_hash == persist_sample(regenerated_ids, tmp_path / "sample2.parquet")
    assert first_hash == digest(tmp_path / "sample1.parquet")
    with pytest.raises(ValueError, match="eligible count"):
        sample_eligible_coordinates(tiles, 1, 8, 4)


def test_five_coordinate_centroids_and_stable_relabeling():
    rng = np.random.default_rng(19)
    centers = np.array([[-10, 0], [0, 10], [10, 0], [20, 5], [30, -5]], dtype=float)
    xy = np.vstack([center + rng.normal(0, 0.1, size=(30, 2)) for center in centers])
    fitted = fit_coordinate_sample(xy)
    assert fitted.shape == (5, 2)
    assert np.array_equal(fitted, fit_coordinate_sample(xy))
    assert np.all(np.diff(fitted[:, 0]) >= 0)
    ordered, mapping = relabel_centroids(np.array([[2, 4], [1, 9], [1, 3], [3, 0], [0, 0]], dtype=float))
    assert ordered.tolist() == [[0, 0], [1, 3], [1, 9], [2, 4], [3, 0]]
    assert mapping.tolist() == [4, 3, 2, 5, 1]
    assert nearest_centroid(np.array([[0, 0], [2, 4]], dtype=float), ordered).tolist() == [1, 4]
    with pytest.raises(ValueError, match="only x_5070"):
        fit_coordinate_sample(np.zeros((10, 3)))


def test_assignment_schema_eligibility_and_target_independence(tmp_path):
    ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    xy = np.array([[0, 0], [1, 0], [5, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
    centers = np.array([[0, 0], [5, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
    eligible = np.array([True, False, True, True, False, True])
    labels = eligible_labels(xy, eligible, centers)
    assert labels.tolist() == [1, 0, 2, 3, 0, 5]
    assert np.count_nonzero(labels) == eligible.sum()
    assert not np.any(labels[~eligible])
    assert sum(np.count_nonzero(labels == k) for k in range(1, 6)) == eligible.sum()
    for k in range(1, 6):
        validation = labels == k
        training = eligible & ~validation
        assert not np.any(validation & training)
        assert np.array_equal(validation | training, eligible)
        assert not np.any((validation | training) & ~eligible)
    table = inner_assignment_table(ids, {j: labels for j in range(1, 6)})
    assert table.column_names == ["cell_id", *(f"inner_fold_outer_{j}" for j in range(1, 6))]
    assert table["inner_fold_outer_1"].null_count == 2
    assert table.schema.field("inner_fold_outer_1").type == pa.uint8()
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(table, first, compression="zstd")
    pq.write_table(inner_assignment_table(ids, {j: labels for j in range(1, 6)}), second, compression="zstd")
    assert digest(first) == digest(second)
    bp = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    valid = np.array([False, True, True, True, False, True])
    bucket = new_bucket()
    add_bucket(bucket, labels == 2, xy, bp, np.ones(6), np.zeros(6), valid)
    assert bucket["rows"] == 1 and bucket["bp_positive"] == 1
    assert np.array_equal(labels, eligible_labels(xy, eligible, centers))
    altered_bp = bp * 1000
    assert np.array_equal(labels, eligible_labels(xy, eligible, centers))
    assert not np.array_equal(bp, altered_bp)
    total = new_bucket()
    total.update({"rows": 4, "bp_positive": 4, "flp_valid": 3})
    train = training_complement(total, bucket)
    assert train["rows"] == 3
