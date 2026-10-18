"""Small source-backed fixtures for the mmap cache and frozen stream semantics."""

import csv

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from wildfire_risk_uncertainty.training import cache as cache_module
from wildfire_risk_uncertainty.training import data as data_module
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    cache_binding,
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.preprocessing import fit_preprocessing


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    rows, parts = 12, 2
    inputs = ("HUDEN2020", "POPDEN2020", "OCCHUDEN2020", "VACHUDEN2020",
              *(f"feature_{j}" for j in range(28)), "wui_failure")
    root = tmp_path
    canonical = root / "artifacts/dataset/canonical/partitions"
    outer = root / "artifacts/spatial_validation/05b/partitions"
    inner = root / "artifacts/spatial_validation/06c/partitions"
    for path in (canonical, outer, inner):
        path.mkdir(parents=True)
    expected = np.zeros(5, dtype=int)
    for part in range(parts):
        ids = np.arange(part * 6, part * 6 + 6, dtype=np.int64)
        features = {name: ids.astype(float) + j for j, name in enumerate(inputs[:-1])}
        features["HUDEN2020"][0] = np.nan
        features["wui_failure"] = (ids % 2 == 0)
        target = {"cell_id": ids, **features, "BP": np.full(6, 0.2),
                  **{f"FLP{k}": np.full(6, 1 / 6) for k in range(1, 7)},
                  "flp_target_valid": ids % 4 != 0}
        target["FLP1"][0] = np.nan
        outer_columns = {"cell_id": ids}
        inner_columns = {"cell_id": ids}
        for study in range(1, 6):
            eligible = ids % 3 != study % 3
            labels = np.where(eligible, (ids + study) % 5 + 1, np.nan)
            outer_columns[f"train_eligible_fold_{study}"] = eligible
            inner_columns[f"inner_fold_outer_{study}"] = labels
            expected[study - 1] += eligible.sum()
        for directory, table in ((canonical, target), (outer, outer_columns), (inner, inner_columns)):
            pq.write_table(pa.table(table), directory / f"part_{part:02d}.parquet")
    report = root / "reports/spatial_validation/05b/tables"
    report.mkdir(parents=True)
    with (report / "fold_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "train_rows"])
        writer.writeheader()
        writer.writerows({"fold": study, "train_rows": count}
                         for study, count in enumerate(expected, start=1))
    monkeypatch.setattr(cache_module, "ROWS", rows)
    monkeypatch.setattr(cache_module, "PARTITIONS", parts)
    monkeypatch.setattr(cache_module, "ARRAYS", {
        "cell_id": (np.int64, (rows,)), "features": (np.float32, (rows, 33)),
        "bp": (np.float32, (rows,)), "flp": (np.float32, (rows, 6)),
        "flp_valid": (np.uint8, (rows,)), "inner_fold": (np.uint8, (rows, 5)),
        "partition_offsets": (np.int64, (parts + 1,)),
    })
    monkeypatch.setattr(cache_module, "source_hashes", lambda _root: {"fixture": "source-v1"})
    monkeypatch.setattr(cache_module, "model_inputs", lambda _path: inputs)
    paths = sorted(canonical.glob("*.parquet"))
    monkeypatch.setattr(cache_module, "inner_partition_paths", lambda *_args: paths)
    monkeypatch.setattr(data_module, "inner_partition_paths", lambda *_args: paths)
    cache_path = root / "artifacts/training/hpo_cache_v1"
    manifest = cache_module.build(root, cache_path)
    return root, cache_path, inputs, manifest


def test_cache_schema_dtypes_offsets_eligibility_and_nan(source_fixture):
    root, cache_path, inputs, manifest = source_fixture
    cache = cache_module.HpoCache(root, cache_path)
    assert manifest["rows"] == 12 and manifest["partitions"] == 2
    assert cache.arrays["features"].dtype == np.float32
    assert cache.arrays["flp"].dtype == np.float32
    assert cache.arrays["inner_fold"].dtype == np.uint8
    assert cache.arrays["partition_offsets"].tolist() == [0, 6, 12]
    assert np.isnan(cache.arrays["features"][0, 0])
    assert np.isnan(cache.arrays["flp"][0, 0])
    assert cache.arrays["flp_valid"][0] == 0
    assert cache.manifest["inputs"] == list(inputs)
    for study in range(1, 6):
        expected = cache.arrays["cell_id"] % 3 != study % 3
        assert np.array_equal(cache.arrays["inner_fold"][:, study - 1] != 0, expected)


@pytest.mark.parametrize("role,epoch", [("train", 1), ("train", 2), ("validation", None)])
def test_exact_reader_order_values_and_preprocessing(source_fixture, role, epoch):
    root, cache_path, inputs, _manifest = source_fixture
    cache = cache_module.HpoCache(root, cache_path)
    args = (root / "artifacts/dataset/canonical/partitions",
            root / "artifacts/spatial_validation/05b/partitions",
            root / "artifacts/spatial_validation/06c/partitions", inputs)
    old = list(data_module.inner_chunks(*args, role=role, epoch=epoch, outer_study=3,
                                        inner_fold=1, include_ids=True))
    new = list(data_module.inner_chunks(*args, role=role, epoch=epoch, outer_study=3,
                                        inner_fold=1, include_ids=True, cache=cache))
    assert len(old) == len(new)
    for left, right in zip(old, new, strict=True):
        for a, b in zip(left, right, strict=True):
            np.testing.assert_array_equal(a, b, strict=True)
    if role == "train":
        old_state = fit_preprocessing((part[:4] for part in old), inputs, "fixture-schema")
        new_state = fit_preprocessing((part[:4] for part in new), inputs, "fixture-schema")
        assert old_state.digest() == new_state.digest()


def test_cache_provenance_and_file_hash_rejection(source_fixture, monkeypatch):
    root, cache_path, _inputs, _manifest = source_fixture
    monkeypatch.setattr(cache_module, "source_hashes", lambda _root: {"fixture": "changed"})
    with pytest.raises(ValueError, match="provenance mismatch"):
        cache_module.HpoCache(root, cache_path)
    monkeypatch.setattr(cache_module, "source_hashes", lambda _root: {"fixture": "source-v1"})
    with (cache_path / "bp.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="file hash mismatch"):
        cache_module.HpoCache(root, cache_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Checkpoint loader uses frozen CUDA device")
def test_checkpoint_rejects_changed_cache_partition_and_preprocessing(source_fixture, tmp_path):
    root, cache_path, inputs, _manifest = source_fixture
    cache = cache_module.HpoCache(root, cache_path)
    args = (root / "artifacts/dataset/canonical/partitions",
            root / "artifacts/spatial_validation/05b/partitions",
            root / "artifacts/spatial_validation/06c/partitions", inputs)
    state = fit_preprocessing(data_module.inner_chunks(*args, role="train", outer_study=3,
                                                         inner_fold=1, cache=cache), inputs, "fixture")
    model = torch.nn.Linear(1, 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "trajectory.pt"
    save_cache_checkpoint(path, model, optimizer, completed_epoch=1, global_step=7,
                          cache=cache, state=state, metadata={"trial": 0})
    assert load_cache_checkpoint(path, model, optimizer, cache=cache, state=state,
                                 expected_metadata={"trial": 0}) == (1, 7)
    original = cache.manifest
    try:
        cache.manifest = {**original, "files_sha256": {
            **original["files_sha256"], "partition_offsets.npy": "changed"}}
        with pytest.raises(ValueError, match="Checkpoint provenance"):
            load_cache_checkpoint(path, model, optimizer, cache=cache, state=state,
                                  expected_metadata={"trial": 0})
    finally:
        cache.manifest = original
    changed_state = type(state)(state.inputs, tuple(x + 1 for x in state.means),
                                state.standard_deviations, state.missing_counts,
                                state.rows, state.flp_valid_rows, state.schema_sha256)
    assert cache_binding(cache, changed_state) != cache_binding(cache, state)
    with pytest.raises(ValueError, match="Checkpoint provenance"):
        load_cache_checkpoint(path, model, optimizer, cache=cache, state=changed_state,
                              expected_metadata={"trial": 0})
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_cache_checkpoint(path, model, optimizer, cache=cache, state=state,
                              expected_metadata={"trial": 0})
