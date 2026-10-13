# Canonical dataset pilot (Checkpoint 4A)

The constructor reads seven aligned Dillon LF2020 rasters on their native EPSG:5070, 270 m grid. It reads SILVIS Version 4 polygons through a FileGDB spatial filter, projects them to EPSG:5070, and intersects them with BP-valid cells. The scientific rules remain in [dataset_specification.md](dataset_specification.md).

Reproduce the fixed pilot from the repository root in the locked environment:

```bash
PYTHONHASHSEED=42 uv run --locked wildfire-create-dataset --pilot --seed 42 \
  --output artifacts/dataset/pilot --reports reports/dataset/pilot
PYTHONHASHSEED=42 uv run --locked wildfire-validate-pilot \
  --output artifacts/dataset/pilot --reports reports/dataset/pilot
```

Use `--no-progress` for CI or redirected logs. The constructor also supports `--full --tile-size 256`, but Checkpoint 4A does not authorize running that build. Each window produces one sorted Parquet file. The schema and deterministic manifest are in `reports/dataset/pilot`; Parquet files stay ignored under `artifacts/dataset/pilot`. The manifest names the exact six pilot windows, source checksums, locked package versions, tolerances and writer settings.

The pilot selected fixed geographic examples near Seattle, the Pacific coast, Chicago/Lake Michigan, Denver, and rural Kansas. The Denver window is 64 × 64 cells to exercise denser polygon processing. The other windows are 8 × 8 cells. Selection uses no randomness. A BP-positive anomalous FLP location was not available without another broad source search, so that case is covered synthetically.

Numerical overlap and coverage discrepancies within `1e-6` of cell area are retained without renormalization and recorded in `tolerated_topology_summary.csv`. Invalid source geometry is repaired only by the projected-area-gated `shapely.make_valid` rule in the specification; accepted and rejected OBJECTIDs appear in `geometry_repair_summary.csv`. Material failures remain in `topology_summary.csv` with WUI predictors null. The independent validator rereads all seven native rasters for every pilot cell, verifies row identity and FLP masks, and checks area-share and null accounting.

The pre-policy source investigation is preserved under `reports/dataset/pilot/hitl_topology/`. The constructor has not run the full-CONUS build, created data splits, or trained models.
