# Full canonical dataset (Checkpoint 4B)

The pre-split CONUS dataset contains 110,817,231 BP-valid Dillon LF2020 cells on the native 270 m EPSG:5070 grid. Construction used the frozen [Checkpoint 3D specification](dataset_specification.md) and Checkpoint 4A constructor at commit `b4dc1bee7d9a870999323e8ca73eb63b40ad1060`, with seed 42, `PYTHONHASHSEED=42`, and 256-cell tiles. No training split, model preprocessing, or model artifact is part of this checkpoint.

The 3,060 sorted Parquet partitions are in ignored `artifacts/dataset/canonical/partitions/`. Detailed cell-level topology and repair records are in ignored `artifacts/dataset/canonical/run_reports/`. Compact national QA and reproducibility reports are in `reports/dataset/canonical/`, including the partition SHA256 list and full manifest. Runtime measurements are kept in `runtime_telemetry.json` separately from deterministic canonical metadata.

The independent validator rereads the seven native target rasters tile by tile, checks row identity and strict target masks, validates schema and all partition hashes, checks coverage and null accounting, streams spatial diagnostics, and compares all six Checkpoint 4A windows column by column. All 4,359 pilot rows match exactly. The target counts reconcile with [Checkpoint 3D support accounting](../reports/eda/raw/decision_resolution/target_support_accounting.csv).

Three cells have material polygon overlap and coverage excess. Under the existing specification they remain in BP support with null WUI predictors and their implicated source IDs preserved in the ignored row-level topology report. There were no new failure types or rejected geometry repairs.

Rebuild from the repository root with the locked 4A environment and unchanged source archives:

```bash
PYTHONHASHSEED=42 uv run wildfire-create-dataset \
  --full --tile-size 256 --seed 42 \
  --output artifacts/dataset/canonical/partitions \
  --reports artifacts/dataset/canonical/run_reports
PYTHONHASHSEED=42 uv run python -m wildfire_risk_uncertainty.dataset.validate_full
```

The partition hashes in `reports/dataset/canonical/partition_hashes.csv` allow a future rerun to be compared without repeating the full build in this checkpoint.
