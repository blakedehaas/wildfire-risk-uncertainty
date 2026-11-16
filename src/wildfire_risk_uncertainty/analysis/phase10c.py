"""Phase 10C: sampled DEAD_ZONE inference and spatial reliability analysis."""
import argparse
import csv
import itertools
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from wildfire_risk_uncertainty.analysis.phase10a import (
    exact_tail_mask,
    rejection_metrics,
)
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.phase9 import (
    PHASE8,
    ROOT,
    RUN,
    infer,
    models_for,
    state_for,
    validate_targets,
)

OUT = ROOT / "artifacts/evaluation/phase10c/phase10c-20261116-v1"
REPORT = ROOT / "reports/uncertainty/10c"
FOLD_DIR = ROOT / "artifacts/spatial_validation/05b/partitions"
CANONICAL_DIR = ROOT / "artifacts/dataset/canonical/partitions"
HEIGHT, WIDTH, CELL_M = 11283, 17372, 270.0
SAMPLE_DENOMINATOR = 100
DISTANCE_BINS_KM = (0, 25, 50, 100, 150, 211.1509902731, 300, 500, 750, 1000, 2000)
BP_BINS = (0, .001, .005, .01, 1.000001)
BATCH = 8192


def cache_for():
    return HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)


def sample_mask(ids, study):
    """SplitMix64 Bernoulli ~1% selection, stable across tiles and runs."""
    z = np.asarray(ids, dtype=np.uint64) + np.uint64((0x9E3779B97F4A7C15 * study) & ((1 << 64) - 1))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z % np.uint64(SAMPLE_DENOMINATOR) == 0


def paths(study, zone):
    return OUT / "sample_plan" / f"outer_{study}_{zone}.parquet"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def fold_counts():
    with (ROOT / "reports/spatial_validation/05b/tables/fold_summary.csv").open(newline="") as handle:
        return {int(r["fold"]): {k: int(r[k]) for k in ("test_rows", "dead_zone_rows", "train_rows")}
                for r in csv.DictReader(handle)}


def audit(cache):
    if not (RUN / "predictions_complete.json").exists():
        raise ValueError("Complete Phase 9 TEST exports are required")
    phase9_gate = json.loads((RUN / "predictions_complete.json").read_text())
    if phase9_gate["studies"] != 5:
        raise ValueError("Incomplete Phase 9 TEST exports")
    counts = fold_counts()
    manifest = json.loads((PHASE8 / "manifest.json").read_text())
    for study in range(1, 6):
        state = state_for(study, cache)
        if state.rows != counts[study]["train_rows"] or cache.manifest["outer_train_counts"][study-1] != state.rows:
            raise ValueError("Preprocessing was not fitted on exact outer TRAIN support")
        for member in range(5):
            path = PHASE8 / "models" / f"outer_{study}_member_{member}" / "checkpoints/epoch_100.pt"
            if not path.is_file() or not path.with_suffix(".pt.sha256.json").is_file():
                raise ValueError("Missing frozen ensemble member")
    return counts, manifest, phase9_gate


def prepare():
    cache = cache_for()
    counts, _manifest, gate = audit(cache)
    if (OUT / "sample_plan/complete.json").exists():
        raise ValueError("Sample plan already complete; reuse it")
    started = time.perf_counter()
    columns = ("global_row", "tile_index", "prediction_row", "cell_id", "row", "column", "x_5070", "y_5070", "phase5b_test_distance_m")
    pieces = {(study, zone): {name: [] for name in columns} for study in range(1, 6) for zone in ("dead", "test")}
    totals = {study: {"test": 0, "dead": 0, "train": 0} for study in range(1, 6)}
    offsets = cache.arrays["partition_offsets"]
    for tile, name in enumerate(cache.names):
        start, stop = map(int, offsets[tile:tile+2])
        f = pq.read_table(FOLD_DIR / name, columns=["cell_id", "outer_fold", "oof_nearest_training_distance_m",
            *(f"train_eligible_fold_{s}" for s in range(1, 6))])
        c = pq.read_table(CANONICAL_DIR / name, columns=["cell_id", "row", "column", "x_5070", "y_5070"])
        ids = f["cell_id"].to_numpy()
        if not np.array_equal(ids, c["cell_id"].to_numpy()) or not np.array_equal(ids, cache.arrays["cell_id"][start:stop]):
            raise ValueError("5B/canonical/cache cell ID mismatch")
        outer = f["outer_fold"].to_numpy()
        if np.any((outer < 1) | (outer > 5)):
            raise ValueError("Invalid frozen outer fold")
        row = c["row"].to_numpy(); col = c["column"].to_numpy()
        if np.any((row < 0) | (row >= HEIGHT) | (col < 0) | (col >= WIDTH)):
            raise ValueError("Canonical grid coordinate out of bounds")
        for study in range(1, 6):
            eligible = f[f"train_eligible_fold_{study}"].to_numpy()
            if not np.array_equal(eligible, cache.arrays["inner_fold"][start:stop, study-1] != 0):
                raise ValueError("Outer TRAIN eligibility differs from cached training traversal")
            test = outer == study
            dead = ~test & ~eligible
            if np.any(test & eligible) or int(test.sum()+dead.sum()+eligible.sum()) != len(ids):
                raise ValueError("TRAIN/TEST/DEAD_ZONE partition invalid")
            totals[study]["test"] += int(test.sum()); totals[study]["dead"] += int(dead.sum()); totals[study]["train"] += int(eligible.sum())
            selected = sample_mask(ids, study)
            test_rank = np.cumsum(test, dtype=np.int32) - 1
            for zone, membership in (("dead", dead), ("test", test)):
                choose = selected & membership
                if not choose.any():
                    continue
                p = pieces[(study, zone)]
                p["global_row"].append(np.flatnonzero(choose).astype(np.int64) + start)
                p["tile_index"].append(np.full(int(choose.sum()), tile, dtype=np.int16))
                p["prediction_row"].append(test_rank[choose] if zone == "test" else np.full(int(choose.sum()), -1, dtype=np.int32))
                for key, value in (("cell_id", ids), ("row", row), ("column", col),
                                   ("x_5070", c["x_5070"].to_numpy()), ("y_5070", c["y_5070"].to_numpy())):
                    p[key].append(value[choose])
                distance = f["oof_nearest_training_distance_m"].to_numpy()[choose] if zone == "test" else np.full(int(choose.sum()), np.nan)
                p["phase5b_test_distance_m"].append(distance)
        if tile % 500 == 0:
            print(f"sample plan tile {tile}/{len(cache.names)}", flush=True)
    results = {}
    for study in range(1, 6):
        if totals[study] != {"test": counts[study]["test_rows"], "dead": counts[study]["dead_zone_rows"], "train": counts[study]["train_rows"]}:
            raise ValueError("Fold totals do not reproduce frozen 5B report")
        for zone in ("dead", "test"):
            data = {name: np.concatenate(pieces[(study, zone)][name]) for name in columns}
            if len(np.unique(data["cell_id"])) != len(data["cell_id"]):
                raise ValueError("Duplicate sampled cell ID within study and zone")
            path = paths(study, zone)
            write_table(path, pa.table(data))
            results[f"outer_{study}_{zone}"] = {"population_n": totals[study][zone], "sample_n": len(data["cell_id"]), "sha256": sha256(path)}
    write_json(OUT / "sample_plan/complete.json", {"method": "SplitMix64(cell_id, study) modulo 100 equals zero",
        "sample_denominator": SAMPLE_DENOMINATOR, "phase9_manifest_sha256": gate["manifest_sha256"],
        "phase5b_manifest_sha256": sha256(ROOT / "artifacts/spatial_validation/05b/manifest.json"),
        "cache_manifest_sha256": sha256(cache.path / "manifest.json"), "seconds": time.perf_counter()-started,
        "studies": results})
    print("Sample plan complete", flush=True)


def plan_table(study, zone):
    marker = json.loads((OUT / "sample_plan/complete.json").read_text())
    path = paths(study, zone)
    if sha256(path) != marker["studies"][f"outer_{study}_{zone}"]["sha256"]:
        raise ValueError("Sample plan hash mismatch")
    return pq.read_table(path)


def distance_path(study):
    return OUT / "distances" / f"outer_{study}.npy"


def compute_distances(study, cache):
    plan = plan_table(study, "dead")
    out = distance_path(study)
    sidecar = out.with_suffix(".json")
    if out.exists() and sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if metadata["plan_sha256"] != sha256(paths(study, "dead")) or metadata["sha256"] != sha256(out):
            raise ValueError("Recovered distance artifact hash mismatch")
        return np.load(out, mmap_mode="r"), metadata
    started = time.perf_counter()
    train = np.zeros((HEIGHT, WIDTH), dtype=bool)
    count = 0
    for name in cache.names:
        f = pq.read_table(FOLD_DIR / name, columns=["cell_id", f"train_eligible_fold_{study}"])
        c = pq.read_table(CANONICAL_DIR / name, columns=["cell_id", "row", "column"])
        if not np.array_equal(f["cell_id"].to_numpy(), c["cell_id"].to_numpy()):
            raise ValueError("Distance geometry ID mismatch")
        eligible = f[f"train_eligible_fold_{study}"].to_numpy()
        train[c["row"].to_numpy()[eligible], c["column"].to_numpy()[eligible]] = True
        count += int(eligible.sum())
    if count != fold_counts()[study]["train_rows"] or np.count_nonzero(train) != count:
        raise ValueError("TRAIN geometry count mismatch")
    grid_seconds = time.perf_counter() - started
    dt_started = time.perf_counter()
    distance = distance_transform_edt(~train, sampling=CELL_M)
    transform_seconds = time.perf_counter()-dt_started
    test = plan_table(study, "test")
    test_values = distance[test["row"].to_numpy(), test["column"].to_numpy()]
    reference = test["phase5b_test_distance_m"].to_numpy()
    if not np.array_equal(test_values, reference):
        raise ValueError("Nearest-TRAIN EDT does not reproduce Phase 5B TEST distances")
    selected = distance[plan["row"].to_numpy(), plan["column"].to_numpy()]
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise ValueError("Invalid sampled DEAD_ZONE nearest-TRAIN distance")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npy")
    np.save(temporary, selected)
    os.replace(temporary, out)
    metadata = {"study": study, "n": len(selected), "train_n": count,
        "plan_sha256": sha256(paths(study, "dead")), "sha256": sha256(out),
        "grid_seconds": grid_seconds, "distance_transform_seconds": transform_seconds,
        "phase5b_test_distance_validation_n": len(test_values), "phase5b_test_distance_max_abs_diff_m": 0.0}
    write_json(sidecar, metadata)
    del distance, train
    return selected, metadata


def pilot():
    cache = cache_for()
    counts, manifest, _ = audit(cache)
    distance, timing = compute_distances(1, cache)
    plan = plan_table(1, "dead")
    rows = plan["global_row"].to_numpy()[:BATCH]
    if np.any(cache.arrays["inner_fold"][rows, 0] != 0):
        raise ValueError("Pilot DEAD_ZONE row entered outer TRAIN")
    state = state_for(1, cache)
    started = time.perf_counter()
    models = models_for(1, cache, state, manifest)
    load_seconds = time.perf_counter()-started
    started = time.perf_counter()
    pb, pf = infer(models, np.asarray(cache.arrays["features"][rows]), state)
    inference_seconds = time.perf_counter()-started
    if pb.shape != (5, len(rows)) or pf.shape != (5, len(rows), 6):
        raise ValueError("Pilot inference shape mismatch")
    write_json(OUT / "pilot.json", {"study": 1, "n": len(rows), "dead_population_n": counts[1]["dead_zone_rows"],
        "sample_distance_n": len(distance), "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds, "rows_per_second": len(rows)/inference_seconds,
        "estimated_five_study_inference_seconds": sum(int(fold_counts()[s]["dead_zone_rows"]*.01) for s in range(1,6))/(len(rows)/inference_seconds),
        "grid_seconds": timing["grid_seconds"], "distance_transform_seconds": timing["distance_transform_seconds"],
        "estimated_remaining_four_distance_seconds": 4*(timing["grid_seconds"]+timing["distance_transform_seconds"]),
        "nearest_train_distance_min_km": float(np.min(distance)/1000),
        "nearest_train_distance_max_km": float(np.max(distance)/1000)})
    print("Pilot benchmark complete", flush=True)


def output_path(study, zone):
    return OUT / "predictions" / f"outer_{study}_{zone}.parquet"


def table_for(plan, study, bp, flp, valid, pred_bp, variance, pred_flp, distance):
    result = {"cell_id": plan["cell_id"].to_numpy(), "outer_study": np.full(len(bp), study, dtype=np.uint8)}
    result["global_row"] = plan["global_row"].to_numpy()
    for key in ("row", "column", "x_5070", "y_5070"):
        result[key] = plan[key].to_numpy()
    result.update({"nearest_train_distance_m": distance, "bp_target": bp,
        "bp_prediction": pred_bp.astype(np.float32), "bp_between_variance": variance.astype(np.float32),
        "flp_valid": valid})
    for k in range(6):
        result[f"flp{k+1}_target"] = flp[:, k]
        result[f"flp{k+1}_prediction"] = pred_flp[:, k].astype(np.float32)
    return pa.table(result)


def infer_dead(study, cache, manifest):
    out = output_path(study, "dead")
    sidecar = out.with_suffix(".json")
    if out.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["sha256"] != sha256(out) or record["plan_sha256"] != sha256(paths(study, "dead")):
            raise ValueError("Recovered DEAD_ZONE prediction hash mismatch")
        return record
    distance, distance_info = compute_distances(study, cache)
    plan = plan_table(study, "dead")
    rows = plan["global_row"].to_numpy()
    ids = plan["cell_id"].to_numpy()
    if not np.array_equal(ids, cache.arrays["cell_id"][rows]) or np.any(cache.arrays["inner_fold"][rows, study-1] != 0):
        raise ValueError("DEAD_ZONE sample/cache/training exclusion mismatch")
    state = state_for(study, cache)
    models = models_for(study, cache, state, manifest)
    started = time.perf_counter()
    batches = []
    for lo in range(0, len(rows), BATCH):
        hi = min(lo+BATCH, len(rows))
        chosen = rows[lo:hi]
        target = np.asarray(cache.arrays["bp"][chosen]); flp = np.asarray(cache.arrays["flp"][chosen])
        valid = np.asarray(cache.arrays["flp_valid"][chosen], dtype=bool)
        validate_targets(target, flp, valid)
        pb, pf = infer(models, np.asarray(cache.arrays["features"][chosen]), state)
        subplan = plan.slice(lo, hi-lo)
        batch = table_for(subplan, study, target, flp, valid,
            pb.mean(axis=0), pb.var(axis=0), pf.mean(axis=0), np.asarray(distance[lo:hi]))
        batches.append(batch)
    write_table(out, pa.concat_tables(batches))
    record = {"study": study, "zone": "DEAD_ZONE", "n": len(rows), "sha256": sha256(out),
        "plan_sha256": sha256(paths(study, "dead")), "distance_sha256": distance_info["sha256"],
        "checkpoint_count": 5, "inference_seconds": time.perf_counter()-started}
    write_json(sidecar, record)
    print(f"DEAD_ZONE study {study}: {len(rows):,} sampled rows", flush=True)
    return record


def export_test(study, cache):
    out = output_path(study, "test")
    sidecar = out.with_suffix(".json")
    if out.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["sha256"] != sha256(out) or record["plan_sha256"] != sha256(paths(study, "test")):
            raise ValueError("Recovered TEST comparison hash mismatch")
        return record
    plan = plan_table(study, "test")
    rows = plan["global_row"].to_numpy(); tile = plan["tile_index"].to_numpy()
    pred_bp = np.empty(len(rows), dtype=np.float32); variance = np.empty(len(rows), dtype=np.float32)
    pred_flp = np.empty((len(rows), 6), dtype=np.float32)
    for tile_id in np.unique(tile):
        selected = np.flatnonzero(tile == tile_id)
        path = RUN / "predictions" / f"outer_{study}" / f"{tile_id:04d}.parquet"
        columns = ["cell_id", "bp_mean", "bp_between_variance", *(f"flp{k}_mean" for k in range(1,7))]
        p = pq.read_table(path, columns=columns)
        index = plan["prediction_row"].to_numpy()[selected]
        if not np.array_equal(p["cell_id"].to_numpy()[index], plan["cell_id"].to_numpy()[selected]):
            raise ValueError("Phase 9 TEST prediction sample alignment mismatch")
        pred_bp[selected] = p["bp_mean"].to_numpy()[index]
        variance[selected] = p["bp_between_variance"].to_numpy()[index]
        for k in range(6):
            pred_flp[selected, k] = p[f"flp{k+1}_mean"].to_numpy()[index]
    if not np.array_equal(plan["cell_id"].to_numpy(), cache.arrays["cell_id"][rows]) or np.any(cache.arrays["inner_fold"][rows, study-1] != 0):
        raise ValueError("TEST sample/cache/training exclusion mismatch")
    target = np.asarray(cache.arrays["bp"][rows]); flp = np.asarray(cache.arrays["flp"][rows])
    valid = np.asarray(cache.arrays["flp_valid"][rows], dtype=bool)
    validate_targets(target, flp, valid)
    table = table_for(plan, study, target, flp, valid, pred_bp, variance, pred_flp,
        plan["phase5b_test_distance_m"].to_numpy())
    write_table(out, table)
    record = {"study": study, "zone": "TEST", "n": len(rows), "sha256": sha256(out),
        "plan_sha256": sha256(paths(study, "test")), "prediction_source": "frozen Phase 9 full-support exports"}
    write_json(sidecar, record)
    print(f"TEST study {study}: {len(rows):,} matched Phase 9 rows", flush=True)
    return record


def run():
    if not (OUT / "pilot.json").exists():
        raise ValueError("Pilot benchmark required before five-study inference")
    cache = cache_for()
    _counts, manifest, _gate = audit(cache)
    records = []
    for study in range(1, 6):
        records.append(infer_dead(study, cache, manifest))
        records.append(export_test(study, cache))
    write_json(OUT / "predictions/complete.json", {"studies": 5, "records": records})


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def read_prediction(study, zone):
    out = output_path(study, zone)
    record = json.loads(out.with_suffix(".json").read_text())
    if record["sha256"] != sha256(out) or record["plan_sha256"] != sha256(paths(study, zone)):
        raise ValueError("Prediction integrity mismatch")
    table = pq.read_table(out)
    if table.num_rows != record["n"] or not np.array_equal(table["cell_id"].to_numpy(), plan_table(study, zone)["cell_id"].to_numpy()):
        raise ValueError("Sampled prediction row order mismatch")
    return table


def summarize(study, zone, table):
    target = table["bp_target"].to_numpy(); pred = table["bp_prediction"].to_numpy()
    var = table["bp_between_variance"].to_numpy(); distance = table["nearest_train_distance_m"].to_numpy()
    error = np.abs(pred.astype(np.float64) - target)
    valid = table["flp_valid"].to_numpy().astype(bool)
    flp_target = np.column_stack([table[f"flp{k}_target"].to_numpy() for k in range(1,7)])
    flp_pred = np.column_stack([table[f"flp{k}_prediction"].to_numpy() for k in range(1,7)])
    flp_tv = .5 * np.abs(flp_pred[valid] - flp_target[valid]).sum(axis=1)
    if np.any(distance <= 0) or not np.isfinite(distance).all() or not np.isfinite(pred).all() or not np.isfinite(var).all():
        raise ValueError("Invalid sampled predictions or distances")
    if zone == "test" and np.any(distance <= 211150.99027309995):
        raise ValueError("TEST sample violates exclusion radius")
    summary = {"study": study, "zone": zone.upper(), "n": len(error),
        "distance_median_km": float(np.median(distance)/1000),
        "distance_p10_km": float(np.quantile(distance,.1)/1000),
        "distance_p90_km": float(np.quantile(distance,.9)/1000),
        "bp_mae": float(error.mean()), "bp_bias": float((pred.astype(np.float64)-target).mean()),
        "mean_disagreement": float(var.mean(dtype=np.float64)),
        "mean_target_bp": float(target.mean(dtype=np.float64)), "mean_predicted_bp": float(pred.mean(dtype=np.float64)),
        "disagreement_error_spearman": float(spearmanr(var, error).statistic),
        "distance_error_spearman": float(spearmanr(distance, error).statistic),
        "distance_disagreement_spearman": float(spearmanr(distance, var).statistic),
        "flp_valid_n": int(valid.sum()), "flp_mean_tv": float(flp_tv.mean())}
    bins = []
    distance_bp_strata = []
    for lo, hi in itertools.pairwise(DISTANCE_BINS_KM):
        selected = (distance/1000 >= lo) & (distance/1000 < hi)
        bins.append({"study": study, "zone": zone.upper(), "distance_lower_km": lo, "distance_upper_km": hi,
            "n": int(selected.sum()), "distance_median_km": float(np.median(distance[selected])/1000) if selected.any() else None,
            "bp_mae": float(error[selected].mean()) if selected.any() else None,
            "bp_bias": float((pred[selected].astype(np.float64)-target[selected]).mean()) if selected.any() else None,
            "mean_disagreement": float(var[selected].mean(dtype=np.float64)) if selected.any() else None,
            "mean_predicted_bp": float(pred[selected].mean(dtype=np.float64)) if selected.any() else None,
            "disagreement_error_spearman": float(spearmanr(var[selected], error[selected]).statistic) if selected.sum() >= 100 else None})
        for bp_lo, bp_hi in itertools.pairwise(BP_BINS):
            both = selected & (pred >= bp_lo) & (pred < bp_hi)
            distance_bp_strata.append({"study": study, "zone": zone.upper(), "distance_lower_km": lo,
                "distance_upper_km": hi, "predicted_bp_lower": bp_lo, "predicted_bp_upper": bp_hi,
                "n": int(both.sum()), "bp_mae": float(error[both].mean()) if both.any() else None,
                "mean_disagreement": float(var[both].mean(dtype=np.float64)) if both.any() else None})
    bp_strata = []
    for lo, hi in itertools.pairwise(BP_BINS):
        selected = (pred >= lo) & (pred < hi)
        bp_strata.append({"study": study, "zone": zone.upper(), "predicted_bp_lower": lo, "predicted_bp_upper": hi,
            "n": int(selected.sum()), "bp_mae": float(error[selected].mean()) if selected.any() else None,
            "mean_disagreement": float(var[selected].mean(dtype=np.float64)) if selected.any() else None,
            "median_distance_km": float(np.median(distance[selected])/1000) if selected.any() else None})
    high = exact_tail_mask(error, .10, largest=True)
    coverage = []
    for method, score in (("disagreement", var), ("predicted_bp", pred), ("random", None)):
        for fraction in (0, .05, .10, .20, .50):
            mae, recall, mass = (float(error.mean()), fraction, fraction) if score is None else rejection_metrics(score,error,high,fraction)
            coverage.append({"study": study, "zone": zone.upper(), "method": method,
                "rejected_fraction": fraction, "remaining_mae": mae, "high_error_recall": recall,
                "error_mass_rejected": mass})
    return summary, bins, bp_strata, coverage, distance_bp_strata


def standardize_bp_strata(strata):
    output = []
    for study in range(1, 6):
        dead = [r for r in strata if r["study"] == study and r["zone"] == "DEAD"]
        test = [r for r in strata if r["study"] == study and r["zone"] == "TEST"]
        common = [(d,t) for d,t in zip(dead,test,strict=True) if d["n"] and t["n"]]
        total = sum(d["n"]+t["n"] for d,t in common)
        for zone, index in (("DEAD",0), ("TEST",1)):
            mae = sum((pair[0]["n"]+pair[1]["n"])/total*pair[index]["bp_mae"] for pair in common)
            variance = sum((pair[0]["n"]+pair[1]["n"])/total*pair[index]["mean_disagreement"] for pair in common)
            output.append({"study": study, "zone": zone, "common_strata_n": len(common),
                "excluded_sample_n": sum(r["n"] for r in (dead if zone == "DEAD" else test) if not any(r is pair[index] for pair in common)),
                "common_mix_bp_mae": mae, "common_mix_mean_disagreement": variance})
    return output


def plot_analysis(summary, bins, coverage):
    REPORT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for metric, ylabel, filename in (("bp_mae", "Mean absolute BP error", "error_vs_distance"),
                                      ("mean_disagreement", "Mean between-member BP variance", "disagreement_vs_distance")):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        for study, ax in enumerate(axes.flat[:5], 1):
            for zone, color in (("DEAD", "#1b9e77"), ("TEST", "#d95f02")):
                selected = [r for r in bins if r["study"] == study and r["zone"] == zone and r["n"] >= 100]
                ax.plot([r["distance_median_km"] for r in selected], [r[metric] for r in selected], "o-", lw=1.8,
                    color=color, label="DEAD_ZONE" if zone == "DEAD" else "TEST")
            ax.axvline(211.1509902731, color="0.5", ls="--", lw=1)
            ax.set(title=f"Study {study}", xscale="log", xlabel="Nearest eligible TRAIN distance (km)", ylabel=ylabel)
        axes.flat[0].legend(frameon=False, fontsize=8)
        axes.flat[5].axis("off")
        fig.suptitle(f"Sampled DEAD_ZONE and matched TEST: {ylabel}")
        fig.tight_layout(); fig.savefig(REPORT / f"{filename}.png", dpi=240); fig.savefig(REPORT / f"{filename}.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(1,6); width=.34
    for j, zone in enumerate(("DEAD", "TEST")):
        values = [next(r for r in summary if r["study"] == study and r["zone"] == zone) for study in range(1,6)]
        for ax, metric in ((axes[0], "bp_mae"), (axes[1], "mean_disagreement")):
            ax.bar(x + (j-.5)*width, [r[metric] for r in values], width=width,
                color="#1b9e77" if zone == "DEAD" else "#d95f02", label="DEAD_ZONE" if j==0 else "TEST")
    axes[0].set(xlabel="Geographic study", ylabel="BP MAE", xticks=x)
    axes[1].set(xlabel="Geographic study", ylabel="Mean between-member BP variance", xticks=x)
    axes[0].legend(frameon=False)
    fig.suptitle("Matched hash samples: zone-level reconstruction and disagreement")
    fig.tight_layout(); fig.savefig(REPORT / "zone_comparison.png", dpi=240); fig.savefig(REPORT / "zone_comparison.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for zone, linestyle in (("DEAD", "-"), ("TEST", "--")):
        for method, color in (("disagreement", "#205493"), ("predicted_bp", "#d95f02"), ("random", "#555555")):
            subset = [r for r in coverage if r["zone"] == zone and r["method"] == method]
            xs = (0,.05,.1,.2,.5)
            y1 = [np.mean([r["remaining_mae"] / next(s["remaining_mae"] for s in subset if s["study"]==r["study"] and s["rejected_fraction"]==0)
                for r in subset if r["rejected_fraction"]==fraction]) for fraction in xs]
            y2 = [np.mean([r["high_error_recall"] for r in subset if r["rejected_fraction"]==fraction]) for fraction in xs]
            label = f"{zone} {method.replace('_',' ')}"
            axes[0].plot(np.array(xs)*100, y1, linestyle=linestyle, marker="o", ms=3, color=color, label=label)
            axes[1].plot(np.array(xs)*100, y2, linestyle=linestyle, marker="o", ms=3, color=color)
    axes[0].set(xlabel="Cells rejected within study and zone (%)", ylabel="Remaining MAE / initial MAE")
    axes[1].set(xlabel="Cells rejected within study and zone (%)", ylabel="Top-decile error recall", ylim=(0,1))
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Error ranking by ensemble disagreement and predicted BP")
    fig.tight_layout(); fig.savefig(REPORT / "risk_coverage.png", dpi=240); fig.savefig(REPORT / "risk_coverage.pdf"); plt.close(fig)


def analyze():
    if not (OUT / "predictions/complete.json").exists():
        raise ValueError("All five DEAD_ZONE and TEST comparison datasets are required")
    summary, bins, strata, coverage, distance_bp_strata = [], [], [], [], []
    for study in range(1, 6):
        for zone in ("dead", "test"):
            result = summarize(study, zone, read_prediction(study, zone))
            summary.append(result[0]); bins.extend(result[1]); strata.extend(result[2]); coverage.extend(result[3])
            distance_bp_strata.extend(result[4])
    standardized = standardize_bp_strata(strata)
    for filename, rows in (("study_zone_summary.csv", summary), ("distance_bins.csv", bins),
                           ("predicted_bp_strata.csv", strata), ("bp_standardized_comparison.csv", standardized),
                           ("risk_coverage.csv", coverage), ("distance_bp_strata.csv", distance_bp_strata)):
        write_csv(REPORT / filename, rows)
    plot_analysis(summary, bins, coverage)
    print("Phase 10C analysis complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "pilot", "run", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare, "pilot": pilot, "run": run, "analyze": analyze}[args.action]()


if __name__ == "__main__":
    main()
