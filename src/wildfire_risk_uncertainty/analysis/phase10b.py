"""Phase 10B: geographic separation and ensemble BP agreement failures."""
import csv
import hashlib
import itertools
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from wildfire_risk_uncertainty.analysis.phase10a import EXPECTED, RUN, exact_tail_mask
from wildfire_risk_uncertainty.training.phase9 import ROOT

OUT = ROOT / "reports/uncertainty/10b"
RADIUS_M = 211150.99027309995
BP_BINS = (0., .001, .005, .01, 1.000001)
COVERAGE = ("wui_covered_fraction", "HUDEN2020__valid_area_fraction",
            "POPDEN2020__valid_area_fraction", "VEG2019PC__valid_area_fraction")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_study(study):
    """One study in memory, one validated tile at a time."""
    fields = {key: [] for key in ("bp", "pred", "variance", "distance", "geo_code", "wui_failure", *COVERAGE)}
    paths = sorted((RUN / "predictions" / f"outer_{study}").glob("*.parquet"))
    if not paths:
        raise ValueError("Missing Phase 9 full-support predictions")
    for path in paths:
        part = json.loads((RUN / "partitions" / f"outer_{study}" / f"{path.stem}.json").read_text())
        name = part["tile_name"]
        fold = pq.read_table(ROOT / "artifacts/spatial_validation/05b/partitions" / name,
            columns=["cell_id", "outer_fold", "oof_nearest_training_distance_m", f"train_eligible_fold_{study}"])
        mask = fold["outer_fold"].to_numpy() == study
        ids = fold["cell_id"].to_numpy()[mask]
        if np.any(mask & fold[f"train_eligible_fold_{study}"].to_numpy()) or len(ids) != part["n"]:
            raise ValueError("TEST support or TRAIN exclusion mismatch")
        if hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest() != part["test_cell_id_sha256"]:
            raise ValueError("Phase 9 TEST ID hash mismatch")
        distance = fold["oof_nearest_training_distance_m"].to_numpy()[mask]
        if not np.isfinite(distance).all() or np.any(distance <= RADIUS_M):
            raise ValueError("Invalid OOF nearest-training distance")
        canonical = pq.read_table(ROOT / "artifacts/dataset/canonical/partitions" / name,
            columns=["cell_id", "BP", "x_5070", "y_5070", "wui_failure", *COVERAGE])
        prediction = pq.read_table(path, columns=["cell_id", "bp_mean", "bp_between_variance"])
        if not np.array_equal(canonical["cell_id"].to_numpy()[mask], ids) or not np.array_equal(prediction["cell_id"].to_numpy(), ids):
            raise ValueError("Canonical or prediction IDs do not match frozen TEST")
        x = canonical["x_5070"].to_numpy()[mask]
        y = canonical["y_5070"].to_numpy()[mask]
        gx, gy = np.floor(x / 100000).astype(np.int32), np.floor(y / 100000).astype(np.int32)
        if np.any((gx < -100) | (gx > 155) | (gy < -100) | (gy > 155)):
            raise ValueError("Geographic bin range exceeded")
        fields["geo_code"].append(((gx + 100) * 256 + gy + 100).astype(np.int32))
        fields["bp"].append(canonical["BP"].to_numpy()[mask].astype(np.float32))
        fields["pred"].append(prediction["bp_mean"].to_numpy())
        fields["variance"].append(prediction["bp_between_variance"].to_numpy())
        fields["distance"].append(distance)
        fields["wui_failure"].append(canonical["wui_failure"].to_numpy()[mask])
        for key in COVERAGE:
            fields[key].append(canonical[key].to_numpy()[mask].astype(np.float32))
    data = {key: np.concatenate(value) for key, value in fields.items()}
    if len(data["bp"]) != EXPECTED[study - 1]:
        raise ValueError("Study support mismatch")
    if any(not np.isfinite(data[key]).all() for key in ("bp", "pred", "variance", "distance")):
        raise ValueError("Nonfinite BP, prediction, variance, or distance")
    return data


def quantile_labels(values, groups=5):
    """Equal-count bins; stable ordering handles distance ties reproducibly."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=np.uint8)
    for k, indices in enumerate(np.array_split(order, groups)):
        labels[indices] = k
    return labels


def summarize_group(study, label, data, selected, error):
    n = int(selected.sum())
    if not n:
        return {"study": study, "group": label, "n": 0, "distance_median_km": None,
            "distance_min_km": None, "distance_max_km": None, "mean_disagreement": None,
            "mean_absolute_error": None, "mean_predicted_bp": None, "mean_target_bp": None}
    return {"study": study, "group": label, "n": n,
        "distance_median_km": float(np.median(data["distance"][selected]) / 1000),
        "distance_min_km": float(data["distance"][selected].min() / 1000),
        "distance_max_km": float(data["distance"][selected].max() / 1000),
        "mean_disagreement": float(data["variance"][selected].mean(dtype=np.float64)),
        "mean_absolute_error": float(error[selected].mean(dtype=np.float64)),
        "mean_predicted_bp": float(data["pred"][selected].mean(dtype=np.float64)),
        "mean_target_bp": float(data["bp"][selected].mean(dtype=np.float64))}


def study_analysis(study, expected_failures):
    data = load_study(study)
    n = len(data["bp"])
    error = np.abs(data["pred"].astype(np.float64) - data["bp"]).astype(np.float32)
    phase9_mae = json.loads((RUN / "summaries" / f"outer_{study}.json").read_text())["bp"]["mae"]
    if abs(error.mean(dtype=np.float64) - phase9_mae) > 2e-7:
        raise ValueError("Phase 9 MAE mismatch")
    high = exact_tail_mask(error, .10, largest=True)
    low = exact_tail_mask(data["variance"], .20, largest=False)
    failure = high & low
    if int(failure.sum()) != expected_failures:
        raise ValueError("Phase 10A agreement-failure definition mismatch")
    distance = data["distance"]
    q = np.quantile(distance, [0, .1, .25, .5, .75, .9, 1]) / 1000
    association = {"study": study, "n": n, "distance_min_km": float(q[0]),
        "distance_p10_km": float(q[1]), "distance_p25_km": float(q[2]),
        "distance_median_km": float(q[3]), "distance_p75_km": float(q[4]),
        "distance_p90_km": float(q[5]), "distance_max_km": float(q[6]),
        "distance_disagreement_spearman": float(spearmanr(distance, data["variance"]).statistic),
        "distance_error_spearman": float(spearmanr(distance, error).statistic),
        "distance_prediction_spearman": float(spearmanr(distance, data["pred"]).statistic),
        "distance_disagreement_pearson": float(np.corrcoef(distance, data["variance"])[0, 1]),
        "distance_error_pearson": float(np.corrcoef(distance, error)[0, 1])}
    labels = quantile_labels(distance)
    quintiles = [summarize_group(study, f"Q{k+1}", data, labels == k, error) for k in range(5)]
    strata = []
    for lo, hi in itertools.pairwise(BP_BINS):
        bp_mask = (data["pred"] >= lo) & (data["pred"] < hi)
        for k in range(5):
            row = summarize_group(study, f"Q{k+1}", data, bp_mask & (labels == k), error)
            row["predicted_bp_lower"] = lo
            row["predicted_bp_upper"] = hi
            strata.append(row)
    failure_distance = [summarize_group(study, f"Q{k+1}", data, failure & (labels == k), error) for k in range(5)]
    for k, row in enumerate(failure_distance):
        row["all_cells_n"] = quintiles[k]["n"]
        row["failure_rate"] = row["n"] / row["all_cells_n"]
        row["share_of_study_failures"] = row["n"] / int(failure.sum())
    failure_summary = {"study": study, "n": n, "failure_n": int(failure.sum()),
        "failure_rate": float(failure.mean()), "failure_distance_median_km": float(np.median(distance[failure]) / 1000),
        "all_distance_median_km": float(np.median(distance) / 1000),
        "failure_mean_target_bp": float(data["bp"][failure].mean(dtype=np.float64)),
        "failure_mean_predicted_bp": float(data["pred"][failure].mean(dtype=np.float64))}
    coverage_rows = []
    conditions = {"WUI failure flag": data["wui_failure"]}
    for key in COVERAGE:
        values = data[key]
        conditions[f"{key} < 1"] = np.isfinite(values) & (values < 1 - 1e-6)
        conditions[f"{key} = 0 or null"] = (~np.isfinite(values)) | (values <= 1e-6)
    for name, mask in conditions.items():
        coverage_rows.append({"study": study, "condition": name,
            "all_n": int(mask.sum()), "condition_rate_all_cells": float(mask.mean()),
            "failure_n": int((mask & failure).sum()), "condition_rate_among_failures": float(mask[failure].mean()),
            "failure_to_all_rate_ratio": float(mask[failure].mean() / mask.mean()) if mask.any() else None})
    code = data["geo_code"]
    all_counts = np.bincount(code, minlength=65536)
    fail_counts = np.bincount(code[failure], minlength=65536)
    geo = []
    for code_value in np.flatnonzero(all_counts):
        gx = code_value // 256 - 100
        gy = code_value % 256 - 100
        geo.append({"study": study, "x_bin_100km": int(gx), "y_bin_100km": int(gy),
            "all_n": int(all_counts[code_value]), "failure_n": int(fail_counts[code_value]),
            "failure_rate": float(fail_counts[code_value] / all_counts[code_value])})
    return association, quintiles, strata, failure_distance, failure_summary, coverage_rows, geo


def geographic_concentration(geography):
    """Descriptive concentration in the highest-count tenth of 100 km bins."""
    output = []
    for study in range(1, 6):
        rows = sorted((r for r in geography if r["study"] == study and r["all_n"] >= 1000),
            key=lambda r: r["failure_n"], reverse=True)
        top_count = math.ceil(len(rows) * .10)
        top = rows[:top_count]
        output.append({"study": study, "eligible_bin_n": len(rows), "top_bin_n": top_count,
            "fraction_of_failures_in_top_bins": sum(r["failure_n"] for r in top) / sum(r["failure_n"] for r in rows),
            "fraction_of_cells_in_top_bins": sum(r["all_n"] for r in top) / sum(r["all_n"] for r in rows),
            "maximum_bin_failure_rate": max(r["failure_rate"] for r in rows)})
    return output


def figures(quintiles, failure_distance, geography):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = plt.cm.tab10(np.arange(5))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for study in range(1, 6):
        rows = [r for r in quintiles if r["study"] == study]
        x = [r["distance_median_km"] for r in rows]
        axes[0].plot(x, [r["mean_disagreement"] for r in rows], "o-", color=colors[study-1], label=f"Study {study}")
        axes[1].plot(x, [r["mean_absolute_error"] for r in rows], "o-", color=colors[study-1])
    axes[0].set(xlabel="Nearest eligible TRAIN distance (km; quintile median)", ylabel="Mean between-member BP variance", title="Distance versus disagreement")
    axes[1].set(xlabel="Nearest eligible TRAIN distance (km; quintile median)", ylabel="Mean absolute BP error", title="Distance versus FSim error")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "distance_disagreement_error.png", dpi=240); fig.savefig(OUT / "distance_disagreement_error.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for study in range(1, 6):
        q = [r for r in quintiles if r["study"] == study]
        f = [r for r in failure_distance if r["study"] == study]
        x = np.arange(1, 6)
        axes[0].plot(x, [r["mean_disagreement"] / q[0]["mean_disagreement"] for r in q], "o-", color=colors[study-1], label=f"Study {study}")
        axes[1].plot(x, [r["mean_absolute_error"] / q[0]["mean_absolute_error"] for r in q], "o-", color=colors[study-1])
        axes[1].scatter(x, [r["failure_rate"] / f[0]["failure_rate"] if f[0]["failure_rate"] else np.nan for r in f],
            marker="x", color=colors[study-1], s=30)
    axes[0].set(xlabel="Within-study distance quintile", ylabel="Disagreement / Q1", title="Relative disagreement")
    axes[1].set(xlabel="Within-study distance quintile", ylabel="Error or failure rate / Q1", title="Error (line) and agreement failures (×)")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes: ax.axhline(1, color="0.6", ls="--", lw=1); ax.set_xticks(range(1, 6))
    fig.tight_layout(); fig.savefig(OUT / "joint_distance_trends.png", dpi=240); fig.savefig(OUT / "joint_distance_trends.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    for study, ax in enumerate(axes, 1):
        rows = [r for r in geography if r["study"] == study and r["all_n"] >= 1000]
        sc = ax.scatter([r["x_bin_100km"]*100 for r in rows], [r["y_bin_100km"]*100 for r in rows],
            c=[r["failure_rate"] for r in rows], s=24, marker="s", cmap="magma", vmin=0, vmax=.10)
        ax.set(title=f"Study {study}", xlabel="EPSG:5070 x (km)", aspect="equal")
        ax.tick_params(axis="both", labelsize=8)
    axes[0].set_ylabel("EPSG:5070 y (km)")
    fig.subplots_adjust(left=.045, right=.88, bottom=.19, top=.86, wspace=.22)
    color_axis = fig.add_axes([.91, .22, .012, .58])
    fig.colorbar(sc, cax=color_axis, label="Agreement-failure rate (capped at 10%)")
    fig.savefig(OUT / "agreement_failure_geography.png", dpi=240); fig.savefig(OUT / "agreement_failure_geography.pdf"); plt.close(fig)


def main():
    gate = json.loads((RUN / "predictions_complete.json").read_text())
    if gate["studies"] != 5 or gate["manifest_sha256"] != json.loads((RUN / "complete.json").read_text())["manifest_sha256"]:
        raise ValueError("Phase 9 provenance mismatch")
    inputs = set(json.loads((ROOT / "artifacts/training/hpo_cache_v1/manifest.json").read_text())["inputs"])
    if not (set(COVERAGE) | {"wui_failure"}).issubset(inputs):
        raise ValueError("Coverage diagnostics are not frozen model inputs")
    ten_a = ROOT / "reports/uncertainty/10a"
    with (ten_a / "low_disagreement_high_error.csv").open() as handle:
        expected = {int(r["study"]): int(r["failure_n"]) for r in csv.DictReader(handle)}
    OUT.mkdir(parents=True, exist_ok=True)
    associations, quintiles, strata, failure_distance, failures, coverage, geo = [], [], [], [], [], [], []
    for study in range(1, 6):
        a, q, s, fd, f, c, g = study_analysis(study, expected[study])
        associations.append(a); quintiles.extend(q); strata.extend(s)
        failure_distance.extend(fd); failures.append(f); coverage.extend(c); geo.extend(g)
        print(f"Study {study}: {a['n']:,} validated TEST cells; distance–variance ρ={a['distance_disagreement_spearman']:.3f}; distance–error ρ={a['distance_error_spearman']:.3f}", flush=True)
    for filename, rows in (("distance_associations.csv", associations), ("distance_quintiles.csv", quintiles),
                           ("predicted_bp_distance_strata.csv", strata), ("failure_distance.csv", failure_distance),
                           ("failure_summary.csv", failures), ("coverage_conditions.csv", coverage),
                           ("failure_geography_100km.csv", geo)):
        write_csv(OUT / filename, rows)
    write_csv(OUT / "geographic_concentration.csv", geographic_concentration(geo))
    figures(quintiles, failure_distance, geo)
    (OUT / "provenance.json").write_text(json.dumps({"phase9_manifest_sha256": gate["manifest_sha256"],
        "spatial_05b_manifest_sha256": hashlib.sha256((ROOT / "artifacts/spatial_validation/05b/manifest.json").read_bytes()).hexdigest(),
        "phase10a_failure_table_sha256": hashlib.sha256((ten_a / "low_disagreement_high_error.csv").read_bytes()).hexdigest(),
        "phase10a_failure_counts": expected, "exclusion_radius_m": RADIUS_M,
        "distance_source": "artifacts/spatial_validation/05b/partitions/oof_nearest_training_distance_m",
        "bp_strata": BP_BINS, "distance_bins": "five equal-count bins within each study",
        "geography": "100 km EPSG:5070 bins; map displays bins with at least 1,000 TEST cells"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
