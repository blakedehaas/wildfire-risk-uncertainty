"""Paired BP variance/Jensen–Shannon comparison on frozen outer TEST cells."""

import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm
from scipy.special import xlogy
from scipy.stats import pearsonr, spearmanr

from wildfire_risk_uncertainty.analysis.phase10c import OUT as PHASE10C_OUT
from wildfire_risk_uncertainty.analysis.phase10c import cache_for
from wildfire_risk_uncertainty.training.cache import sha256
from wildfire_risk_uncertainty.training.phase9 import (
    PHASE8,
    ROOT,
    RUN,
    infer,
    models_for,
    state_for,
)

OUT = ROOT / "artifacts/evaluation/phase11a/phase11a-20261116-v1"
REPORT = ROOT / "reports/uncertainty/11a"
N_PER_STUDY = 100_000
SEED = 11_011
BATCH = 8192
FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.50)
METHODS = ("variance", "jensen_shannon", "predicted_bp", "random")
CONDITIONAL_FRACTIONS = (0.10, 0.20)


def bernoulli_entropy(p):
    """Bernoulli entropy in nats, including exact zero and one endpoints."""
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return -xlogy(p, p) - xlogy(1 - p, 1 - p)


def jensen_shannon(members):
    """Mean KL(member Bernoulli || mixture Bernoulli), in nats."""
    p = np.asarray(members, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != 5:
        raise ValueError("Expected five member-probability rows")
    mean = p.mean(axis=0)
    value = bernoulli_entropy(mean) - bernoulli_entropy(p).mean(axis=0)
    if np.min(value) < -1e-13:
        raise ValueError("Negative Jensen–Shannon divergence beyond roundoff")
    return np.maximum(value, 0)


def mean_kl(members):
    """Independent KL form used to verify the entropy-difference calculation."""
    p = np.asarray(members, dtype=np.float64)
    mean = p.mean(axis=0)
    return (
        xlogy(p, p) - xlogy(p, mean)
        + xlogy(1 - p, 1 - p) - xlogy(1 - p, 1 - mean)
    ).mean(axis=0)


def selected_rows(population_n, study, n=N_PER_STUDY):
    """Uniform second-stage selection from Phase 10C's uniform hash TEST sample."""
    if population_n < n:
        raise ValueError("Insufficient source TEST sample")
    return np.sort(np.random.default_rng(SEED + study).choice(population_n, n, replace=False))


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _parquet(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def plan_path(study):
    return OUT / "sample" / f"outer_{study}.parquet"


def prediction_path(study):
    return OUT / "predictions" / f"outer_{study}.parquet"


def source_table(study):
    path = PHASE10C_OUT / "predictions" / f"outer_{study}_test.parquet"
    sidecar = json.loads(path.with_suffix(".json").read_text())
    source_plan = PHASE10C_OUT / "sample_plan" / f"outer_{study}_test.parquet"
    if (sidecar["zone"] != "TEST" or sidecar["study"] != study
            or sha256(path) != sidecar["sha256"]
            or sha256(source_plan) != sidecar["plan_sha256"]):
        raise ValueError("Phase 10C TEST export provenance mismatch")
    return pq.read_table(path), sidecar


def prepare():
    gate = json.loads((RUN / "predictions_complete.json").read_text())
    manifest_hash = sha256(RUN / "evaluation_manifest.json")
    if gate != {"studies": 5, "manifest_sha256": manifest_hash}:
        raise ValueError("Phase 9 full TEST export provenance changed")
    records = []
    for study in range(1, 6):
        phase9_tiles = sorted((RUN / "predictions" / f"outer_{study}").glob("*.parquet"))
        if not phase9_tiles:
            raise ValueError("Missing Phase 9 full-support predictions")
        phase9_columns = pq.ParquetFile(phase9_tiles[0]).schema_arrow.names
        if any("member" in name for name in phase9_columns) or not {
            "cell_id", "bp_mean", "bp_between_variance"
        }.issubset(phase9_columns):
            raise ValueError("Unexpected Phase 9 export schema")
        source, sidecar = source_table(study)
        if len(source) < N_PER_STUDY or set(source.column_names).intersection({"bp_member_0", "bp_member_1"}):
            raise ValueError("Unexpected Phase 10C TEST source")
        choice = selected_rows(len(source), study)
        plan = source.take(pa.array(choice)).select([
            "cell_id", "outer_study", "global_row", "bp_target",
            "bp_prediction", "bp_between_variance",
        ])
        if not np.all(plan["outer_study"].to_numpy() == study):
            raise ValueError("Sample includes another outer study")
        if np.unique(plan["cell_id"].to_numpy()).size != N_PER_STUDY:
            raise ValueError("Duplicate sampled cell IDs")
        path = plan_path(study)
        _parquet(path, plan)
        records.append({"study": study, "source_n": len(source), "n": len(plan),
                        "source_sha256": sidecar["sha256"], "sample_sha256": sha256(path),
                        "phase9_export_columns": phase9_columns})
    _json(OUT / "sample/complete.json", {
        "seed": SEED, "n_per_study": N_PER_STUDY,
        "method": "Uniform without replacement from Phase 10C's study-specific SplitMix64 1% TEST sample",
        "outcome_independent": True, "phase9_manifest_sha256": manifest_hash,
        "records": records,
    })


def read_plan(study):
    marker = json.loads((OUT / "sample/complete.json").read_text())
    record = marker["records"][study - 1]
    path = plan_path(study)
    if record["study"] != study or record["sample_sha256"] != sha256(path):
        raise ValueError("Sample provenance mismatch")
    return pq.read_table(path)


def _inputs(cache, plan, study):
    rows = plan["global_row"].to_numpy()
    ids = plan["cell_id"].to_numpy()
    if not np.array_equal(np.asarray(cache.arrays["cell_id"][rows]), ids):
        raise ValueError("Cell IDs no longer align with frozen feature cache")
    if np.any(np.asarray(cache.arrays["inner_fold"][rows, study - 1]) != 0):
        raise ValueError("Outer TEST sample contains TRAIN-eligible rows")
    if not np.array_equal(np.asarray(cache.arrays["bp"][rows]), plan["bp_target"].to_numpy()):
        raise ValueError("Sample BP targets differ from frozen feature cache")
    return rows


def _models(study, cache):
    manifest = json.loads((PHASE8 / "manifest.json").read_text())
    state = state_for(study, cache)
    if state.rows != cache.manifest["outer_train_counts"][study - 1]:
        raise ValueError("Study preprocessing TRAIN count mismatch")
    return models_for(study, cache, state, manifest), state


def _match_phase9_batch_shape(members, rows, stored_mean, stored_var, study, cache, models, state):
    """Re-evaluate rare bfloat16 shape-sensitive rows at their original tile batch size."""
    initial_mean = np.abs(members.mean(axis=0) - stored_mean)
    initial_var = np.abs(members.var(axis=0) - stored_var)
    bad = np.flatnonzero((initial_mean > 1e-8) | (initial_var > 1e-11))
    if len(bad) == 0:
        return 0, float(initial_mean.max()), float(initial_var.max())
    offsets = np.asarray(cache.arrays["partition_offsets"])
    tiles = np.searchsorted(offsets, rows[bad], side="right") - 1
    sizes = np.array([
        json.loads((RUN / "partitions" / f"outer_{study}" / f"{tile:04d}.json").read_text())["n"]
        for tile in tiles
    ])
    for original_size in np.unique(sizes):
        local = bad[sizes == original_size]
        if original_size < len(local) or original_size > 65536:
            raise ValueError("Invalid original Phase 9 tile inference batch size")
        inputs = np.asarray(cache.arrays["features"][rows[local]])
        if len(inputs) < original_size:
            inputs = np.concatenate((inputs, np.repeat(inputs[:1], original_size - len(inputs), axis=0)))
        corrected, _ = infer(models, inputs, state)
        members[:, local] = corrected[:, :len(local)]
    return len(bad), float(initial_mean.max()), float(initial_var.max())


def pilot():
    cache = cache_for()
    plan = read_plan(1).slice(0, BATCH)
    rows = _inputs(cache, plan, 1)
    started = time.perf_counter()
    models, state = _models(1, cache)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    members, _ = infer(models, np.asarray(cache.arrays["features"][rows]), state)
    infer_seconds = time.perf_counter() - started
    js = jensen_shannon(members)
    kl = mean_kl(members)
    if not np.allclose(js, kl, atol=1e-13, rtol=1e-10):
        raise ValueError("Jensen–Shannon / mean KL mismatch")
    mean_diff = np.abs(members.mean(axis=0) - plan["bp_prediction"].to_numpy())
    variance_diff = np.abs(members.var(axis=0) - plan["bp_between_variance"].to_numpy())
    if mean_diff.max() > 1e-8 or variance_diff.max() > 1e-11:
        raise ValueError("Pilot does not reproduce Phase 9 summaries")
    _json(OUT / "pilot.json", {"n": len(rows), "study": 1, "load_seconds": load_seconds,
                              "inference_seconds": infer_seconds,
                              "estimated_inference_seconds": infer_seconds * (5 * N_PER_STUDY / len(rows)),
                              "max_mean_difference": float(mean_diff.max()),
                              "max_variance_difference": float(variance_diff.max()),
                              "max_js_kl_difference": float(np.max(np.abs(js - kl)))})
    print(f"Pilot: {len(rows):,} cells in {infer_seconds:.2f}s; five-study estimate {infer_seconds * (5 * N_PER_STUDY / len(rows)):.1f}s")


def run_study(study, cache):
    path = prediction_path(study)
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["sha256"] == sha256(path) and record["sample_sha256"] == sha256(plan_path(study)):
            return record
        raise ValueError("Existing prediction artifact failed recovery check")
    plan = read_plan(study)
    rows = _inputs(cache, plan, study)
    models, state = _models(study, cache)
    member_chunks = []
    started = time.perf_counter()
    for lo in range(0, len(rows), BATCH):
        chosen = rows[lo:lo + BATCH]
        members, _ = infer(models, np.asarray(cache.arrays["features"][chosen]), state)
        member_chunks.append(members)
    elapsed = time.perf_counter() - started
    members = np.concatenate(member_chunks, axis=1)
    target = plan["bp_target"].to_numpy().astype(np.float64)
    stored_mean = plan["bp_prediction"].to_numpy().astype(np.float64)
    stored_var = plan["bp_between_variance"].to_numpy().astype(np.float64)
    corrections, initial_mean_difference, initial_variance_difference = _match_phase9_batch_shape(
        members, rows, stored_mean, stored_var, study, cache, models, state)
    mean = members.mean(axis=0)
    variance = members.var(axis=0)
    js = jensen_shannon(members)
    mean_diff = np.abs(mean - stored_mean)
    variance_diff = np.abs(variance - stored_var)
    if mean_diff.max() > 1e-8 or variance_diff.max() > 1e-11:
        raise ValueError(f"Regenerated Phase 9 moments differ: mean={mean_diff.max():.3g}, variance={variance_diff.max():.3g}")
    # Validate the independent KL expression on a fixed slice without a second full pass.
    kl_difference = float(np.max(np.abs(js[::101] - mean_kl(members[:, ::101]))))
    if kl_difference > 1e-13:
        raise ValueError("Jensen–Shannon and mean KL expressions disagree")
    data = {"cell_id": plan["cell_id"].to_numpy(), "outer_study": plan["outer_study"].to_numpy(),
            "global_row": rows, "bp_target": target.astype(np.float32),
            "bp_mean": mean.astype(np.float32), "bp_between_variance": variance.astype(np.float32),
            "bp_jensen_shannon": js.astype(np.float64),
            "phase9_bp_mean": stored_mean.astype(np.float32),
            "phase9_bp_between_variance": stored_var.astype(np.float32),
            "absolute_bp_error": np.abs(mean - target).astype(np.float32)}
    for member in range(5):
        data[f"member_{member}_bp"] = members[member].astype(np.float32)
    _parquet(path, pa.table(data))
    record = {"study": study, "n": len(rows), "sha256": sha256(path),
              "sample_sha256": sha256(plan_path(study)), "inference_seconds": elapsed,
              "max_phase9_mean_difference": float(mean_diff.max()),
              "max_phase9_variance_difference": float(variance_diff.max()),
              "batch_shape_corrections": corrections,
              "initial_max_mean_difference": initial_mean_difference,
              "initial_max_variance_difference": initial_variance_difference,
              "max_js_kl_difference": kl_difference,
              "preprocessing_sha256": state.digest(),
              "phase8_manifest_sha256": sha256(PHASE8 / "manifest.json"),
              "checkpoint_sha256": [sha256(PHASE8 / "models" / f"outer_{study}_member_{member}"
                                         / "checkpoints/epoch_100.pt") for member in range(5)]}
    _json(sidecar, record)
    print(f"Outer {study}: {len(rows):,} cells in {elapsed:.2f}s")
    return record


def run():
    if not (OUT / "pilot.json").exists():
        raise ValueError("Run the bounded pilot before five-study inference")
    cache = cache_for()
    records = [run_study(study, cache) for study in range(1, 6)]
    _json(OUT / "predictions/complete.json", {"studies": 5, "records": records,
        "phase9_manifest_sha256": sha256(RUN / "evaluation_manifest.json")})


def _top_indices(score, ids, fraction):
    n = len(score)
    k = round(n * fraction)
    return np.lexsort((ids, -score))[:k] if k else np.empty(0, dtype=np.int64)


def rejection(error, score, ids, high_error, fraction):
    """Phase 10A risk and top-decile recall, with explicit cell-ID tie breaking."""
    if score is None:
        return float(error.mean()), float(fraction), float(fraction)
    selected = _top_indices(score, ids, fraction)
    k = len(selected)
    if k == 0:
        return float(error.mean()), 0.0, 0.0
    return (float((error.sum(dtype=np.float64) - error[selected].sum(dtype=np.float64)) / (len(error) - k)),
            float(high_error[selected].sum() / high_error.sum()),
            float(error[selected].sum(dtype=np.float64) / error.sum(dtype=np.float64)))


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _study(study):
    path = prediction_path(study)
    marker = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != marker["sha256"]:
        raise ValueError("Prediction output hash mismatch")
    table = pq.read_table(path)
    if len(table) != N_PER_STUDY or not np.all(table["outer_study"].to_numpy() == study):
        raise ValueError("Prediction study/sample mismatch")
    ids = table["cell_id"].to_numpy()
    target = table["bp_target"].to_numpy().astype(np.float64)
    pred = table["bp_mean"].to_numpy().astype(np.float64)
    error = np.abs(pred - target)
    if np.max(np.abs(error - table["absolute_bp_error"].to_numpy())) > 1e-8:
        raise ValueError("Stored error mismatch")
    score = {"variance": table["bp_between_variance"].to_numpy().astype(np.float64),
             "jensen_shannon": table["bp_jensen_shannon"].to_numpy().astype(np.float64),
             "predicted_bp": pred, "phase9_variance": table["phase9_bp_between_variance"].to_numpy().astype(np.float64)}
    high = np.zeros(len(error), dtype=bool)
    high[_top_indices(error, ids, .10)] = True
    return ids, target, pred, error, score, high


def analyze():
    if not (OUT / "predictions/complete.json").exists():
        raise ValueError("Five-study inference is incomplete")
    performances, associations, conditional, summaries = [], [], [], []
    plots = []
    for study in range(1, 6):
        ids, _target, pred, error, scores, high = _study(study)
        plots.append((scores["variance"], scores["jensen_shannon"]))
        for method in (*METHODS, "phase9_variance"):
            score = scores.get(method)
            if method != "random":
                associations.append({"study": study, "method": method, "n": len(ids),
                    "pearson_error": float(pearsonr(score, error).statistic),
                    "spearman_error": float(spearmanr(score, error).statistic)})
            for fraction in FRACTIONS:
                mae, recall, mass = rejection(error, score, ids, high, fraction)
                performances.append({"study": study, "method": method, "rejected_fraction": fraction,
                    "remaining_mae": mae, "high_error_recall": recall,
                    "error_mass_rejected": mass, "baseline_mae": float(error.mean()),
                    "relative_mae_change_pct": float(100 * (mae / error.mean() - 1))})
        rank = np.lexsort((ids, pred))
        for decile, members in enumerate(np.array_split(rank, 10), 1):
            e = error[members]
            local_high = np.zeros(len(e), dtype=bool)
            local_high[_top_indices(e, ids[members], .10)] = True
            for method in ("variance", "jensen_shannon", "random"):
                score = scores.get(method)
                for fraction in CONDITIONAL_FRACTIONS:
                    mae, recall, mass = rejection(e, score[members] if score is not None else None,
                                                  ids[members], local_high, fraction)
                    conditional.append({"study": study, "predicted_bp_decile": decile,
                        "n": len(members), "predicted_bp_min": float(pred[members].min()),
                        "predicted_bp_max": float(pred[members].max()), "method": method,
                        "rejected_fraction": fraction, "baseline_mae": float(e.mean()),
                        "remaining_mae": mae, "relative_mae_change_pct": float(100 * (mae / e.mean() - 1)),
                        "high_error_recall": recall, "error_mass_rejected": mass})
        summaries.append({"study": study, "n": len(ids), "sample_bp_mae": float(error.mean()),
            "phase9_full_bp_mae": float(json.loads((RUN / "summaries" / f"outer_{study}.json").read_text())["bp"]["mae"]),
            "variance_js_spearman": float(spearmanr(scores["variance"], scores["jensen_shannon"]).statistic),
            "max_phase9_mean_difference": float(np.max(np.abs(pred - _phase9_mean(study)))),
            "max_phase9_variance_difference": float(np.max(np.abs(scores["variance"] - scores["phase9_variance"])))})
    _csv(REPORT / "rejection_performance.csv", performances)
    _csv(REPORT / "associations.csv", associations)
    _csv(REPORT / "within_predicted_bp_deciles.csv", conditional)
    _csv(REPORT / "within_predicted_bp_summary.csv", _conditional_summary(conditional))
    _csv(REPORT / "study_summary.csv", summaries)
    _plots(performances, plots)
    _json(REPORT / "provenance.json", {"sample": json.loads((OUT / "sample/complete.json").read_text()),
        "predictions": json.loads((OUT / "predictions/complete.json").read_text()),
        "phase8_manifest_sha256": sha256(PHASE8 / "manifest.json"),
        "checkpoint_sha256": {f"outer_{study}_member_{member}": sha256(
            PHASE8 / "models" / f"outer_{study}_member_{member}" / "checkpoints/epoch_100.pt")
            for study in range(1, 6) for member in range(5)},
        "definitions": {"variance": "Population variance of five member BP probabilities",
            "jensen_shannon": "Bernoulli entropy of member mean minus mean member entropy (natural-log nats)",
            "high_error": "Top 10% absolute BP error within each sampled outer study",
            "conditional_strata": "Ten equal-count predicted-BP strata within each sampled outer study",
            "ties": "Descending score, ascending cell ID"}})


def _phase9_mean(study):
    return pq.read_table(prediction_path(study), columns=["phase9_bp_mean"])["phase9_bp_mean"].to_numpy()


def _conditional_summary(rows):
    output = []
    for study in range(1, 6):
        for fraction in CONDITIONAL_FRACTIONS:
            for method in ("variance", "jensen_shannon", "random"):
                selected = [r for r in rows if r["study"] == study and r["rejected_fraction"] == fraction
                            and r["method"] == method]
                n = sum(r["n"] for r in selected)
                baseline = sum(r["n"] * r["baseline_mae"] for r in selected) / n
                remaining = sum(r["n"] * r["remaining_mae"] for r in selected) / n
                output.append({"study": study, "method": method, "rejected_fraction_within_decile": fraction,
                    "n": n, "baseline_mae": baseline, "remaining_mae": remaining,
                    "relative_mae_change_pct": 100 * (remaining / baseline - 1),
                    "mean_within_decile_high_error_recall": float(np.mean([r["high_error_recall"] for r in selected]))})
    return output


def _plots(performance, pairs):
    colors = {"variance": "#225ea8", "jensen_shannon": "#ba4a00",
              "predicted_bp": "#238b45", "random": "#555555"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for method in METHODS:
        selected = [r for r in performance if r["method"] == method]
        for ax, metric in zip(axes, ("remaining_mae", "high_error_recall"), strict=True):
            y = [np.mean([r[metric] / r["baseline_mae"] if metric == "remaining_mae" else r[metric]
                          for r in selected if r["rejected_fraction"] == fraction]) for fraction in FRACTIONS]
            ax.plot(np.array(FRACTIONS) * 100, y, marker="o", color=colors[method],
                    label=method.replace("_", " "))
    axes[0].set(ylabel="Remaining MAE / initial MAE", ylim=(.5, 1.03))
    axes[1].set(ylabel="Recall of within-study top-error decile", ylim=(0, 1))
    for ax in axes:
        ax.set_xlabel("Rejected cells within each study (%)")
        ax.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Paired 100,000-cell outer TEST samples; equal-study mean")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(REPORT / f"risk_coverage.{suffix}", dpi=240)
    plt.close(fig)
    fig, grid = plt.subplots(2, 3, figsize=(11.5, 7))
    axes = list(grid.flat)[:5]
    last_hexbin = None
    for study, (ax, (var, js)) in enumerate(zip(axes, pairs, strict=True), 1):
        positive = (var > 0) & (js > 0)
        last_hexbin = ax.hexbin(var[positive], js[positive], xscale="log", yscale="log",
                               gridsize=45, mincnt=1, cmap="viridis", norm=LogNorm(vmin=1, vmax=1000))
        ax.set_title(f"Study {study}\nρ={spearmanr(var, js).statistic:.3f}")
        ax.set_xlim(1e-12, 1e-4)
        ax.set_ylim(1e-9, 1e-2)
        ax.tick_params(labelbottom=study > 3, labelleft=study in (1, 4))
        ax.grid(alpha=.15)
    grid[0, 0].set_ylabel("Jensen–Shannon divergence (nats)")
    grid[1, 0].set_ylabel("Jensen–Shannon divergence (nats)")
    fig.colorbar(last_hexbin, cax=grid[1, 2], label="Cells per hexagon")
    fig.supxlabel("Between-member BP variance (probability squared)")
    fig.suptitle("Variance and Jensen–Shannon rank agreement on identical TEST cells")
    fig.tight_layout(rect=(0, .03, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(REPORT / f"variance_vs_js.{suffix}", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "run", "analyze"))
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == "__main__":
    main()
