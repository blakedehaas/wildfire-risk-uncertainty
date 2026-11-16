"""Phase 10A: exact study-level BP rejection analysis of frozen Phase 9 exports."""
import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from wildfire_risk_uncertainty.training.phase9 import ROOT, RUN

OUT = ROOT / "reports/uncertainty/10a"
FRACTIONS = (0, .05, .10, .20, .50)
METHODS = ("disagreement", "predicted_bp", "entropy", "random")
EXPECTED = (19595079, 19594646, 21984912, 22597733, 27044861)


def entropy(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log1p(-p)


def correlation_sums(x, y):
    return np.array([len(x), x.sum(dtype=np.float64), y.sum(dtype=np.float64),
        np.dot(x.astype(np.float64), x.astype(np.float64)),
        np.dot(y.astype(np.float64), y.astype(np.float64)),
        np.dot(x.astype(np.float64), y.astype(np.float64))], dtype=np.float64)


def pearson_from_sums(s):
    n, sx, sy, sxx, syy, sxy = s
    return float((sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n)))


def rejection_metrics(score, error, high_error, fraction):
    """Reject the largest score; ties at the boundary have arbitrary order."""
    n = len(error)
    k = round(n * fraction)
    if k == 0:
        return float(error.mean()), 0., 0.
    rejected = np.argpartition(score, n - k)[n - k:]
    # The fractional threshold has no material tie ambiguity for these exports.
    remain_mae = float((error.sum(dtype=np.float64) - error[rejected].sum(dtype=np.float64)) / (n - k))
    hit = float(high_error[rejected].sum() / high_error.sum())
    error_share = float(error[rejected].sum(dtype=np.float64) / error.sum(dtype=np.float64))
    return remain_mae, hit, error_share


def exact_tail_mask(values, fraction, largest):
    """Select an exact fraction; partition order resolves threshold ties."""
    n = len(values)
    k = round(n * fraction)
    result = np.zeros(n, dtype=bool)
    if k:
        selected = np.argpartition(values, n - k)[n - k:] if largest else np.argpartition(values, k - 1)[:k]
        result[selected] = True
    return result


def load_study(study):
    """Read one tile at a time, checking frozen TEST membership and row order."""
    folder = RUN / "predictions" / f"outer_{study}"
    paths = sorted(folder.glob("*.parquet"))
    if not paths:
        raise ValueError(f"Missing full-support predictions for study {study}")
    ids_all, bp_all, pred_all, var_all = [], [], [], []
    for path in paths:
        tile = path.stem
        part = json.loads((RUN / "partitions" / f"outer_{study}" / f"{tile}.json").read_text())
        canonical = ROOT / "artifacts/dataset/canonical/partitions" / part["tile_name"]
        fold = ROOT / "artifacts/spatial_validation/05b/partitions" / part["tile_name"]
        f = pq.read_table(fold, columns=["cell_id", "outer_fold", f"train_eligible_fold_{study}"])
        mask = f["outer_fold"].to_numpy() == study
        if np.any(mask & f[f"train_eligible_fold_{study}"].to_numpy()):
            raise ValueError("TRAIN/TEST overlap")
        ids = f["cell_id"].to_numpy()[mask]
        if len(ids) != part["n"] or hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest() != part["test_cell_id_sha256"]:
            raise ValueError("Frozen TEST support changed")
        target = pq.read_table(canonical, columns=["cell_id", "BP"])
        if not np.array_equal(target["cell_id"].to_numpy()[mask], ids):
            raise ValueError("Canonical target alignment changed")
        prediction = pq.read_table(path, columns=["cell_id", "bp_mean", "bp_between_variance"])
        if not np.array_equal(prediction["cell_id"].to_numpy(), ids):
            raise ValueError("Prediction cell order does not match TEST")
        ids_all.append(ids)
        bp_all.append(target["BP"].to_numpy()[mask].astype(np.float32))
        pred_all.append(prediction["bp_mean"].to_numpy())
        var_all.append(prediction["bp_between_variance"].to_numpy())
    ids, bp, pred, var = (np.concatenate(x) for x in (ids_all, bp_all, pred_all, var_all))
    if len(ids) != EXPECTED[study - 1] or not np.isfinite(bp).all() or not np.isfinite(pred).all() or not np.isfinite(var).all():
        raise ValueError("Invalid full-support arrays")
    if np.any((pred < 0) | (pred > 1) | (var < 0)):
        raise ValueError("Invalid prediction range")
    return ids, bp, pred, var


def study_analysis(study):
    ids, bp, pred, var = load_study(study)
    error = np.abs(pred.astype(np.float64) - bp).astype(np.float32)
    previous = json.loads((RUN / "summaries" / f"outer_{study}.json").read_text())
    mae = float(error.mean(dtype=np.float64))
    if abs(mae - previous["bp"]["mae"]) > 2e-7:
        raise ValueError("Phase 9 MAE reproduction failed")
    # Fixed definitions: top error decile and bottom disagreement quintile, within study.
    high = exact_tail_mask(error, .10, largest=True)
    low = exact_tail_mask(var, .20, largest=False)
    cutoff = float(error[high].min())
    low_cutoff = float(var[low].max())
    failures = high & low
    predicted_bins = [0, .001, .005, .01, .02, .05, .1, 1.000001]
    ranges = []
    for lo, hi in itertools.pairwise(predicted_bins):
        region = (pred >= lo) & (pred < hi)
        ranges.append({"study": study, "predicted_bp_lower": lo, "predicted_bp_upper": hi,
            "n": int(region.sum()), "high_error_low_disagreement_n": int((region & failures).sum()),
            "failure_rate": float((region & failures).sum() / region.sum()) if region.any() else None})
    sample = ids % 101 == 0
    sampled = (var[sample], error[sample], pred[sample])
    score_arrays = {"disagreement": var, "predicted_bp": pred, "entropy": entropy(pred)}
    rows = []
    for method in METHODS:
        score = score_arrays.get(method)
        for fraction in FRACTIONS:
            if method == "random":
                metrics = (mae, fraction, fraction)
            else:
                metrics = rejection_metrics(score, error, high, fraction)
            rows.append({"study": study, "method": method, "rejected_fraction": fraction,
                "remaining_mae": metrics[0], "high_error_recall": metrics[1], "error_mass_rejected": metrics[2]})
    associations = {"study": study, "n": len(error), "pearson": pearson_from_sums(correlation_sums(var, error)),
        "spearman": float(spearmanr(var, error).statistic),
        "predicted_bp_spearman": float(spearmanr(pred, error).statistic),
        "entropy_spearman": float(spearmanr(score_arrays["entropy"], error).statistic),
        "predicted_bp_at_or_above_half": int((pred >= .5).sum())}
    failure = {"study": study, "n": len(error), "high_error_threshold": cutoff,
        "low_disagreement_threshold": low_cutoff, "high_error_n": int(high.sum()), "low_disagreement_n": int(low.sum()),
        "failure_n": int(failures.sum()), "failure_fraction": float(failures.mean()),
        "fraction_of_high_errors_missed": float(failures.sum() / high.sum()),
        "fraction_of_total_absolute_error": float(error[failures].sum(dtype=np.float64) / error.sum(dtype=np.float64)),
        "mean_predicted_bp": float(pred[failures].mean()) if failures.any() else None,
        "mean_target_bp": float(bp[failures].mean()) if failures.any() else None}
    return rows, associations, failure, ranges, correlation_sums(var, error), sampled


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def conditional_sample_rows(samples):
    """Check disagreement utility within predeclared predicted-BP deciles on fixed sample."""
    output = []
    for study, (variance, error, prediction) in enumerate(samples, 1):
        order = np.argsort(prediction, kind="stable")
        for decile, indices in enumerate(np.array_split(order, 10), 1):
            v, e, p = variance[indices], error[indices], prediction[indices]
            rejected = exact_tail_mask(v, .10, largest=True)
            output.append({"study": study, "predicted_bp_decile": decile, "n": len(indices),
                "predicted_bp_min": float(p.min()), "predicted_bp_max": float(p.max()),
                "disagreement_error_spearman": float(spearmanr(v, e).statistic),
                "baseline_mae": float(e.mean()), "remaining_mae_after_10pct_disagreement_rejection": float(e[~rejected].mean()),
                "relative_mae_change_pct": float(100 * (e[~rejected].mean() / e.mean() - 1))})
    return output


def conditional_summary(rows):
    output = []
    for study in range(1, 6):
        selected = [r for r in rows if r["study"] == study]
        n = sum(r["n"] for r in selected)
        baseline = sum(r["n"] * r["baseline_mae"] for r in selected) / n
        remaining = sum(r["n"] * r["remaining_mae_after_10pct_disagreement_rejection"] for r in selected) / n
        output.append({"study": study, "sample_n": n, "baseline_mae": baseline,
            "remaining_mae": remaining, "relative_mae_change_pct": 100 * (remaining / baseline - 1),
            "positive_within_decile_spearman_count": sum(r["disagreement_error_spearman"] > 0 for r in selected)})
    return output


def figures(rows, associations, samples):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    colors = {"disagreement": "#205493", "predicted_bp": "#d95f02", "entropy": "#7b3294", "random": "#555555"}
    labels = {"disagreement": "Ensemble BP variance", "predicted_bp": "Predicted BP / entropy", "random": "Random (expected)"}
    for method in ("disagreement", "predicted_bp", "random"):
        for metric, ax in (("remaining_mae", axes[0]), ("high_error_recall", axes[1])):
            y = [np.mean([r[metric] for r in rows if r["method"] == method and r["rejected_fraction"] == f]) for f in FRACTIONS]
            ax.plot(np.array(FRACTIONS) * 100, y, "o-", lw=2, ms=4, color=colors[method], label=labels[method] if metric == "remaining_mae" else None)
    axes[0].set(ylabel="Remaining BP MAE (equal-study mean)", xlabel="Cells rejected within each study (%)")
    axes[1].set(ylabel="Top-decile errors found (equal-study mean)", xlabel="Cells rejected within each study (%)", ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out FSim BP reconstruction: exact study-level rejection")
    fig.tight_layout(); fig.savefig(OUT / "risk_coverage.png", dpi=240); fig.savefig(OUT / "risk_coverage.pdf"); plt.close(fig)
    v = np.concatenate([s[0] for s in samples]); e = np.concatenate([s[1] for s in samples])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    h = axes[0].hexbin(v, e, gridsize=75, bins="log", mincnt=1, cmap="viridis")
    fig.colorbar(h, ax=axes[0], label="Cell count (log scale)")
    axes[0].set(xlabel="Between-member BP variance", ylabel="Absolute BP error", title="Fixed 1/101 diagnostic sample")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for a in associations[:5]:
        axes[1].scatter(a["study"], a["spearman"], color="#205493", s=40)
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].set(xlabel="Geographic study", ylabel="Full-support Spearman ρ", xticks=range(1, 6), title="Disagreement versus absolute error")
    fig.tight_layout(); fig.savefig(OUT / "disagreement_vs_error.png", dpi=240); fig.savefig(OUT / "disagreement_vs_error.pdf"); plt.close(fig)


def main():
    if not (RUN / "predictions_complete.json").exists() or not (RUN / "complete.json").exists():
        raise ValueError("Complete frozen Phase 9 predictions required")
    gate = json.loads((RUN / "predictions_complete.json").read_text())
    if gate["studies"] != 5 or gate["manifest_sha256"] != json.loads((RUN / "complete.json").read_text())["manifest_sha256"]:
        raise ValueError("Phase 9 provenance mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, associations, failures, ranges, sums, samples = [], [], [], [], np.zeros(6), []
    for study in range(1, 6):
        r, a, f, b, s, sample = study_analysis(study)
        rows.extend(r); associations.append(a); failures.append(f); ranges.extend(b); sums += s; samples.append(sample)
        print(f"Study {study}: {a['n']:,} TEST cells; ρ={a['spearman']:.3f}", flush=True)
    v = np.concatenate([s[0] for s in samples]); e = np.concatenate([s[1] for s in samples])
    associations.append({"study": "pooled", "n": int(sums[0]), "pearson": pearson_from_sums(sums),
        "spearman": float(spearmanr(v, e).statistic), "spearman_basis": "cell_id % 101 == 0; diagnostic sample",
        "sample_n": len(v)})
    for r in rows: r["relative_mae_change_pct"] = 100 * (r["remaining_mae"] / next(x["remaining_mae"] for x in rows if x["study"] == r["study"] and x["method"] == r["method"] and x["rejected_fraction"] == 0) - 1)
    write_csv(OUT / "rejection_performance.csv", rows)
    write_csv(OUT / "associations.csv", [{k: a.get(k) for k in ("study", "n", "pearson", "spearman", "predicted_bp_spearman", "entropy_spearman", "predicted_bp_at_or_above_half", "spearman_basis", "sample_n")} for a in associations])
    write_csv(OUT / "low_disagreement_high_error.csv", failures)
    write_csv(OUT / "failure_by_predicted_bp.csv", ranges)
    conditional = conditional_sample_rows(samples)
    write_csv(OUT / "conditional_bp_deciles_sample.csv", conditional)
    write_csv(OUT / "conditional_summary_sample.csv", conditional_summary(conditional))
    figures(rows, associations, samples)
    (OUT / "provenance.json").write_text(json.dumps({"phase9_manifest_sha256": gate["manifest_sha256"],
        "source": str(RUN.relative_to(ROOT)), "full_support_bp_n": int(sums[0]), "pooled_spearman_sample_n": len(v),
        "definitions": {"disagreement": "population variance of five BP member probabilities", "error": "absolute difference between ensemble mean BP and FSim BP", "high_error": "top 10% absolute errors within each study", "low_disagreement": "bottom 20% between-member BP variance within each study", "rejection": "largest ranking score within each study", "random": "expected remaining MAE and recall under uniform random rejection"}}, indent=2) + "\n")


if __name__ == "__main__": main()
