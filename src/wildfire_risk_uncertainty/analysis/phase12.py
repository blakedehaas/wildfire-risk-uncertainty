"""Phase 12: conditional FLP reconstruction and ensemble-disagreement utility."""

import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from wildfire_risk_uncertainty.training.phase9 import ROOT, RUN

OUT = ROOT / "reports/uncertainty/12"
CANONICAL = ROOT / "artifacts/dataset/canonical/partitions"
FOLDS = ROOT / "artifacts/spatial_validation/05b/partitions"
CLASSES = tuple(f"FLP{i}" for i in range(1, 7))
FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.50)
METHODS = ("disagreement", "entropy", "random")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def columns(table, names):
    return np.column_stack([table[name].to_numpy(zero_copy_only=False) for name in names])


def validate_distribution(values):
    if values.ndim != 2 or values.shape[1] != 6 or not len(values):
        raise ValueError("Expected nonempty six-class FLP distributions")
    if not np.isfinite(values).all() or np.any((values < -1e-7) | (values > 1 + 1e-7)):
        raise ValueError("Invalid FLP probabilities")
    if np.max(np.abs(values.sum(axis=1, dtype=np.float64) - 1)) > 1e-5:
        raise ValueError("FLP distribution is not normalized")


def total_variation(pred, target):
    return 0.5 * np.abs(pred.astype(np.float64) - target).sum(axis=1)


def disagreement(variances):
    if not np.isfinite(variances).all() or np.any(variances < 0):
        raise ValueError("Invalid between-member variances")
    return variances.sum(axis=1, dtype=np.float64)


def categorical_entropy(pred):
    validate_distribution(pred)
    p = pred.astype(np.float64)
    return -np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0).sum(axis=1)


def train_constant(targets, eligible, valid):
    mask = eligible & valid
    if not mask.any():
        raise ValueError("No eligible FLP-valid TRAIN cells")
    selected = targets[mask]
    validate_distribution(selected)
    return selected.sum(axis=0, dtype=np.float64), int(mask.sum())


def stable_descending(score, ids):
    """Largest score first, with smaller cell_id first at exact score ties."""
    return np.lexsort((ids, -score))


def high_error_mask(error, ids):
    mask = np.zeros(len(error), dtype=bool)
    mask[stable_descending(error, ids)[:round(len(error) * .10)]] = True
    return mask


def rejection_rows(study, method, score, error, high, ids=None):
    n = len(error)
    if method == "random":
        return [{"study": study, "method": method, "rejected_fraction": fraction,
                 "rejected_n": round(n * fraction), "remaining_mean_tv": float(error.mean()),
                 "high_error_recall": round(n * fraction) / n} for fraction in FRACTIONS]
    order = stable_descending(score, ids if ids is not None else np.arange(n, dtype=np.int64))
    ordered_error = error[order]
    ordered_high = high[order]
    cumulative_error = np.cumsum(ordered_error, dtype=np.float64)
    cumulative_high = np.cumsum(ordered_high, dtype=np.int64)
    total_error = float(cumulative_error[-1])
    high_n = int(high.sum())
    rows = []
    for fraction in FRACTIONS:
        k = round(n * fraction)
        removed_error = float(cumulative_error[k - 1]) if k else 0.0
        found = int(cumulative_high[k - 1]) if k else 0
        rows.append({"study": study, "method": method, "rejected_fraction": fraction,
                     "rejected_n": k, "remaining_mean_tv": (total_error - removed_error) / (n - k),
                     "high_error_recall": found / high_n})
    return rows


def quintile_rows(study, score, error, ids):
    order = np.lexsort((ids, score))
    rows = []
    for group, chosen in enumerate(np.array_split(order, 5), 1):
        rows.append({"study": study, "quintile": group, "n": len(chosen),
                     "mean_tv": float(error[chosen].mean(dtype=np.float64)),
                     "min_disagreement": float(score[chosen].min()),
                     "max_disagreement": float(score[chosen].max())})
    return rows


def aggregate(rows):
    result = []
    for method in METHODS:
        for fraction in FRACTIONS:
            selected = [r for r in rows if r["method"] == method and r["rejected_fraction"] == fraction]
            for label in ("equal_study", "pooled_cell_weighted"):
                tv_weights = (np.ones(5) if label == "equal_study" else
                              np.array([r["study_n"] - r["rejected_n"] for r in selected], dtype=np.float64))
                recall_weights = (np.ones(5) if label == "equal_study" else
                                  np.array([r["high_error_n"] for r in selected], dtype=np.float64))
                result.append({"aggregation": label, "method": method, "rejected_fraction": fraction,
                               "remaining_mean_tv": float(np.average([r["remaining_mean_tv"] for r in selected], weights=tv_weights)),
                               "high_error_recall": float(np.average([r["high_error_recall"] for r in selected], weights=recall_weights))})
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def study_paths(study):
    return sorted((RUN / "predictions" / f"outer_{study}").glob("*.parquet"))


def fold_and_target(path, study):
    fold = pq.read_table(FOLDS / path, columns=["cell_id", "outer_fold", f"train_eligible_fold_{study}"])
    target = pq.read_table(CANONICAL / path, columns=["cell_id", "flp_target_valid", *CLASSES])
    ids = fold["cell_id"].to_numpy()
    if not np.array_equal(ids, target["cell_id"].to_numpy()):
        raise ValueError(f"Canonical/fold cell-ID mismatch: {path}")
    test = fold["outer_fold"].to_numpy() == study
    eligible = fold[f"train_eligible_fold_{study}"].to_numpy().astype(bool)
    if np.any(test & eligible):
        raise ValueError("TRAIN/TEST overlap")
    valid = target["flp_target_valid"].to_numpy().astype(bool)
    return ids, test, eligible, valid, columns(target, CLASSES)


def study_analysis(study, pilot=False):
    paths = study_paths(study)
    if not paths:
        raise ValueError(f"Missing Phase 9 predictions for study {study}")
    # The frozen TRAIN eligibility is defined over all canonical partitions, including
    # tiles without this study's TEST predictions.
    sums = np.zeros(6, dtype=np.float64)
    train_n = 0
    for path in sorted(FOLDS.glob("*.parquet")):
        _, _, eligible, valid, target = fold_and_target(path.name, study)
        piece, count = train_constant(target, eligible, valid) if np.any(eligible & valid) else (np.zeros(6), 0)
        sums += piece
        train_n += count
    baseline = sums / train_n
    validate_distribution(baseline[None, :])

    ids_parts, error_parts, score_parts, entropy_parts = [], [], [], []
    total_tv = baseline_tv = 0.0
    test_n = valid_n = 0
    for path in paths[:2] if pilot else paths:
        tile = path.stem
        part = json.loads((RUN / "partitions" / f"outer_{study}" / f"{tile}.json").read_text())
        sidecar = json.loads(path.with_suffix(".sha256.json").read_text())
        if (sha256(path) != sidecar["sha256"] or sidecar["n"] != part["n"] or
                sidecar["input_cell_id_sha256"] != part["test_cell_id_sha256"] or
                part["manifest_sha256"] != json.loads((RUN / "complete.json").read_text())["manifest_sha256"]):
            raise ValueError(f"Prediction hash or sidecar mismatch: {path}")
        ids, test, _, valid, target = fold_and_target(part["tile_name"], study)
        selected_ids = ids[test]
        if len(selected_ids) != part["n"] or hashlib.sha256(np.ascontiguousarray(selected_ids).tobytes()).hexdigest() != part["test_cell_id_sha256"]:
            raise ValueError("Phase 9 TEST support changed")
        pred = pq.read_table(path, columns=["cell_id", *[f"flp{k}_mean" for k in range(1, 7)],
                                             *[f"flp{k}_between_variance" for k in range(1, 7)]])
        if not np.array_equal(pred["cell_id"].to_numpy(), selected_ids):
            raise ValueError("Prediction cell IDs do not match TEST")
        all_predictions = columns(pred, [f"flp{k}_mean" for k in range(1, 7)])
        validate_distribution(all_predictions)
        keep = valid[test]
        test_n += len(selected_ids)
        valid_n += int(keep.sum())
        if not keep.any():
            continue
        y = target[test][keep]
        p = all_predictions[keep]
        v = columns(pred, [f"flp{k}_between_variance" for k in range(1, 7)])[keep]
        validate_distribution(y)
        validate_distribution(p)
        error = total_variation(p, y)
        tile_flp = part["stats"]["flp"]
        if int(keep.sum()) != tile_flp["n"] or abs(float(error.mean()) - tile_flp["tv"] / tile_flp["n"]) > 2e-7:
            raise ValueError(f"Phase 9 tile FLP support or TV changed: {path}")
        base_error = total_variation(np.broadcast_to(baseline, y.shape), y)
        total_tv += error.sum(dtype=np.float64)
        baseline_tv += base_error.sum(dtype=np.float64)
        ids_parts.append(selected_ids[keep])
        error_parts.append(error.astype(np.float32))
        score_parts.append(disagreement(v).astype(np.float32))
        entropy_parts.append(categorical_entropy(p).astype(np.float32))
    if pilot:
        return {"pilot_tiles": len(paths[:2]), "pilot_test_n": test_n, "pilot_flp_n": valid_n,
                "pilot_mean_tv": total_tv / valid_n, "train_n": train_n}
    previous = json.loads((RUN / "summaries" / f"outer_{study}.json").read_text())["flp"]
    if valid_n != previous["n"] or test_n != json.loads((RUN / "summaries" / f"outer_{study}.json").read_text())["bp"]["n"]:
        raise ValueError("Phase 9 FLP or TEST support mismatch")
    mean_tv = total_tv / valid_n
    if abs(mean_tv - previous["mean_tv"]) > 2e-7:
        raise ValueError("Phase 9 ensemble FLP TV reproduction failed")
    ids = np.concatenate(ids_parts)
    error = np.concatenate(error_parts)
    score = np.concatenate(score_parts)
    entropy = np.concatenate(entropy_parts)
    high = high_error_mask(error, ids)
    rows = []
    for method, ranking in (("disagreement", score), ("entropy", entropy), ("random", None)):
        rows.extend(rejection_rows(study, method, ranking, error, high, ids))
    for row in rows:
        row["study_n"] = valid_n
        row["high_error_n"] = int(high.sum())
    summary = {"study": study, "test_n": test_n, "flp_valid_n": valid_n, "train_flp_valid_n": train_n,
               "ensemble_mean_tv": mean_tv, "train_constant_mean_tv": baseline_tv / valid_n,
               "tv_improvement_pct": 100 * (1 - total_tv / baseline_tv),
               "disagreement_tv_spearman": float(spearmanr(score, error).statistic),
               "high_error_n": int(high.sum()), "high_error_min_tv": float(error[high].min()),
               **{f"train_constant_{name}": float(baseline[i]) for i, name in enumerate(CLASSES)},
               **{f"{name}_mae": previous["class_mae"][i] for i, name in enumerate(CLASSES)},
               **{f"{name}_bias": previous["class_bias"][i] for i, name in enumerate(CLASSES)}}
    quintiles = quintile_rows(study, score, error, ids)
    return summary, rows, quintiles


def figures(summaries, rejection, quintiles):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    colors = ["#225ea8", "#41b6c4", "#238b45", "#88419d", "#d95f0e"]
    studies = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(studies - .18, [r["ensemble_mean_tv"] for r in summaries], width=.36, label="Ensemble", color="#225ea8")
    ax.bar(studies + .18, [r["train_constant_mean_tv"] for r in summaries], width=.36, label="TRAIN-constant", color="#fdae6b")
    ax.set(xlabel="Geographic study", ylabel="Mean TV = ½ Σ |predicted FLP − FSim FLP|", xticks=studies,
           title="Conditional FLP distribution reconstruction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "reconstruction_performance.png", dpi=240)
    fig.savefig(OUT / "reconstruction_performance.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for study, color in zip(studies, colors, strict=True):
        rows = [r for r in quintiles if r["study"] == study]
        ax.plot(studies[:5], [r["mean_tv"] for r in rows], marker="o", lw=2, color=color, label=f"Study {study}")
    ax.set(xlabel="FLP disagreement quintile (low → high; equal cell counts)", ylabel="Mean TV error",
           xticks=studies, title="Between-member FLP disagreement versus reconstruction error")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "disagreement_vs_error.png", dpi=240)
    fig.savefig(OUT / "disagreement_vs_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    x = np.arange(6)
    labels = [f"Study {i}" for i in studies] + ["Equal-study\nmean"]
    styles = [("disagreement", "#225ea8", -0.24), ("entropy", "#88419d", 0), ("random", "#999999", .24)]
    for method, color, offset in styles:
        selected = [r for r in rejection if r["method"] == method and r["rejected_fraction"] == .20]
        values = [r["high_error_recall"] for r in selected]
        ax.bar(x + offset, values + [float(np.mean(values))], width=.23, color=color,
               label={"disagreement": "FLP disagreement", "entropy": "Predicted FLP entropy",
                      "random": "Random expectation"}[method])
    ax.set(xticks=x, xticklabels=labels, ylabel="Recall of study's top 10% TV errors",
           xlabel="20% of FLP-valid TEST cells rejected", ylim=(0, 1),
           title="Detection of large FLP reconstruction errors")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "high_error_recall_20pct.png", dpi=240)
    fig.savefig(OUT / "high_error_recall_20pct.pdf")
    plt.close(fig)


def main():
    gate = json.loads((RUN / "predictions_complete.json").read_text())
    complete = json.loads((RUN / "complete.json").read_text())
    if gate["studies"] != 5 or gate["manifest_sha256"] != complete["manifest_sha256"]:
        raise ValueError("Phase 9 completion or manifest mismatch")
    if sha256(RUN / "evaluation_manifest.json") != gate["manifest_sha256"]:
        raise ValueError("Phase 9 evaluation manifest hash changed")
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = study_analysis(1, pilot=True)
    (OUT / "pilot.json").write_text(json.dumps(pilot, indent=2) + "\n")
    summaries, rejection, quintiles = [], [], []
    for study in range(1, 6):
        summary, rows, groups = study_analysis(study)
        summaries.append(summary)
        rejection.extend(rows)
        quintiles.extend(groups)
        print(f"Study {study}: {summary['flp_valid_n']:,} FLP cells; TV={summary['ensemble_mean_tv']:.4f}; ρ={summary['disagreement_tv_spearman']:.3f}", flush=True)
    write_csv(OUT / "study_summary.csv", summaries)
    write_csv(OUT / "rejection_performance.csv", rejection)
    write_csv(OUT / "disagreement_quintiles.csv", quintiles)
    # Rejection counts differ by at most rounding; use the original study support
    # for the pooled cell-weighted descriptive average.
    aggregates = aggregate(rejection)
    write_csv(OUT / "rejection_aggregates.csv", aggregates)
    figures(summaries, rejection, quintiles)
    provenance = {"phase9_run": str(RUN.relative_to(ROOT)), "phase9_manifest_sha256": gate["manifest_sha256"],
                  "phase9_predictions_complete_sha256": sha256(RUN / "predictions_complete.json"),
                  "canonical_manifest_sha256": sha256(ROOT / "reports/dataset/canonical/manifest.json"),
                  "fold_manifest_sha256": sha256(ROOT / "reports/spatial_validation/05b/manifest.json"),
                  "phase9_summary_sha256": {str(i): sha256(RUN / "summaries" / f"outer_{i}.json") for i in range(1, 6)},
                  "prediction_partition_count": {str(i): len(study_paths(i)) for i in range(1, 6)},
                  "canonical_targets": str(CANONICAL.relative_to(ROOT)), "folds": str(FOLDS.relative_to(ROOT)),
                  "pilot": pilot, "definitions": {"valid": "canonical flp_target_valid on frozen Phase 9 outer TEST cell IDs",
                  "tv": "0.5 times the sum of six absolute predicted-minus-reference FLP differences",
                  "disagreement": "sum of six population between-member FLP variances exported by Phase 9",
                  "baseline": "mean six-class target distribution among FLP-valid, train_eligible_fold_i cells",
                  "entropy": "negative sum of p log p over the six ensemble-mean FLP probabilities",
                  "high_error": "largest round(0.10*n) TV errors within study",
                  "rejection": "largest round(fraction*n) ranking scores within study at 5%, 10%, 20%, 50%",
                  "ties": "exact ties broken by ascending cell_id; exact counts via Python round",
                  "random": "uniform random rejection expectation: unchanged remaining mean TV and recall equal to rejected fraction"}}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
