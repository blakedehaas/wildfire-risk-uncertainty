"""Read-only analysis of the completed outer-3 GradNorm HPO pilot."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = "wildfire_outer_3_gradnorm_hpo_v3"
DB = ROOT / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
RECORDS = ROOT / "artifacts/hpo/records" / STUDY
OUT = ROOT / "reports/hpo/07c_outer3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def study_trials(db: Path) -> list[dict]:
    """Use SQLite's read-only URI; never create or update the study database."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("""SELECT t.trial_id, t.number, t.state,
            t.datetime_start, t.datetime_complete, v.value
            FROM trials t JOIN studies s ON s.study_id=t.study_id
            LEFT JOIN trial_values v ON v.trial_id=t.trial_id AND v.objective=0
            WHERE s.study_name=? ORDER BY t.number""", (STUDY,)).fetchall()
        trials = []
        for row in rows:
            params = connection.execute("SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?", (row["trial_id"],)).fetchall()
            decoded = {}
            for param in params:
                distribution = json.loads(param["distribution_json"])
                value = param["param_value"]
                if distribution["name"] == "CategoricalDistribution":
                    value = distribution["attributes"]["choices"][int(value)]
                decoded[param["param_name"]] = value
            trials.append({"number": row["number"], "state": row["state"], "value": row["value"],
                           "start": row["datetime_start"], "end": row["datetime_complete"], **decoded})
        return trials
    finally:
        connection.close()


def summarize_folds(folds: list[dict]) -> dict:
    """Recompute the frozen equal-fold and equal-task objective from final records."""
    if len(folds) != 5 or {f["inner_fold"] for f in folds} != set(range(1, 6)):
        raise ValueError("Exactly five distinct final folds required")
    ordered = sorted(folds, key=lambda f: f["inner_fold"])
    skills = []
    bp_skills = []
    flp_skills = []
    for fold in ordered:
        reference = fold["validation_log_skill"]
        bp = 1 - fold["validation"]["bp_loss_mean"] / reference["bp_reference_ce"]
        flp = 1 - fold["validation"]["flp_loss_mean"] / reference["flp_reference_ce"]
        if not (math.isclose(bp, reference["bp"], abs_tol=1e-12) and
                math.isclose(flp, reference["flp"], abs_tol=1e-12)):
            raise ValueError("Saved task skill disagrees with frozen CE ratio")
        skills.append((bp + flp) / 2)
        bp_skills.append(bp)
        flp_skills.append(flp)
    return {"objective": statistics.fmean(skills), "fold_sd": statistics.pstdev(skills),
            "worst_fold": min(skills), "best_fold": max(skills),
            "bp_skill": statistics.fmean(bp_skills), "flp_skill": statistics.fmean(flp_skills),
            "fold_skills": skills, "fold_bp_skills": bp_skills, "fold_flp_skills": flp_skills,
            "train_seconds": sum(f["training_wall_seconds"] for f in ordered),
            "validation_seconds": sum(f["validation_wall_seconds"] for f in ordered)}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def style() -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2})


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / "figures" / name, bbox_inches="tight")
    plt.close()


def figures(rows: list[dict], audit: list[dict]) -> None:
    style()
    x = np.array([r["number"] for r in rows])
    y = np.array([r["objective"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(x, y, c=["#607d8b" if r["sampler_phase"] == "startup" else "#c75b39" for r in rows], s=45)
    ax[0].scatter([a["number"] for a in audit if a["state"] == "FAIL"], [0] * sum(a["state"] == "FAIL" for a in audit), marker="x", c="black", label="failed (axis marker)")
    ax[0].set(xlabel="Optuna trial number", ylabel="Five-fold validation skill", title="Completed trial objectives")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].step(x, np.maximum.accumulate(y), where="post", color="#315b7d")
    ax[1].set(xlabel="Optuna trial number", ylabel="Best completed skill so far", title="Cumulative best")
    save("convergence.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([r["objective"] for r in rows if r["sampler_phase"] == "startup"], bins=8, alpha=.65, label="random startup")
    ax.hist([r["objective"] for r in rows if r["sampler_phase"] == "TPE"], bins=8, alpha=.65, label="TPE")
    ax.set(xlabel="Five-fold validation skill", ylabel="Completed trials", title="Objective distribution by sampler phase")
    ax.legend(frameon=False)
    save("distribution.png")

    params = ["depth", "width", "learning_rate", "weight_decay", "batch_size"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, key in zip(axes.flat, params):
        values = np.array([r[key] for r in rows], dtype=float)
        ax.scatter(values, y, c=x, cmap="viridis", s=43, edgecolors="white", linewidths=.5)
        if key in {"learning_rate", "weight_decay"}:
            ax.set_xscale("log")
        else:
            ax.set_xticks(sorted(set(values)))
        ax.set(xlabel=key.replace("_", " "), ylabel="Validation skill")
    axes.flat[-1].axis("off")
    fig.suptitle("Observed hyperparameter slices (color: trial number)")
    save("hyperparameter_slices.png")

    matrix = np.array([[r[f"fold_{k}_skill"] for k in range(1, 6)] for r in rows])
    order = np.argsort(-y)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix[order], aspect="auto", cmap="RdBu", vmin=-.1, vmax=.35)
    ax.set(xticks=range(5), xticklabels=range(1, 6), yticks=range(len(rows)),
           yticklabels=[rows[i]["number"] for i in order], xlabel="Inner geographic fold",
           ylabel="Trial (mean skill rank)", title="Geographic validation skill by trial")
    fig.colorbar(image, ax=ax, label="Equal-task skill")
    save("fold_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([r["worst_fold"] for r in rows], y, c=x, cmap="viridis", s=55)
    ax.set(xlabel="Worst fold skill", ylabel="Five-fold mean skill", title="Mean versus worst-fold validation")
    save("robustness.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([r["bp_skill"] for r in rows], [r["flp_skill"] for r in rows], c=x, cmap="viridis", s=55)
    ax.set(xlabel="Mean BP skill", ylabel="Mean FLP skill", title="Task balance across completed trials")
    save("task_balance.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ["depth", "width", "batch_size"]):
        groups = sorted({r[key] for r in rows})
        ax.boxplot([[r["runtime_hours"] for r in rows if r[key] == group] for group in groups], tick_labels=groups)
        ax.set(xlabel=key.replace("_", " "), ylabel="Trial runtime (hours)")
    fig.suptitle("Runtime by configuration (observed, confounded)")
    save("runtime.png")


def main() -> None:
    trials = study_trials(DB)
    if len(trials) != 22 or Counter(t["state"] for t in trials) != {"COMPLETE": 20, "FAIL": 2}:
        raise ValueError("Unexpected study state; review before analysis")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    audit, summary, fold_rows = [], [], []
    prior_complete = 0
    for trial in trials:
        number = trial["number"]
        path = RECORDS / f"trial_{number:03d}"
        phase = "startup" if prior_complete < 10 else "TPE"
        audit.append({**trial, "sampler_phase": phase, "prior_complete": prior_complete,
                      "local_record_exists": path.exists()})
        if trial["state"] != "COMPLETE":
            continue
        prior_complete += 1
        manifest = read_json(path / "manifest.json")
        folds = [read_json(path / f"fold_{k}_epoch_100.json") for k in range(1, 6)]
        aggregate = read_json(path / "aggregate_epoch_100.json")
        state = read_json(path / "state.json")
        result = summarize_folds(folds)
        references = manifest["provenance"]["inner_fold_constant_references"]
        for fold in folds:
            reference = references[str(fold["inner_fold"])]
            if (fold["validation"]["bp_count"] != reference["bp_validation_rows"] or
                    fold["validation"]["flp_valid_count"] != reference["flp_validation_valid_rows"] or
                    not math.isclose(fold["validation_log_skill"]["bp_reference_ce"],
                                     reference["bp_validation_reference_ce"], abs_tol=1e-12) or
                    not math.isclose(fold["validation_log_skill"]["flp_reference_ce"],
                                     reference["flp_validation_reference_ce"], abs_tol=1e-12)):
                raise ValueError(f"Trial {number}: reference/support mismatch")
        if not (math.isclose(result["objective"], trial["value"], abs_tol=1e-12) and
                math.isclose(result["objective"], aggregate["J_e"], abs_tol=1e-12)):
            raise ValueError(f"Trial {number}: Optuna/local objective mismatch")
        if manifest["configuration"] != {k: trial[k] for k in ["depth", "width", "learning_rate", "weight_decay", "batch_size"]}:
            raise ValueError(f"Trial {number}: configuration mismatch")
        row = {"number": number, "sampler_phase": phase, "successful_order": prior_complete,
               **{k: trial[k] for k in ["depth", "width", "learning_rate", "weight_decay", "batch_size"]},
               "objective": result["objective"], "bp_skill": result["bp_skill"],
               "flp_skill": result["flp_skill"], "fold_sd": result["fold_sd"],
               "worst_fold": result["worst_fold"], "best_fold": result["best_fold"],
               "runtime_hours": state["total_runtime_seconds"] / 3600,
               "train_hours": result["train_seconds"] / 3600,
               "validation_hours": result["validation_seconds"] / 3600,
               **{f"fold_{k}_skill": result["fold_skills"][k-1] for k in range(1, 6)}}
        summary.append(row)
        for fold in folds:
            k = fold["inner_fold"]
            ref = fold["validation_log_skill"]
            fold_rows.append({"number": number, "fold": k, "objective": result["fold_skills"][k-1],
                              "bp_skill": ref["bp"], "flp_skill": ref["flp"],
                              "bp_ce": fold["validation"]["bp_loss_mean"],
                              "flp_ce": fold["validation"]["flp_loss_mean"],
                              "bp_reference_ce": ref["bp_reference_ce"],
                              "flp_reference_ce": ref["flp_reference_ce"],
                              "bp_rows": fold["validation"]["bp_count"],
                              "flp_rows": fold["validation"]["flp_valid_count"]})
    write_csv(OUT / "trial_audit.csv", audit)
    write_csv(OUT / "completed_trials.csv", summary)
    write_csv(OUT / "trial_fold_metrics.csv", fold_rows)
    reference = read_json(RECORDS / "trial_019/manifest.json")["provenance"]["inner_fold_constant_references"]
    support = list(csv.DictReader((ROOT / "reports/spatial_validation/06c/tables/target_support_by_inner_fold.csv").open()))
    centroids = list(csv.DictReader((ROOT / "reports/spatial_validation/06c/tables/inner_fold_centroids.csv").open()))
    fold_summary = []
    for k in range(1, 6):
        subset = [r for r in fold_rows if r["fold"] == k]
        valid = next(r for r in support if r["outer_study"] == "3" and r["inner_fold"] == str(k) and r["role"] == "validation")
        train = next(r for r in support if r["outer_study"] == "3" and r["inner_fold"] == str(k) and r["role"] == "train")
        center = next(r for r in centroids if r["outer_study"] == "3" and r["inner_fold"] == str(k))
        fold_summary.append({"fold": k, "mean_skill": statistics.fmean(r["objective"] for r in subset),
                             "sd_skill": statistics.pstdev(r["objective"] for r in subset),
                             "min_skill": min(r["objective"] for r in subset),
                             "max_skill": max(r["objective"] for r in subset),
                             "mean_bp_skill": statistics.fmean(r["bp_skill"] for r in subset),
                             "mean_flp_skill": statistics.fmean(r["flp_skill"] for r in subset),
                             "validation_rows": int(valid["rows"]), "train_rows": int(train["rows"]),
                             "validation_bp_mean": float(valid["bp_mean"]),
                             "train_bp_mean": float(train["bp_mean"]),
                             "validation_flp_valid_fraction": int(valid["flp_target_valid"]) / int(valid["rows"]),
                             "train_flp_valid_fraction": int(train["flp_target_valid"]) / int(train["rows"]),
                             "bp_reference_ce": reference[str(k)]["bp_validation_reference_ce"],
                             "flp_reference_ce": reference[str(k)]["flp_validation_reference_ce"],
                             "centroid_x_5070": float(center["x_5070"]),
                             "centroid_y_5070": float(center["y_5070"])})
    write_csv(OUT / "fold_summary.csv", fold_summary)
    coverage = []
    for k in range(1, 6):
        state = read_json(ROOT / f"artifacts/hpo/preprocessing_cache_v1/outer_3_inner_{k}.json")
        counts = state["state"]["missing_counts"]
        for feature, count in counts.items():
            coverage.append({"fold": k, "role": "train", "feature": feature,
                             "missing_rows": count, "train_rows": state["state"]["rows"],
                             "missing_fraction": count / state["state"]["rows"]})
    write_csv(OUT / "train_predictor_missingness.csv", coverage)
    figures(summary, audit)
    print(json.dumps({"states": dict(Counter(t["state"] for t in trials)),
                      "best": max(summary, key=lambda r: r["objective"]),
                      "best_worst_fold": max(summary, key=lambda r: r["worst_fold"]),
                      "startup": [r["objective"] for r in summary if r["sampler_phase"] == "startup"],
                      "tpe": [r["objective"] for r in summary if r["sampler_phase"] == "TPE"],
                      "folds": fold_summary}, indent=2))


if __name__ == "__main__":
    main()
