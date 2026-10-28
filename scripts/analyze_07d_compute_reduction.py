"""Read-only retrospective of the completed outer-3 GradNorm HPO study.

Run from the repository root with ``uv run python scripts/analyze_07d_compute_reduction.py``.
Only reports/hpo/07d_compute_reduction is written. Original SQLite and records
are opened/read without mutation. No outer-test data are loaded.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import Counter
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_07c_outer3_hpo import DB, RECORDS, ROOT, study_trials, summarize_folds
from scipy.stats import kendalltau, spearmanr

OUT = ROOT / "reports/hpo/07d_compute_reduction"
EPOCHS = (10, 25, 50, 75, 100)
BUDGETS = (3, 5, 8, 10, 15, 20)
CANDIDATES = (19, 20, 6, 8, 4)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_comparison(scores: dict[int, float], final: dict[int, float]) -> dict:
    """Compare a screening ranking to five-fold epoch-100 truth, with stable ties."""
    if scores.keys() != final.keys() or len(scores) < 2:
        raise ValueError("Rankings need the same candidates")
    selected = max(scores, key=lambda key: (scores[key], -key))
    winner = max(final, key=lambda key: (final[key], -key))
    keys = sorted(scores)
    return {"selected_trial": selected, "final_winner": winner,
            "top_agreement": int(selected == winner),
            "spearman": float(spearmanr([scores[k] for k in keys], [final[k] for k in keys]).statistic),
            "kendall": float(kendalltau([scores[k] for k in keys], [final[k] for k in keys]).statistic),
            "regret": final[winner] - final[selected]}


def budget_trajectory(rows: list[dict]) -> list[dict]:
    """Original completion order; failures consume no successful-trial slots."""
    ordered = sorted(rows, key=lambda row: row["successful_order"])
    if [row["successful_order"] for row in ordered] != list(range(1, len(rows) + 1)):
        raise ValueError("Nonconsecutive successful order")
    best = max(ordered, key=lambda row: row["objective"])["objective"]
    incumbent = None
    result = []
    for row in ordered:
        if incumbent is None or row["objective"] > incumbent["objective"]:
            incumbent = row
        result.append({"successful_count": row["successful_order"],
                       "last_trial": row["number"], "last_phase": row["sampler_phase"],
                       "best_trial": incumbent["number"], "best_objective": incumbent["objective"],
                       "gap_to_final_best": best - incumbent["objective"]})
    return result


def subset_budget(rows: list[dict], budgets: tuple[int, ...]) -> list[dict]:
    """Exact unordered subsets; descriptive coverage, never a TPE replay."""
    values = [row["objective"] for row in rows]
    final_best = max(values)
    result = []
    for budget in budgets:
        regrets = [final_best - max(values[i] for i in members)
                   for members in itertools.combinations(range(len(rows)), budget)]
        result.append({"budget": budget, "subset_count": len(regrets),
                       "median_regret": statistics.median(regrets),
                       "p90_regret": float(np.quantile(regrets, .9)),
                       "max_regret": max(regrets),
                       "winner_included_fraction": sum(math.isclose(x, 0, abs_tol=1e-14)
                                                       for x in regrets) / len(regrets)})
    return result


def load() -> tuple[list[dict], dict[int, dict[int, list[dict]]], list[dict]]:
    trials = study_trials(DB)
    if Counter(t["state"] for t in trials) != {"COMPLETE": 20, "FAIL": 2}:
        raise ValueError("Unexpected Optuna study states")
    rows, records, audit = [], {}, []
    successful = 0
    for trial in trials:
        phase = "startup" if successful < 10 else "TPE"
        number = trial["number"]
        audit.append({"number": number, "state": trial["state"], "phase_at_suggestion": phase})
        if trial["state"] != "COMPLETE":
            continue
        successful += 1
        path = RECORDS / f"trial_{number:03d}"
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["configuration"] != {k: trial[k] for k in
                                        ("depth", "width", "batch_size", "learning_rate", "weight_decay")}:
            raise ValueError(f"Configuration mismatch in trial {number}")
        records[number] = {}
        for epoch in EPOCHS:
            folds = [json.loads((path / f"fold_{fold}_epoch_{epoch:03d}.json").read_text())
                     for fold in range(1, 6)]
            aggregate = json.loads((path / f"aggregate_epoch_{epoch:03d}.json").read_text())
            result = summarize_folds(folds)
            if not math.isclose(result["objective"], aggregate["J_e"], abs_tol=1e-12):
                raise ValueError(f"Epoch objective mismatch: {number}/{epoch}")
            records[number][epoch] = folds
        final = summarize_folds(records[number][100])
        if not math.isclose(final["objective"], trial["value"], abs_tol=1e-12):
            raise ValueError(f"Optuna mismatch in trial {number}")
        state = json.loads((path / "state.json").read_text())
        wall = (datetime.fromisoformat(trial["end"]) - datetime.fromisoformat(trial["start"])).total_seconds()
        rows.append({"number": number, "successful_order": successful, "sampler_phase": phase,
                     **manifest["configuration"], "objective": final["objective"],
                     "bp_skill": final["bp_skill"], "flp_skill": final["flp_skill"],
                     "worst_fold": final["worst_fold"],
                     "local_runtime_hours": state["total_runtime_seconds"] / 3600,
                     "optuna_wall_hours": wall / 3600})
    return rows, records, audit


def fold_analysis(rows: list[dict], records: dict) -> list[dict]:
    final = {r["number"]: r["objective"] for r in rows}
    output = []
    for size in (2, 3, 4):
        for folds in itertools.combinations(range(1, 6), size):
            scores = {number: statistics.fmean(
                (record["validation_log_skill"]["bp"] +
                 record["validation_log_skill"]["flp"]) / 2
                for record in records[number][100] if record["inner_fold"] in folds)
                for number in final}
            output.append({"size": size, "folds": "-".join(map(str, folds)),
                           "contains_fold_2": int(2 in folds), "contains_fold_5": int(5 in folds),
                           **rank_comparison(scores, final)})
    return output


def epoch_analysis(rows: list[dict], records: dict) -> tuple[list[dict], list[dict]]:
    final = {r["number"]: r["objective"] for r in rows}
    summaries, details = [], []
    for epoch in EPOCHS:
        scores = {number: summarize_folds(records[number][epoch])["objective"] for number in final}
        comparisons = rank_comparison(scores, final)
        final_order = sorted(final, key=lambda n: (-final[n], n))
        early_order = sorted(scores, key=lambda n: (-scores[n], n))
        summaries.append({"epoch": epoch, **comparisons,
                          "top_5_overlap": len(set(final_order[:5]) & set(early_order[:5])),
                          "mean_abs_rank_change": statistics.fmean(
                              abs(final_order.index(n) - early_order.index(n)) for n in final)})
        for row in rows:
            number = row["number"]
            details.append({"epoch": epoch, "number": number, "depth": row["depth"],
                            "width": row["width"], "batch_size": row["batch_size"],
                            "objective": scores[number], "final_objective": final[number],
                            "rank": early_order.index(number) + 1,
                            "final_rank": final_order.index(number) + 1,
                            "improvement_to_100": final[number] - scores[number],
                            **{f"fold_{k}_skill": (records[number][epoch][k - 1]["validation_log_skill"]["bp"] +
                                                     records[number][epoch][k - 1]["validation_log_skill"]["flp"]) / 2
                               for k in range(1, 6)}})
    return summaries, details


def runtime_analysis(rows: list[dict], records: dict) -> tuple[list[dict], list[dict]]:
    local = sum(r["local_runtime_hours"] for r in rows)
    optuna = sum(r["optuna_wall_hours"] for r in rows)
    mean = optuna / len(rows)
    final_train_rows = sum(f["validation_rows"] for f in records[19][100])
    fits = []
    for number in (19, 20, 6, 8, 4):
        folds = records[number][100]
        # Full outer TRAIN refit has no inner validation. Scale measured epoch-100
        # training throughput by row count; initialization/inference are excluded.
        projected = statistics.fmean(f["training_wall_seconds"] / f["train_rows"]
                                     for f in folds) * final_train_rows * 100 / 3600
        fits.append({"trial": number, "depth": folds[0]["configuration"]["depth"],
                     "width": folds[0]["configuration"]["width"],
                     "full_train_rows_outer_3": final_train_rows,
                     "estimated_one_full_fit_hours": projected,
                     "estimated_five_member_hours": 5 * projected,
                     "estimated_four_study_ensemble_hours": 20 * projected,
                     "estimated_five_study_ensemble_hours": 25 * projected})
    scenarios = []
    for label, candidates, folds in (("original_20x5x4", 20, 5),
                                     ("reduced_5x5x4", 5, 5),
                                     ("reduced_3x5x4", 3, 5),
                                     ("fixed_no_hpo", 0, 0)):
        scenarios.append({"strategy": label, "candidates_per_study": candidates,
                          "folds_per_candidate": folds, "outer_studies": 4,
                          "hpo_fits": candidates * folds * 4,
                          "hpo_hours_pilot_optuna_mean": candidates * 4 * mean,
                          "hpo_hours_pilot_local_mean": candidates * 4 * local / len(rows),
                          "ensemble_four_study_hours_trial19": fits[0]["estimated_four_study_ensemble_hours"],
                          "total_four_study_hours_trial19": candidates * 4 * mean +
                          fits[0]["estimated_four_study_ensemble_hours"]})
    return scenarios, fits


def figures(trajectory: list[dict], subsets: list[dict], folds: list[dict],
            epochs: list[dict], details: list[dict], scenarios: list[dict]) -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 240, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    figures_dir = OUT / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["successful_count"] for r in trajectory],
            [r["gap_to_final_best"] for r in trajectory], marker="o", color="#224e70")
    ax.axvline(10.5, color="#b85c38", linestyle="--", label="startup → TPE")
    ax.set(xlabel="Completed trials in original order", ylabel="Gap to final best skill",
           title="Observed budget trajectory")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures_dir / "budget.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["budget"] for r in subsets], [r["median_regret"] for r in subsets],
            marker="o", label="Median unordered subset")
    ax.plot([r["budget"] for r in subsets], [r["p90_regret"] for r in subsets],
            marker="s", label="90th percentile unordered subset")
    ax.set(xlabel="Subset size from completed candidates", ylabel="Five-fold regret",
           title="Candidate coverage, not an adaptive-search replay")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures_dir / "candidate_subsets.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for size, color in ((2, "#b85c38"), (3, "#348a80"), (4, "#224e70")):
        data = [r for r in folds if r["size"] == size]
        ax.scatter([r["spearman"] for r in data], [r["regret"] for r in data],
                   label=f"{size} folds", s=55, color=color, alpha=.8)
    ax.set(xlabel="Spearman rank correlation with five folds", ylabel="Selected-trial five-fold regret",
           title="All 25 geographic fold subsets")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures_dir / "fold_subsets.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["epoch"] for r in epochs], [r["spearman"] for r in epochs], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Spearman correlation with epoch 100",
                title="Candidate rank stability")
    axes[1].plot([r["epoch"] for r in epochs], [r["regret"] for r in epochs], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Full-epoch regret of early selection",
                title="Early winner versus final winner")
    fig.tight_layout(); fig.savefig(figures_dir / "epoch_stability.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for number in (19, 20, 6, 8, 4):
        data = [r for r in details if r["number"] == number]
        ax.plot([r["epoch"] for r in data], [r["objective"] for r in data],
                marker="o", label=f"Trial {number}")
    ax.set(xlabel="Epoch", ylabel="Five-fold validation skill",
           title="Convergence of diverse candidate configurations")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(figures_dir / "candidate_epochs.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r["strategy"].replace("_", " ") for r in scenarios]
    y = np.arange(len(scenarios))
    ax.barh(y, [r["hpo_hours_pilot_optuna_mean"] for r in scenarios],
            color="#224e70", label="New HPO")
    ax.barh(y, [r["ensemble_four_study_hours_trial19"] for r in scenarios],
            left=[r["hpo_hours_pilot_optuna_mean"] for r in scenarios],
            color="#b85c38", label="Five-member ensemble refits")
    ax.set(yticks=y, yticklabels=labels, xlabel="Projected serial wall hours, four studies",
           title="Compute forecast at pilot rates")
    ax.invert_yaxis(); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures_dir / "runtime.png"); plt.close(fig)


def main() -> None:
    rows, records, audit = load()
    trajectory = budget_trajectory(rows)
    subsets = subset_budget(rows, BUDGETS)
    folds = fold_analysis(rows, records)
    epochs, details = epoch_analysis(rows, records)
    scenarios, fits = runtime_analysis(rows, records)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("trial_audit.csv", audit)
    write_csv("completed_trials.csv", rows)
    write_csv("budget_trajectory.csv", trajectory)
    write_csv("candidate_subset_budgets.csv", subsets)
    write_csv("fold_subsets.csv", folds)
    write_csv("epoch_summary.csv", epochs)
    write_csv("epoch_candidate_details.csv", details)
    write_csv("runtime_scenarios.csv", scenarios)
    write_csv("ensemble_refit_estimates.csv", fits)
    write_csv("design_comparison.csv", [
        {"design": "A_unrestricted_nested_five", "new_hpo_hours_forecast": scenarios[1]["hpo_hours_pilot_optuna_mean"],
         "new_hpo_fits": 100, "remaining_outer_test_isolated_from_own_selection": True,
         "cross_partition_pilot_dependence": "budget_choice_only",
         "claim": "within_study_held_out_conditional_on_revised_algorithm",
         "pooled_five_study_claim": "revised_protocol_sensitivity_not_fully_prospective"},
        {"design": "B_pilot_informed_five_candidates", "new_hpo_hours_forecast": scenarios[1]["hpo_hours_pilot_optuna_mean"],
         "new_hpo_fits": 100, "remaining_outer_test_isolated_from_own_selection": False,
         "cross_partition_pilot_dependence": "candidate_identification",
         "claim": "exploratory_geographic_transfer",
         "pooled_five_study_claim": "not_unbiased_nested_cv"},
        {"design": "B_pilot_informed_fixed_one", "new_hpo_hours_forecast": 0,
         "new_hpo_fits": 0, "remaining_outer_test_isolated_from_own_selection": False,
         "cross_partition_pilot_dependence": "fixed_candidate_identification",
         "claim": "exploratory_fixed_model_transfer",
         "pooled_five_study_claim": "not_unbiased_nested_cv"},
        {"design": "C_outer_3_primary", "new_hpo_hours_forecast": 0,
         "new_hpo_fits": 0, "remaining_outer_test_isolated_from_own_selection": "not_applicable",
         "cross_partition_pilot_dependence": "none_for_outer_3_test",
         "claim": "confirmatory_single_outer_3_held_out",
         "pooled_five_study_claim": "not_a_five_study_estimate"},
    ])
    sweep = []
    reasons = {
        19: ("Best frozen objective; low-rate depth-3 anchor", "Can the BP-led winner transfer?"),
        20: ("Near-neighbor with better FLP mean", "Is task balance stable near the winner?"),
        6: ("Best worst-fold skill; fast depth-2 large-batch alternative", "Does weak-region robustness transfer?"),
        8: ("Strong startup depth-2 option", "Is depth-3 complexity necessary?"),
        4: ("Width-128 startup comparator", "Does a wider architecture improve other geographies?"),
    }
    for number in CANDIDATES:
        row = next(r for r in rows if r["number"] == number)
        sweep.append({"trial": number, "status": "tested_outer_3_only",
                      **{k: row[k] for k in ("depth", "width", "batch_size", "learning_rate", "weight_decay")},
                      "outer_3_objective": row["objective"], "outer_3_bp_skill": row["bp_skill"],
                      "outer_3_flp_skill": row["flp_skill"], "outer_3_worst_fold": row["worst_fold"],
                      "observed_outer_3_local_runtime_hours": row["local_runtime_hours"],
                      "reason": reasons[number][0], "question": reasons[number][1]})
    write_csv("proposed_pilot_informed_sweep.csv", sweep)
    figures(trajectory, subsets, folds, epochs, details, scenarios)
    print(json.dumps({"budgets": [trajectory[b - 1] for b in BUDGETS],
                      "folds_by_size": {size: {"top_agreement": sum(r["top_agreement"] for r in folds if r["size"] == size),
                                               "count": sum(r["size"] == size for r in folds),
                                               "median_regret": statistics.median(r["regret"] for r in folds if r["size"] == size)}
                                        for size in (2, 3, 4)},
                      "epochs": epochs, "runtime_scenarios": scenarios,
                      "runtime_accounting": {"local_hours": sum(r["local_runtime_hours"] for r in rows),
                                             "optuna_wall_hours": sum(r["optuna_wall_hours"] for r in rows)}}, indent=2))


if __name__ == "__main__":
    main()
