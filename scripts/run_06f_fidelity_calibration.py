"""No-pruning outer-study-1 fidelity calibration for the 11 epoch-4 trials."""

import csv
import hashlib
import itertools
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wildfire_risk_uncertainty.hpo.config import Configuration
from wildfire_risk_uncertainty.hpo.fold_workers import PersistentFoldProvider
from wildfire_risk_uncertainty.hpo.records import (
    aggregate,
    fold_epoch_record,
    read_json,
    write_json,
)
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "reports/training/06f_diagnostics"
REPORT = ROOT / "reports/training/06f_fidelity_calibration"
ARTIFACT = ROOT / "artifacts/hpo/06f_fidelity_calibration"
CACHE_PATH = ROOT / "artifacts/training/hpo_cache_v1"
FIDELITIES = (1, 4, 8, 16)
GATE_EPOCHS = (1, 4)
FOLDS = tuple(range(1, 6))


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def selected_configurations() -> list[dict]:
    rows = read_csv(DIAGNOSTIC / "trial_summary.csv")
    selected = [row for row in rows if int(row["epochs_recorded"]) >= 4]
    configs = []
    for row in selected:
        configs.append(
            {
                "trial_number": int(row["trial_number"]),
                "historical_state": row["state"],
                "historical_pruning_epoch": (
                    int(row["pruning_epoch"]) if row["pruning_epoch"] else None
                ),
                "configuration": Configuration(
                    depth=int(row["depth"]),
                    width=int(row["width"]),
                    learning_rate=float(row["learning_rate"]),
                    weight_decay=float(row["weight_decay"]),
                    batch_size=int(row["batch_size"]),
                ),
            }
        )
    identities = {
        tuple(item["configuration"].serializable().items()) for item in configs
    }
    if len(configs) != 11 or len(identities) != 11:
        raise ValueError("Expected exactly 11 unique configurations reaching epoch 4")
    return configs


def historical_records() -> dict[tuple[int, int], dict]:
    records = {}
    for row in read_csv(DIAGNOSTIC / "learning_curves.csv"):
        trial = int(row["trial_number"])
        epoch = int(row["epoch"])
        if epoch not in GATE_EPOCHS:
            continue
        folds = []
        for fold in FOLDS:
            prefix = f"fold_{fold}_validation_"
            folds.append(
                {
                    "bp_count": int(row[prefix + "bp_count"]),
                    "bp_loss_sum": float(row[prefix + "bp_loss_sum"]),
                    "flp_valid_count": int(row[prefix + "flp_valid_count"]),
                    "flp_loss_sum": float(row[prefix + "flp_loss_sum"]),
                    "bp_loss_mean": float(row[prefix + "bp_loss_mean"]),
                    "flp_loss_mean": float(row[prefix + "flp_loss_mean"]),
                    "equal_task_objective": float(
                        row[prefix + "equal_task_objective"]
                    ),
                }
            )
        records[(trial, epoch)] = {"J_e": float(row["J_e"]), "folds": folds}
    return records


def manifest(configs: list[dict], cache: HpoCache) -> dict:
    diagnostic_metadata = read_json(DIAGNOSTIC / "metadata.json")
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "experiment": "checkpoint-06f-hpo-fidelity-calibration-v1",
        "status": "RUNNING",
        "outer_study": 1,
        "outer_test_inspected": False,
        "pruning": False,
        "target_epoch": 16,
        "fidelities": list(FIDELITIES),
        "historical_gate_epochs": list(GATE_EPOCHS),
        "execution": {
            "persistent_cuda_workers": 2,
            "common_epoch_barrier": True,
            "separate_optuna_trials": False,
        },
        "source_git_sha": source_sha,
        "diagnostic_source_git_sha": diagnostic_metadata["source_git_sha"],
        "cache_version": cache.manifest["version"],
        "cache_manifest_sha256": hashlib.sha256(
            stable_json_bytes(cache.manifest)
        ).hexdigest(),
        "configurations": [
            {
                "trial_number": item["trial_number"],
                "historical_state": item["historical_state"],
                "historical_pruning_epoch": item["historical_pruning_epoch"],
                **item["configuration"].serializable(),
            }
            for item in configs
        ],
    }


def trial_root(trial_number: int) -> Path:
    return ARTIFACT / f"trial_{trial_number:03d}"


def completed_epoch(trial_number: int) -> int:
    record_root = trial_root(trial_number) / "records"
    by_fold = []
    for fold in FOLDS:
        epochs = sorted(
            int(path.stem.split("_")[-1])
            for path in record_root.glob(f"fold_{fold}_epoch_*.json")
        )
        if epochs != list(range(1, len(epochs) + 1)):
            raise ValueError(f"Trial {trial_number} fold {fold} records are not contiguous")
        by_fold.append(len(epochs))
    if len(set(by_fold)) != 1:
        raise ValueError(f"Trial {trial_number} lacks a completed common-epoch barrier")
    aggregates = sorted(record_root.glob("aggregate_epoch_*.json"))
    if len(aggregates) != by_fold[0]:
        raise ValueError(f"Trial {trial_number} aggregate/fold records disagree")
    return by_fold[0]


def compare_historical(
    trial_number: int,
    epoch: int,
    folds: list[dict],
    spatial: dict,
    historical: dict[tuple[int, int], dict],
) -> dict:
    expected = historical.get((trial_number, epoch))
    if expected is None:
        raise ValueError(f"Missing historical trial {trial_number} epoch {epoch}")
    comparisons = {"aggregate_J_e": spatial["J_e"] == expected["J_e"], "folds": {}}
    for fold, record in zip(FOLDS, folds, strict=True):
        actual = record["validation"]
        expected_fold = expected["folds"][fold - 1]
        comparisons["folds"][str(fold)] = {
            key: actual[key] == expected_fold[key] for key in expected_fold
        }
    passed = comparisons["aggregate_J_e"] and all(
        all(values.values()) for values in comparisons["folds"].values()
    )
    result = {
        "trial_number": trial_number,
        "epoch": epoch,
        "passed": passed,
        "actual_J_e": spatial["J_e"],
        "historical_J_e": expected["J_e"],
        "comparisons": comparisons,
    }
    gate_path = REPORT / "gate_checks" / f"trial_{trial_number:03d}_epoch_{epoch:03d}.json"
    write_json(gate_path, result)
    if not passed:
        write_json(
            REPORT / "STOPPED_MISMATCH.json",
            {"status": "STOPPED", "reason": "historical mismatch", **result},
        )
        raise RuntimeError(
            f"Historical equivalence mismatch at trial {trial_number}, epoch {epoch}"
        )
    return result


def run_to_epoch(
    item: dict,
    target: int,
    historical: dict[tuple[int, int], dict],
) -> None:
    trial_number = item["trial_number"]
    configuration = item["configuration"]
    complete = completed_epoch(trial_number)
    if complete >= target:
        return
    record_root = trial_root(trial_number) / "records"
    checkpoints = trial_root(trial_number) / "checkpoints"
    steps = {}
    for fold in FOLDS:
        if complete:
            prior = read_json(record_root / f"fold_{fold}_epoch_{complete:03d}.json")
            steps[fold] = prior["global_optimizer_step"]
        else:
            steps[fold] = 0
    provider = PersistentFoldProvider(
        ROOT,
        1,
        configuration,
        cache_path=CACHE_PATH,
        trial_number=trial_number,
        checkpoint_root=checkpoints,
    )
    try:
        provider.restore_many({fold: complete for fold in FOLDS}, steps)
        for epoch in range(complete + 1, target + 1):
            started = time.perf_counter()
            results = provider.run_epoch(epoch, list(FOLDS))
            folds = []
            for fold in FOLDS:
                training, validation = results[fold]
                steps[fold] += training["optimizer_steps"]
                folds.append(
                    fold_epoch_record(
                        outer_study=1,
                        trial_number=trial_number,
                        inner_fold=fold,
                        epoch=epoch,
                        configuration=configuration.serializable(),
                        training=training,
                        validation=validation,
                        global_step=steps[fold],
                        milestone=epoch in FIDELITIES,
                    )
                )
            provider.save_many({fold: (epoch, steps[fold]) for fold in FOLDS})
            for record in folds:
                write_json(
                    record_root
                    / f"fold_{record['inner_fold']}_epoch_{record['epoch']:03d}.json",
                    record,
                )
            spatial = {
                **aggregate(epoch, folds),
                "trial_number": trial_number,
                "configuration": configuration.serializable(),
                "epoch_wall_seconds": time.perf_counter() - started,
            }
            write_json(record_root / f"aggregate_epoch_{epoch:03d}.json", spatial)
            if epoch in GATE_EPOCHS:
                compare_historical(trial_number, epoch, folds, spatial, historical)
            provider.cleanup_many({fold: epoch for fold in FOLDS})
            print(
                f"trial={trial_number:02d} epoch={epoch:02d} "
                f"J={spatial['J_e']:.15f} wall={spatial['epoch_wall_seconds']:.3f}s",
                flush=True,
            )
    finally:
        provider.close()


def validate_full_gate(configs: list[dict]) -> dict:
    checks = []
    for item in configs:
        for epoch in GATE_EPOCHS:
            check = read_json(
                REPORT
                / "gate_checks"
                / f"trial_{item['trial_number']:03d}_epoch_{epoch:03d}.json"
            )
            checks.append(check)
    result = {
        "status": "PASSED",
        "exact": True,
        "configurations": len(configs),
        "fold_epochs": len(configs) * len(GATE_EPOCHS) * len(FOLDS),
        "aggregate_epochs": len(checks),
        "checks": checks,
    }
    write_json(REPORT / "historical_equivalence_gate.json", result)
    return result


def collect_metrics(configs: list[dict]) -> list[dict]:
    rows = []
    for item in configs:
        trial_number = item["trial_number"]
        config = item["configuration"].serializable()
        for epoch in range(1, 17):
            spatial = read_json(
                trial_root(trial_number)
                / "records"
                / f"aggregate_epoch_{epoch:03d}.json"
            )
            rows.append(
                {
                    "trial_number": trial_number,
                    **config,
                    "epoch": epoch,
                    "J_e": spatial["J_e"],
                    "validation_bp_mean": spatial["validation_bp_mean"],
                    "validation_flp_mean": spatial["validation_flp_mean"],
                    "fold_objective_std_population": spatial[
                        "fold_objective_std_population"
                    ],
                    **{
                        f"fold_{fold}_J": spatial["fold_validation_objectives"][fold - 1]
                        for fold in FOLDS
                    },
                    "epoch_wall_seconds": spatial["epoch_wall_seconds"],
                }
            )
    return rows


def analyze(configs: list[dict], metrics: list[dict], total_runtime: float) -> dict:
    selected = [row for row in metrics if row["epoch"] in FIDELITIES]
    by_trial_epoch = {
        (row["trial_number"], row["epoch"]): row for row in selected
    }
    trials = [item["trial_number"] for item in configs]
    values = {
        epoch: np.array([by_trial_epoch[(trial, epoch)]["J_e"] for trial in trials])
        for epoch in FIDELITIES
    }
    ranks = {epoch: rankdata(values[epoch], method="average") for epoch in FIDELITIES}
    correlations = []
    for first, second in itertools.combinations(FIDELITIES, 2):
        spearman = spearmanr(values[first], values[second])
        kendall = kendalltau(values[first], values[second])
        correlations.append(
            {
                "fidelity_a": first,
                "fidelity_b": second,
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_pvalue": float(kendall.pvalue),
            }
        )
    ranking_rows = []
    for index, trial in enumerate(trials):
        ranking_rows.append(
            {
                "trial_number": trial,
                **{f"J_{epoch}": float(values[epoch][index]) for epoch in FIDELITIES},
                **{f"rank_{epoch}": float(ranks[epoch][index]) for epoch in FIDELITIES},
            }
        )
    top_sets = {
        (epoch, k): {
            trials[index] for index in np.argsort(values[epoch], kind="stable")[:k]
        }
        for epoch in FIDELITIES
        for k in (1, 3, 5)
    }
    overlap = []
    for k in (1, 3, 5):
        for first, second in itertools.combinations(FIDELITIES, 2):
            common = sorted(top_sets[(first, k)] & top_sets[(second, k)])
            overlap.append(
                {
                    "k": k,
                    "fidelity_a": first,
                    "fidelity_b": second,
                    "overlap_count": len(common),
                    "overlap_fraction": len(common) / k,
                    "common_trials": common,
                }
            )
    winner_index = int(np.argmin(values[16]))
    winner = trials[winner_index]
    winner_history = next(item for item in configs if item["trial_number"] == winner)
    regrets = []
    best_16 = float(values[16][winner_index])
    for epoch in FIDELITIES:
        chosen_index = int(np.argmin(values[epoch]))
        regrets.append(
            {
                "selection_fidelity": epoch,
                "selected_trial": trials[chosen_index],
                "selected_J_at_fidelity": float(values[epoch][chosen_index]),
                "selected_J_16": float(values[16][chosen_index]),
                "best_J_16": best_16,
                "regret": float(values[16][chosen_index] - best_16),
            }
        )
    result = {
        "status": "COMPLETE",
        "outer_study": 1,
        "configuration_count": len(configs),
        "epochs_per_configuration": 16,
        "total_runtime_seconds": total_runtime,
        "correlations": correlations,
        "rankings": ranking_rows,
        "top_k_overlaps": overlap,
        "epoch_16_winner": {
            "trial_number": winner,
            "J_16": best_16,
            "historical_state": winner_history["historical_state"],
            "historical_pruning_epoch": winner_history["historical_pruning_epoch"],
            "survived_original_epoch_1_decision": True,
            "survived_original_epoch_4_decision": (
                winner_history["historical_pruning_epoch"] is None
                or winner_history["historical_pruning_epoch"] > 4
            ),
        },
        "selection_regret": regrets,
    }
    write_json(REPORT / "analysis.json", result)
    return result


def make_plots(metrics: list[dict], analysis: dict) -> None:
    trials = sorted({row["trial_number"] for row in metrics})
    selected = {
        (row["trial_number"], row["epoch"]): row
        for row in metrics
        if row["epoch"] in FIDELITIES
    }
    rankings = {row["trial_number"]: row for row in analysis["rankings"]}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for trial in trials:
        ax.plot(
            FIDELITIES,
            [rankings[trial][f"rank_{epoch}"] for epoch in FIDELITIES],
            marker="o",
            linewidth=1.4,
            label=f"trial {trial}",
        )
    ax.invert_yaxis()
    ax.set(xticks=FIDELITIES, xlabel="Epoch", ylabel="Rank (1 = best)")
    ax.set_title("Hyperparameter rank trajectories")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORT / "rank_trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for trial in trials:
        rows = sorted(
            (row for row in metrics if row["trial_number"] == trial),
            key=lambda row: row["epoch"],
        )
        ax.plot(
            [row["epoch"] for row in rows],
            [row["J_e"] for row in rows],
            linewidth=1.4,
            label=f"trial {trial}",
        )
    ax.scatter(
        [epoch for trial in trials for epoch in FIDELITIES],
        [selected[(trial, epoch)]["J_e"] for trial in trials for epoch in FIDELITIES],
        s=9,
        color="black",
        alpha=0.4,
    )
    ax.set(xlabel="Epoch", ylabel="Five-fold aggregate J")
    ax.set_title("No-pruning learning curves")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORT / "learning_curves.png", dpi=180)
    plt.close(fig)

    matrices = {"Spearman rho": np.eye(4), "Kendall tau": np.eye(4)}
    index = {epoch: position for position, epoch in enumerate(FIDELITIES)}
    for row in analysis["correlations"]:
        first = index[row["fidelity_a"]]
        second = index[row["fidelity_b"]]
        matrices["Spearman rho"][first, second] = matrices["Spearman rho"][second, first] = row[
            "spearman_rho"
        ]
        matrices["Kendall tau"][first, second] = matrices["Kendall tau"][second, first] = row[
            "kendall_tau"
        ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, (title, matrix) in zip(axes, matrices.items(), strict=True):
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(4), FIDELITIES)
        ax.set_yticks(range(4), FIDELITIES)
        ax.set(xlabel="Epoch", ylabel="Epoch", title=title)
        for row in range(4):
            for column in range(4):
                ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(REPORT / "rank_correlations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pairs = [f"{row['fidelity_a']}–{row['fidelity_b']}" for row in analysis["top_k_overlaps"] if row["k"] == 1]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    for offset, k in zip((-0.25, 0, 0.25), (1, 3, 5), strict=True):
        rows = [row for row in analysis["top_k_overlaps"] if row["k"] == k]
        ax.bar(x + offset, [row["overlap_fraction"] for row in rows], width=0.24, label=f"top {k}")
    ax.set_xticks(x, pairs)
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="Epoch pair", ylabel="Overlap fraction", title="Top-k stability")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT / "top_k_overlap.png", dpi=180)
    plt.close(fig)


def export_tables(metrics: list[dict], analysis: dict) -> None:
    metric_fields = list(metrics[0])
    write_csv(REPORT / "epoch_metrics.csv", metrics, metric_fields)
    write_csv(
        REPORT / "fidelity_metrics.csv",
        [row for row in metrics if row["epoch"] in FIDELITIES],
        metric_fields,
    )
    write_csv(
        REPORT / "rankings.csv",
        analysis["rankings"],
        list(analysis["rankings"][0]),
    )
    correlation_rows = analysis["correlations"]
    write_csv(
        REPORT / "rank_correlations.csv",
        correlation_rows,
        list(correlation_rows[0]),
    )
    overlap_rows = [
        {**row, "common_trials": ";".join(map(str, row["common_trials"]))}
        for row in analysis["top_k_overlaps"]
    ]
    write_csv(REPORT / "top_k_overlap.csv", overlap_rows, list(overlap_rows[0]))
    regret_rows = analysis["selection_regret"]
    write_csv(REPORT / "selection_regret.csv", regret_rows, list(regret_rows[0]))


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    configs = selected_configurations()
    historical = historical_records()
    cache = HpoCache(ROOT, CACHE_PATH)
    load_states(ROOT, cache, 1)
    expected_manifest = manifest(configs, cache)
    manifest_path = REPORT / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("status") == "COMPLETE":
            raise FileExistsError("Completed epoch-16 calibration is frozen")
        for key, value in expected_manifest.items():
            if key != "status" and existing.get(key) != value:
                raise ValueError(f"Existing calibration manifest differs at {key}")
    else:
        write_json(manifest_path, {**expected_manifest, "started_utc": utc_now()})

    run_started = time.perf_counter()
    prior_runtime = 0.0
    runtime_path = REPORT / "runtime_progress.json"
    if runtime_path.exists():
        prior_runtime = read_json(runtime_path)["accumulated_runtime_seconds"]
    try:
        for item in configs:
            run_to_epoch(item, 4, historical)
        validate_full_gate(configs)
        print("HISTORICAL_EQUIVALENCE_GATE=PASSED", flush=True)
        for item in configs:
            run_to_epoch(item, 16, historical)
    finally:
        total_runtime = prior_runtime + time.perf_counter() - run_started
        write_json(
            runtime_path,
            {"accumulated_runtime_seconds": total_runtime, "updated_utc": utc_now()},
        )

    metrics = collect_metrics(configs)
    analysis = analyze(configs, metrics, total_runtime)
    export_tables(metrics, analysis)
    make_plots(metrics, analysis)
    current = read_json(manifest_path)
    write_json(
        manifest_path,
        {**current, "status": "COMPLETE", "completed_utc": utc_now()},
    )
    print(f"CALIBRATION_COMPLETE runtime={total_runtime:.3f}s", flush=True)


if __name__ == "__main__":
    main()
