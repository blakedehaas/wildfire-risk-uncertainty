"""Continue the accepted 11-trial outer-1 calibration from epoch 16 to 32."""

import itertools
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import kendalltau, rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_06f_fidelity_calibration as base

from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.cache_checkpoint import cache_binding
from wildfire_risk_uncertainty.training.cache_states import load_states

OUT = base.REPORT / "epoch32"
FIDELITIES = (1, 4, 8, 16, 32)
REPORTED_EPOCHS = (1, 4, 8, 16, 24, 32)
TRIALS = (0, 1, 2, 7, 13, 15, 16, 19, 22, 28, 29)


def now() -> str:
    return datetime.now(UTC).isoformat()


def table(path: Path, rows: list[dict]) -> None:
    base.write_csv(path, rows, list(rows[0]))


def validate_latest_checkpoints(configs: list[dict]) -> dict:
    original = read_json(base.REPORT / "manifest.json")
    gate = read_json(base.REPORT / "historical_equivalence_gate.json")
    if original["status"] != "COMPLETE" or original["target_epoch"] != 16:
        raise ValueError("Epoch-16 calibration is not complete")
    if gate["status"] != "PASSED" or len(gate["checks"]) != 22:
        raise ValueError("Historical equivalence gate is not complete")
    if not all(check["passed"] for check in gate["checks"]):
        raise ValueError("Historical equivalence gate contains a mismatch")
    if [item["trial_number"] for item in configs] != list(TRIALS):
        raise ValueError("Selected configurations differ from the accepted 11")
    if original["configurations"] != [
        {
            "trial_number": item["trial_number"],
            "historical_state": item["historical_state"],
            "historical_pruning_epoch": item["historical_pruning_epoch"],
            **item["configuration"].serializable(),
        }
        for item in configs
    ]:
        raise ValueError("Configuration manifest differs from the epoch-16 cohort")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=base.ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != original["source_git_sha"]:
        raise ValueError("Frozen training implementation commit differs")
    cache = HpoCache(base.ROOT, base.CACHE_PATH)
    current = base.manifest(configs, cache)
    if current["cache_manifest_sha256"] != original["cache_manifest_sha256"]:
        raise ValueError("Cache manifest differs from epoch-16 calibration")
    states = load_states(base.ROOT, cache, 1)
    original_rows = {
        int(row["trial_number"]): row
        for row in base.read_csv(base.REPORT / "fidelity_metrics.csv")
        if int(row["epoch"]) == 16
    }
    checkpoints = []
    completed = {}
    for item in configs:
        trial = item["trial_number"]
        last = base.completed_epoch(trial)
        if not 16 <= last <= 32:
            raise ValueError(f"Trial {trial} completed epoch {last}; expected 16–32")
        completed[str(trial)] = last
        epoch_16 = read_json(
            base.trial_root(trial) / "records" / "aggregate_epoch_016.json"
        )
        expected_16 = original_rows[trial]
        if epoch_16["J_e"] != float(expected_16["J_e"]):
            raise ValueError(f"Trial {trial} epoch-16 aggregate differs from frozen result")
        for fold in base.FOLDS:
            if epoch_16["fold_validation_objectives"][fold - 1] != float(
                expected_16[f"fold_{fold}_J"]
            ):
                raise ValueError(f"Trial {trial} fold {fold} epoch-16 value changed")
            record = read_json(
                base.trial_root(trial)
                / "records"
                / f"fold_{fold}_epoch_{last:03d}.json"
            )
            if record["configuration"] != item["configuration"].serializable():
                raise ValueError("Fold record configuration mismatch")
            path = (
                base.trial_root(trial)
                / "checkpoints"
                / f"fold_{fold}_epoch_{last:03d}.pt"
            )
            recorded_sha = read_json(
                path.with_suffix(path.suffix + ".sha256.json")
            )["sha256"]
            actual_sha = sha256(path)
            if recorded_sha != actual_sha:
                raise ValueError(f"Checkpoint integrity mismatch: trial {trial}, fold {fold}")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            expected_metadata = {
                "outer_study": 1,
                "trial_number": trial,
                "inner_fold": fold,
                "configuration": item["configuration"].serializable(),
                "seed": 42,
                "cache_binding": cache_binding(cache, states[fold]),
            }
            if saved["metadata"] != expected_metadata:
                raise ValueError(f"Checkpoint provenance mismatch: trial {trial}, fold {fold}")
            if saved["completed_epoch"] != last or saved["global_step"] != record[
                "global_optimizer_step"
            ]:
                raise ValueError(f"Checkpoint step mismatch: trial {trial}, fold {fold}")
            for required in (
                "model_state",
                "optimizer_state",
                "python_rng_state",
                "numpy_rng_state",
                "torch_cpu_rng_state",
                "torch_cuda_rng_state",
            ):
                if required not in saved:
                    raise ValueError(f"Checkpoint missing {required}: trial {trial}, fold {fold}")
            checkpoints.append(
                {
                    "trial_number": trial,
                    "fold": fold,
                    "epoch": last,
                    "global_optimizer_step": saved["global_step"],
                    "sha256": actual_sha,
                }
            )
    result = {
        "status": "PASSED",
        "source_git_sha": head,
        "cache_manifest_sha256": current["cache_manifest_sha256"],
        "historical_equivalence_gate": "PASSED",
        "completed_epochs_before_resume": completed,
        "validated_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "validated_utc": now(),
    }
    write_json(OUT / "checkpoint_validation.json", result)
    return result


def collect_metrics(configs: list[dict]) -> list[dict]:
    rows = []
    for item in configs:
        trial = item["trial_number"]
        for epoch in range(1, 33):
            spatial = read_json(
                base.trial_root(trial)
                / "records"
                / f"aggregate_epoch_{epoch:03d}.json"
            )
            folds = [
                read_json(
                    base.trial_root(trial)
                    / "records"
                    / f"fold_{fold}_epoch_{epoch:03d}.json"
                )
                for fold in base.FOLDS
            ]
            if spatial["fold_validation_objectives"] != [
                fold["validation"]["equal_task_objective"] for fold in folds
            ]:
                raise ValueError(f"Trial {trial} epoch {epoch} fold aggregate mismatch")
            rows.append(
                {
                    "trial_number": trial,
                    **item["configuration"].serializable(),
                    "epoch": epoch,
                    "J_e": spatial["J_e"],
                    "validation_bp_mean": spatial["validation_bp_mean"],
                    "validation_flp_mean": spatial["validation_flp_mean"],
                    "fold_objective_std_population": spatial[
                        "fold_objective_std_population"
                    ],
                    **{
                        f"fold_{fold}_J": spatial["fold_validation_objectives"][fold - 1]
                        for fold in base.FOLDS
                    },
                    **{
                        f"fold_{fold}_bp_loss_mean": folds[fold - 1]["validation"][
                            "bp_loss_mean"
                        ]
                        for fold in base.FOLDS
                    },
                    **{
                        f"fold_{fold}_flp_loss_mean": folds[fold - 1]["validation"][
                            "flp_loss_mean"
                        ]
                        for fold in base.FOLDS
                    },
                    **{
                        f"fold_{fold}_bp_loss_sum": folds[fold - 1]["validation"][
                            "bp_loss_sum"
                        ]
                        for fold in base.FOLDS
                    },
                    **{
                        f"fold_{fold}_flp_loss_sum": folds[fold - 1]["validation"][
                            "flp_loss_sum"
                        ]
                        for fold in base.FOLDS
                    },
                    **{
                        f"fold_{fold}_bp_count": folds[fold - 1]["validation"][
                            "bp_count"
                        ]
                        for fold in base.FOLDS
                    },
                    **{
                        f"fold_{fold}_flp_valid_count": folds[fold - 1]["validation"][
                            "flp_valid_count"
                        ]
                        for fold in base.FOLDS
                    },
                    "epoch_wall_seconds": spatial["epoch_wall_seconds"],
                }
            )
    return rows


def analyze(configs: list[dict], metrics: list[dict], runtime: float) -> dict:
    trials = [item["trial_number"] for item in configs]
    by_trial_epoch = {
        (row["trial_number"], row["epoch"]): row for row in metrics
    }
    values = {
        epoch: np.array([by_trial_epoch[trial, epoch]["J_e"] for trial in trials])
        for epoch in FIDELITIES
    }
    ranks = {epoch: rankdata(values[epoch], method="average") for epoch in FIDELITIES}
    correlations = []
    for first, second in itertools.combinations(FIDELITIES, 2):
        rho = spearmanr(values[first], values[second])
        tau = kendalltau(values[first], values[second])
        correlations.append(
            {
                "fidelity_a": first,
                "fidelity_b": second,
                "spearman_rho": float(rho.statistic),
                "spearman_pvalue": float(rho.pvalue),
                "kendall_tau": float(tau.statistic),
                "kendall_pvalue": float(tau.pvalue),
            }
        )
    ranking_rows = [
        {
            "trial_number": trial,
            **{f"J_{epoch}": float(values[epoch][index]) for epoch in FIDELITIES},
            **{f"rank_{epoch}": float(ranks[epoch][index]) for epoch in FIDELITIES},
        }
        for index, trial in enumerate(trials)
    ]
    top = {
        (epoch, k): {
            trials[index] for index in np.argsort(values[epoch], kind="stable")[:k]
        }
        for epoch in FIDELITIES
        for k in (1, 3, 5)
    }
    overlap = [
        {
            "k": k,
            "fidelity_a": first,
            "fidelity_b": second,
            "overlap_count": len(top[first, k] & top[second, k]),
            "overlap_fraction": len(top[first, k] & top[second, k]) / k,
            "common_trials": sorted(top[first, k] & top[second, k]),
        }
        for k in (1, 3, 5)
        for first, second in itertools.combinations(FIDELITIES, 2)
    ]
    winner_index = int(np.argmin(values[32]))
    winner = trials[winner_index]
    historical = next(item for item in configs if item["trial_number"] == winner)
    best = float(values[32][winner_index])
    regrets = []
    for epoch in FIDELITIES:
        chosen = int(np.argmin(values[epoch]))
        regrets.append(
            {
                "selection_fidelity": epoch,
                "selected_trial": trials[chosen],
                "selected_J_at_fidelity": float(values[epoch][chosen]),
                "selected_J_32": float(values[32][chosen]),
                "best_J_32": best,
                "regret": float(values[32][chosen] - best),
            }
        )
    winner_survival = {
        "original_epoch_1": True,
        "original_epoch_4": (
            historical["historical_pruning_epoch"] is None
            or historical["historical_pruning_epoch"] > 4
        ),
        "hypothetical_top_k": {
            str(epoch): {str(k): winner in top[epoch, k] for k in (1, 3, 5)}
            for epoch in (8, 16)
        },
    }
    result = {
        "status": "COMPLETE",
        "outer_study": 1,
        "configuration_count": 11,
        "epochs_per_configuration": 32,
        "additional_runtime_seconds": runtime,
        "rank_correlations": correlations,
        "rankings": ranking_rows,
        "top_k_overlaps": overlap,
        "epoch_32_winner": {
            "trial_number": winner,
            "J_32": best,
            "historical_state": historical["historical_state"],
            "historical_pruning_epoch": historical["historical_pruning_epoch"],
            "survival": winner_survival,
        },
        "selection_regret": regrets,
    }
    write_json(OUT / "analysis.json", result)
    return result


def make_plots(metrics: list[dict], analysis: dict) -> None:
    rankings = {row["trial_number"]: row for row in analysis["rankings"]}
    trials = sorted(rankings)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for trial in trials:
        ax.plot(
            FIDELITIES,
            [rankings[trial][f"rank_{epoch}"] for epoch in FIDELITIES],
            marker="o",
            label=f"trial {trial}",
        )
    ax.invert_yaxis()
    ax.set(xticks=FIDELITIES, xlabel="Epoch", ylabel="Rank (1 = best)")
    ax.set_title("Rank trajectories through epoch 32")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "rank_trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for trial in trials:
        rows = [row for row in metrics if row["trial_number"] == trial]
        ax.plot(
            [row["epoch"] for row in rows],
            [row["J_e"] for row in rows],
            label=f"trial {trial}",
        )
    ax.set(xlabel="Epoch", ylabel="Five-fold aggregate J")
    ax.set_title("No-pruning learning curves through epoch 32")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, key, title in (
        (axes[0], "spearman_rho", "Spearman ρ"),
        (axes[1], "kendall_tau", "Kendall τ"),
    ):
        matrix = np.eye(len(FIDELITIES))
        for row in analysis["rank_correlations"]:
            first = FIDELITIES.index(row["fidelity_a"])
            second = FIDELITIES.index(row["fidelity_b"])
            matrix[first, second] = matrix[second, first] = row[key]
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(FIDELITIES)), FIDELITIES)
        ax.set_yticks(range(len(FIDELITIES)), FIDELITIES)
        ax.set(xlabel="Epoch", ylabel="Epoch", title=title)
        for row in range(len(FIDELITIES)):
            for column in range(len(FIDELITIES)):
                ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(OUT / "rank_correlations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    relative = [row for row in analysis["top_k_overlaps"] if row["fidelity_b"] == 32]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(4)
    for offset, k in zip((-0.25, 0.0, 0.25), (1, 3, 5), strict=True):
        rows = [row for row in relative if row["k"] == k]
        ax.bar(x + offset, [row["overlap_fraction"] for row in rows], 0.24, label=f"top {k}")
    ax.set_xticks(x, (1, 4, 8, 16))
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="Earlier epoch", ylabel="Overlap with epoch 32", title="Top-k stability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "top_k_overlap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_manifest_path = OUT / "manifest.json"
    if run_manifest_path.exists():
        status = read_json(run_manifest_path).get("status")
        if status == "SUSPENDED_INCOMPLETE":
            raise RuntimeError("Epoch-32 calibration is suspended pending a new HITL decision")
        if status == "COMPLETE":
            raise FileExistsError("Completed epoch-32 calibration is frozen")
    configs = base.selected_configurations()
    try:
        validation = validate_latest_checkpoints(configs)
    except Exception as error:
        write_json(OUT / "STOPPED_CHECKPOINT_MISMATCH.json", {"status": "STOPPED", "reason": str(error), "utc": now()})
        raise
    print(f"CHECKPOINT_VALIDATION=PASSED count={validation['validated_checkpoints']}", flush=True)
    expected_manifest = {
        "experiment": "checkpoint-06f-hpo-fidelity-calibration-epoch32-v1",
        "parent_manifest": "../manifest.json",
        "parent_source_git_sha": validation["source_git_sha"],
        "outer_study": 1,
        "trials": list(TRIALS),
        "resume_epoch": 16,
        "target_epoch": 32,
        "pruning": False,
        "outer_test_inspected": False,
        "concurrent_fold_workers": 2,
    }
    if run_manifest_path.exists():
        existing = read_json(run_manifest_path)
        for key, value in expected_manifest.items():
            if existing.get(key) != value:
                raise ValueError(f"Epoch-32 manifest differs at {key}")
    else:
        write_json(run_manifest_path, {**expected_manifest, "status": "RUNNING", "started_utc": now()})
    started = time.perf_counter()
    progress_path = OUT / "runtime_progress.json"
    prior = read_json(progress_path)["accumulated_runtime_seconds"] if progress_path.exists() else 0.0
    try:
        for item in configs:
            base.run_to_epoch(item, 32, {})
            write_json(
                progress_path,
                {
                    "accumulated_runtime_seconds": prior + time.perf_counter() - started,
                    "completed_epochs": {str(config["trial_number"]): base.completed_epoch(config["trial_number"]) for config in configs},
                    "updated_utc": now(),
                },
            )
    finally:
        runtime = prior + time.perf_counter() - started
        write_json(
            progress_path,
            {
                "accumulated_runtime_seconds": runtime,
                "completed_epochs": {str(config["trial_number"]): base.completed_epoch(config["trial_number"]) for config in configs},
                "updated_utc": now(),
            },
        )
    if any(base.completed_epoch(item["trial_number"]) != 32 for item in configs):
        raise ValueError("Not all configurations reached epoch 32")
    metrics = collect_metrics(configs)
    result = analyze(configs, metrics, runtime)
    table(OUT / "epoch_metrics.csv", metrics)
    table(OUT / "reported_epoch_metrics.csv", [row for row in metrics if row["epoch"] in REPORTED_EPOCHS])
    table(OUT / "rankings.csv", result["rankings"])
    table(OUT / "rank_correlations.csv", result["rank_correlations"])
    table(
        OUT / "top_k_overlap.csv",
        [{**row, "common_trials": ";".join(map(str, row["common_trials"]))} for row in result["top_k_overlaps"]],
    )
    table(OUT / "selection_regret.csv", result["selection_regret"])
    make_plots(metrics, result)
    current_manifest = read_json(run_manifest_path)
    write_json(run_manifest_path, {**current_manifest, "status": "COMPLETE", "completed_utc": now()})
    print(f"EPOCH32_COMPLETE additional_runtime={runtime:.3f}s", flush=True)


if __name__ == "__main__":
    main()
