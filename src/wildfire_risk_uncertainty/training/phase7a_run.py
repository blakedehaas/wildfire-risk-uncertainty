"""Checkpointed per-fold Pareto training and outer-TRAIN-only selection."""

import csv
import hashlib
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.phase7a import (
    aggregate_folds,
    select_pareto_knee,
)
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)


def tasks_from_proposals(proposals: dict) -> list[dict]:
    tasks = []
    for outer in range(1, 6):
        candidates = proposals[str(outer)]["candidates"]
        if len(candidates) != 6 or len(set(candidates)) != 6 or 0.5 not in candidates:
            raise ValueError(f"Outer {outer} must have six unique alphas including 0.5")
        for alpha in candidates:
            if not 0 < alpha < 1:
                raise ValueError("Pareto alpha must be strictly between 0 and 1")
            for fold in range(1, 6):
                tasks.append({"id": len(tasks) + 1, "outer_study": outer,
                              "alpha": float(alpha), "inner_fold": fold})
    return tasks


def _progress(path: Path, task_id: int, epoch: int, phase: str) -> None:
    write_json(path / f"fold_{task_id}.progress.json",
               {"task_id": task_id, "epoch": epoch, "phase": phase,
                "time_unix": time.time()})


def _check_metrics(metrics: dict, alpha: float) -> None:
    bp = metrics["bp_loss_sum"] / metrics["bp_rows"]
    flp = metrics["flp_loss_sum"] / metrics["flp_valid_rows"]
    combined = alpha * bp + (1 - alpha) * flp
    if (abs(bp - metrics["bp_loss"]) > 1e-12 or
            abs(flp - metrics["flp_loss"]) > 1e-12 or
            abs(combined - metrics["combined_loss"]) > 1e-12):
        raise ValueError("Pareto task means/objective disagree with raw sums")


def _reference_check(root: Path, task: dict, training: dict, state_sha: str) -> None:
    if task["outer_study"] != 3 or task["alpha"] != 0.5:
        return
    reference = read_json(root / "artifacts/benchmark/07a_detached/07a-20261020T193344Z-f965d452" /
                          f"fold_{task['inner_fold']}_epoch_001.json")
    keys = ("bp_loss_sum", "flp_loss_sum", "bp_rows", "flp_valid_rows", "optimizer_steps")
    if any(training[key] != reference["training"][key] for key in keys):
        raise ValueError("Alpha-0.5 production training differs from frozen 07A epoch-1 reference")
    old_state = read_json(root / "artifacts/benchmark/07a_detached/07a-20261020T193344Z-f965d452" /
                          f"fold_{task['inner_fold']}.json")["state_sha256"]
    if state_sha != old_state:
        raise ValueError("Alpha-0.5 preprocessing state differs from 07A reference")


def run_task(root: Path, path: Path, config: dict, task_id: int) -> None:
    task = config["tasks"][task_id - 1]
    if task["id"] != task_id:
        raise ValueError("Task ID/order mismatch")
    outer, fold, alpha = task["outer_study"], task["inner_fold"], task["alpha"]
    cache = HpoCache(root, root / "artifacts/training/hpo_cache_v1", check_files=False)
    state = load_states(root, cache, outer)[fold]
    counts = support_counts(root, outer)[fold]
    seed_everything(config["seed"])
    model, optimizer = make_model(len(state.inputs), config["depth"], config["width"],
                                  config["learning_rate"], config["weight_decay"])
    write_json(path / f"fold_{task_id}.ready.json",
               {"task": task, "state_sha256": state.digest(), "cache_source_hashes": cache.manifest["source_hashes"]})
    metadata = {"phase": "07a_pareto", "outer_study": outer, "inner_fold": fold,
                "alpha": alpha, "configuration_sha256": hashlib.sha256(
                    stable_json_bytes(config["model_configuration"])).hexdigest(),
                "seed": config["seed"]}
    completed = 0
    global_step = 0
    for epoch in range(1, config["max_epochs"] + 1):
        record_path = path / f"fold_{task_id}_epoch_{epoch:03d}.json"
        if not record_path.exists():
            break
        record = read_json(record_path)
        if record["task"] != task or record["epoch"] != epoch:
            raise ValueError("Existing Pareto epoch record provenance mismatch")
        completed = epoch
        global_step = record["global_step"]
    if completed:
        checkpoint = path / f"fold_{task_id}_epoch_{completed:03d}.pt"
        restored = load_cache_checkpoint(checkpoint, model, optimizer, cache=cache,
                                         state=state, expected_metadata=metadata)
        if restored != (completed, global_step):
            raise ValueError("Pareto resume checkpoint epoch/step mismatch")
    for epoch in range(completed + 1, config["max_epochs"] + 1):
        if (path / "stop.request.json").exists():
            break
        _progress(path, task_id, epoch, "train")
        training = train_epoch(root, state, model, optimizer, epoch, outer_study=outer,
                               inner_fold=fold, batch_size=config["batch_size"],
                               cache=cache, alpha=alpha)
        _check_metrics(training, alpha)
        if epoch == 1:
            _reference_check(root, task, training, state.digest())
        global_step += training["optimizer_steps"]
        _progress(path, task_id, epoch, "validation")
        validation = validate(root, state, model, counts["validation"]["rows"],
                              counts["validation"]["flp_valid"], outer_study=outer,
                              inner_fold=fold, batch_size=config["batch_size"], cache=cache,
                              alpha=alpha, report_excess=True)
        _check_metrics(validation, alpha)
        if (training["bp_rows"] != counts["train"]["rows"] or
                training["flp_valid_rows"] != counts["train"]["flp_valid"] or
                validation["bp_rows"] != counts["validation"]["rows"] or
                validation["flp_valid_rows"] != counts["validation"]["flp_valid"] or
                any(not math.isfinite(value) for value in (
                    training["bp_loss"], training["flp_loss"],
                    validation["bp_excess_log_loss"], validation["flp_excess_log_loss"]))):
            raise ValueError("Pareto training/validation support or finite check failed")
        _progress(path, task_id, epoch, "checkpoint")
        checkpoint = path / f"fold_{task_id}_epoch_{epoch:03d}.pt"
        started = time.perf_counter()
        save_cache_checkpoint(checkpoint, model, optimizer, completed_epoch=epoch,
                              global_step=global_step, cache=cache, state=state,
                              metadata=metadata)
        checkpoint_seconds = time.perf_counter() - started
        write_json(path / f"fold_{task_id}_epoch_{epoch:03d}.json",
                   {"task": task, "epoch": epoch, "training": training,
                    "validation": validation, "global_step": global_step,
                    "checkpoint_save_seconds": checkpoint_seconds,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": read_json(checkpoint.with_suffix(".pt.sha256.json"))["sha256"],
                    "preprocessing_sha256": state.digest()})
        completed = epoch
        if (path / "stop.request.json").exists():
            break
    stopped = completed < config["max_epochs"]
    _progress(path, task_id, completed, "stopped" if stopped else "complete")
    write_json(path / f"fold_{task_id}.json", {"task": task, "completed_epochs": completed,
                                                "stopped_early": stopped,
                                                "preprocessing_sha256": state.digest()})


def analyze(path: Path) -> dict:
    """Analyze only complete five-inner-fold alpha results; no outer TEST reads."""
    config = read_json(path / "config.json")
    if config["kind"] != "pareto":
        raise ValueError("Not a Pareto experiment")
    if any(not (path / f"fold_{task['id']}_epoch_{config['max_epochs']:03d}.json").exists()
           for task in config["tasks"]):
        raise ValueError("All 150 full-fidelity task results are required before selection")
    figure_dir = path / "figures"
    figure_dir.mkdir(exist_ok=True)
    selections = []
    table = []
    for outer in range(1, 6):
        candidates = tuple(config["proposals"][str(outer)]["candidates"])
        records = []
        for task in config["tasks"]:
            if task["outer_study"] != outer:
                continue
            epoch = read_json(path / f"fold_{task['id']}_epoch_{config['max_epochs']:03d}.json")
            valid = epoch["validation"]
            records.append({"alpha": task["alpha"], "fold": task["inner_fold"],
                            "bp_rows": valid["bp_rows"], "flp_valid_rows": valid["flp_valid_rows"],
                            "bp_excess_log_loss": valid["bp_excess_log_loss"],
                            "flp_excess_log_loss": valid["flp_excess_log_loss"],
                            "bp_loss": valid["bp_loss"], "flp_loss": valid["flp_loss"]})
        points = aggregate_folds(records, candidates)
        selected = select_pareto_knee(points)
        selected["outer_study"] = outer
        selections.append(selected)
        for point in selected["points"]:
            table.append({"outer_study": outer, **point,
                          "selected": point["alpha"] == selected["selected_alpha"]})
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter([p["bp_excess_log_loss"] for p in points],
                   [p["flp_excess_log_loss"] for p in points], c="gray", label="Candidates")
        frontier = sorted((p for p in points if p["alpha"] in selected["frontier_alphas"]),
                          key=lambda p: p["bp_excess_log_loss"])
        ax.plot([p["bp_excess_log_loss"] for p in frontier],
                [p["flp_excess_log_loss"] for p in frontier], color="tab:blue", label="Pareto frontier")
        chosen = next(p for p in points if p["alpha"] == selected["selected_alpha"])
        ax.scatter(chosen["bp_excess_log_loss"], chosen["flp_excess_log_loss"],
                   marker="*", s=160, color="tab:red", label="Selected")
        for point in points:
            ax.annotate(str(point["alpha"]), (point["bp_excess_log_loss"],
                                                  point["flp_excess_log_loss"]))
        ax.set(xlabel="BP excess log loss (nats)", ylabel="FLP excess log loss (nats)",
               title=f"Outer TRAIN study {outer}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"outer_{outer}_pareto.png", dpi=160)
        plt.close(fig)
    with (path / "pareto_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    result = {"configuration_sha256": hashlib.sha256(stable_json_bytes(config)).hexdigest(),
              "max_epochs": config["max_epochs"], "selections": selections,
              "table": "pareto_table.csv", "figures": "figures/outer_N_pareto.png"}
    write_json(path / "pareto_selection.json", result)
    return result
