"""Deterministic static fold, spatial-CV, ASHA, and study plots from local records."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, metadata={"Software": "wildfire-risk-uncertainty checkpoint 6E"})
    plt.close()


def trial_figures(trial_dir: Path, output_dir: Path, label: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    folds = {}
    for fold in range(1, 6):
        records = [_read(p) for p in sorted(trial_dir.glob(f"fold_{fold}_epoch_*.json"))]
        if not records:
            continue
        folds[fold] = records
        epochs = [r["epoch"] for r in records]
        for component, metric in (("objective", "equal_task_objective"),
                                  ("bp", "bp_loss_mean"), ("flp", "flp_loss_mean")):
            plt.figure(figsize=(7, 4))
            for role in ("training", "validation"):
                plt.plot(epochs, [r[role][metric] for r in records], marker="o", markersize=2,
                         label=role)
            path = output_dir / f"{label}_fold_{fold}_{component}.png"
            _save(path, f"{label}: inner fold {fold} {component}", "Epoch", metric)
            written.append(str(path))
    aggregate = [_read(p) for p in sorted(trial_dir.glob("aggregate_epoch_*.json"))]
    if aggregate:
        plt.figure(figsize=(7, 4))
        for fold, records in folds.items():
            plt.plot([r["epoch"] for r in records],
                     [r["validation"]["equal_task_objective"] for r in records],
                     label=f"inner fold {fold}")
        path = output_dir / f"{label}_fold_validation_overlay.png"
        _save(path, f"{label}: spatial validation folds", "Epoch", "Validation equal-task objective")
        written.append(str(path))

        epochs = np.array([r["resource_epoch"] for r in aggregate])
        means = np.array([r["J_e"] for r in aggregate])
        std = np.array([r["fold_objective_std_population"] for r in aggregate])
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, means, label="five-fold mean J_e")
        plt.fill_between(epochs, means - std, means + std, alpha=0.2, label="± population SD")
        path = output_dir / f"{label}_spatial_mean_dispersion.png"
        _save(path, f"{label}: spatial validation mean and dispersion", "Common resource epoch",
              "Validation equal-task objective")
        written.append(str(path))

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [r["training_objective_mean"] for r in aggregate], label="five-fold training mean")
        plt.plot(epochs, means, label="five-fold validation mean")
        path = output_dir / f"{label}_train_validation_spatial_summary.png"
        _save(path, f"{label}: spatial training and validation", "Common resource epoch",
              "Equal-task objective")
        written.append(str(path))

        rungs = [r for r in aggregate if r["resource_milestone"]]
        if rungs:
            plt.figure(figsize=(7, 4))
            plt.plot([r["resource_epoch"] for r in rungs], [r["J_e"] for r in rungs],
                     marker="o", label="five-fold J_e")
            for r in rungs:
                plt.annotate(r["pruning_decision"].lower(),
                             (r["resource_epoch"], r["J_e"]), xytext=(3, 5),
                             textcoords="offset points", fontsize=7)
            path = output_dir / f"{label}_asha_resource_trace.png"
            _save(path, f"{label}: ASHA comparison trace", "Resource epoch", "J_e")
            written.append(str(path))
    return written


def study_figures(study: optuna.Study, output_dir: Path, label: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = study.get_trials(deepcopy=True)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    written = []
    if complete:
        plt.figure(figsize=(7, 4))
        plt.plot([t.number for t in complete], [t.value for t in complete], marker="o",
                 label="completed objective")
        path = output_dir / f"{label}_trial_objective_history.png"
        _save(path, f"{label}: completed trial objectives", "Trial number", "Final objective")
        written.append(str(path))

        plt.figure(figsize=(7, 4))
        plt.step([t.number for t in complete], np.minimum.accumulate([t.value for t in complete]),
                 where="post", label="best so far")
        path = output_dir / f"{label}_best_so_far.png"
        _save(path, f"{label}: best completed objective so far", "Trial number", "Objective")
        written.append(str(path))

        for parameter in ("depth", "width", "learning_rate", "weight_decay", "batch_size"):
            pairs = [(t.params[parameter], t.value) for t in complete if parameter in t.params]
            if not pairs:
                continue
            plt.figure(figsize=(7, 4))
            plt.scatter([x for x, _ in pairs], [y for _, y in pairs], label=parameter)
            if parameter in {"learning_rate", "weight_decay"}:
                plt.xscale("log")
            path = output_dir / f"{label}_slice_{parameter}.png"
            _save(path, f"{label}: {parameter} slice", parameter, "Final objective")
            written.append(str(path))

        if len(complete) >= 2:
            keys = [k for k in ("depth", "width", "learning_rate", "weight_decay", "batch_size")
                    if all(k in t.params for t in complete)]
            if keys:
                plt.figure(figsize=(8, 4))
                for trial in complete:
                    values = [float(trial.params[k]) for k in keys]
                    normalized = [(v - min(float(t.params[k]) for t in complete)) /
                                  max(1e-12, max(float(t.params[k]) for t in complete) -
                                      min(float(t.params[k]) for t in complete))
                                  for v, k in zip(values, keys, strict=True)]
                    plt.plot(range(len(keys)), normalized, alpha=0.5, label=f"trial {trial.number}")
                plt.xticks(range(len(keys)), keys, rotation=25)
                path = output_dir / f"{label}_parallel_coordinates.png"
                _save(path, f"{label}: normalized hyperparameter paths", "Hyperparameter",
                      "Within-study normalized value")
                written.append(str(path))
        if len(complete) >= 10:
            importance = optuna.importance.get_param_importances(study)
            plt.figure(figsize=(7, 4))
            plt.bar(importance.keys(), importance.values(), label="completed trials")
            plt.xticks(rotation=25)
            path = output_dir / f"{label}_hyperparameter_importance.png"
            _save(path, f"{label}: hyperparameter importance", "Hyperparameter", "Importance")
            written.append(str(path))

    plt.figure(figsize=(7, 4))
    resources = [max(t.intermediate_values, default=0) for t in trials]
    plt.scatter([t.number for t in trials], resources, label="last ASHA resource")
    path = output_dir / f"{label}_trial_resource.png"
    _save(path, f"{label}: resource reached", "Trial number", "Last reported epoch")
    written.append(str(path))

    plt.figure(figsize=(7, 4))
    durations = [(t.datetime_complete - t.datetime_start).total_seconds()
                 if t.datetime_start and t.datetime_complete else 0.0 for t in trials]
    plt.scatter([t.number for t in trials], durations, label="trial duration")
    path = output_dir / f"{label}_trial_duration.png"
    _save(path, f"{label}: trial durations", "Trial number", "Elapsed seconds")
    written.append(str(path))

    counts = Counter(t.state.name for t in trials)
    plt.figure(figsize=(6, 4))
    states = ["COMPLETE", "PRUNED", "FAIL"]
    plt.bar(states, [counts[s] for s in states], label="trial count")
    path = output_dir / f"{label}_trial_state_counts.png"
    _save(path, f"{label}: trial states", "State", "Trials")
    written.append(str(path))
    return written


def curated_real_figures(trial_dir: Path, output_dir: Path) -> list[str]:
    """Summarize the fixed smoke trial from authoritative local records only."""
    records = [_read(trial_dir / f"fold_{fold}_epoch_001.json") for fold in range(1, 6)]
    aggregate = _read(trial_dir / "aggregate_epoch_001.json")
    folds = np.arange(1, 6)
    written = []

    plt.figure(figsize=(7, 4))
    plt.plot(folds, aggregate["fold_validation_objectives"], "o-", label="validation objective")
    plt.xticks(folds)
    path = output_dir / "real_validation_objective_by_fold.png"
    _save(path, "Fixed smoke: validation objective by inner fold", "Inner fold",
          "Validation equal-task objective")
    written.append(str(path))

    for component, field in (("bp", "bp_loss_mean"), ("flp", "flp_loss_mean"),
                             ("equal_task", "equal_task_objective")):
        plt.figure(figsize=(7, 4))
        for role in ("training", "validation"):
            plt.plot(folds, [record[role][field] for record in records], "o-", label=role)
        plt.xticks(folds)
        path = output_dir / f"real_{component}_train_validation_by_fold.png"
        _save(path, f"Fixed smoke: {component} train and validation, epoch 1", "Inner fold",
              field)
        written.append(str(path))

    plt.figure(figsize=(7, 4))
    values = aggregate["fold_validation_objectives"]
    plt.scatter(folds, values, label="inner-fold validation")
    plt.axhline(aggregate["J_e"], color="black", label="five-fold mean J₁")
    plt.axhspan(aggregate["J_e"] - aggregate["fold_objective_std_population"],
                aggregate["J_e"] + aggregate["fold_objective_std_population"],
                alpha=0.15, label="mean ± population SD")
    plt.xticks(folds)
    path = output_dir / "real_spatial_cv_summary.png"
    _save(path, "Fixed smoke: spatial CV at epoch 1", "Inner fold",
          "Validation equal-task objective")
    written.append(str(path))

    plt.figure(figsize=(7, 4))
    plt.bar(folds - 0.18, [r["training_wall_seconds"] for r in records], width=0.36,
            label="training")
    plt.bar(folds + 0.18, [r["validation_wall_seconds"] for r in records], width=0.36,
            label="validation")
    plt.xticks(folds)
    path = output_dir / "real_runtime_by_fold.png"
    _save(path, "Fixed smoke: epoch 1 runtime", "Inner fold", "Wall seconds")
    written.append(str(path))
    return written


def curated_synthetic_figures(records_root: Path, output_dir: Path) -> list[str]:
    """Show ASHA decisions and trial states without loading or rerunning a study."""
    trial_dirs = sorted(records_root.glob("trial_*"))
    states = [(int(path.name.split("_")[1]), _read(path / "state.json"))
              for path in trial_dirs]
    plt.figure(figsize=(7, 4))
    for number, state in states:
        points = [_read(p) for p in sorted((records_root / f"trial_{number:03d}").glob(
            "aggregate_epoch_*.json")) if _read(p)["resource_milestone"]]
        if points:
            plt.plot([p["resource_epoch"] for p in points], [p["J_e"] for p in points],
                     marker="o", label=f"trial {number}: {state['trial_state']}")
    path = output_dir / "synthetic_asha_pruning_trace.png"
    _save(path, "Synthetic ASHA five-fold resource decisions", "Resource epoch", "Jₑ")
    written = [str(path)]

    plt.figure(figsize=(7, 4))
    colors = {"COMPLETE": "#287e4b", "PRUNED": "#df9a27", "FAIL": "#b64b4b",
              "FAILED": "#b64b4b"}
    for state_name in ("COMPLETE", "PRUNED", "FAIL", "FAILED"):
        selected = [(number, state) for number, state in states
                    if state["trial_state"] == state_name]
        if selected:
            resources = []
            for number, _ in selected:
                epochs = [int(p.stem.split("_")[-1]) for p in
                          (records_root / f"trial_{number:03d}").glob("aggregate_epoch_*.json")]
                resources.append(max(epochs, default=0))
            plt.scatter([n for n, _ in selected], resources, color=colors[state_name],
                        label=f"{state_name} ({len(selected)})")
    path = output_dir / "synthetic_trial_state_resource.png"
    _save(path, "Synthetic trial states and common resource", "Trial number",
          "Last common resource epoch")
    written.append(str(path))
    return written
