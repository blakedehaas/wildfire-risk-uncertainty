"""Explicit 6E validation commands; no default real HPO execution."""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pyarrow
import torch
import wandb

from wildfire_risk_uncertainty.hpo.config import (
    MAX_EPOCHS,
    MILESTONES,
    SEARCH_SPACE,
    SEED,
    search_space_hash,
)
from wildfire_risk_uncertainty.hpo.real import STUDY_NAME as REAL_NAME
from wildfire_risk_uncertainty.hpo.real import run_real_hpo, run_smoke
from wildfire_risk_uncertainty.hpo.reconcile import reconcile_trial
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.reporting import real_smoke_tables
from wildfire_risk_uncertainty.hpo.synthetic import STUDY_NAME as SYNTHETIC_NAME
from wildfire_risk_uncertainty.hpo.synthetic import run as run_synthetic
from wildfire_risk_uncertainty.hpo.visualization import (
    curated_real_figures,
    curated_synthetic_figures,
    study_figures,
    trial_figures,
)
from wildfire_risk_uncertainty.hpo.wandb_tracking import require_online_authentication


def write_figure_manifest(root: Path) -> None:
    """Keep only curated figures in Git; detailed plots remain regenerable artifacts."""
    report = root / "reports/hpo/06e"
    artifact = root / "artifacts/hpo/06e"
    committed = sorted(str(path.relative_to(root)) for path in (report / "figures").glob("*.png"))
    detailed = sorted(str(path.relative_to(root)) for path in (artifact / "figures").glob("*.png"))
    write_json(artifact / "detailed_figures_manifest.json", {
        "files": detailed, "count": len(detailed), "regenerable_from": "local raw metric records"})
    write_json(report / "figures_manifest.json", {
        "committed_report_figures": committed,
        "detailed_artifact_figures": {
            "directory": "artifacts/hpo/06e/figures",
            "count": len(detailed),
            "manifest": "artifacts/hpo/06e/detailed_figures_manifest.json",
            "git_ignored": True,
            "regenerable_from": "authoritative local raw metric records",
        },
    })


def report_setup(root: Path) -> None:
    report = root / "reports/hpo/06e"
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "environment.json", {
        "base_git_sha": "d719dfafbbbe0475960e19a13149fa5d255ebcb0",
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "bf16": torch.cuda.is_bf16_supported(), "numpy": np.__version__,
        "pyarrow": pyarrow.__version__, "optuna": optuna.__version__, "wandb": wandb.__version__,
    })
    write_json(report / "search_space.json", {"dimensions": SEARCH_SPACE,
        "sha256": search_space_hash(), "sampler": {"name": "TPESampler", "seed": SEED},
        "pruner": {"name": "SuccessiveHalvingPruner", "min_resource": 1,
                   "reduction_factor": 4, "min_early_stopping_rate": 0},
        "resource_unit": "epoch", "milestones": MILESTONES, "maximum_epochs": MAX_EPOCHS,
        "planned_real_hpo_trials_per_outer_study": 30,
        "planned_independent_outer_studies": 5, "n_jobs": 1})
    write_json(report / "tracking_schema.json", {
        "authority": "atomic local JSON records under ignored artifacts/hpo",
        "mirror": "one online W&B run per Optuna trial",
        "hierarchy": ["study", "trial", "inner_fold", "epoch"],
        "fold_epoch_file": "trial_NNN/fold_K_epoch_EEE.json",
        "aggregate_file": "trial_NNN/aggregate_epoch_EEE.json",
        "raw_task_fields": ["bp_loss_sum", "bp_count", "flp_loss_sum", "flp_valid_count"],
        "derived_fields": ["bp_loss_mean", "flp_loss_mean", "equal_task_objective"],
        "spatial_objective": "arithmetic mean of five fold validation equal-task objectives",
        "dispersion": "population standard deviation across five folds",
        "wandb_axes": "fold_K/epoch per fold; spatial_cv/resource_epoch for aggregates",
        "serialization_comparison_tolerance": {"relative": 1e-12, "absolute": 1e-12},
    })


def synthetic_validation(root: Path) -> None:
    report_setup(root)
    for phase in ("synthetic-phase1", "synthetic-phase2"):
        subprocess.run([sys.executable, "-m", "wildfire_risk_uncertainty.hpo.cli", phase],
                       cwd=root, check=True)
    artifact = root / "artifacts/hpo/06e"
    study = optuna.load_study(study_name=SYNTHETIC_NAME,
                              storage=f"sqlite:///{(artifact / 'synthetic.sqlite3').resolve()}")
    detailed_dir = artifact / "figures"
    checks = []
    for trial in study.get_trials(deepcopy=True):
        local = artifact / "records" / SYNTHETIC_NAME / f"trial_{trial.number:03d}"
        checks.append(reconcile_trial(trial, local))
        trial_figures(local, detailed_dir, f"synthetic_trial_{trial.number:03d}")
    study_figures(study, detailed_dir, "synthetic_study")
    curated_synthetic_figures(artifact / "records" / SYNTHETIC_NAME,
                              root / "reports/hpo/06e/figures")
    write_json(root / "reports/hpo/06e/metric_reconciliation.json", {
        "synthetic_trials": checks, "real_smoke": None})
    write_json(root / "reports/hpo/06e/wandb_validation.json", {
        "online_auth_check": read_json(artifact / "auth_check.json"),
        "synthetic_online_runs": [{"trial_number": x["trial_number"], "run_id": x["wandb_run_id"],
                                   "url": x["wandb_run_url"]} for x in checks],
        "real_smoke_online_run": None,
        "online_spatial_history_reconciled": True})
    write_figure_manifest(root)


def real_smoke_validation(root: Path) -> None:
    report_setup(root)
    run_smoke(root)
    real_smoke_tables(root)
    artifact = root / "artifacts/hpo/06e"
    study = optuna.load_study(study_name=REAL_NAME,
                              storage=f"sqlite:///{(artifact / 'fixed_real_smoke.sqlite3').resolve()}")
    trial = study.get_trials(deepcopy=True)[0]
    local = artifact / "records" / REAL_NAME / "trial_000"
    check = reconcile_trial(trial, local, check_fold_history=True)
    trial_figures(local, artifact / "figures", "real_fixed_smoke")
    study_figures(study, artifact / "figures", "real_fixed_smoke_study")
    curated_real_figures(local, root / "reports/hpo/06e/figures")
    report = root / "reports/hpo/06e"
    metric = read_json(report / "metric_reconciliation.json")
    metric["real_smoke"] = check
    write_json(report / "metric_reconciliation.json", metric)
    online = read_json(report / "wandb_validation.json")
    online["real_smoke_online_run"] = {"trial_number": trial.number,
        "run_id": check["wandb_run_id"], "url": check["wandb_run_url"],
        "fold_specific_axes_reconciled": True,
        "spatial_resource_axis_reconciled": True}
    write_json(report / "wandb_validation.json", online)
    write_figure_manifest(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested spatial HPO infrastructure")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-validate")
    commands.add_parser("synthetic-phase1", help=argparse.SUPPRESS)
    commands.add_parser("synthetic-phase2", help=argparse.SUPPRESS)
    commands.add_parser("real-smoke")
    future = commands.add_parser("real-hpo", help="Future 6F use only; explicit authorization required")
    future.add_argument("--outer-study", type=int, choices=range(1, 6), required=True)
    future.add_argument("--cache-dir", type=Path, required=True)
    future.add_argument("--pause-after-epoch", type=int, choices=range(1, 101))
    future.add_argument("--checkpoint-6f-authorized", action="store_true", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "synthetic-validate":
        synthetic_validation(root)
    elif args.command == "synthetic-phase1":
        run_synthetic(root, phase="first")
    elif args.command == "synthetic-phase2":
        run_synthetic(root, phase="second")
    elif args.command == "real-smoke":
        real_smoke_validation(root)
    else:
        require_online_authentication()
        run_real_hpo(root, args.outer_study, cache_dir=args.cache_dir,
                     pause_after_epoch=args.pause_after_epoch)


if __name__ == "__main__":
    main()
