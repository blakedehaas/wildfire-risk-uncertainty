"""Read-only Phase 7E outer-study-1 prelaunch check; no GPU work or W&B writes."""

import json
import sqlite3
import subprocess
from pathlib import Path

import torch

from wildfire_risk_uncertainty.hpo.phase7e import validate_manifest
from wildfire_risk_uncertainty.hpo.status import status

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path = ROOT) -> dict:
    manifest = validate_manifest(root, 1)
    study = manifest["studies"]["1"]
    db = root / study["sqlite"]
    records = root / study["record_root"]
    if db.exists() or records.exists():
        raise RuntimeError("Outer-study-1 GradNorm artifacts already exist; inspect before launch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise RuntimeError("Launch requires a clean committed worktree")
    pilot = status(root, 3, gradnorm=True)
    if pilot["trial_states"]["COMPLETE"] != 20 or pilot["trial_states"]["FAIL"] != 2:
        raise RuntimeError("Completed pilot study differs from Phase 7D audit")
    with sqlite3.connect(f"file:{(root / 'artifacts/hpo/outer_3_gradnorm_v3.sqlite3').resolve()}?mode=ro",
                         uri=True) as connection:
        if connection.execute("SELECT count(*) FROM trials").fetchone()[0] != 22:
            raise RuntimeError("Pilot trial count differs")
    active = []
    for path in (root / "artifacts/benchmark/07a_detached").glob("07a-*/config.json"):
        config = json.loads(path.read_text())
        if config.get("kind") == "gradnorm_hpo" and config.get("outer_study") == 1:
            state = json.loads((path.parent / "status.json").read_text())
            if state.get("state") in {"starting", "running", "stopping"}:
                active.append(path.parent.name)
    if active:
        raise RuntimeError(f"Existing outer-study-1 controller: {active}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    available, total = torch.cuda.mem_get_info()
    if available < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free GPU memory")
    return {"ready_for_review": True, "outer_study": 1,
            "study": study["optuna_study"], "sqlite": study["sqlite"],
            "record_root": study["record_root"],
            "success_target": manifest["success_target"],
            "attempt_cap": manifest["attempt_cap"],
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_free_bytes": available, "cuda_total_bytes": total,
            "wandb_online_auth": "checked_by_launcher_at_execution"}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, sort_keys=True))
