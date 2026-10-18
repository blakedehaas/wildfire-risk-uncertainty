"""Two-process synthetic check of W&B's online same-run resume contract."""

import argparse
import time
import uuid
from pathlib import Path

import wandb

from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.hpo.wandb_tracking import require_online_authentication

PROJECT = "wildfire-risk-uncertainty"
ARTIFACT = Path("artifacts/hpo/06f/live_wandb_resume/run.json")
REPORT = Path("reports/training/06f_prelaunch/live_wandb_resume_validation.json")
METRIC = "prelaunch_resume_test/sequence"


def start() -> None:
    if ARTIFACT.exists():
        raise ValueError("Live resume test already exists; refusing to create another run")
    require_online_authentication()
    run_id = uuid.uuid4().hex[:8]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=PROJECT, group="checkpoint-6f-prelaunch",
                     job_type="synthetic-resume-test",
                     name=f"prelaunch-live-resume-{run_id}", id=run_id,
                     resume="never", mode="online", dir=str(ARTIFACT.parent),
                     settings=wandb.Settings(init_timeout=45))
    if run is None or run.id != run_id:
        raise RuntimeError("Initial online W&B run identity mismatch")
    write_json(ARTIFACT, {"run_id": run_id, "entity": run.entity,
                          "project": run.project, "run_url": run.url})
    run.log({METRIC: 1, "prelaunch_resume_test/value": 0.125})
    run.finish()


def resume() -> None:
    identity = read_json(ARTIFACT)
    require_online_authentication()
    run = wandb.init(project=identity["project"], entity=identity["entity"],
                     group="checkpoint-6f-prelaunch", job_type="synthetic-resume-test",
                     name=f"prelaunch-live-resume-{identity['run_id']}",
                     id=identity["run_id"], resume="must", mode="online",
                     dir=str(ARTIFACT.parent), settings=wandb.Settings(init_timeout=45))
    if run is None or run.id != identity["run_id"]:
        raise RuntimeError("Resumed online W&B run identity mismatch")
    run.log({METRIC: 2, "prelaunch_resume_test/value": 0.25})
    run.finish()


def verify() -> None:
    identity = read_json(ARTIFACT)
    api = wandb.Api(timeout=30)
    project = f"{identity['entity']}/{identity['project']}"
    last_observed = None
    for _ in range(12):
        runs = list(api.runs(project, filters={"name": identity["run_id"]}))
        history = (list(runs[0].scan_history(keys=[METRIC])) if len(runs) == 1 else [])
        values = [record[METRIC] for record in history if METRIC in record]
        last_observed = {"matching_run_count": len(runs), "sequence_values": values}
        if len(runs) == 1 and runs[0].id == identity["run_id"] and values == [1, 2]:
            write_json(REPORT, {"passed": True, "synthetic_only": True,
                                "two_separate_processes": True,
                                "resume_semantics": "id plus resume=must",
                                "run_id": identity["run_id"], "run_url": identity["run_url"],
                                "entity": identity["entity"], "project": identity["project"],
                                "matching_run_count": 1, "sequence_values": values,
                                "duplicate_metric_records": False,
                                "credentials_recorded": False})
            return
        time.sleep(5)
    raise RuntimeError(f"W&B same-run history verification failed: {last_observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "resume", "verify"))
    phase = parser.parse_args().phase
    {"start": start, "resume": resume, "verify": verify}[phase]()


if __name__ == "__main__":
    main()
