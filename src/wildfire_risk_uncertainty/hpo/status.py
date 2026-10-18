"""Read-only local progress view for one frozen outer-study HPO run."""

import argparse
import json
import sqlite3
from pathlib import Path


def status(root: Path, outer_study: int) -> dict:
    name = f"wildfire_outer_{outer_study}_hpo_v1"
    database = root / "artifacts/hpo" / f"outer_{outer_study}.sqlite3"
    if not database.exists():
        return {"study": name, "state": "not_started", "trials": 0}
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT t.number, t.state FROM trials AS t "
            "JOIN studies AS s ON t.study_id = s.study_id "
            "WHERE s.study_name = ? ORDER BY t.number", (name,),
        ).fetchall()
    counts = {state: sum(value == state for _, value in rows)
              for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")}
    running = [number for number, state in rows if state == "RUNNING"]
    active = None
    if running:
        record = root / "artifacts/hpo/records" / name / f"trial_{running[0]:03d}"
        aggregates = sorted(record.glob("aggregate_epoch_*.json"))
        active = {"trial_number": running[0],
                  "last_common_epoch": int(aggregates[-1].stem.split("_")[-1]) if aggregates else 0,
                  "fold_epoch_records": len(list(record.glob("fold_*_epoch_*.json")))}
    return {"study": name, "state": "in_progress" if len(rows) < 30 else "budget_reached",
            "trials": len(rows), "trial_states": counts, "active_trial": active}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only local HPO study status")
    parser.add_argument("--outer-study", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    print(json.dumps(status(Path.cwd(), args.outer_study), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
