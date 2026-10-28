"""Read-only local progress view for one frozen outer-study HPO run."""

import argparse
import json
import sqlite3
from pathlib import Path


def status(root: Path, outer_study: int, *, gradnorm: bool = False) -> dict:
    if outer_study not in range(1, 6):
        raise ValueError("Outer study must be 1..5")
    name = (f"wildfire_outer_{outer_study}_gradnorm_hpo_v3" if gradnorm else
            f"wildfire_outer_{outer_study}_hpo_v1")
    database = root / "artifacts/hpo" / (f"outer_{outer_study}_gradnorm_v3.sqlite3" if gradnorm
                                         else f"outer_{outer_study}.sqlite3")
    if not database.exists():
        return {"study": name, "state": "not_started", "trials": 0,
                "trial_limit": None if gradnorm else 30}
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT t.number, t.state FROM trials AS t "
            "JOIN studies AS s ON t.study_id = s.study_id "
            "WHERE s.study_name = ? ORDER BY t.number", (name,),
        ).fetchall()
        limit = 30
        success_budget = None
        if gradnorm:
            attributes = connection.execute(
                "SELECT a.value_json FROM study_user_attributes AS a "
                "JOIN studies AS s ON a.study_id = s.study_id "
                "WHERE s.study_name = ? AND a.key = 'trial_budget'", (name,),
            ).fetchone()
            if attributes is None:
                success_attr = connection.execute(
                    "SELECT a.value_json FROM study_user_attributes AS a "
                    "JOIN studies AS s ON a.study_id = s.study_id "
                    "WHERE s.study_name = ? AND a.key = 'success_budget'", (name,),
                ).fetchone()
                if success_attr is None:
                    raise ValueError("GradNorm study lacks a persisted budget")
                success_budget = json.loads(success_attr[0])
                limit = int(success_budget["attempt_cap"])
            else:
                limit = int(json.loads(attributes[0])["current"])
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
    state = ("budget_reached" if success_budget and counts["COMPLETE"] == success_budget["target"]
             else "attempt_cap_reached" if success_budget and len(rows) == limit and not running
             else "failed" if counts["FAIL"] and not success_budget
             else "in_progress" if running or len(rows) < limit else "budget_reached")
    return {"study": name, "state": state, "trials": len(rows),
            "trial_limit": limit, "pending_trial_slots": max(0, limit - len(rows)),
            "trial_states": counts, "active_trial": active,
            **({"success_target": success_budget["target"],
                "remaining_successes": max(0, success_budget["target"] - counts["COMPLETE"])}
               if success_budget else {})}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only local HPO study status")
    parser.add_argument("--outer-study", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gradnorm", action="store_true")
    args = parser.parse_args()
    print(json.dumps(status(Path.cwd(), args.outer_study, gradnorm=args.gradnorm),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
