"""Read-only pilot selection comparison under the frozen five-success rule."""

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "artifacts/hpo/outer_3_gradnorm_v3.sqlite3"
OUTPUT = ROOT / "reports/hpo/07e_protocol/pilot_selection.csv"


def selections(database: Path) -> list[dict]:
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT t.number, t.state, v.value FROM trials t "
            "JOIN studies s ON t.study_id=s.study_id "
            "LEFT JOIN trial_values v ON v.trial_id=t.trial_id AND v.objective=0 "
            "WHERE s.study_name='wildfire_outer_3_gradnorm_hpo_v3' ORDER BY t.number"
        ).fetchall()
    if len(rows) != 22 or sum(state == "COMPLETE" for _, state, _ in rows) != 20:
        raise ValueError("Pilot study history differs from audited 20-success study")
    completed = [(number, value) for number, state, value in rows if state == "COMPLETE"]
    result = []
    for count, label in ((5, "revised_comparable_first_five"), (20, "original_pilot_all_twenty")):
        selected = max(completed[:count], key=lambda item: (item[1], -item[0]))
        result.append({"analysis": label, "completed_successes": count,
                       "selection_rule": "highest_frozen_objective_earliest_trial_tie",
                       "selected_optuna_trial": selected[0], "inner_validation_objective": selected[1],
                       "outer_test_used": False})
    return result


if __name__ == "__main__":
    rows = selections(DATABASE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
