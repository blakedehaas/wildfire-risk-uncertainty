"""Audit saved benchmark outputs without starting training or changing evidence."""

import csv
import hashlib
import json
import math
from pathlib import Path

from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import stable_json_bytes

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    original = list(OUT.glob("c[123]_fold*")) + [
        OUT / name for name in ("metadata.json", "summary.json", "resource_samples.csv")
    ]
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original}
    metadata = read(OUT / "metadata.json")
    summary = read(OUT / "summary.json")
    history = ROOT / "artifacts/hpo/records/wildfire_outer_1_hpo_v1/trial_000"
    manifest = read(history / "manifest.json")
    assert metadata["configuration"] == manifest["configuration"]
    assert metadata["source_git_sha"] == manifest["provenance"]["source_git_sha"]
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    assert (
        hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest()
        == metadata["cache_manifest_sha256"]
    )
    states, counts = load_states(ROOT, cache, 1), support_counts(ROOT, 1)
    keys = (
        "bp_loss",
        "flp_loss",
        "combined_loss",
        "bp_loss_sum",
        "flp_loss_sum",
        "bp_rows",
        "flp_valid_rows",
        "optimizer_steps",
    )
    mapping = {
        "bp_loss": "bp_loss_mean",
        "flp_loss": "flp_loss_mean",
        "combined_loss": "equal_task_objective",
        "bp_loss_sum": "bp_loss_sum",
        "flp_loss_sum": "flp_loss_sum",
        "bp_rows": "bp_count",
        "flp_valid_rows": "flp_valid_count",
    }
    folds = []
    objectives = []
    for fold in range(1, 6):
        a, b = [read(OUT / f"c{c}_fold{fold}.json") for c in (1, 2)]
        historical = read(history / f"fold_{fold}_epoch_001.json")
        assert a["fold"] == b["fold"] == fold
        assert a["concurrency"] == 1 and b["concurrency"] == 2
        for role, support_role in [("training", "train"), ("validation", "validation")]:
            for key in keys:
                assert math.isfinite(a[role][key]) and a[role][key] == b[role][key], (
                    fold,
                    role,
                    key,
                )
            for key, previous_key in mapping.items():
                assert a[role][key] == historical[role][previous_key], (fold, role, key)
            assert a[role]["bp_rows"] == counts[fold][support_role]["rows"]
            assert a[role]["flp_valid_rows"] == counts[fold][support_role]["flp_valid"]
            assert a[role]["optimizer_steps"] == math.ceil(
                a[role]["bp_rows"] / metadata["configuration"]["batch_size"]
            )
        assert a["training"]["optimizer_steps"] == historical["optimizer_steps"]
        for key in ("model_sha256", "optimizer_sha256", "state_sha256"):
            assert a[key] == b[key] and len(a[key]) == 64, (fold, key)
        assert a["state_sha256"] == states[fold].digest()
        objectives.append(a["validation"]["combined_loss"])
        folds.append(
            {
                "fold": fold,
                "exact_c1_c2_metrics": True,
                "exact_historical_epoch1_metrics": True,
                "model_sha256": a["model_sha256"],
                "optimizer_sha256": a["optimizer_sha256"],
                "preprocessing_sha256": a["state_sha256"],
            }
        )
    objective = sum(objectives) / 5
    assert objective == read(history / "aggregate_epoch_001.json")["J_e"]
    with (OUT / "resource_samples.csv").open() as stream:
        samples = [
            {k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)
        ]
    resources = []
    for record in summary:
        c = record["concurrency"]
        rows = [r for r in samples if r["concurrency"] == c]
        assert max(r["gpu_vram_mib"] for r in rows) == record["peak_gpu_vram_mib"]
        assert max(r["swap_used_bytes"] for r in rows) == record["peak_swap_used_bytes"]
        previous = lower_bound = 0
        for row in rows:
            lower_bound += max(0, row["cpu_seconds"] - previous)
            previous = row["cpu_seconds"]
        resources.append(
            {
                "concurrency": c,
                "samples": len(rows),
                "sampled_cpu_seconds_lower_bound": lower_bound,
                "average_cpu_cores_lower_bound": lower_bound / record["wall_seconds"],
            }
        )
    assert (
        summary[2]["status"] == "STOPPED"
        and summary[2]["stop_reason"] == "VRAM pressure"
    )
    assert summary[2]["peak_gpu_vram_mib"] > 15500
    assert not list(OUT.glob("c3_fold*.json"))
    assert all(p.stat().st_size == 0 for p in OUT.glob("c3_fold*.stderr"))
    result = {
        "audit_passed": True,
        "folds": folds,
        "J_1": objective,
        "observed_c2_speedup": summary[0]["wall_seconds"] / summary[1]["wall_seconds"],
        "resources": resources,
        "existing_artifact_sha256": checksums,
        "c3_disposition": "Safety stopped; not an interruption; no retry under original gates",
        "c4_c5_disposition": "Not run after c3 safety stop",
        "limitations": [
            "One full epoch per fold, one measurement per level; no repeated warm timing",
            "State equality uses stored deterministic tensor/state digests; tensors were not retained",
            "No new traversal ID hashes; frozen reader and exact support/trajectory checks used",
            "Cache manifest/source provenance checked; full cache content not rehashed by benchmark",
            "Original cpu_seconds is not total CPU time: terminated workers disappear from samples",
            "CPU lower bound sums positive sample deltas; worker exits can hide CPU increments",
            "RSS sum includes shared mmap pages and overstates unique resident RAM",
            "GPU samples are device-wide, not attributable solely to benchmark workers",
            "Validation optimizer_steps field counts batches; validation makes no optimizer updates",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "J_1": objective,
                "observed_c2_speedup": result["observed_c2_speedup"],
            }
        )
    )


if __name__ == "__main__":
    main()
