"""CPU-only split-conformal evaluation on frozen BP reconstruction exports."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from wildfire_risk_uncertainty.training.cache import HpoCache, sha256

ROOT = Path(__file__).resolve().parents[3]
TEST_SOURCE = ROOT / "artifacts/evaluation/phase11a/phase11a-20261116-v1"
DEAD_SOURCE = ROOT / "artifacts/evaluation/phase10c/phase10c-20261116-v1"
REPORT = ROOT / "reports/uncertainty/11b"
SEED = 11_012
NOMINAL = (0.80, 0.90, 0.95)
BP_EDGES = (0.0, 0.001, 0.005, 0.01, 1.0)
BP_LABELS = ("[0, 0.001)", "[0.001, 0.005)", "[0.005, 0.01)", "[0.01, 1]")
TEST_COLUMNS = ("cell_id", "outer_study", "global_row", "bp_target", "bp_mean",
                "absolute_bp_error")
DEAD_COLUMNS = ("cell_id", "outer_study", "global_row", "bp_target", "bp_prediction")


def split_indices(n: int, study: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Outcome-independent, disjoint half split; odd extra goes to evaluation."""
    if n < 2 or study not in range(1, 6):
        raise ValueError("Need at least two cells and a study in 1..5")
    permutation = np.random.default_rng(seed + study).permutation(n)
    half = n // 2
    return np.sort(permutation[:half]), np.sort(permutation[half:])


def conformal_quantile(scores: np.ndarray, nominal: float) -> float:
    """Finite-sample split-conformal order statistic with an infinite sentinel."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Calibration scores must be a nonempty finite nonnegative vector")
    if not 0 < nominal < 1:
        raise ValueError("Nominal coverage must be strictly between zero and one")
    rank = math.ceil((len(values) + 1) * nominal)
    if rank > len(values):
        return math.inf
    return float(np.partition(values, rank - 1)[rank - 1])


def intervals(prediction: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 1 or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("BP predictions must be finite probabilities")
    if quantile < 0 or np.isnan(quantile):
        raise ValueError("Conformal quantile must be nonnegative")
    if math.isinf(quantile):
        return np.zeros_like(prediction), np.ones_like(prediction)
    return np.maximum(0.0, prediction - quantile), np.minimum(1.0, prediction + quantile)


def interval_metrics(target: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     nominal: float) -> dict[str, float]:
    """Inclusive coverage and central interval score (Winkler score)."""
    y, lo, hi = (np.asarray(a, dtype=np.float64) for a in (target, lower, upper))
    if y.ndim != 1 or not len(y) or y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("Targets and interval bounds must be matching nonempty vectors")
    if not all(np.isfinite(a).all() for a in (y, lo, hi)) or np.any((y < 0) | (y > 1)):
        raise ValueError("Targets and bounds must be finite; targets must be probabilities")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi) or not 0 < nominal < 1:
        raise ValueError("Invalid interval or nominal coverage")
    alpha = 1.0 - nominal
    score = (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return {"coverage": float(np.mean((lo <= y) & (y <= hi))),
            "mean_width": float(np.mean(hi - lo)), "interval_score": float(np.mean(score))}


def _digest_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def validate_pair(test: dict, dead: dict, study: int, cache: HpoCache) -> None:
    """Check source identity, BP alignment, and study-specific TRAIN exclusion."""
    for zone, data in (("TEST", test), ("DEAD_ZONE", dead)):
        ids = np.asarray(data["cell_id"])
        rows = np.asarray(data["global_row"])
        target = np.asarray(data["bp_target"], dtype=np.float64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        if not len(ids) or len(np.unique(ids)) != len(ids) or len(rows) != len(ids):
            raise ValueError(f"{zone} has missing or duplicate cell IDs")
        if np.any(np.asarray(data["outer_study"]) != study) or np.any(rows < 0) or np.any(rows >= len(cache.arrays["cell_id"])):
            raise ValueError(f"{zone} study/global-row mismatch")
        if not np.array_equal(ids, np.asarray(cache.arrays["cell_id"][rows])):
            raise ValueError(f"{zone} cell IDs do not align with frozen cache")
        if not np.array_equal(target, np.asarray(cache.arrays["bp"][rows]).astype(np.float64)):
            raise ValueError(f"{zone} BP targets do not align with frozen cache")
        if np.any(np.asarray(cache.arrays["inner_fold"][rows, study - 1]) != 0):
            raise ValueError(f"{zone} contains study-specific TRAIN-eligible cells")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all() or np.any((prediction < 0) | (prediction > 1)) or np.any((target < 0) | (target > 1)):
            raise ValueError(f"{zone} has invalid BP probabilities")
    if np.intersect1d(test["cell_id"], dead["cell_id"]).size:
        raise ValueError("Study TEST and DEAD_ZONE cell IDs overlap")
    if "absolute_bp_error" in test and not np.allclose(
        test["absolute_bp_error"], np.abs(test["bp_target"] - test["prediction"]),
        atol=1e-8, rtol=0,
    ):
        raise ValueError("Stored TEST BP error is inconsistent")


def _read_source(path: Path, columns: tuple[str, ...], expected: dict) -> tuple[dict, dict]:
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if any(sidecar.get(key) != value for key, value in expected.items()) or sha256(path) != sidecar["sha256"]:
        raise ValueError(f"Source provenance mismatch: {path}")
    table = pq.read_table(path, columns=list(columns))
    if len(table) != sidecar["n"]:
        raise ValueError(f"Source row count mismatch: {path}")
    return {name: table[name].to_numpy() for name in columns}, sidecar


def load_study(study: int, cache: HpoCache) -> tuple[dict, dict, dict]:
    test_path = TEST_SOURCE / "predictions" / f"outer_{study}.parquet"
    dead_path = DEAD_SOURCE / "predictions" / f"outer_{study}_dead.parquet"
    test, test_meta = _read_source(test_path, TEST_COLUMNS, {"study": study})
    dead, dead_meta = _read_source(dead_path, DEAD_COLUMNS, {"study": study, "zone": "DEAD_ZONE"})
    test["prediction"] = test.pop("bp_mean")
    dead["prediction"] = dead.pop("bp_prediction")
    sample_path = TEST_SOURCE / "sample" / f"outer_{study}.parquet"
    dead_plan = DEAD_SOURCE / "sample_plan" / f"outer_{study}_dead.parquet"
    if sha256(sample_path) != test_meta["sample_sha256"] or sha256(dead_plan) != dead_meta["plan_sha256"]:
        raise ValueError("Source sample-plan provenance mismatch")
    test_plan = pq.read_table(sample_path, columns=["cell_id", "global_row", "bp_target"])
    dead_plan_table = pq.read_table(dead_plan, columns=["cell_id", "global_row"])
    for key in ("cell_id", "global_row", "bp_target"):
        if not np.array_equal(test[key], test_plan[key].to_numpy()):
            raise ValueError(f"Phase 11A TEST prediction/plan {key} mismatch")
    for key in ("cell_id", "global_row"):
        if not np.array_equal(dead[key], dead_plan_table[key].to_numpy()):
            raise ValueError(f"Phase 10C DEAD_ZONE prediction/plan {key} mismatch")
    validate_pair(test, dead, study, cache)
    provenance = json.loads(
        (ROOT / "reports/uncertainty/11a/provenance.json").read_text()
    )
    record = next(
        (r for r in provenance["predictions"]["records"]
         if r["study"] == study),
        None,
    )
    if record is None or any(
        record[key] != test_meta[key]
        for key in ("sha256", "sample_sha256", "preprocessing_sha256")
    ):
        raise ValueError("Phase 11A global/sidecar provenance mismatch")
    checkpoints = [
        provenance["checkpoint_sha256"][f"outer_{study}_member_{member}"]
        for member in range(5)
    ]
    return test, dead, {"test_prediction_sha256": test_meta["sha256"],
                        "test_sample_sha256": test_meta["sample_sha256"],
                        "dead_prediction_sha256": dead_meta["sha256"],
                        "dead_sample_sha256": dead_meta["plan_sha256"],
                        "phase8_manifest_sha256": provenance["phase8_manifest_sha256"],
                        "checkpoint_sha256": checkpoints,
                        "preprocessing_sha256": test_meta["preprocessing_sha256"]}


def evaluate_study(test: dict, dead: dict, study: int, seed: int = SEED) -> tuple[list[dict], list[dict], dict]:
    cal_idx, eval_idx = split_indices(len(test["cell_id"]), study, seed)
    cal_ids, eval_ids = test["cell_id"][cal_idx], test["cell_id"][eval_idx]
    if np.intersect1d(cal_ids, eval_ids).size or len(np.unique(np.concatenate((cal_ids, eval_ids)))) != len(test["cell_id"]):
        raise ValueError("TEST calibration and evaluation IDs must partition the sample")
    y_eval = np.asarray(test["bp_target"][eval_idx], dtype=np.float64)
    p_eval = np.asarray(test["prediction"][eval_idx], dtype=np.float64)
    cal_sources = {
        "within_test": (np.asarray(test["bp_target"][cal_idx], dtype=np.float64),
                        np.asarray(test["prediction"][cal_idx], dtype=np.float64)),
        "dead_zone_transfer": (np.asarray(dead["bp_target"], dtype=np.float64),
                               np.asarray(dead["prediction"], dtype=np.float64)),
    }
    summary, strata = [], []
    for method, (y_cal, p_cal) in cal_sources.items():
        scores = np.abs(y_cal - p_cal)
        for nominal in NOMINAL:
            q = conformal_quantile(scores, nominal)
            lower, upper = intervals(p_eval, q)
            summary.append({"row_type": "study", "study": study, "calibration": method,
                            "nominal_coverage": nominal, "n_calibration": len(scores),
                            "n_evaluation": len(y_eval), "conformal_quantile": q,
                            **interval_metrics(y_eval, lower, upper, nominal)})
            for i, (left, right) in enumerate(itertools.pairwise(BP_EDGES)):
                mask = (p_eval >= left) & (p_eval <= right if i == len(BP_LABELS) - 1 else p_eval < right)
                if not np.any(mask):
                    continue
                strata.append({"row_type": "study", "study": study, "calibration": method,
                               "nominal_coverage": nominal, "bp_stratum": BP_LABELS[i],
                               "n_evaluation": int(mask.sum()),
                               **interval_metrics(y_eval[mask], lower[mask], upper[mask], nominal)})
    metadata = {"study": study, "seed": seed + study, "n_test": len(test["cell_id"]),
                "n_dead_zone": len(dead["cell_id"]), "n_test_calibration": len(cal_idx),
                "n_test_evaluation": len(eval_idx), "test_calibration_ids_sha256": _digest_ids(cal_ids),
                "test_evaluation_ids_sha256": _digest_ids(eval_ids),
                "dead_zone_ids_sha256": _digest_ids(dead["cell_id"])}
    return summary, strata, metadata


def equal_study_rows(rows: list[dict], *, by_stratum: bool = False) -> list[dict]:
    keys = ("calibration", "nominal_coverage", "bp_stratum") if by_stratum else ("calibration", "nominal_coverage")
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key, group in groups.items():
        item = {"row_type": "equal_study_mean", "study": "all", **dict(zip(keys, key)),
                "n_studies": len(group), "n_evaluation": sum(r["n_evaluation"] for r in group)}
        for metric in ("coverage", "mean_width", "interval_score"):
            item[metric] = float(np.mean([r[metric] for r in group]))
        if not by_stratum:
            item["n_calibration"] = ""
            item["conformal_quantile"] = ""
        output.append(item)
    return output


def add_transfer_deltas(rows: list[dict]) -> None:
    """Add paired DEAD_ZONE minus within-TEST differences on the same evaluation cells."""
    baseline = {(r["study"], r["nominal_coverage"]): r for r in rows
                if r["calibration"] == "within_test"}
    for row in rows:
        if row["calibration"] != "dead_zone_transfer":
            continue
        reference = baseline[row["study"], row["nominal_coverage"]]
        if row["n_evaluation"] != reference["n_evaluation"]:
            raise ValueError("Transfer and within-TEST evaluation counts differ")
        for key in ("coverage", "mean_width", "interval_score"):
            row[f"delta_{key}"] = row[key] - reference[key]


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _figures(rows: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 200})
    fig, ax = plt.subplots(figsize=(6.2, 4.5), layout="constrained")
    ax.plot([0.75, 1.0], [0.75, 1.0], color="0.45", linestyle="--", label="Nominal = empirical")
    for method, label, color in (("within_test", "Within TEST", "#0072B2"),
                                 ("dead_zone_transfer", "DEAD_ZONE transfer", "#D55E00")):
        points = sorted((r for r in rows if r["row_type"] == "equal_study_mean" and r["calibration"] == method),
                        key=lambda r: r["nominal_coverage"])
        ax.plot([r["nominal_coverage"] for r in points], [r["coverage"] for r in points],
                marker="o", linewidth=2, color=color, label=label)
    ax.set(xlabel="Nominal BP interval coverage", ylabel="Empirical BP interval coverage",
           title="FSim BP reconstruction: equal-study mean coverage", xlim=(0.77, 0.98), ylim=(0, 1.0))
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(REPORT / "calibration_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    x = np.arange(5)
    for offset, method, label, color in ((-0.18, "within_test", "Within TEST", "#0072B2"),
                                         (0.18, "dead_zone_transfer", "DEAD_ZONE transfer", "#D55E00")):
        values = [next(r["mean_width"] for r in rows if r["row_type"] == "study" and r["study"] == study
                       and r["calibration"] == method and r["nominal_coverage"] == .90) for study in range(1, 6)]
        ax.bar(x + offset, values, width=.35, color=color, label=label)
    ax.set(xlabel="Geographic outer study", ylabel="Mean clipped BP interval width",
           title="90% FSim BP reconstruction intervals", xticks=x, xticklabels=[str(i) for i in range(1, 6)])
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(REPORT / "interval_width_comparison.png")
    plt.close(fig)


def _report(rows: list[dict], metadata: dict) -> None:
    selected = [r for r in rows if r["nominal_coverage"] == .90]
    lines = ["# Phase 11B — split-conformal FSim BP reconstruction intervals", "",
             "## Method", "",
             "This exploratory follow-up uses frozen Phase 11A TEST predictions (100,000 sampled cells per geographic study) and Phase 10C DEAD_ZONE predictions. No model is refit. For each study, a seeded outcome-independent permutation places half of the TEST sample in calibration and half in evaluation. The identical TEST evaluation subset is used for both calibration sources. The study-specific DEAD_ZONE sample is the geographic-transfer calibration source. All reported intervals evaluate FSim BP reconstruction, not observed wildfire events.", "",
             "The nonconformity score is `|BP_FSim − BP_predicted|`. At nominal coverage `c`, with `n` calibration scores, the quantile is the value at one-based rank `ceil((n+1)c)`; a rank above `n` gives infinity. Evaluation intervals are `[max(0, prediction−q), min(1, prediction+q)]`. Primary coverage is 90%; secondary levels are 80% and 95%. Coverage includes both endpoints. Width is `upper−lower`. The central interval score is `width + 2/alpha*(lower−target)` below the interval, or `width + 2/alpha*(target−upper)` above it, where `alpha=1−c`.", "",
             "Predicted-BP diagnostic strata are fixed from Phase 10C: `[0,.001)`, `[.001,.005)`, `[.005,.01)`, `[.01,1]`. They use predictions available at inference time; per-stratum coverage is descriptive because quantiles were calibrated globally within each study. Equal-study rows average study metrics without pixel weighting; stratum aggregates average contributing studies and report `n_studies`. Spatial dependence and overlapping cells across outer studies limit independence interpretations.", "",
             "Split conformal's usual marginal coverage statement requires exchangeability of calibration and evaluation examples. Here the within-TEST random split supports an empirical check for this specified sampled TEST design; geographic dependence and alternative regional deployments require care. Conventional exchangeability-based guarantees do not apply to DEAD_ZONE-to-TEST transfer. Clipping retains the target support `[0,1]` and may make widths asymmetric. See [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511) for split-conformal foundations.", "",
             "## Results", "",
             "Primary 90% nominal intervals, evaluated on the same held-out TEST half in each comparison:", "",
             "| Study | Calibration | Calibration n | Evaluation n | Coverage | Mean width | Interval score | Δ coverage | Δ width | Δ score |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for study in [1, 2, 3, 4, 5, "all"]:
        for method in ("within_test", "dead_zone_transfer"):
            r = next(r for r in selected if r["study"] == study and r["calibration"] == method)
            n_cal = r["n_calibration"] if study != "all" else "—"
            differences = (f"{r['delta_coverage']:+.4f} | {r['delta_mean_width']:+.6f} | {r['delta_interval_score']:+.6f}"
                           if method == "dead_zone_transfer" else "— | — | —")
            lines.append(f"| {study} | {method} | {n_cal} | {r['n_evaluation']} | {r['coverage']:.4f} | {r['mean_width']:.6f} | {r['interval_score']:.6f} | {differences} |")
    lines += ["", "![Empirical versus nominal coverage](calibration_curve.png)", "",
              "![Study-level interval widths](interval_width_comparison.png)", "",
              "Full results: [coverage comparison](coverage_comparison.csv), [coverage by predicted BP](coverage_by_bp.csv), and [provenance](provenance.json). The tables are descriptive; no test of causal geographic effects or real-world risk calibration is implied.", "",
              "## Interpretation and limitations", "",
              "Compare the two calibrations study by study: transfer may change coverage, width, and interval score because the residual distributions differ. Within-TEST marginal coverage does not imply conditional coverage within BP strata, reliability on a new geographic region, or calibrated probabilities of model error. DEAD_ZONE cells were excluded from each study's TRAIN data; they are not a model trained without a spatial buffer. FSim supplies probabilistic targets, while these intervals quantify reconstruction error relative to those targets. Cross-study samples can repeat physical cells and are not independent replications.", "",
              "## Reproducibility", "",
              f"Run `.venv/bin/python -m wildfire_risk_uncertainty.analysis.phase11b --seed {metadata['seed']}` from the repository root. Input hashes, checkpoint/preprocessing provenance, split ID digests, exact sample sizes, and method definitions are in [provenance.json](provenance.json). The script reads one study at a time and never performs neural-network inference. If a source artifact or frozen cache alignment check fails, it stops before reporting results.", ""]
    (REPORT / "README.md").write_text("\n".join(lines))


def run(seed: int = SEED) -> None:
    sample_marker = json.loads((TEST_SOURCE / "sample/complete.json").read_text())
    test_gate = json.loads((TEST_SOURCE / "predictions/complete.json").read_text())
    dead_gate = json.loads((DEAD_SOURCE / "predictions/complete.json").read_text())
    if not sample_marker.get("outcome_independent") or sample_marker.get("n_per_study") != 100_000:
        raise ValueError("Phase 11A outcome-independent 100,000-cell sample is required")
    if test_gate.get("studies") != 5 or dead_gate.get("studies") != 5:
        raise ValueError("Five complete geographic studies are required")
    if len(test_gate.get("records", [])) != 5 or len(dead_gate.get("records", [])) != 10:
        raise ValueError("Completed-run record counts changed")
    phase9_manifest = ROOT / "artifacts/evaluation/phase9/phase9-20261104-v1/evaluation_manifest.json"
    phase8_manifest = ROOT / "artifacts/training/phase8/phase8-20261102T181708Z/manifest.json"
    if (test_gate.get("phase9_manifest_sha256") != sha256(phase9_manifest)
            or sample_marker.get("phase9_manifest_sha256") != test_gate["phase9_manifest_sha256"]):
        raise ValueError("Frozen Phase 9 evaluation manifest provenance mismatch")
    phase8_digest = sha256(phase8_manifest)
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    summary, strata, records = [], [], []
    for study in range(1, 6):
        test, dead, source_meta = load_study(study, cache)
        if len(test["cell_id"]) != 100_000:
            raise ValueError("Phase 11A study sample size changed")
        expected_test = next((r for r in test_gate["records"] if r["study"] == study), None)
        expected_dead = next((r for r in dead_gate["records"]
                              if r["study"] == study and r["zone"] == "DEAD_ZONE"), None)
        expected_sample = next((r for r in sample_marker["records"] if r["study"] == study), None)
        if (expected_test is None or expected_dead is None or expected_sample is None
                or expected_test["sha256"] != source_meta["test_prediction_sha256"]
                or expected_dead["sha256"] != source_meta["dead_prediction_sha256"]
                or expected_sample["sample_sha256"] != source_meta["test_sample_sha256"]
                or source_meta["phase8_manifest_sha256"] != phase8_digest):
            raise ValueError("Completed-run and sample-manifest provenance mismatch")
        a, b, split_meta = evaluate_study(test, dead, study, seed)
        summary.extend(a)
        strata.extend(b)
        records.append({**source_meta, **split_meta})
        print(f"Validated and evaluated study {study}: {len(test['cell_id']):,} TEST, {len(dead['cell_id']):,} DEAD_ZONE")
    summary.extend(equal_study_rows(summary))
    add_transfer_deltas(summary)
    strata.extend(equal_study_rows(strata, by_stratum=True))
    REPORT.mkdir(parents=True, exist_ok=True)
    _csv(REPORT / "coverage_comparison.csv", summary)
    _csv(REPORT / "coverage_by_bp.csv", strata)
    _figures(summary)
    provenance = {"phase": "11b", "seed": seed, "study_split_seed": "seed + study",
                  "split": "seeded random permutation, floor(n/2) calibration, remainder evaluation",
                  "sampling": sample_marker["method"], "nominal_coverage": NOMINAL,
                  "bp_strata_edges": BP_EDGES, "score": "absolute BP reconstruction error",
                  "quantile": "ceil((n+1)*coverage) one-based order statistic; infinity if rank > n",
                  "interval": "[max(0,p-q), min(1,p+q)]",
                  "interval_score": "width + 2/alpha*(lower-y) if y<lower + 2/alpha*(y-upper) if y>upper",
                  "test_sample_marker_sha256": sha256(TEST_SOURCE / "sample/complete.json"),
                  "test_complete_sha256": sha256(TEST_SOURCE / "predictions/complete.json"),
                  "dead_complete_sha256": sha256(DEAD_SOURCE / "predictions/complete.json"),
                  "phase9_evaluation_manifest_sha256": test_gate["phase9_manifest_sha256"],
                  "phase8_manifest_sha256": phase8_digest,
                  "cache_manifest_sha256": sha256(ROOT / "artifacts/training/hpo_cache_v1/manifest.json"),
                  "records": records}
    _json(REPORT / "provenance.json", provenance)
    _report(summary, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED, help="Base seed for study-specific TEST splits")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
