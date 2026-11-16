"""Phase 10D: bounded gradient SHAP for frozen BP ensembles."""

import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shap
import torch
from scipy.stats import spearmanr
from torch import nn

from wildfire_risk_uncertainty.analysis.phase10a import exact_tail_mask, load_study
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.phase9 import (
    PHASE8,
    ROOT,
    models_for,
    state_for,
)
from wildfire_risk_uncertainty.training.preprocessing import transform

SOURCE = ROOT / "artifacts/evaluation/phase10c/phase10c-20261116-v1"
OUT = ROOT / "artifacts/evaluation/phase10d/phase10d-20261116-v1"
REPORT = ROOT / "reports/uncertainty/10d"
FAILURES = ROOT / "reports/uncertainty/10a/low_disagreement_high_error.csv"
BACKGROUND_N = 128
REPRESENTATIVE_N = 110
ELEVATED_N = 50
FAILURE_N = 40
MEMBER_N = 32
NSAMPLES = 1024
SEED = 1042016


class BPWrapper(nn.Module):
    """Differentiable ensemble mean of member BP probabilities."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, features):
        # Reproduce Phase 9's bfloat16 forward path, then average float32 member probabilities.
        with torch.autocast(features.device.type, dtype=torch.bfloat16):
            probabilities = [m.bp_head(m.backbone(features)).sigmoid().float() for m in self.models]
        return torch.stack(probabilities).mean(0)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def cache_for():
    return HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)


def source_table(study):
    path = SOURCE / "predictions" / f"outer_{study}_test.parquet"
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != sidecar["sha256"]:
        raise ValueError("Phase 10C TEST export SHA mismatch")
    return pq.read_table(path), sidecar


def failure_counts():
    with FAILURES.open(newline="") as handle:
        return {int(r["study"]): int(r["failure_n"]) for r in csv.DictReader(handle)}


def exact_sample_failures(table, study, expected_count):
    """Map Phase 10A's exact full-support tail intersection onto sampled IDs."""
    ids, target, pred, variance = load_study(study)
    error = np.abs(pred.astype(np.float64) - target).astype(np.float32)
    failures = exact_tail_mask(error, .10, largest=True) & exact_tail_mask(variance, .20, largest=False)
    if int(failures.sum()) != expected_count:
        raise ValueError("Phase 10A full-support failure count mismatch")
    order = np.argsort(ids)
    if np.any(np.diff(ids[order]) <= 0):
        raise ValueError("Duplicate or unsorted full-support TEST IDs")
    sample_ids = table["cell_id"].to_numpy()
    positions = np.searchsorted(ids[order], sample_ids)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[order[positions]], sample_ids):
        raise ValueError("Sample IDs do not align with Phase 10A full TEST support")
    mapped = failures[order[positions]]
    if not np.array_equal(pred[order[positions]], table["bp_prediction"].to_numpy()):
        raise ValueError("Sample predictions differ from Phase 10A full TEST support")
    return mapped


def select_test(table, study, failure_membership):
    """Fixed-size diagnostic sample with matched low-error controls."""
    rng = np.random.default_rng(SEED + study)
    pred = table["bp_prediction"].to_numpy().astype(np.float64)
    target = table["bp_target"].to_numpy().astype(np.float64)
    error = np.abs(pred - target)
    if len(failure_membership) != len(pred):
        raise ValueError("Failure membership length mismatch")
    failure_pool = np.flatnonzero(failure_membership)
    if len(failure_pool) < FAILURE_N:
        raise ValueError(f"Study {study} has too few Phase 10A agreement failures in the 10C sample")
    failure = np.sort(rng.choice(failure_pool, FAILURE_N, replace=False))
    control_pool = np.flatnonzero(error <= np.median(error))
    if len(control_pool) < FAILURE_N:
        raise ValueError("Insufficient below-median-error controls")
    used = np.zeros(len(pred), dtype=bool)
    controls = []
    for i in failure:
        available = control_pool[~used[control_pool]]
        j = available[np.argmin(np.abs(pred[available] - pred[i]))]
        used[j] = True
        controls.append(j)
    controls = np.array(controls, dtype=np.int64)
    representative = np.sort(rng.choice(len(pred), REPRESENTATIVE_N, replace=False))
    elevated_pool = np.flatnonzero(pred >= np.quantile(pred, .9))
    elevated = np.sort(rng.choice(elevated_pool, ELEVATED_N, replace=False))
    # Category priority ensures one explanation per cell.
    category = np.full(len(pred), "", dtype="U15")
    category[representative] = "representative"
    category[elevated] = "elevated_bp"
    category[controls] = "control"
    category[failure] = "failure"
    selected = np.flatnonzero(category != "")
    if not np.array_equal(table["outer_study"].to_numpy()[selected], np.full(len(selected), study)):
        raise ValueError("TEST study mismatch")
    return selected, category[selected], failure, controls


def select_background(cache, study):
    rng = np.random.default_rng(SEED + 10_000 + study)
    total = len(cache.arrays["cell_id"])
    candidates = rng.choice(total, 1_000, replace=False)
    eligible = candidates[cache.arrays["inner_fold"][candidates, study - 1] != 0]
    if len(eligible) < BACKGROUND_N:
        raise ValueError("Insufficient eligible frozen TRAIN candidates")
    rows = eligible[:BACKGROUND_N]
    if len(np.unique(cache.arrays["cell_id"][rows])) != BACKGROUND_N:
        raise ValueError("Duplicate TRAIN background cell")
    return rows


def explain(explainer, model, x, background, seed, nsamples=NSAMPLES):
    """Return probability-scale expected gradients and completeness residual."""
    values = np.asarray(explainer.shap_values(x, nsamples=nsamples, rseed=seed))
    values = values.reshape(len(x), x.shape[1])
    with torch.no_grad():
        prediction = model(x).detach().cpu().numpy().reshape(-1)
        baseline = float(model(background).mean().item())
    residual = prediction - baseline - values.sum(axis=1)
    if values.shape != (len(x), x.shape[1]) or not np.isfinite(values).all() or not np.isfinite(residual).all():
        raise ValueError("Invalid gradient SHAP attribution or residual")
    return values, prediction, baseline, residual


def prepare():
    cache = cache_for()
    expected_failures = failure_counts()
    records = []
    for study in range(1, 6):
        state = state_for(study, cache)
        if state.inputs != tuple(cache.manifest["inputs"]) or state.rows != cache.manifest["outer_train_counts"][study - 1]:
            raise ValueError("Frozen feature order or TRAIN-only preprocessing mismatch")
        rows = select_background(cache, study)
        table, source = source_table(study)
        exact_failures = exact_sample_failures(table, study, expected_failures[study])
        selected, category, failure, controls = select_test(table, study, exact_failures)
        test_rows = table["global_row"].to_numpy()[selected]
        if np.any(cache.arrays["inner_fold"][test_rows, study - 1] != 0):
            raise ValueError("TEST row entered corresponding TRAIN")
        if not np.array_equal(table["cell_id"].to_numpy()[selected], cache.arrays["cell_id"][test_rows]):
            raise ValueError("TEST cell ID/cache mismatch")
        background = pa.table({"global_row": rows, "cell_id": cache.arrays["cell_id"][rows]})
        sample = table.take(pa.array(selected)).append_column("selection_category", pa.array(category))
        bg_path = OUT / f"outer_{study}_background.parquet"
        sample_path = OUT / f"outer_{study}_sample.parquet"
        write_table(bg_path, background); write_table(sample_path, sample)
        record = {"study": study, "background_n": len(rows), "sample_n": len(selected),
            "failure_n": len(failure), "control_n": len(controls),
            "sampled_exact_failure_pool_n": int(exact_failures.sum()),
            "failure_control_median_bp_difference": float(np.median(np.abs(
                table["bp_prediction"].to_numpy()[failure] - table["bp_prediction"].to_numpy()[controls]))),
            "failure_control_max_bp_difference": float(np.max(np.abs(
                table["bp_prediction"].to_numpy()[failure] - table["bp_prediction"].to_numpy()[controls]))),
            "background_sha256": sha256(bg_path), "sample_sha256": sha256(sample_path),
            "source_sha256": source["sha256"], "state_sha256": state.digest(),
            "phase8_manifest_sha256": sha256(PHASE8 / "manifest.json"),
            "checkpoint_count": 5}
        records.append(record)
    write_json(OUT / "sampling.json", {"seed": SEED, "background_policy": "random eligible TRAIN rows per outer study",
        "test_source": "Phase 10C 1% deterministic TEST export, from Phase 9 full-support predictions",
        "failure_definition": "Phase 10A study-specific top error decile and bottom disagreement quintile",
        "control_definition": "at-or-below-median absolute error, nearest predicted BP without replacement within study",
        "records": records})


def pilot():
    if not (OUT / "sampling.json").exists():
        raise ValueError("Run prepare first")
    cache = cache_for(); state = state_for(1, cache)
    manifest = json.loads((PHASE8 / "manifest.json").read_text())
    models = models_for(1, cache, state, manifest)
    wrapper = BPWrapper(models).cuda().eval()
    sample = pq.read_table(OUT / "outer_1_sample.parquet")
    bg = pq.read_table(OUT / "outer_1_background.parquet")
    x = torch.from_numpy(transform(np.asarray(cache.arrays["features"][sample["global_row"].to_numpy()[:16]]), state)).cuda()
    background = torch.from_numpy(transform(np.asarray(cache.arrays["features"][bg["global_row"].to_numpy()]), state)).cuda()
    started = time.perf_counter()
    values, prediction, baseline, residual = explain(shap.GradientExplainer(wrapper, background),
        wrapper, x, background, SEED, NSAMPLES)
    seconds = time.perf_counter() - started
    reference = sample["bp_prediction"].to_numpy()[:16]
    if np.max(np.abs(prediction - reference)) > 2e-6:
        raise ValueError("Frozen wrapper disagrees with Phase 9 prediction export")
    record = {"n": len(x), "background_n": len(background), "nsamples": NSAMPLES, "seconds": seconds,
        "estimated_ensemble_seconds": seconds / len(x) * 1_200,
        "mean_absolute_completeness_residual": float(np.abs(residual).mean()),
        "mean_absolute_prediction_shift": float(np.abs(prediction - baseline).mean()),
        "mean_absolute_shap": float(np.abs(values).mean())}
    write_json(OUT / "pilot.json", record)
    print(json.dumps(record, indent=2), flush=True)


def run_study(study):
    output = OUT / f"outer_{study}_ensemble.parquet"
    if output.exists():
        record = json.loads(output.with_suffix(".json").read_text())
        if sha256(output) != record["sha256"]:
            raise ValueError("Recovered SHAP output mismatch")
        complete_members = True
        for member in range(5):
            member_path = OUT / f"outer_{study}_member_{member}.parquet"
            member_sidecar = member_path.with_suffix(".json")
            if not member_path.exists() or not member_sidecar.exists():
                complete_members = False
                break
            if sha256(member_path) != json.loads(member_sidecar.read_text())["sha256"]:
                raise ValueError("Recovered member SHAP output mismatch")
        if complete_members:
            return
    cache = cache_for(); state = state_for(study, cache)
    manifest = json.loads((PHASE8 / "manifest.json").read_text())
    models = models_for(study, cache, state, manifest)
    wrapper = BPWrapper(models).cuda().eval()
    sample = pq.read_table(OUT / f"outer_{study}_sample.parquet")
    bg = pq.read_table(OUT / f"outer_{study}_background.parquet")
    rows = sample["global_row"].to_numpy(); bg_rows = bg["global_row"].to_numpy()
    if np.any(cache.arrays["inner_fold"][rows, study - 1] != 0) or np.any(cache.arrays["inner_fold"][bg_rows, study - 1] == 0):
        raise ValueError("TRAIN/TEST SHAP provenance mismatch")
    raw = np.asarray(cache.arrays["features"][rows])
    transformed = transform(raw, state)
    x = torch.from_numpy(transformed).cuda()
    background = torch.from_numpy(transform(np.asarray(cache.arrays["features"][bg_rows]), state)).cuda()
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = np.empty_like(transformed, dtype=np.float64)
    predictions = np.empty(len(x)); residuals = np.empty(len(x))
    started = time.perf_counter()
    for lo in range(0, len(x), 16):
        hi = min(lo + 16, len(x))
        values, prediction, baseline, residual = explain(explainer, wrapper, x[lo:hi], background,
            SEED + study * 1000 + lo, NSAMPLES)
        shap_values[lo:hi] = values; predictions[lo:hi] = prediction; residuals[lo:hi] = residual
    source_prediction = sample["bp_prediction"].to_numpy()
    max_difference = float(np.max(np.abs(predictions - source_prediction)))
    if max_difference > 2e-6:
        raise ValueError("SHAP wrapper disagrees with frozen Phase 9 predictions")
    data = {"cell_id": sample["cell_id"].to_numpy(), "outer_study": np.full(len(x), study),
        "selection_category": sample["selection_category"].to_numpy(),
        "global_row": rows, "x_5070": sample["x_5070"].to_numpy(), "y_5070": sample["y_5070"].to_numpy(),
        "bp_target": sample["bp_target"].to_numpy(), "bp_prediction": source_prediction,
        "bp_between_variance": sample["bp_between_variance"].to_numpy(),
        "baseline_bp": np.full(len(x), baseline), "completeness_residual": residuals}
    for j, name in enumerate(state.inputs):
        data[f"shap_{j:02d}"] = shap_values[:, j]
        data[f"transformed_{j:02d}"] = transformed[:, j]
        data[f"raw_{j:02d}"] = raw[:, j]
    write_table(output, pa.table(data))
    write_json(output.with_suffix(".json"), {"study": study, "n": len(x), "nsamples": NSAMPLES,
        "seconds": time.perf_counter() - started, "sha256": sha256(output), "baseline_bp": baseline,
        "max_phase9_prediction_difference": max_difference,
        "mean_absolute_completeness_residual": float(np.abs(residuals).mean()),
        "mean_absolute_prediction_shift": float(np.abs(predictions - baseline).mean()),
        "mean_absolute_relative_residual": float(np.mean(np.abs(residuals) / np.maximum(np.abs(predictions - baseline), 1e-6)))})
    print(f"Ensemble SHAP study {study}: {len(x)} rows", flush=True)
    # Identical matched diagnostic cases and the same TRAIN background for every member.
    category = sample["selection_category"].to_numpy()
    member_rows = np.r_[np.flatnonzero(category == "failure")[:MEMBER_N // 2],
        np.flatnonzero(category == "control")[:MEMBER_N // 2]]
    if len(member_rows) != MEMBER_N:
        raise ValueError("Insufficient matched member diagnostic cases")
    for member, model in enumerate(models):
        path = OUT / f"outer_{study}_member_{member}.parquet"
        member_model = BPWrapper([model]).cuda().eval()
        explainer = shap.GradientExplainer(member_model, background)
        member_values = []; member_residuals = []
        for lo in range(0, len(member_rows), 16):
            hi = min(lo + 16, len(member_rows))
            v, _p, _b, res = explain(explainer, member_model, x[lo:hi], background,
                SEED + study * 10_000 + member * 100 + lo, NSAMPLES)
            member_values.append(v); member_residuals.append(res)
        values = np.vstack(member_values)
        write_table(path, pa.table({"cell_id": sample["cell_id"].to_numpy()[member_rows],
            "outer_study": np.full(len(member_rows), study), "member": np.full(len(member_rows), member),
            "selection_category": category[member_rows],
            "completeness_residual": np.concatenate(member_residuals),
            **{f"shap_{j:02d}": values[:, j] for j in range(len(state.inputs))}}))
        write_json(path.with_suffix(".json"), {"sha256": sha256(path), "study": study,
            "member": member, "n": len(member_rows), "nsamples": NSAMPLES})


def run():
    if not (OUT / "pilot.json").exists():
        raise ValueError("Pilot benchmark required")
    for study in range(1, 6):
        run_study(study)
    write_json(OUT / "complete.json", {"studies": 5, "nsamples": NSAMPLES, "background_n": BACKGROUND_N})


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def analyze():
    if not (OUT / "complete.json").exists():
        raise ValueError("All five SHAP studies required")
    cache = cache_for(); names = cache.manifest["inputs"]
    importance = []; group_rows = []; consistency = []; member_rows = []; all_values = []; all_features = []
    for study in range(1, 6):
        path = OUT / f"outer_{study}_ensemble.parquet"
        if sha256(path) != json.loads(path.with_suffix(".json").read_text())["sha256"]:
            raise ValueError("Ensemble SHAP hash mismatch")
        table = pq.read_table(path)
        category = table["selection_category"].to_numpy()
        values = np.column_stack([table[f"shap_{j:02d}"].to_numpy() for j in range(len(names))])
        features = np.column_stack([table[f"transformed_{j:02d}"].to_numpy() for j in range(len(names))])
        raw_features = np.column_stack([table[f"raw_{j:02d}"].to_numpy() for j in range(len(names))])
        representative = category == "representative"
        all_values.append(values[representative]); all_features.append(features[representative])
        for j, name in enumerate(names):
            importance.append({"study": study, "feature": name, "feature_index": j,
                "representative_n": int(representative.sum()),
                "mean_absolute_shap_bp": float(np.mean(np.abs(values[representative, j]))),
                "mean_signed_shap_bp": float(np.mean(values[representative, j])),
                "feature_shap_spearman": float(spearmanr(features[representative, j], values[representative, j]).statistic)
                    if np.ptp(features[representative, j]) > 0 and np.ptp(values[representative, j]) > 0 else None})
            failure = category == "failure"; control = category == "control"
            group_rows.append({"study": study, "feature": name, "feature_index": j,
                "failure_n": int(failure.sum()), "control_n": int(control.sum()),
                "failure_mean_shap_bp": float(values[failure, j].mean()),
                "control_mean_shap_bp": float(values[control, j].mean()),
                "failure_mean_abs_shap_bp": float(np.abs(values[failure, j]).mean()),
                "control_mean_abs_shap_bp": float(np.abs(values[control, j]).mean()),
                "failure_mean_transformed": float(features[failure, j].mean()),
                "control_mean_transformed": float(features[control, j].mean()),
                "failure_missing_fraction": float((~np.isfinite(raw_features[failure, j])).mean()),
                "control_missing_fraction": float((~np.isfinite(raw_features[control, j])).mean())})
        member_importance = []; member_values = []; expected_member_ids = None; diagnostic_group = None
        for member in range(5):
            member_path = OUT / f"outer_{study}_member_{member}.parquet"
            if sha256(member_path) != json.loads(member_path.with_suffix(".json").read_text())["sha256"]:
                raise ValueError("Member SHAP hash mismatch")
            mt = pq.read_table(member_path)
            mv = np.column_stack([mt[f"shap_{j:02d}"].to_numpy() for j in range(len(names))])
            member_ids = mt["cell_id"].to_numpy()
            category_member = mt["selection_category"].to_numpy()
            if expected_member_ids is None:
                expected_member_ids = member_ids; diagnostic_group = category_member
            elif not np.array_equal(member_ids, expected_member_ids) or not np.array_equal(category_member, diagnostic_group):
                raise ValueError("Member explanations do not share identical cells")
            member_importance.append(np.abs(mv).mean(axis=0))
            member_values.append(mv)
        member_importance = np.array(member_importance)
        for left in range(5):
            for right in range(left + 1, 5):
                a = member_values[left]; b = member_values[right]
                cosine = np.sum(a * b, axis=1) / np.maximum(
                    np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-15)
                for group in ("failure", "control"):
                    consistency.append({"study": study, "member_a": left, "member_b": right,
                        "group": group,
                        "importance_spearman": float(spearmanr(member_importance[left], member_importance[right]).statistic),
                        "top5_overlap": len(set(np.argsort(member_importance[left])[-5:]) & set(np.argsort(member_importance[right])[-5:])),
                        "mean_signed_attribution_cosine": float(cosine[diagnostic_group == group].mean())})
        for j, name in enumerate(names):
            for member in range(5):
                member_rows.append({"study": study, "member": member, "feature": name, "feature_index": j,
                    "diagnostic_n": MEMBER_N, "mean_abs_shap_bp": float(member_importance[member, j])})
    write_csv(REPORT / "study_feature_importance.csv", importance)
    write_csv(REPORT / "failure_control_attributions.csv", group_rows)
    write_csv(REPORT / "member_consistency.csv", consistency)
    write_csv(REPORT / "member_feature_importance.csv", member_rows)
    ranked = []
    for j, name in enumerate(names):
        per_study = [r for r in importance if r["feature_index"] == j]
        ranked.append({"feature": name, "feature_index": j,
            "equal_study_mean_abs_shap_bp": float(np.mean([r["mean_absolute_shap_bp"] for r in per_study])),
            "study_min_abs_shap_bp": float(np.min([r["mean_absolute_shap_bp"] for r in per_study])),
            "study_max_abs_shap_bp": float(np.max([r["mean_absolute_shap_bp"] for r in per_study])),
            "equal_study_mean_signed_shap_bp": float(np.mean([r["mean_signed_shap_bp"] for r in per_study])),
            "equal_study_mean_feature_shap_spearman": float(np.mean(
                [r["feature_shap_spearman"] for r in per_study if r["feature_shap_spearman"] is not None]))
                if any(r["feature_shap_spearman"] is not None for r in per_study) else None})
    ranked.sort(key=lambda r: -r["equal_study_mean_abs_shap_bp"])
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    write_csv(REPORT / "global_feature_importance.csv", ranked)
    comparison = []
    for j, name in enumerate(names):
        rows = [r for r in group_rows if r["feature_index"] == j]
        comparison.append({"feature": name, "feature_index": j,
            "equal_study_failure_minus_control_mean_shap_bp": float(np.mean([
                r["failure_mean_shap_bp"] - r["control_mean_shap_bp"] for r in rows])),
            "equal_study_failure_minus_control_mean_abs_shap_bp": float(np.mean([
                r["failure_mean_abs_shap_bp"] - r["control_mean_abs_shap_bp"] for r in rows])),
            "equal_study_failure_minus_control_mean_transformed": float(np.mean([
                r["failure_mean_transformed"] - r["control_mean_transformed"] for r in rows]))})
    comparison.sort(key=lambda r: -abs(r["equal_study_failure_minus_control_mean_shap_bp"]))
    write_csv(REPORT / "failure_control_summary.csv", comparison)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    top = ranked[:15][::-1]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh([r["feature"] for r in top], [r["equal_study_mean_abs_shap_bp"] for r in top], color="#205493")
    ax.set(xlabel="Mean |gradient SHAP| on BP probability scale (equal-study mean)",
        title="Frozen ensemble BP feature attribution · TRAIN background")
    fig.tight_layout(); fig.savefig(REPORT / "feature_importance.png", dpi=240); fig.savefig(REPORT / "feature_importance.pdf"); plt.close(fig)
    all_values = np.vstack(all_values); all_features = np.vstack(all_features)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    rng = np.random.default_rng(SEED)
    for rank, row in enumerate(ranked[:15][::-1]):
        j = row["feature_index"]
        v = all_values[:, j]; f = all_features[:, j]
        lo, hi = np.quantile(f, [.05, .95]); color = np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1)
        ax.scatter(v, rank + rng.uniform(-.28, .28, len(v)), c=color, cmap="coolwarm", vmin=0, vmax=1,
            s=5, alpha=.45, rasterized=True)
    ax.axvline(0, color="0.5", lw=1)
    ax.set(yticks=np.arange(15), yticklabels=[r["feature"] for r in ranked[:15][::-1]],
        xlabel="Gradient SHAP contribution to ensemble mean BP", title="Representative TEST cells · TRAIN background")
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="coolwarm"), ax=ax,
        label="Relative feature value (5th–95th percentile)")
    fig.tight_layout(); fig.savefig(REPORT / "shap_summary.png", dpi=240); fig.savefig(REPORT / "shap_summary.pdf"); plt.close(fig)
    top_difference = comparison[:12][::-1]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh([r["feature"] for r in top_difference],
        [r["equal_study_failure_minus_control_mean_shap_bp"] for r in top_difference], color="#a23b45")
    ax.axvline(0, color="0.5", lw=1)
    ax.set(xlabel="Mean SHAP in failures − matched controls (BP probability)",
        title="Diagnostic agreement failures: equal-study attribution differences")
    fig.tight_layout(); fig.savefig(REPORT / "failure_control_attributions.png", dpi=240)
    fig.savefig(REPORT / "failure_control_attributions.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, row in zip(axes, ranked[:3], strict=True):
        j = row["feature_index"]
        ax.scatter(all_features[:, j], all_values[:, j], s=8, alpha=.35, color="#205493", rasterized=True)
        ax.axhline(0, color="0.5", lw=1)
        ax.set(title=row["feature"], xlabel="Model input after frozen preprocessing", ylabel="Gradient SHAP contribution to BP")
    fig.tight_layout(); fig.savefig(REPORT / "top_feature_dependence.png", dpi=240); fig.savefig(REPORT / "top_feature_dependence.pdf"); plt.close(fig)
    write_json(REPORT / "provenance.json", {"shap_version": shap.__version__, "seed": SEED,
        "background_n_per_study": BACKGROUND_N, "integration_samples_per_cell": NSAMPLES,
        "representative_n_per_study": REPRESENTATIVE_N, "elevated_n_target_per_study": ELEVATED_N,
        "failure_n_per_study": FAILURE_N, "member_diagnostic_n_per_study": MEMBER_N,
        "phase10a_failure_summary_sha256": sha256(FAILURES),
        "sampling_manifest_sha256": sha256(OUT / "sampling.json"),
        "studies": [{"study": study,
            "ensemble_sha256": sha256(OUT / f"outer_{study}_ensemble.parquet"),
            "member_sha256": [sha256(OUT / f"outer_{study}_member_{member}.parquet") for member in range(5)]}
            for study in range(1, 6)]})
    print("Phase 10D analysis complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "pilot", "run", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare, "pilot": pilot, "run": run, "analyze": analyze}[args.action]()


if __name__ == "__main__":
    main()
