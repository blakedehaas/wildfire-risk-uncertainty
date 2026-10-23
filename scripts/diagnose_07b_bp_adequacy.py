"""Read-only TRAIN/geographic VALIDATION scores for saved Phase 7A checkpoints."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.cache_checkpoint import cache_binding
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.preprocessing import transform
from wildfire_risk_uncertainty.training.trainer import make_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/benchmark/07a_detached/07a-20261022T054112Z-5270cc63"
OUTPUT = ROOT / "reports/training/07b"
EPOCHS = (1, 3, 25, 50, 100)
STRATA = ((0.0, 0.001), (0.001, 0.01), (0.01, 0.02), (0.02, 1.000001))
MIN_TILE_ROWS = 100


def empty() -> dict:
    return dict.fromkeys(("n", "target", "prediction", "target2", "prediction2",
                          "cross", "absolute", "square", "ce", "entropy", "brier_base"), 0.0)


def summarize(s: dict, climate: float) -> dict:
    n = s["n"]
    if not n:
        return {"rows": 0}
    target, prediction = s["target"] / n, s["prediction"] / n
    target_var = s["target2"] - s["target"] ** 2 / n
    prediction_var = s["prediction2"] - s["prediction"] ** 2 / n
    covariance = s["cross"] - s["target"] * s["prediction"] / n
    correlation = (covariance / math.sqrt(target_var * prediction_var)
                   if target_var > 0 and prediction_var > 0 else None)
    base_ce = -target * math.log(climate) - (1 - target) * math.log1p(-climate)
    return {"rows": int(n), "target_mean": target, "prediction_mean": prediction,
            "bias": prediction - target, "prediction_target_pearson": correlation,
            "mae": s["absolute"] / n, "rmse": math.sqrt(s["square"] / n),
            "brier": s["square"] / n, "climatology_brier": s["brier_base"] / n,
            "cross_entropy": s["ce"] / n, "excess_log_loss": (s["ce"] - s["entropy"]) / n,
            "climatology_excess_log_loss": base_ce - s["entropy"] / n,
            "log_skill": base_ce - s["ce"] / n}


@torch.no_grad()
def score(cache: HpoCache, state, model, role: str, climate: float) -> dict:
    """Score each frozen cache partition with the training inference precision."""
    buckets = [empty() for _ in range(5)]
    tile_pairs = [[] for _ in buckets]
    flp_ce = flp_entropy = flp_rows = 0.0
    offsets = cache.arrays["partition_offsets"]
    model.eval()
    for number in range(len(cache.names)):
        start, stop = map(int, offsets[number:number + 2])
        labels = cache.arrays["inner_fold"][start:stop, 2]
        chosen = (labels != 0) & (labels != 1) if role == "train" else labels == 1
        rows_in_tile = np.flatnonzero(chosen) + start
        if not len(rows_in_tile):
            continue
        tile = [{"n": 0, "target": 0.0, "prediction": 0.0} for _ in buckets]
        for offset in range(0, len(rows_in_tile), 65536):
            rows = rows_in_tile[offset:offset + 65536]
            features = transform(np.asarray(cache.arrays["features"][rows]), state)
            if not np.isfinite(features).all():
                raise FloatingPointError("Nonfinite features")
            x = torch.from_numpy(features).cuda()
            raw_y = torch.from_numpy(np.asarray(cache.arrays["bp"][rows])).cuda()
            y = raw_y.double()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bp_logits, flp_logits = model(x)
            z = bp_logits.double()
            p = z.sigmoid()
            if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(flp_logits).all()):
                raise FloatingPointError("Nonfinite logits")
            valid = torch.from_numpy(np.asarray(cache.arrays["flp_valid"][rows]).astype(bool)).cuda()
            if bool(valid.any()):
                target = torch.from_numpy(np.asarray(cache.arrays["flp"][rows])).cuda()[valid].float()
                log_prob = F.log_softmax(flp_logits[valid].float(), dim=-1)
                flp_ce += float(-(target * log_prob).sum())
                flp_entropy += float(-torch.special.xlogy(target, target).sum())
                flp_rows += target.shape[0]
            masks = [torch.ones_like(raw_y, dtype=torch.bool),
                     *((raw_y >= low) & (raw_y < high) for low, high in STRATA)]
            for index, mask in enumerate(masks):
                a, b, logits = y[mask], p[mask], z[mask]
                if not a.numel():
                    continue
                s, t = buckets[index], tile[index]
                n, ay, bp = a.numel(), float(a.sum()), float(b.sum())
                s["n"] += n
                s["target"] += ay
                s["prediction"] += bp
                s["target2"] += float(a.square().sum())
                s["prediction2"] += float(b.square().sum())
                s["cross"] += float((a * b).sum())
                s["absolute"] += float((b - a).abs().sum())
                s["square"] += float((b - a).square().sum())
                s["ce"] += float(F.binary_cross_entropy_with_logits(logits, a, reduction="sum"))
                s["entropy"] += float(-(torch.special.xlogy(a, a) +
                                        torch.special.xlogy(1 - a, 1 - a)).sum())
                s["brier_base"] += float((a - climate).square().sum())
                t["n"] += n
                t["target"] += ay
                t["prediction"] += bp
        for index, item in enumerate(tile):
            if item["n"] >= MIN_TILE_ROWS:
                tile_pairs[index].append((item["target"] / item["n"],
                                          item["prediction"] / item["n"]))
    bp = []
    for bucket, pairs in zip(buckets, tile_pairs, strict=True):
        result = summarize(bucket, climate)
        points = np.asarray(pairs)
        result["spatial_tile_count"] = len(pairs)
        result["spatial_tile_mean_pearson"] = (
            float(np.corrcoef(points.T)[0, 1]) if len(pairs) > 1 and
            np.std(points[:, 0]) > 0 and np.std(points[:, 1]) > 0 else None)
        bp.append(result)
    return {"bp_overall": bp[0], "bp_strata": bp[1:], "flp_valid_rows": int(flp_rows),
            "flp_cross_entropy": flp_ce / flp_rows,
            "flp_excess_log_loss": (flp_ce - flp_entropy) / flp_rows}


def checkpoint(epoch: int) -> Path:
    if epoch <= 3:
        return ROOT / f"artifacts/benchmark/07a_gradnorm_ce_reference_fold1_validated/fold_1_epoch_{epoch:03d}.pt"
    return RUN / f"fold_1_epoch_{epoch:03d}.pt"


def plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for role in ("train", "validation"):
        selected = [row for row in rows if row["role"] == role]
        axes[0].plot([r["epoch"] for r in selected],
                     [r["bp_overall"]["excess_log_loss"] for r in selected], "o-", label=role)
        axes[1].plot([r["epoch"] for r in selected],
                     [r["bp_strata"][3]["prediction_mean"] /
                      r["bp_strata"][3]["target_mean"] for r in selected], "o-", label=role)
    axes[0].set(ylabel="BP excess log loss", xlabel="Saved epoch")
    axes[1].set(ylabel="BP ≥ 0.02 prediction / target", xlabel="Saved epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "bp_adequacy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    config = json.loads((RUN / "config.json").read_text())
    if (config["kind"], config["outer_study"], config["folds"]) != ("gradnorm_full_fold", 3, [1]):
        raise ValueError("Unexpected experiment")
    if config["bp_risk_strata"] != [list(item) for item in STRATA]:
        raise ValueError("Risk strata differ from Phase 7A")
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    state = load_states(ROOT, cache, 3)[1]
    if state.digest() != config["continuation"]["preprocessing_sha256"]:
        raise ValueError("Preprocessing state differs")
    counts = support_counts(ROOT, 3)[1]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    source = json.loads((ROOT / "reports/training/07a/gradnorm_ce_reference_fold1.json").read_text())
    for epoch in EPOCHS:
        path = checkpoint(epoch)
        digest = sha256(path)
        if digest != json.loads(path.with_suffix(".pt.sha256.json").read_text())["sha256"]:
            raise ValueError(f"Checkpoint hash differs: {epoch}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["completed_epoch"] != epoch or saved["metadata"]["cache_binding"] != cache_binding(cache, state):
            raise ValueError(f"Checkpoint provenance differs: {epoch}")
        model, _ = make_model(33, 3, 128, config["learning_rate"], config["weight_decay"])
        model.load_state_dict(saved["model_state"])
        for role in ("train", "validation"):
            result = score(cache, state, model, role, config["bp_climatology"])
            if (result["bp_overall"]["rows"] != counts[role]["rows"] or
                    result["flp_valid_rows"] != counts[role]["flp_valid"] or
                    sum(item["rows"] for item in result["bp_strata"]) != counts[role]["rows"]):
                raise ValueError(f"Frozen support differs: {epoch}/{role}")
            if role == "validation":
                previous = (source["epochs"][epoch - 1] if epoch <= 3 else
                            json.loads((RUN / f"fold_1_epoch_{epoch:03d}.json").read_text()))
                if (abs(result["bp_overall"]["excess_log_loss"] -
                        previous["validation"]["bp_excess_log_loss"]) > 1e-5 or
                        abs(result["flp_excess_log_loss"] -
                            previous["validation"]["flp_excess_log_loss"]) > 1e-5):
                    raise ValueError(f"Phase 7A validation score differs: {epoch}")
            record = {"epoch": epoch, "role": role, "checkpoint_sha256": digest, **result}
            records.append(record)
            print(f"epoch={epoch} role={role} BP_KL={result['bp_overall']['excess_log_loss']:.6f}", flush=True)
    (OUTPUT / "scores.json").write_text(json.dumps({"experiment": RUN.name, "epochs": EPOCHS,
        "outer_study": 3, "inner_fold": 1, "preprocessing_sha256": state.digest(),
        "cache_manifest_sha256": config["continuation"]["cache_manifest_sha256"],
        "climatology": config["bp_climatology"], "strata": STRATA,
        "spatial_unit": "frozen cache partition with >=100 role/stratum rows; equal partition weighting",
        "records": records}, indent=2) + "\n")
    with (OUTPUT / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("epoch", "role", "stratum", "rows", "target_mean", "prediction_mean",
                         "bias", "bp_kl", "bp_log_skill", "bp_brier", "bp_climatology_brier",
                         "bp_mae", "bp_rmse", "cell_pearson", "tile_count", "tile_pearson",
                         "flp_valid_rows", "flp_kl"))
        for row in records:
            for label, value in [("overall", row["bp_overall"]), *zip(
                    ("0-.001", ".001-.01", ".01-.02", ">=.02"), row["bp_strata"], strict=True)]:
                writer.writerow((row["epoch"], row["role"], label, value["rows"],
                                 value["target_mean"], value["prediction_mean"], value["bias"],
                                 value["excess_log_loss"], value["log_skill"], value["brier"],
                                 value["climatology_brier"], value["mae"], value["rmse"],
                                 value["prediction_target_pearson"], value["spatial_tile_count"],
                                 value["spatial_tile_mean_pearson"], row["flp_valid_rows"],
                                 row["flp_excess_log_loss"]))
    plot(records)


if __name__ == "__main__":
    main()
