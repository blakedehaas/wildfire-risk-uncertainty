"""One frozen inner-fold, 100-epoch GradNorm continuation for detached execution."""

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from wildfire_risk_uncertainty.hpo.real import support_counts
from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.cache_checkpoint import (
    load_cache_checkpoint,
    save_cache_checkpoint,
)
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.preprocessing import (
    stable_json_bytes,
    transform,
)
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/training/07a/gradnorm_ce_reference_fold1.json"
SOURCE_CHECKPOINT = ROOT / "artifacts/benchmark/07a_gradnorm_ce_reference_fold1_validated/fold_1_epoch_003.pt"
STRATA = ((0.0, 0.001), (0.001, 0.01), (0.01, 0.02), (0.02, 1.000001))
ELEVATED_MINIMUM = 0.02
MIN_SPATIAL_TILE_ROWS = 100


def digest_state(value: object) -> str:
    digest = hashlib.sha256()

    def add(item: object) -> None:
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous().numpy()
            digest.update(stable_json_bytes({"dtype": str(data.dtype), "shape": data.shape}))
            digest.update(data.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
        else:
            digest.update(stable_json_bytes(item))

    add(value)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(stable_json_bytes(value))
    temporary.replace(path)


def source_compatibility(config: dict) -> dict:
    """Fail before launch if the epoch-3 source cannot continue exactly."""
    source = json.loads(SOURCE.read_text())
    final = source["epochs"][-1]
    expected = {"outer_study": 3, "inner_fold": 1}
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("Source fold or precision differs")
    if final["epoch"] != 3 or not final["checkpoint_restored_exactly"]:
        raise ValueError("Source epoch-3 restoration was not verified")
    if config["outer_study"] != 3 or config["folds"] != [1] or config["epochs"] != 100:
        raise ValueError("Full-fold run must retain the representative fold and 100 epochs")
    for key, value in (("depth", 3), ("width", 128), ("batch_size", 65536),
                       ("learning_rate", 0.001), ("weight_decay", 0.000316227766),
                       ("precision", "BF16"), ("seed", 42), ("threads_each", 4)):
        if config[key] != value:
            raise ValueError(f"Source configuration differs: {key}")
    if sha256(SOURCE_CHECKPOINT) != final["checkpoint_sha256"]:
        raise ValueError("Source checkpoint SHA-256 mismatch")
    for filename, expected_hash in source["implementation_files_sha256"].items():
        if sha256(ROOT / filename) != expected_hash:
            raise ValueError(f"Production source changed since epoch 3: {filename}")
    if sha256(ROOT / "scripts/validate_gradnorm_ce_reference.py") != source["source_file_sha256"]:
        raise ValueError("Source diagnostic runner changed since epoch 3")
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    state = load_states(ROOT, cache, 3)[1]
    if state.digest() != source["preprocessing_sha256"]:
        raise ValueError("Frozen preprocessing differs from epoch-3 source")
    seed_everything(42)
    model, optimizer = make_model(33, 3, 128, 0.001, 0.000316227766)
    gradnorm = GradNorm("cuda")
    source_metadata = {"diagnostic": "gradnorm-ce-first-epoch-v2", "outer_study": 3,
                       "inner_fold": 1, "depth": 3, "width": 128, "seed": 42,
                       "batch_size": 65536, "learning_rate": 0.001,
                       "weight_decay": 0.000316227766, "precision": "BF16"}
    completed, step = load_cache_checkpoint(SOURCE_CHECKPOINT, model, optimizer,
                                            cache=cache, state=state,
                                            expected_metadata=source_metadata,
                                            gradnorm=gradnorm)
    if (completed, step) != (3, final["trace_end"]):
        raise ValueError("Source checkpoint epoch/global step mismatch")
    raw = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    for saved_key, actual in (("model_state", model.state_dict()),
                              ("optimizer_state", optimizer.state_dict()),
                              ("gradnorm_state", gradnorm.state_dict()),
                              ("gradnorm_optimizer_state", gradnorm.optimizer.state_dict())):
        if digest_state(raw[saved_key]) != digest_state(actual):
            raise ValueError(f"Source checkpoint restoration differs: {saved_key}")
    if (random.getstate() != raw["python_rng_state"] or
            not torch.equal(torch.get_rng_state(), raw["torch_cpu_rng_state"]) or
            any(not torch.equal(actual, saved) for actual, saved in zip(
                torch.cuda.get_rng_state_all(), raw["torch_cuda_rng_state"], strict=True))):
        raise ValueError("Source checkpoint RNG restoration differs")
    numpy_saved = raw["numpy_rng_state"]
    numpy_actual = np.random.get_state()
    if (numpy_actual[0] != numpy_saved[0] or
            not np.array_equal(numpy_actual[1], numpy_saved[1]) or
            numpy_actual[2:] != numpy_saved[2:]):
        raise ValueError("Source checkpoint NumPy RNG restoration differs")
    if not torch.allclose(gradnorm.weights.detach().cpu(), torch.tensor([
            final["training"]["gradnorm"]["bp_weight_final"],
            final["training"]["gradnorm"]["flp_weight_final"]]), atol=0, rtol=0):
        raise ValueError("Source adaptive weights differ from result")
    if gradnorm.initial_ce.detach().cpu().tolist() != source["epochs"][0]["training"]["gradnorm"][
            "first_epoch_initial_ce"]:
        raise ValueError("Source first-epoch CE references differ")
    return {"source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "source_checkpoint_sha256": final["checkpoint_sha256"],
            "source_result_sha256": sha256(SOURCE),
            "source_completed_epoch": completed, "source_global_step": step,
            "preprocessing_sha256": state.digest(),
            "cache_manifest_sha256": hashlib.sha256(stable_json_bytes(cache.manifest)).hexdigest(),
            "exact_load_verified": True}


def train_bp_mean() -> float:
    with (ROOT / "reports/spatial_validation/06c/tables/target_support_by_inner_fold.csv").open() as handle:
        matches = [row for row in csv.DictReader(handle)
                   if row["outer_study"] == "3" and row["inner_fold"] == "1" and row["role"] == "train"]
    if len(matches) != 1:
        raise ValueError("Frozen inner TRAIN BP mean unavailable")
    mean = float(matches[0]["bp_mean"])
    if not 0 < mean < 1:
        raise ValueError("Invalid training-derived BP climatology")
    return mean


def _summary(sums: dict, climate: float) -> dict:
    n = sums["n"]
    if n == 0:
        return {"rows": 0}
    target, predicted = sums["target"] / n, sums["predicted"] / n
    tvar = sums["target2"] - sums["target"]**2 / n
    pvar = sums["predicted2"] - sums["predicted"]**2 / n
    cov = sums["cross"] - sums["target"] * sums["predicted"] / n
    corr = cov / math.sqrt(tvar * pvar) if tvar > 0 and pvar > 0 else None
    base_ce = -target * math.log(climate) - (1 - target) * math.log1p(-climate)
    model_kl = (sums["model_ce"] - sums["entropy"]) / n
    base_kl = base_ce - sums["entropy"] / n
    return {"rows": int(n), "target_mean": target, "prediction_mean": predicted,
            "prediction_sd": math.sqrt(max(0, pvar / n)),
            "prediction_target_pearson": corr,
            "mae": sums["abs_error"] / n, "rmse": math.sqrt(sums["squared_error"] / n),
            "excess_log_loss": model_kl, "climatology_prediction": climate,
            "climatology_mae": sums["climate_abs_error"] / n,
            "climatology_rmse": math.sqrt(sums["climate_squared_error"] / n),
            "climatology_excess_log_loss": base_kl,
            "excess_log_loss_skill_vs_climatology": base_kl - model_kl}


def risk_masks(raw_target: torch.Tensor) -> list[torch.Tensor]:
    """Apply frozen boundaries in stored float32 target precision."""
    return [torch.ones_like(raw_target, dtype=torch.bool),
            *((raw_target >= low) & (raw_target < high) for low, high in STRATA)]


@torch.no_grad()
def bp_validation_diagnostics(cache: HpoCache, state, model, climate: float) -> dict:
    """Full-support BP strata plus tile-mean geographic discrimination."""
    fields = ("n", "target", "predicted", "target2", "predicted2", "cross", "abs_error",
              "squared_error", "model_ce", "entropy", "climate_abs_error",
              "climate_squared_error")
    buckets = [{field: 0.0 for field in fields} for _ in range(len(STRATA) + 1)]
    tile_pairs: list[list[tuple[float, float]]] = [[] for _ in buckets]
    offsets = cache.arrays["partition_offsets"]
    model.eval()
    for number in range(len(cache.names)):
        start, stop = map(int, offsets[number:number + 2])
        labels = cache.arrays["inner_fold"][start:stop, 2]
        selected = np.flatnonzero(labels == 1) + start
        if not len(selected):
            continue
        tile = [{"n": 0.0, "target": 0.0, "predicted": 0.0} for _ in buckets]
        for position in range(0, len(selected), 65536):
            rows = selected[position:position + 65536]
            features = transform(np.asarray(cache.arrays["features"][rows]), state)
            if not np.isfinite(features).all():
                raise FloatingPointError("Nonfinite validation features")
            x = torch.from_numpy(features).cuda()
            raw_target = torch.from_numpy(np.asarray(cache.arrays["bp"][rows])).cuda()
            target = raw_target.double()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            z = logits.double()
            predicted = z.sigmoid()
            if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(predicted).all()):
                raise FloatingPointError("Nonfinite BP validation prediction")
            for index, mask in enumerate(risk_masks(raw_target)):
                y, p, selected_logits = target[mask], predicted[mask], z[mask]
                if not y.numel():
                    continue
                bucket = buckets[index]
                count = y.numel()
                ysum, psum = float(y.sum()), float(p.sum())
                bucket["n"] += count
                bucket["target"] += ysum
                bucket["predicted"] += psum
                bucket["target2"] += float(y.square().sum())
                bucket["predicted2"] += float(p.square().sum())
                bucket["cross"] += float((y * p).sum())
                bucket["abs_error"] += float((p - y).abs().sum())
                bucket["squared_error"] += float((p - y).square().sum())
                bucket["model_ce"] += float(F.binary_cross_entropy_with_logits(
                    selected_logits, y, reduction="sum"))
                bucket["entropy"] += float((-(torch.special.xlogy(y, y) +
                                              torch.special.xlogy(1-y, 1-y))).sum())
                bucket["climate_abs_error"] += float((y - climate).abs().sum())
                bucket["climate_squared_error"] += float((y - climate).square().sum())
                tile[index]["n"] += count
                tile[index]["target"] += ysum
                tile[index]["predicted"] += psum
        for index, item in enumerate(tile):
            if item["n"] >= MIN_SPATIAL_TILE_ROWS:
                tile_pairs[index].append((item["target"] / item["n"],
                                          item["predicted"] / item["n"]))
    summaries = []
    for index, bucket in enumerate(buckets):
        summary = _summary(bucket, climate)
        pairs = tile_pairs[index]
        if len(pairs) > 1:
            values = np.asarray(pairs)
            correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]) if (
                np.std(values[:, 0]) > 0 and np.std(values[:, 1]) > 0) else None
        else:
            correlation = None
        summary.update(spatial_tile_count=len(pairs), spatial_tile_mean_pearson=correlation,
                       spatial_tile_minimum_rows=MIN_SPATIAL_TILE_ROWS)
        summaries.append(summary)
    return {"overall": summaries[0],
            "strata": [{"target_interval": [low, high], **summaries[index + 1]}
                       for index, (low, high) in enumerate(STRATA)],
            "spatial_unit": "frozen canonical cache partition, equal tile weighting",
            "climatology_source": "outer_3_inner_1_TRAIN_bp_mean_full_support"}


def run(path: Path, config: dict, progress) -> None:
    if config["kind"] != "gradnorm_full_fold" or config["folds"] != [1]:
        raise ValueError("Only the one-fold GradNorm run is supported")
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[:4])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cache = HpoCache(ROOT, ROOT / "artifacts/training/hpo_cache_v1", check_files=False)
    state = load_states(ROOT, cache, 3)[1]
    counts = support_counts(ROOT, 3)[1]
    climate = train_bp_mean()
    if climate != config["bp_climatology"] or config["bp_risk_strata"] != [list(x) for x in STRATA]:
        raise ValueError("Climatology or risk strata differ from frozen launch configuration")
    seed_everything(42)
    model, optimizer = make_model(33, 3, 128, 0.001, 0.000316227766)
    gradnorm = GradNorm("cuda")
    metadata = {"benchmark": "phase_07a_gradnorm_ce_full_fold", "outer_study": 3,
                "inner_fold": 1, "source_git_sha": config["source_git_sha"],
                "source_checkpoint_sha256": config["continuation"]["source_checkpoint_sha256"],
                "configuration": {key: config[key] for key in (
                    "depth", "width", "batch_size", "learning_rate", "weight_decay", "seed")},
                "gradnorm_rate_version": "ce-first-epoch-v2"}
    existing = sorted(path.glob("fold_1_epoch_*.json"))
    if existing:
        latest = json.loads(existing[-1].read_text())
        epoch, step = load_cache_checkpoint(path / latest["checkpoint"], model, optimizer,
                                            cache=cache, state=state, expected_metadata=metadata,
                                            gradnorm=gradnorm)
        if (epoch, step) != (latest["epoch"], latest["global_step"]):
            raise ValueError("Run checkpoint and epoch record disagree")
        completed = epoch
        global_step = step
    else:
        source = config["continuation"]
        if source["source_checkpoint_sha256"] != sha256(SOURCE_CHECKPOINT):
            raise ValueError("Original epoch-3 checkpoint changed")
        raw = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
        source_metadata = {key: value for key, value in raw["metadata"].items()
                           if key != "cache_binding"}
        completed, global_step = load_cache_checkpoint(
            SOURCE_CHECKPOINT, model, optimizer, cache=cache, state=state,
            expected_metadata=source_metadata, gradnorm=gradnorm)
        if (completed, global_step) != (3, source["source_global_step"]):
            raise ValueError("Initial continuation step mismatch")
        source_results = json.loads(SOURCE.read_text())
        if sha256(SOURCE) != source["source_result_sha256"]:
            raise ValueError("Original three-epoch results changed")
        write(path / "source_epochs_001_to_003.json", {
            "source_result": str(SOURCE.relative_to(ROOT)),
            "source_result_sha256": source["source_result_sha256"],
            "epochs": source_results["epochs"],
        })
    write(path / "fold_1.ready.json", {"fold": 1, "ready_at": time.time(),
                                       "pid": os.getpid(), "completed_epoch": completed,
                                       "global_step": global_step,
                                       "source_checkpoint_sha256": config["continuation"][
                                           "source_checkpoint_sha256"]})
    trace: list[dict] = []
    original_step = gradnorm.step

    def traced_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        trace.append(result)
        return result

    gradnorm.step = traced_step
    for epoch in range(completed + 1, 101):
        if (path / "stop.request.json").exists():
            break
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        trace.clear()
        progress(path, 1, epoch, "train")
        training = train_epoch(ROOT, state, model, optimizer, epoch,
                               outer_study=3, inner_fold=1, batch_size=65536,
                               cache=cache, gradnorm=gradnorm)
        global_step += training["optimizer_steps"]
        train_seconds = time.perf_counter() - epoch_started
        progress(path, 1, epoch, "validation")
        validation = validate(ROOT, state, model, counts["validation"]["rows"],
                              counts["validation"]["flp_valid"], outer_study=3,
                              inner_fold=1, batch_size=65536, cache=cache,
                              report_excess=True)
        validation_seconds = time.perf_counter() - epoch_started - train_seconds
        progress(path, 1, epoch, "bp_diagnostics")
        bp = bp_validation_diagnostics(cache, state, model, climate)
        bp_diagnostic_seconds = time.perf_counter() - epoch_started - train_seconds - validation_seconds
        if bp["overall"]["rows"] != validation["bp_rows"] or abs(
                bp["overall"]["excess_log_loss"] - validation["bp_excess_log_loss"]) > 1e-4:
            raise ValueError("BP diagnostic and production validation supports/loss disagree")
        if not all(math.isfinite(value) for value in (
                training["bp_loss"], training["flp_loss"],
                validation["bp_excess_log_loss"], validation["flp_excess_log_loss"])):
            raise FloatingPointError("Nonfinite train or validation metric")
        for item in [bp["overall"], *bp["strata"]]:
            if not all(math.isfinite(value) for value in item.values()
                       if isinstance(value, (int, float))):
                raise FloatingPointError("Nonfinite BP risk-stratum metric")
        if len(trace) != training["optimizer_steps"]:
            raise ValueError("GradNorm trajectory step count differs")
        weights = [item["bp_weight_after"] for item in trace]
        shares = sorted(item["bp_shared_gradient_norm"] /
                        (item["bp_shared_gradient_norm"] + item["flp_shared_gradient_norm"])
                        for item in trace if item["gradnorm_weight_updated"] and
                        item["bp_shared_gradient_norm"] + item["flp_shared_gradient_norm"] > 0)
        trajectory = {"bp_weight_min": min(weights), "bp_weight_max": max(weights),
                      "bp_weight_floor_contacts": sum(weight <= 1.1e-8 for weight in weights),
                      "bp_weighted_shared_gradient_share_median": (
                          shares[len(shares) // 2] if shares else None),
                      "samples": [{"step_in_epoch": index + 1, **item}
                                  for index, item in enumerate(trace)
                                  if index == 0 or (index + 1) % 25 == 0 or index == len(trace) - 1]}
        progress(path, 1, epoch, "checkpoint")
        checkpoint_started = time.perf_counter()
        checkpoint = path / f"fold_1_epoch_{epoch:03d}.pt"
        before = [digest_state(item) for item in (model.state_dict(), optimizer.state_dict(),
                                                 gradnorm.state_dict(), gradnorm.optimizer.state_dict())]
        save_cache_checkpoint(checkpoint, model, optimizer, completed_epoch=epoch,
                              global_step=global_step, cache=cache, state=state,
                              metadata=metadata, gradnorm=gradnorm)
        restored_epoch, restored_step = load_cache_checkpoint(
            checkpoint, model, optimizer, cache=cache, state=state,
            expected_metadata=metadata, gradnorm=gradnorm)
        after = [digest_state(item) for item in (model.state_dict(), optimizer.state_dict(),
                                                gradnorm.state_dict(), gradnorm.optimizer.state_dict())]
        if (restored_epoch, restored_step) != (epoch, global_step) or before != after:
            raise ValueError("Checkpoint restoration failed")
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        record = {"epoch": epoch, "finished_at": time.time(), "training": training,
                  "validation": validation, "bp_diagnostics": bp, "gradnorm_trajectory": trajectory,
                  "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                  "checkpoint_restored": True, "global_step": global_step,
                  "timing_seconds": {"training": train_seconds,
                                     "production_validation": validation_seconds,
                                     "bp_diagnostics": bp_diagnostic_seconds,
                                     "checkpoint_save_load_verify": checkpoint_seconds,
                                     "total": time.perf_counter() - epoch_started},
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "current_rss_bytes": int(Path("/proc/self/status").read_text().split(
                      "VmRSS:", 1)[1].split()[0]) * 1024}
        write(path / f"fold_1_epoch_{epoch:03d}.json", record)
        progress(path, 1, epoch, "complete")
    records = sorted(path.glob("fold_1_epoch_*.json"))
    completed = int(records[-1].stem.rsplit("_", 1)[1]) if records else 3
    write(path / "fold_1.json", {"fold": 1, "completed_epoch": completed,
                                  "stopped_early": completed < 100,
                                  "epoch_records": [item.name for item in records],
                                  "continuation": config["continuation"],
                                  "finished_at": time.time()})
