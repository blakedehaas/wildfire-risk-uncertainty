"""Explicit Checkpoint 6D engineering validation; outer TEST is inaccessible."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow
import torch

from wildfire_risk_uncertainty.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from wildfire_risk_uncertainty.training.data import inner_chunks, model_inputs
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    fit_preprocessing,
    stable_json_bytes,
)
from wildfire_risk_uncertainty.training.trainer import (
    BATCH_SIZE,
    DATA_ORDER_VERSION,
    SEED,
    make_model,
    seed_everything,
    train_epoch,
    validate,
)

BASE = "4dcd92acefa7c5bb27fd33c01190d5836a4b886e"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value))


def paths(root: Path) -> tuple[Path, Path, Path]:
    return (root / "artifacts/dataset/canonical/partitions",
            root / "artifacts/spatial_validation/05b/partitions",
            root / "artifacts/spatial_validation/06c/partitions")


def frozen_counts(root: Path) -> dict:
    with (root / "reports/spatial_validation/06c/tables/target_support_by_inner_fold.csv").open() as f:
        entries = [r for r in csv.DictReader(f) if r["outer_study"] == "3" and r["inner_fold"] == "1"]
    by_role = {r["role"]: r for r in entries}
    with (root / "reports/spatial_validation/06c/tables/outer_study_summary.csv").open() as f:
        outer = next(r for r in csv.DictReader(f) if r["outer_study"] == "3")
    actual = {"outer_train_rows": int(outer["outer_train_rows"]),
              "inner_train_rows": int(by_role["train"]["rows"]),
              "inner_train_flp_valid": int(by_role["train"]["flp_target_valid"]),
              "inner_train_bp_positive": int(by_role["train"]["bp_positive"]),
              "inner_train_bp_zero": int(by_role["train"]["bp_zero"]),
              "inner_validation_rows": int(by_role["validation"]["rows"]),
              "inner_validation_flp_valid": int(by_role["validation"]["flp_target_valid"]),
              "inner_validation_bp_positive": int(by_role["validation"]["bp_positive"]),
              "inner_validation_bp_zero": int(by_role["validation"]["bp_zero"])}
    manifest = json.loads((root / "reports/spatial_validation/06c/manifest.json").read_text())
    if actual["outer_train_rows"] != manifest["outer_train_rows"]["3"]:
        raise ValueError("Frozen 6C outer TRAIN count has changed")
    if actual["inner_train_rows"] + actual["inner_validation_rows"] != actual["outer_train_rows"]:
        raise ValueError("Inner populations do not partition the outer TRAIN population")
    return actual


def state_from_json(path: Path) -> PreprocessingState:
    d = json.loads(path.read_text())
    return PreprocessingState(tuple(d["inputs"]), tuple(d["means_after_fixed_transform"]),
                              tuple(d["standard_deviations_after_imputation"]),
                              tuple(d["missing_counts"][name] for name in d["inputs"]),
                              d["rows"], d["flp_valid_rows"], d["schema_sha256"])


def metadata(root: Path, state: PreprocessingState) -> dict:
    return {"source_git_sha": BASE, "seed": SEED, "outer_study": 3, "inner_fold": 1,
            "batch_size": BATCH_SIZE, "precision": "BF16 autocast", "model": {
                "input_count": 33, "hidden_depth": 2, "shared_width": 128,
                "activation": "ReLU", "dropout": 0, "bp_logits": 1, "flp_logits": 6,
                "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4},
            "feature_order_sha256": hashlib.sha256(stable_json_bytes(list(state.inputs))).hexdigest(),
            "preprocessing_sha256": state.digest(), "data_order_version": DATA_ORDER_VERSION,
            "canonical_manifest_sha256": json.loads((root / "reports/spatial_validation/06c/manifest.json").read_text())[
                "canonical_manifest_sha256"],
            "outer_5b_manifest_sha256": json.loads((root / "reports/spatial_validation/06c/manifest.json").read_text())[
                "outer_5b_manifest_sha256"]}


def run_phase(root: Path, phase: str) -> None:
    artifact = root / "artifacts/training/06d"
    counts = frozen_counts(root)
    valid_rows = counts["inner_validation_rows"]
    valid_flp = counts["inner_validation_flp_valid"]
    state = state_from_json(artifact / "preprocessing_state.json")
    if state.rows != counts["inner_train_rows"] or state.flp_valid_rows != counts["inner_train_flp_valid"]:
        raise ValueError("Preprocessing state does not match frozen INNER TRAIN")
    meta = metadata(root, state)
    seed_everything()
    model, optimizer = make_model()
    if phase == "reference":
        records = [{"run": phase, "stage": "initialization", "kind": "validation",
                    **validate(root, state, model, valid_rows, valid_flp)}]
        for epoch in (1, 2):
            records.append({"run": phase, "stage": f"epoch_{epoch}", "kind": "training",
                            **train_epoch(root, state, model, optimizer, epoch)})
            records.append({"run": phase, "stage": f"epoch_{epoch}", "kind": "validation",
                            **validate(root, state, model, valid_rows, valid_flp)})
        save_checkpoint(artifact / "reference_final.pt", model, optimizer,
                        completed_epoch=2, global_step=sum(r["optimizer_steps"] for r in records if r["kind"] == "training"),
                        metadata=meta)
    elif phase == "resume_first":
        records = [{"run": phase, "stage": "epoch_1", "kind": "training",
                    **train_epoch(root, state, model, optimizer, 1)}]
        records.append({"run": phase, "stage": "epoch_1", "kind": "validation",
                        **validate(root, state, model, valid_rows, valid_flp)})
        save_checkpoint(artifact / "resume_boundary.pt", model, optimizer,
                        completed_epoch=1, global_step=records[0]["optimizer_steps"], metadata=meta)
    elif phase == "resume_second":
        completed, step = load_checkpoint(artifact / "resume_boundary.pt", model, optimizer,
                                          expected_metadata=meta)
        if completed != 1:
            raise ValueError("Resume must start immediately after epoch 1")
        records = [{"run": phase, "stage": "epoch_2", "kind": "training",
                    **train_epoch(root, state, model, optimizer, 2)}]
        records.append({"run": phase, "stage": "epoch_2", "kind": "validation",
                        **validate(root, state, model, valid_rows, valid_flp)})
        save_checkpoint(artifact / "resumed_final.pt", model, optimizer,
                        completed_epoch=2, global_step=step + records[0]["optimizer_steps"], metadata=meta)
    else:
        raise ValueError(phase)
    write_json(artifact / f"{phase}_metrics.json", records)


def _tensor_differences(left: object, right: object) -> tuple[bool, float, float]:
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right)), 0.0, 0.0
    if isinstance(left, torch.Tensor):
        a, b = left.cpu(), right.cpu()
        delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
        return bool(torch.equal(a, b)), float(delta.max()) if delta.numel() else 0.0, float(
            (delta / b.to(torch.float64).abs().clamp_min(1e-12)).max()) if delta.numel() else 0.0
    if isinstance(left, dict):
        parts = [_tensor_differences(left[k], right[k]) for k in left]
    elif isinstance(left, (tuple, list)):
        parts = [_tensor_differences(a, b) for a, b in zip(left, right, strict=True)]
    else:
        return left == right, 0.0, 0.0
    return all(p[0] for p in parts), max((p[1] for p in parts), default=0.0), max(
        (p[2] for p in parts), default=0.0)


def compare(root: Path) -> dict:
    artifact = root / "artifacts/training/06d"
    counts = frozen_counts(root)
    a = torch.load(artifact / "reference_final.pt", map_location="cpu", weights_only=False)
    b = torch.load(artifact / "resumed_final.pt", map_location="cpu", weights_only=False)
    model_equal, model_abs, model_rel = _tensor_differences(a["model_state"], b["model_state"])
    optimizer_equal, optimizer_abs, optimizer_rel = _tensor_differences(a["optimizer_state"], b["optimizer_state"])
    ref = json.loads((artifact / "reference_metrics.json").read_text())
    resumed = json.loads((artifact / "resume_second_metrics.json").read_text())
    ref_train = next(x for x in ref if x["stage"] == "epoch_2" and x["kind"] == "training")
    ref_valid = next(x for x in ref if x["stage"] == "epoch_2" and x["kind"] == "validation")
    train = next(x for x in resumed if x["kind"] == "training")
    valid = next(x for x in resumed if x["kind"] == "validation")
    metric_differences = {f"training_{k}": abs(ref_train[k] - train[k]) for k in (
        "bp_loss", "flp_loss", "combined_loss")}
    metric_differences.update({f"validation_{k}": abs(ref_valid[k] - valid[k]) for k in (
        "bp_loss", "flp_loss", "combined_loss")})
    result = {"model_parameters_bitwise_equal": model_equal, "optimizer_state_bitwise_equal": optimizer_equal,
              "max_model_parameter_absolute_difference": model_abs,
              "max_model_parameter_relative_difference": model_rel,
              "max_optimizer_absolute_difference": optimizer_abs,
              "max_optimizer_relative_difference": optimizer_rel,
              "global_step_equal": a["global_step"] == b["global_step"],
              "global_step": a["global_step"],
              "row_counts_equal": ref_train["bp_rows"] == train["bp_rows"] == counts["inner_train_rows"] and
              ref_valid["bp_rows"] == valid["bp_rows"] == counts["inner_validation_rows"],
              "epoch_2_metric_absolute_differences": metric_differences,
              "checkpoint_metadata_equal": a["metadata"] == b["metadata"],
              "rng_states_equal": all(_tensor_differences(a[key], b[key])[0] for key in (
                  "python_rng_state", "numpy_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_state"))}
    if not all((model_equal, optimizer_equal, result["global_step_equal"],
                result["row_counts_equal"], result["checkpoint_metadata_equal"])):
        raise ValueError("Checkpoint/resume changed the training trajectory")
    if any(delta > 1e-8 for delta in metric_differences.values()):
        raise ValueError("Checkpoint/resume changed epoch-2 diagnostics")
    return result


def orchestrate(root: Path) -> None:
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != BASE:
        raise ValueError("Checkpoint 6D requires the frozen 6C base commit")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Pinned CUDA/BF16 RTX 5080 environment is unavailable")
    inputs = model_inputs(root / "reports/dataset/canonical/schema.json")
    counts = frozen_counts(root)
    canonical, outer, inner = paths(root)
    if any(len(list(p.glob("*.parquet"))) != 3060 for p in (canonical, outer, inner)):
        raise ValueError("A frozen source partition set is incomplete")
    artifact = root / "artifacts/training/06d"
    report = root / "reports/training/06d"
    artifact.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    schema_hash = hashlib.sha256((root / "reports/dataset/canonical/schema.json").read_bytes()).hexdigest()
    fits = []
    for _ in range(2):
        start = time.perf_counter()
        state = fit_preprocessing(inner_chunks(canonical, outer, inner, inputs, role="train"), inputs, schema_hash)
        fits.append((state, time.perf_counter() - start))
    if (state.rows != counts["inner_train_rows"] or state.flp_valid_rows != counts["inner_train_flp_valid"]
            or fits[0][0].digest() != fits[1][0].digest()):
        raise ValueError("INNER TRAIN preprocessing count or deterministic refit failed")
    write_json(artifact / "preprocessing_state.json", state.serializable())
    write_json(report / "preprocessing_summary.json", {
        **state.serializable(), "preprocessing_sha256": state.digest(),
        "refit_identical": True, "scan_seconds": [elapsed for _, elapsed in fits]})
    seed_everything()
    model, _ = make_model()
    parameter_count = sum(p.numel() for p in model.parameters())
    write_json(report / "environment.json", {"python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "bf16_supported": torch.cuda.is_bf16_supported(),
        "numpy": np.__version__, "pyarrow": pyarrow.__version__})
    write_json(report / "run_manifest.json", {**metadata(root, state), **counts,
        "ordered_model_inputs": list(inputs), "model_parameter_count": parameter_count,
        "validation_only": True, "maximum_epochs_per_run": 2,
        "checkpoint_directory": str(artifact.relative_to(root))})
    del model
    for phase in ("reference", "resume_first", "resume_second"):
        subprocess.run([sys.executable, "-m", "wildfire_risk_uncertainty.training.cli",
                        "--phase", phase], cwd=root, check=True)
    comparison = compare(root)
    write_json(report / "checkpoint_resume_validation.json", comparison)
    records = []
    for phase in ("reference", "resume_first", "resume_second"):
        records.extend(json.loads((artifact / f"{phase}_metrics.json").read_text()))
    with (report / "epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    write_json(report / "data_traversal_validation.json", {
        "train_rows_per_epoch": counts["inner_train_rows"],
        "train_flp_valid_per_epoch": counts["inner_train_flp_valid"],
        "validation_rows_per_pass": counts["inner_validation_rows"],
        "validation_flp_valid_per_pass": counts["inner_validation_flp_valid"],
        "deterministic_data_order": DATA_ORDER_VERSION,
        "all_recorded_passes_have_exact_expected_counts": all(
            r["bp_rows"] == (counts["inner_train_rows"] if r["kind"] == "training" else counts["inner_validation_rows"]) and
            r["flp_valid_rows"] == (counts["inner_train_flp_valid"] if r["kind"] == "training" else counts["inner_validation_flp_valid"])
            for r in records)})


def main() -> None:
    parser = argparse.ArgumentParser(description="6D fixed training integration validation")
    parser.add_argument("--phase", choices=("reference", "resume_first", "resume_second"))
    args = parser.parse_args()
    if args.phase:
        run_phase(Path.cwd(), args.phase)
    else:
        orchestrate(Path.cwd())


if __name__ == "__main__":
    main()
