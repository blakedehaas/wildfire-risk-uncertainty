"""W&B is an optional online mirror; local JSON records remain authoritative."""

import time
from pathlib import Path

import wandb

from wildfire_risk_uncertainty.hpo.records import write_json


def require_online_authentication() -> None:
    """Verify online access before a real HPO invocation, without exposing credentials."""
    try:
        viewer = wandb.Api(timeout=15).viewer
        if not viewer:
            raise RuntimeError("W&B returned no authenticated viewer")
    except Exception:  # noqa: BLE001 - sanitize any SDK/network error before HITL stop
        raise RuntimeError(
            "W&B online authentication is unavailable; stop for HITL before real HPO"
        ) from None


class DisabledTracker:
    run_id = None
    run_url = None
    sync_state = "disabled"

    def fold(self, record: dict) -> None:
        pass

    def spatial(self, record: dict) -> None:
        pass

    def finish(self, summary: dict) -> None:
        pass


class MockTracker(DisabledTracker):
    def __init__(self) -> None:
        self.run_id = "mock-run"
        self.run_url = None
        self.sync_state = "mock"
        self.events: list[dict] = []
        self.summary: dict = {}

    def fold(self, record: dict) -> None:
        self.events.append(fold_payload(record))

    def spatial(self, record: dict) -> None:
        self.events.append(spatial_payload(record))

    def finish(self, summary: dict) -> None:
        self.summary = summary


def fold_payload(record: dict) -> dict:
    prefix = f"fold_{record['inner_fold']}"
    payload = {f"{prefix}/epoch": record["epoch"],
               f"{prefix}/global_optimizer_step": record["global_optimizer_step"],
               f"{prefix}/training_wall_seconds": record["training_wall_seconds"],
               f"{prefix}/validation_wall_seconds": record["validation_wall_seconds"],
               **({f"{prefix}/checkpoint_wall_seconds": record["checkpoint_wall_seconds"]}
                  if "checkpoint_wall_seconds" in record else {}),
               f"{prefix}/training_rows_per_second": record["training_rows_per_second"],
               f"{prefix}/validation_rows_per_second": record["validation_rows_per_second"]}
    for role in ("training", "validation"):
        metrics = record[role]
        for key in ("bp_loss_sum", "bp_count", "flp_loss_sum", "flp_valid_count",
                    "bp_loss_mean", "flp_loss_mean", "equal_task_objective"):
            payload[f"{prefix}/{role}/{key}"] = metrics[key]
    if "gradnorm" in record:
        for key, value in record["gradnorm"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[f"{prefix}/gradnorm/{key}"] = value
        payload[f"{prefix}/validation/bp_excess_log_loss"] = record["validation_bp_excess_log_loss"]
        payload[f"{prefix}/validation/flp_excess_log_loss"] = record["validation_flp_excess_log_loss"]
    if "validation_log_skill" in record:
        for key, value in record["validation_log_skill"].items():
            payload[f"{prefix}/validation_log_skill/{key}"] = value
    return payload


def spatial_payload(record: dict) -> dict:
    return {"spatial_cv/resource_epoch": record["resource_epoch"],
            **({"spatial_cv/validation_bp_log_skill_mean": record["validation_bp_log_skill_mean"],
                "spatial_cv/validation_flp_log_skill_mean": record["validation_flp_log_skill_mean"]}
               if "validation_bp_log_skill_mean" in record else {}),
            "spatial_cv/validation_objective_mean": record["J_e"],
            "spatial_cv/validation_objective_std": record["fold_objective_std_population"],
            "spatial_cv/validation_bp_mean": record["validation_bp_mean"],
            "spatial_cv/validation_flp_mean": record["validation_flp_mean"],
            "spatial_cv/validation_objective_min": record["fold_objective_min"],
            "spatial_cv/validation_objective_max": record["fold_objective_max"],
            "spatial_cv/training_objective_mean": record["training_objective_mean"],
            "spatial_cv/pruned": int(record.get("pruned", False))}


class OnlineTracker:
    def __init__(self, *, outer_study: int, trial_number: int, configuration: dict,
                 provenance: dict, local_dir: Path, checkpoint: str,
                 abort_on_failure: bool = False, run_id: str | None = None) -> None:
        self.local_dir = local_dir
        self.abort_on_failure = abort_on_failure
        self.errors: list[str] = []
        self.sync_state = "online"
        self.start_time = time.perf_counter()
        resume_kwargs = {"id": run_id, "resume": "must"} if run_id is not None else {}
        pruning_tag = ("nop-pruner" if provenance.get("pruner", {}).get("name") == "NopPruner"
                       else "asha")
        self.run = wandb.init(
            project="wildfire-risk-uncertainty", group=f"outer-{outer_study}",
            job_type="hpo-trial", name=f"outer-{outer_study}-trial-{trial_number:03d}",
            tags=[checkpoint, "hpo", "tpe", pruning_tag, f"outer-{outer_study}"],
            config={**configuration, **provenance, "outer_study": outer_study,
                    "trial_number": trial_number}, mode="online", dir=str(local_dir),
            settings=wandb.Settings(init_timeout=45),
            **resume_kwargs,
        )
        if self.run is None:
            raise RuntimeError("W&B online initialization returned no run")
        self.run_id = self.run.id
        self.run_url = self.run.url
        for fold in range(1, 6):
            self.run.define_metric(f"fold_{fold}/epoch")
            self.run.define_metric(f"fold_{fold}/*", step_metric=f"fold_{fold}/epoch")
        self.run.define_metric("spatial_cv/resource_epoch")
        self.run.define_metric("spatial_cv/*", step_metric="spatial_cv/resource_epoch")

    def _log(self, payload: dict, name: str) -> None:
        try:
            self.run.log(payload)
            write_json(self.local_dir / "wandb_mirror" / name, payload)
        except Exception as exc:
            self.sync_state = "degraded_pending"
            self.errors.append(f"{type(exc).__name__}: {exc}")
            write_json(self.local_dir / "wandb_sync_state.json", {
                "state": self.sync_state, "errors": self.errors})
            if self.abort_on_failure:
                raise

    def fold(self, record: dict) -> None:
        self._log(fold_payload(record), f"fold_{record['inner_fold']}_epoch_{record['epoch']:03d}.json")

    def spatial(self, record: dict) -> None:
        self._log(spatial_payload(record), f"aggregate_epoch_{record['resource_epoch']:03d}.json")

    def finish(self, summary: dict) -> None:
        try:
            self.run.summary.update(summary)
            self.run.finish()
        except Exception as exc:
            self.sync_state = "degraded_pending"
            self.errors.append(f"{type(exc).__name__}: {exc}")
            write_json(self.local_dir / "wandb_sync_state.json", {
                "state": self.sync_state, "errors": self.errors})
            if self.abort_on_failure:
                raise
