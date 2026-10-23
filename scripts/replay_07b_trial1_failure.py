"""Read-only, one-epoch replay of trial 1 from a verified epoch-75 checkpoint."""

import argparse
import hashlib
from pathlib import Path

from wildfire_risk_uncertainty.hpo.config import (
    GRADNORM_TRAINING_VERSION,
    Configuration,
)
from wildfire_risk_uncertainty.hpo.records import read_json, write_json
from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.cache_checkpoint import load_cache_checkpoint
from wildfire_risk_uncertainty.training.cache_states import load_states
from wildfire_risk_uncertainty.training.gradnorm import GradNorm
from wildfire_risk_uncertainty.training.trainer import (
    make_model,
    seed_everything,
    train_epoch,
)


def replay(root: Path, fold: int) -> dict:
    if fold not in range(1, 6):
        raise ValueError("Inner fold must be 1..5")
    local = root / "artifacts/hpo/records/wildfire_outer_3_gradnorm_hpo_v3/trial_001"
    manifest = read_json(local / "manifest.json")
    if manifest["trial_number"] != 1 or manifest["outer_study"] != 3:
        raise ValueError("Unexpected trial identity")
    last = read_json(local / f"fold_{fold}_epoch_075.json")
    if (local / f"fold_{fold}_epoch_076.json").exists():
        raise ValueError("Epoch 76 is already persisted; refusing replay")
    configuration = Configuration(**manifest["configuration"])
    checkpoint = local / "checkpoints" / f"fold_{fold}_epoch_075.pt"
    cache = HpoCache(root, root / "artifacts/training/hpo_cache_v1")
    state = load_states(root, cache, 3)[fold]
    seed_everything(42)
    model, optimizer = make_model(inputs=len(state.inputs), depth=configuration.depth,
                                  width=configuration.width,
                                  learning_rate=configuration.learning_rate,
                                  weight_decay=configuration.weight_decay)
    gradnorm = GradNorm("cuda")
    epoch, step = load_cache_checkpoint(
        checkpoint, model, optimizer, cache=cache, state=state,
        expected_metadata={"outer_study": 3, "trial_number": 1, "inner_fold": fold,
                           "configuration": configuration.serializable(), "seed": 42,
                           "loss_balancing": "gradnorm", "gradnorm_alpha": 1.5,
                           "gradnorm_training_version": GRADNORM_TRAINING_VERSION},
        gradnorm=gradnorm)
    if (epoch, step) != (75, last["global_optimizer_step"]):
        raise ValueError("Checkpoint epoch/step differs from persisted fold record")
    report = {"trial_number": 1, "outer_study": 3, "inner_fold": fold,
              "replayed_epoch": 76, "checkpoint_epoch": epoch,
              "checkpoint_global_step": step,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "configuration": configuration.serializable(),
              "original_source_git_sha": manifest["provenance"]["source_git_sha"],
              "checkpoint_verified": True}
    try:
        training = train_epoch(root, state, model, optimizer, 76, outer_study=3,
                               inner_fold=fold, batch_size=configuration.batch_size,
                               cache=cache, gradnorm=gradnorm, global_step_base=step)
    except FloatingPointError as exc:
        report.update(outcome="numerical_exception", exception_type=type(exc).__name__,
                      exception=str(exc))
    else:
        report.update(outcome="epoch_completed", completed_optimizer_steps=training["optimizer_steps"],
                      bp_ce=training["bp_loss"], flp_ce=training["flp_loss"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = replay(root, args.fold)
    output = args.output or root / "reports/hpo/07b" / f"trial1_replay_fold{args.fold}.json"
    write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
