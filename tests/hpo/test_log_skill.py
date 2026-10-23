"""Inner-fold constant references and task-balanced HPO ranking."""

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pytest

from wildfire_risk_uncertainty.hpo.config import FIXED_SMOKE
from wildfire_risk_uncertainty.hpo.records import aggregate, fold_epoch_record
from wildfire_risk_uncertainty.hpo.resume import run_resumable_trial
from wildfire_risk_uncertainty.hpo.skill import fold_references, log_skill
from wildfire_risk_uncertainty.hpo.storage import open_study
from wildfire_risk_uncertainty.hpo.synthetic import raw_metrics
from wildfire_risk_uncertainty.hpo.wandb_tracking import DisabledTracker


def test_constant_references_use_only_corresponding_inner_train(tmp_path: Path) -> None:
    labels = np.repeat(np.arange(1, 6, dtype=np.uint8), 2)
    bp = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.3, 0.2], dtype=np.float32)
    valid = np.tile([1, 0], 5).astype(np.uint8)
    flp = np.full((10, 6), np.nan, dtype=np.float32)
    for fold in range(5):
        flp[2 * fold] = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.05]) if fold == 0 else np.array(
            [0.1, 0.5, 0.1, 0.1, 0.1, 0.1])
    cache = SimpleNamespace(names=["tile"], arrays={
        "partition_offsets": np.array([0, 10]), "inner_fold": np.tile(labels[:, None], (1, 5)),
        "bp": bp, "flp": flp, "flp_valid": valid})
    directory = tmp_path / "reports/spatial_validation/06c/tables"
    directory.mkdir(parents=True)
    with (directory / "target_support_by_inner_fold.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("outer_study", "inner_fold", "role", "rows", "flp_target_valid",
                         "bp_zero", "bp_positive"))
        for fold in range(1, 6):
            writer.writerow((3, fold, "train", 8, 4, 0, 8))
            writer.writerow((3, fold, "validation", 2, 1, 0, 2))
    references = fold_references(cache, tmp_path, 3)
    first = references[1]
    assert first["bp_train_probability"] == pytest.approx(float(bp[2:].mean()))
    assert first["flp_train_probabilities"] == pytest.approx([0.1, 0.5, 0.1, 0.1, 0.1, 0.1])
    assert first["bp_validation_reference_ce"] == pytest.approx(
        -0.85 * math.log(float(bp[2:].mean())) - 0.15 * math.log1p(-float(bp[2:].mean())))
    assert first["flp_validation_reference_ce"] == pytest.approx(
        -sum(y * math.log(q) for y, q in zip(flp[0], first["flp_train_probabilities"], strict=True)))
    assert first["flp_train_valid_rows"] == 4
    assert first["flp_validation_valid_rows"] == 1


def test_equal_relative_skill_ranking_and_maximization(tmp_path: Path) -> None:
    references = {fold: {"bp_validation_rows": 1000, "flp_validation_valid_rows": 700,
                         "bp_validation_reference_ce": 0.05 * fold,
                         "flp_validation_reference_ce": 1.5 * fold} for fold in range(1, 6)}
    records = []
    for fold in range(1, 6):
        ref = references[fold]
        record = fold_epoch_record(
            outer_study=3, trial_number=0, inner_fold=fold, epoch=100,
            configuration={}, training=raw_metrics(0.02, 1.2),
            validation=raw_metrics(0.8 * ref["bp_validation_reference_ce"],
                                   0.6 * ref["flp_validation_reference_ce"]),
            global_step=1, milestone=False, reference=ref)
        assert record["validation_log_skill"]["bp"] == pytest.approx(0.2)
        assert record["validation_log_skill"]["flp"] == pytest.approx(0.4)
        records.append(record)
    spatial = aggregate(100, records)
    assert spatial["J_e"] == pytest.approx(0.3)
    assert spatial["objective_direction"] == "maximize"
    assert spatial["validation_bp_log_skill_mean"] == pytest.approx(0.2)
    assert spatial["validation_flp_log_skill_mean"] == pytest.approx(0.4)
    assert spatial["validation_flp_mean"] != pytest.approx(spatial["J_e"])
    study = open_study(tmp_path / "new.sqlite3", "skill", {"kind": "gradnorm_hpo"})
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    with pytest.raises(ValueError, match="supports differ"):
        fold_epoch_record(outer_study=3, trial_number=0, inner_fold=1, epoch=1,
                          configuration={}, training=raw_metrics(0.02, 1.2),
                          validation=raw_metrics(0.02, 1.2), global_step=1,
                          milestone=True, reference={**references[1], "bp_validation_rows": 999})
    with pytest.raises(ValueError, match="positive reference"):
        log_skill(0.01, 0.0)


def test_resumable_trial_uses_skill_at_common_epoch(tmp_path: Path) -> None:
    class Provider:
        def restore(self, fold, completed_epoch, global_step):
            assert completed_epoch == global_step

        def __call__(self, fold, epoch):
            return raw_metrics(0.04, 1.0), raw_metrics(0.05 * (1 - 0.1 * epoch),
                                                       1.5 * (1 - 0.2 * epoch))

        def save(self, fold, epoch, global_step):
            assert epoch == global_step

        def cleanup(self, fold, epoch):
            pass

    references = {fold: {"bp_validation_rows": 1000, "flp_validation_valid_rows": 700,
                         "bp_validation_reference_ce": 0.05,
                         "flp_validation_reference_ce": 1.5} for fold in range(1, 6)}
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    trial = study.ask()
    args = {"trial": trial, "outer_study": 3, "configuration": FIXED_SMOKE,
            "provider": Provider(), "record_root": tmp_path, "study_name": "skill_resume",
            "provenance": {"references": {str(k): v for k, v in references.items()}},
            "tracker": DisabledTracker(),
            "max_epochs": 2, "references": references}
    assert run_resumable_trial(**args, pause_after_epoch=1) is None
    result = run_resumable_trial(**args)
    assert result == pytest.approx(0.3)
    study.tell(trial, result)
    assert study.best_value == pytest.approx(0.3)
