"""Inner-TRAIN constant references and two-task validation log skill."""

import math
from pathlib import Path

import numpy as np


def log_skill(model_ce: float, reference_ce: float) -> float:
    if not math.isfinite(model_ce) or not math.isfinite(reference_ce) or reference_ce <= 0:
        raise ValueError("Model and positive reference cross-entropies must be finite")
    return 1 - model_ce / reference_ce


def fold_references(cache, root: Path, outer_study: int) -> dict[int, dict]:
    """Derive fold-constant probabilities strictly from inner TRAIN targets."""
    from wildfire_risk_uncertainty.hpo.real import support_counts

    if outer_study not in range(1, 6):
        raise ValueError("Outer study must be 1..5")
    bp_sums = np.zeros(6, dtype=np.float64)
    bp_counts = np.zeros(6, dtype=np.int64)
    flp_sums = np.zeros((6, 6), dtype=np.float64)
    flp_counts = np.zeros(6, dtype=np.int64)
    offsets = cache.arrays["partition_offsets"]
    for number in range(len(cache.names)):
        start, stop = map(int, offsets[number:number + 2])
        all_labels = np.asarray(cache.arrays["inner_fold"][start:stop, outer_study - 1])
        eligible = np.flatnonzero(all_labels != 0) + start
        labels = all_labels[all_labels != 0]
        bp = np.asarray(cache.arrays["bp"][eligible])
        valid = np.asarray(cache.arrays["flp_valid"][eligible], dtype=bool)
        bp_sums += np.bincount(labels, weights=bp, minlength=6)
        bp_counts += np.bincount(labels, minlength=6)
        groups = labels[valid]
        flp_counts += np.bincount(groups, minlength=6)
        flp = np.asarray(cache.arrays["flp"][eligible[valid]])
        for cls in range(6):
            flp_sums[:, cls] += np.bincount(groups, weights=flp[:, cls], minlength=6)
    frozen = support_counts(root, outer_study)
    references = {}
    for fold in range(1, 6):
        train_bp_count = int(bp_counts[1:].sum() - bp_counts[fold])
        train_flp_count = int(flp_counts[1:].sum() - flp_counts[fold])
        train_bp = (bp_sums[1:].sum() - bp_sums[fold]) / train_bp_count
        train_flp = (flp_sums[1:].sum(axis=0) - flp_sums[fold]) / train_flp_count
        validation_bp = bp_sums[fold] / bp_counts[fold]
        validation_flp = flp_sums[fold] / flp_counts[fold]
        if (train_bp_count != frozen[fold]["train"]["rows"] or
                train_flp_count != frozen[fold]["train"]["flp_valid"] or
                bp_counts[fold] != frozen[fold]["validation"]["rows"] or
                flp_counts[fold] != frozen[fold]["validation"]["flp_valid"]):
            raise ValueError(f"Frozen task support differs for fold {fold}")
        if (not 0 < train_bp < 1 or not np.all(train_flp > 0) or
                not np.isclose(train_flp.sum(), 1, atol=1e-6) or
                not np.isclose(validation_flp.sum(), 1, atol=1e-6)):
            raise ValueError(f"Invalid constant-probability reference for fold {fold}")
        bp_ce = -validation_bp * math.log(train_bp) - (
            1 - validation_bp) * math.log1p(-train_bp)
        flp_ce = float(-np.dot(validation_flp, np.log(train_flp)))
        references[fold] = {
            "bp_train_probability": float(train_bp),
            "flp_train_probabilities": train_flp.tolist(),
            "bp_train_rows": train_bp_count,
            "flp_train_valid_rows": train_flp_count,
            "bp_validation_rows": int(bp_counts[fold]),
            "flp_validation_valid_rows": int(flp_counts[fold]),
            "bp_validation_reference_ce": bp_ce,
            "flp_validation_reference_ce": flp_ce,
        }
    return references
