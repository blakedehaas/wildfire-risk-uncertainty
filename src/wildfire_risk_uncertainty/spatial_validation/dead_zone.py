"""Exact Euclidean grid-center buffer and nearest-training geometry."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def fold_geometry(support: np.ndarray, folds: np.ndarray, fold: int, radius_m: float, cell_m: float = 270.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return TEST, DEAD_ZONE, TRAIN, and TEST-to-nearest-TRAIN distance."""
    test = support & (folds == fold)
    if not test.any():
        raise ValueError(f"Fold {fold} has no test cells")
    dist_to_test = distance_transform_edt(~test, sampling=cell_m)
    dead = support & ~test & (dist_to_test <= radius_m)
    train = support & ~test & (dist_to_test > radius_m)
    if not train.any():
        raise ValueError(f"Fold {fold} has no usable training cells")
    dist_to_train = distance_transform_edt(~train, sampling=cell_m)
    if np.min(dist_to_train[test]) <= radius_m:
        raise ValueError(f"Fold {fold} violates strict geographic separation")
    return test, dead, train, dist_to_train
