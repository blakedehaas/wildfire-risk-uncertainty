"""Full-coverage FLP-vector diagnostics accumulated inside the Dillon scan."""

from dataclasses import dataclass, field

import numpy as np

from .statistics import Accumulator

TOLERANCES = (1e-6, 1e-4, 1e-3, 1e-2)
STATE_TOLERANCE = 1e-4
CATEGORICAL_TOLERANCE = 1e-6
SUM_EDGES = np.linspace(0, 1.1, 111)
DEVIATION_EDGES = np.r_[0., np.geomspace(1e-9, 1.1, 111)]
ENTROPY_EDGES = np.linspace(0, 1.8, 81)


def vector_metrics(flp):
    """Float64 sums; zero vectors have no categorical class."""
    values = np.asarray(flp, dtype=np.float64)
    sums = np.sum(values, axis=0, dtype=np.float64)
    deviation = np.abs(sums - 1)
    zero = sums == 0
    near = (deviation <= STATE_TOLERANCE) & ~zero
    classes = np.argmax(values, axis=0) + 1
    classes = np.where(zero, 0, classes)
    entropy = -np.sum(np.where(values > 0, values * np.log(np.where(values > 0, values, 1)), 0), axis=0)
    return sums, deviation, zero, near, classes, entropy


@dataclass
class FLPDiagnostics:
    sums: Accumulator = field(default_factory=Accumulator)
    deviation: Accumulator = field(default_factory=Accumulator)
    exact_one: int = 0
    positive: int = 0
    tolerances: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    states: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    crosstab: np.ndarray = field(default_factory=lambda: np.zeros((2, 3), dtype=np.int64))
    sum_hist: np.ndarray = field(default_factory=lambda: np.zeros(110, dtype=np.int64))
    deviation_hist: np.ndarray = field(default_factory=lambda: np.zeros(111, dtype=np.int64))
    class_counts: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))
    max_probability: Accumulator = field(default_factory=Accumulator)
    entropy: Accumulator = field(default_factory=Accumulator)
    entropy_hist: np.ndarray = field(default_factory=lambda: np.zeros(80, dtype=np.int64))
    categorical_count: int = 0
    categorical_invalid: int = 0

    def update(self, flp, bp=None):
        """FLP arrays are valid six-vector pixels; BP may contain NaN for mismatch."""
        values = np.asarray(flp, dtype=np.float64)
        if not values.size:
            return
        sums, deviation, zero, near, classes, entropy = vector_metrics(values)
        self.sums.update(sums)
        self.deviation.update(deviation)
        self.exact_one += int(np.count_nonzero(sums == 1))
        self.positive += int(np.count_nonzero(sums > 0))
        self.tolerances += [np.count_nonzero(deviation <= t) for t in TOLERANCES]
        self.states += [np.count_nonzero(zero), np.count_nonzero((sums > 0) & (sums < 1 - STATE_TOLERANCE)),
                        np.count_nonzero(near), np.count_nonzero(sums > 1 + STATE_TOLERANCE)]
        self.sum_hist += np.histogram(sums, SUM_EDGES)[0]
        self.deviation_hist += np.histogram(deviation, DEVIATION_EDGES)[0]
        if bp is not None:
            bp = np.asarray(bp)
            valid_bp = np.isfinite(bp)
            for b, selector in enumerate((bp == 0, bp > 0)):
                self.crosstab[b] += [np.count_nonzero(valid_bp & selector & zero),
                                     np.count_nonzero(valid_bp & selector & near),
                                     np.count_nonzero(valid_bp & selector & ~(zero | near))]
            support = valid_bp & (bp > 0) & (deviation <= CATEGORICAL_TOLERANCE) & np.all((values >= 0) & (values <= 1), axis=0)
            self.categorical_invalid += int(np.count_nonzero(valid_bp & (bp > 0) & ~support))
            if np.any(support):
                self.categorical_count += int(np.count_nonzero(support))
                self.class_counts += np.bincount(classes[support], minlength=7)[1:]
                self.max_probability.update(np.max(values[:, support], axis=0))
                self.entropy.update(entropy[support])
                self.entropy_hist += np.histogram(entropy[support], ENTROPY_EDGES)[0]

    def categorical_justified(self):
        return self.categorical_count > 0
