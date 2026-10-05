"""Float64 streaming moments and exact fixed-bin counts."""

from dataclasses import dataclass, field

import numpy as np

QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
LINEAR_EDGES = np.linspace(0, 1, 101)
LOG_EDGES = np.geomspace(1e-12, 1, 121)


@dataclass
class Accumulator:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = float('inf')
    maximum: float = float('-inf')
    below: int = 0
    above: int = 0
    zero: int = 0
    log_underflow: int = 0
    linear: np.ndarray = field(default_factory=lambda: np.zeros(100, dtype=np.int64))
    logarithmic: np.ndarray = field(default_factory=lambda: np.zeros(120, dtype=np.int64))

    def update(self, values):
        """Values must already be finite and valid; parallel variance merge."""
        x = np.asarray(values, dtype=np.float64)
        n = x.size
        if not n:
            return
        mean = float(x.mean())
        delta = mean - self.mean
        total = self.count + n
        self.m2 += float(np.sum((x - mean) ** 2)) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total
        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))
        self.below += int(np.count_nonzero(x < 0))
        self.above += int(np.count_nonzero(x > 1))
        self.zero += int(np.count_nonzero(x == 0))
        self.log_underflow += int(np.count_nonzero((x > 0) & (x < LOG_EDGES[0])))
        self.linear += np.histogram(x, LINEAR_EDGES)[0]
        self.logarithmic += np.histogram(x, LOG_EDGES)[0]

    def summary(self, total, sample):
        sample = sample[np.isfinite(sample)].astype(np.float64)
        row = {
            'total_pixels': total, 'valid_pixels': self.count,
            'invalid_pixels': total - self.count, 'valid_fraction': self.count / total,
            'min': self.minimum if self.count else None,
            'max': self.maximum if self.count else None,
            'mean': self.mean if self.count else None,
            'std': (self.m2 / self.count)**0.5 if self.count else None,
            'below_0': self.below, 'above_1': self.above,
            'outside_fraction': (self.below + self.above) / self.count if self.count else None,
            'zero_count': self.zero, 'positive_below_log_histogram': self.log_underflow,
            'valid_sample_size': len(sample),
            'moments_method': 'full population; float64 merged central moments; std ddof=0',
            'quantile_method': 'spatial lattice; numpy linear; q0/q100 exact streaming extrema',
        }
        for q in QUANTILES:
            row[f'q{q:02d}'] = (
                row['min'] if q == 0 else row['max'] if q == 100 else
                float(np.quantile(sample, q / 100, method='linear')) if sample.size else None
            )
        return row
