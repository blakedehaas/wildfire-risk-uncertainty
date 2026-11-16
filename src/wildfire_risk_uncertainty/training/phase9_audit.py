"""Backfill and verify partition-level integrity records for Phase 9 outputs."""
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq

from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.phase9 import ROOT, RUN, atomic_json


def main():
    cache = HpoCache(ROOT, ROOT / 'artifacts/training/hpo_cache_v1', check_files=False)
    for outer in range(1, 6):
        for tile, name in enumerate(cache.names):
            out = RUN / 'partitions' / f'outer_{outer}' / f'{tile:04d}.json'
            record = json.loads(out.read_text())
            fold = pq.read_table(ROOT / 'artifacts/spatial_validation/05b/partitions' / name,
                                 columns=['cell_id', 'outer_fold'])
            ids = fold['cell_id'].to_numpy(zero_copy_only=False)
            chosen = fold['outer_fold'].to_numpy(zero_copy_only=False) == outer
            if record['n'] != int(chosen.sum()):
                raise ValueError('Partition support mismatch')
            digest = hashlib.sha256(np.ascontiguousarray(ids[chosen]).tobytes()).hexdigest()
            if 'test_cell_id_sha256' in record and record['test_cell_id_sha256'] != digest:
                raise ValueError('Partition input ID mismatch')
            record['test_cell_id_sha256'] = digest
            atomic_json(out, record)
            sample = RUN / 'samples' / f'outer_{outer}' / f'{tile:04d}.parquet'
            if sample.exists() != bool(record['n']):
                raise ValueError('Partition sample presence mismatch')
            atomic_json(out.with_suffix('.integrity.json'), {
                'record_sha256': sha256(out),
                'sample_sha256': sha256(sample) if sample.exists() else None,
            })
        print(f'audited outer {outer}', flush=True)


if __name__ == '__main__':
    main()
