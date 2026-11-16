import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from wildfire_risk_uncertainty.training.cache import HpoCache
from wildfire_risk_uncertainty.training.phase9 import (
    PHASE8,
    ROOT,
    flp_stats,
    models_for,
    moments,
    score,
    score_flp,
    state_for,
    validate_targets,
)


def test_soft_bp_metrics_and_constant_prediction():
    target = np.array([0.0, 0.25, 0.75, 1.0])
    prediction = np.array([0.1, 0.4, 0.6, 0.9])
    result = score(moments(target, prediction), 0.5)
    assert result['n'] == 4
    assert result['mae'] == pytest.approx(0.125)
    assert result['bias'] == pytest.approx(0)
    assert result['rmse'] == pytest.approx(np.sqrt(np.mean((prediction-target)**2)))
    assert result['excess_ce_kl'] >= 0
    assert result['ce_skill'] > 0
    constant = score(moments(target,np.full(4,0.5)))
    assert constant['pearson'] is None


def test_soft_flp_metrics_and_modal_diagnostic():
    target=np.array([[.7,.3,0,0,0,0],[0,.2,.8,0,0,0]])
    prediction=np.array([[.6,.4,0,0,0,0],[0,.7,.3,0,0,0]])
    result=score_flp(flp_stats(target,prediction))
    assert result['n']==2
    assert result['mean_tv']==pytest.approx(.3)
    assert result['modal_confusion'][0][0]==1
    assert result['modal_confusion'][2][1]==1
    assert result['dominant_class_accuracy']==pytest.approx(.5)
    assert result['excess_ce_kl']>=0


def test_target_masks_and_normalization():
    bp=np.array([0.,.5])
    flp=np.array([[1.,0,0,0,0,0],[np.nan]*6])
    validate_targets(bp,flp,np.array([True,False]))
    with pytest.raises(ValueError):
        validate_targets(bp,np.array([[.8,0,0,0,0,0],[np.nan]*6]),np.array([True,False]))
    with pytest.raises(ValueError):
        validate_targets(bp,flp,np.array([True,True]))


def test_checkpoint_restore_and_training_partition():
    cache=HpoCache(ROOT,ROOT/'artifacts/training/hpo_cache_v1',check_files=False)
    manifest=json.loads((PHASE8/'manifest.json').read_text())
    state=state_for(1,cache)
    models=models_for(1,cache,state,manifest)
    assert len(models)==5
    assert all(all(torch.isfinite(p).all() for p in model.parameters()) for model in models)
    paths=sorted((ROOT/'artifacts/spatial_validation/05b/partitions').glob('*.parquet'))
    path=next(p for p in paths if pq.ParquetFile(p).metadata.num_rows > 0)
    fold=pq.read_table(path,columns=['outer_fold','train_eligible_fold_1'])
    test=fold['outer_fold'].to_numpy()==1
    train=fold['train_eligible_fold_1'].to_numpy()
    assert not np.any(test & train)
