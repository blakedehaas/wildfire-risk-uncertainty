"""Frozen Phase 9 held-out inference and additive, recoverable evaluation."""
import argparse
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from wildfire_risk_uncertainty.training.cache import HpoCache, sha256
from wildfire_risk_uncertainty.training.cache_checkpoint import cache_binding
from wildfire_risk_uncertainty.training.model import RiskMLP
from wildfire_risk_uncertainty.training.phase8 import ROOT, checkpoint_metadata
from wildfire_risk_uncertainty.training.preprocessing import (
    PreprocessingState,
    transform,
)

RUN = ROOT / 'artifacts/evaluation/phase9/phase9-20261104-v1'
PHASE8 = ROOT / 'artifacts/training/phase8/phase8-20261102T181708Z'
BINS = np.array([0, .001, .005, .01, .02, .05, .1, 1.0000001])
EPS = 1e-12


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def state_for(outer, cache):
    record = json.loads((PHASE8 / 'preprocessing' / f'outer_{outer}.json').read_text())
    inputs = tuple(cache.manifest['inputs'])
    state = PreprocessingState(inputs, tuple(record['means_after_fixed_transform']),
        tuple(record['standard_deviations_after_imputation']),
        tuple(record['missing_counts'][n] for n in inputs), record['rows'],
        record['flp_valid_rows'], record['schema_sha256'])
    if state.digest() != record['sha256']:
        raise ValueError('Preprocessing mismatch')
    return state


def models_for(outer, cache, state, manifest):
    config = manifest['selected_trials'][str(outer)]['configuration']
    models = []
    for member in range(5):
        key = f'outer_{outer}_member_{member}'
        path = PHASE8 / 'models' / key / 'checkpoints/epoch_100.pt'
        hash_value = sha256(path)
        if json.loads(path.with_suffix('.pt.sha256.json').read_text())['sha256'] != hash_value:
            raise ValueError(f'Checkpoint hash mismatch: {key}')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        expected = {**checkpoint_metadata(manifest, key, 42000 + outer * 1000 + member, state.digest()),
                    'cache_binding': cache_binding(cache, state)}
        if saved['completed_epoch'] != 100 or saved['metadata'] != expected:
            raise ValueError(f'Checkpoint provenance mismatch: {key}')
        model = RiskMLP(len(state.inputs), config['depth'], config['width']).cuda()
        model.load_state_dict(saved['model_state'], strict=True)
        model.eval()
        models.append(model)
    return models


def infer(models, features, state):
    x = torch.from_numpy(transform(features, state)).cuda()
    bp, flp = [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for model in models:
            a, b = model.probabilities(x)
            bp.append(a.float().cpu().numpy())
            flp.append(b.float().cpu().numpy())
    b = np.stack(bp).astype(np.float64)
    f = np.stack(flp).astype(np.float64)
    if not np.isfinite(b).all() or not np.isfinite(f).all() or np.max(abs(f.sum(axis=2)-1)) > 1e-5:
        raise ValueError('Invalid predictions')
    return b, f


def validate_targets(bp, flp, valid):
    if not np.isfinite(bp).all() or np.any((bp < 0) | (bp > 1)):
        raise ValueError('Invalid BP target')
    if np.any(valid):
        f = flp[valid]
        if not np.isfinite(f).all() or np.any(f < -1e-7) or np.max(abs(f.sum(axis=1)-1)) > 1e-5:
            raise ValueError('Invalid FLP target/mask')


def moments(y, p):
    e = p-y
    return {'n':len(y), 'sum_y':float(y.sum()), 'sum_p':float(p.sum()),
        'sum_yy':float(np.dot(y,y)), 'sum_pp':float(np.dot(p,p)), 'sum_yp':float(np.dot(y,p)),
        'sum_abs':float(np.abs(e).sum()), 'sum_e':float(e.sum()), 'sum_e2':float(np.dot(e,e)),
        'ce':float((-y*np.log(np.clip(p,EPS,1))-(1-y)*np.log(np.clip(1-p,EPS,1))).sum()),
        'entropy':float((-np.where(y>0,y*np.log(np.clip(y,EPS,1)),0)-np.where(y<1,(1-y)*np.log(np.clip(1-y,EPS,1)),0)).sum())}


def flp_stats(y,p):
    e=p-y; n=len(y)
    if n==0:
        return {'n':0,'ce':0.,'entropy':0.,'abs':[0.]*6,'e':[0.]*6,'e2':[0.]*6,'tv':0.,'confusion':[[0]*6 for _ in range(6)]}
    labels=np.argmax(y,axis=1)*6+np.argmax(p,axis=1)
    return {'n':int(n),'ce':float((-y*np.log(np.clip(p,EPS,1))).sum()),
      'entropy':float((-np.where(y>0,y*np.log(np.clip(y,EPS,1)),0)).sum()),
      'abs':np.abs(e).sum(axis=0).tolist(),'e':e.sum(axis=0).tolist(),
      'e2':(e*e).sum(axis=0).tolist(),'tv':float(.5*np.abs(e).sum()),
      'confusion':np.bincount(labels,minlength=36).reshape(6,6).tolist()}


def record(bp,flp,valid,pb,pf,ids,x,y):
    n=len(bp); avg_b=pb.mean(axis=0); avg_f=pf.mean(axis=0)
    v=pb.var(axis=0); fv=pf.var(axis=0)
    out={'bp':moments(bp,avg_b),'flp':flp_stats(flp[valid],avg_f[valid]),
      'members':[{'bp':moments(bp,pb[k]),'flp':flp_stats(flp[valid],pf[k,valid])} for k in range(5)],
      'bins':[], 'strata':{},'uncertainty':{},'geography':{}}
    for j in range(len(BINS)-1):
        mask=(bp>=BINS[j])&(bp<BINS[j+1]); out['bins'].append(moments(bp[mask],avg_b[mask]))
    for threshold in (.01,.02):
        mask=bp>=threshold; out['strata'][str(threshold)]={'bp':moments(bp[mask],avg_b[mask]),
          'flp':flp_stats(flp[mask&valid],avg_f[mask&valid]),
          'variance_sum':float(v[mask].sum())}
    error=np.abs(avg_b-bp)
    out['uncertainty']={'n':n,'sum_v':float(v.sum()),'sum_v2':float(np.dot(v,v)),
      'sum_error':float(error.sum()),'sum_error2':float(np.dot(error,error)),
      'sum_v_error':float(np.dot(v,error)),
      'within_bp_sum':float((pb*(1-pb)).mean(axis=0).sum()),
      'between_bp_sum':float(v.sum()),
      'within_flp_sum':(pf*(1-pf)).mean(axis=0).sum(axis=0).tolist(),
      'between_flp_sum':fv.sum(axis=0).tolist()}
    gx=np.floor(x/100000).astype(np.int32); gy=np.floor(y/100000).astype(np.int32)
    keys=np.stack([gx,gy],axis=1); groups, inverse=np.unique(keys,axis=0,return_inverse=True)
    for j,(a,b) in enumerate(groups):
        z=inverse==j
        out['geography'][f'{a},{b}']={'n':int(z.sum()),'target_bp':float(bp[z].sum()),
          'prediction_bp':float(avg_b[z].sum()),'absolute_error':float(error[z].sum()),
          'variance':float(v[z].sum()),'flp_n':int((z&valid).sum()),
          'target_flp':flp[z&valid].sum(axis=0).tolist(),
          'prediction_flp':avg_f[z&valid].sum(axis=0).tolist()}
    sample=ids%101==0
    tab=pa.table({'cell_id':ids[sample],'x_5070':x[sample],'y_5070':y[sample],
      'bp_target':bp[sample].astype('float32'),'bp_prediction':avg_b[sample].astype('float32'),
      'bp_between_variance':v[sample].astype('float32'),
      'flp_valid':valid[sample],**{f'flp{k+1}_target':flp[sample,k] for k in range(6)},
      **{f'flp{k+1}_prediction':avg_f[sample,k].astype('float32') for k in range(6)}})
    return out,tab


def merge(left,right):
    if isinstance(left,dict):
        for k,v in right.items(): left[k]=merge(left[k],v) if k in left else v
        return left
    if isinstance(left,list):
        return [merge(a,b) for a,b in zip(left,right,strict=True)]
    return left+right


def score(m, baseline=None):
    n=m['n']
    if not n:return {'n':0}
    sy,sp,syy,spp,syp=(m[k] for k in ('sum_y','sum_p','sum_yy','sum_pp','sum_yp'))
    vy=syy-sy*sy/n; vp=spp-sp*sp/n
    cor=(syp-sy*sp/n)/np.sqrt(vy*vp) if vy>0 and vp>0 else None
    result={'n':n,'target_mean':sy/n,'prediction_mean':sp/n,'mae':m['sum_abs']/n,
      'rmse':np.sqrt(m['sum_e2']/n),'bias':m['sum_e']/n,
      'r2':1-m['sum_e2']/vy if vy>0 else None,'pearson':cor,
      'ce':m['ce']/n,'excess_ce_kl':(m['ce']-m['entropy'])/n}
    if baseline is not None:
        bce=(-sy*np.log(np.clip(baseline,EPS,1))-(n-sy)*np.log(np.clip(1-baseline,EPS,1)))/n
        result['train_constant_bp']=baseline; result['baseline_ce']=bce
        result['ce_skill']=1-result['ce']/bce if bce>0 else None
        result['ce_improvement']=bce-result['ce']
    return result


def score_flp(m):
    n=m['n']
    if not n:return {'n':0}
    c=np.array(m['confusion'])
    return {'n':n,'ce':m['ce']/n,'excess_ce_kl':(m['ce']-m['entropy'])/n,
      'class_mae':(np.array(m['abs'])/n).tolist(),
      'class_rmse':np.sqrt(np.array(m['e2'])/n).tolist(),
      'class_bias':(np.array(m['e'])/n).tolist(),'mean_tv':m['tv']/n,
      'modal_confusion':m['confusion'],'dominant_class_accuracy':float(np.trace(c)/n)}


def train_baseline(cache,outer):
    path=RUN/'preflight'/f'outer_{outer}_train_baseline.json'
    if path.exists():return json.loads(path.read_text())['mean']
    total=0.; n=0
    for _,bp,_,_ in cache.outer_train_chunks(outer_study=outer,epoch=None):
        total+=float(bp.astype(np.float64).sum()); n+=len(bp)
    if n != cache.manifest['outer_train_counts'][outer-1]:raise ValueError('TRAIN baseline support mismatch')
    atomic_json(path,{'sum':total,'n':n,'mean':total/n})
    return total/n


def preflight(cache,manifest):
    for outer in range(1,6):
        state=state_for(outer,cache); models=models_for(outer,cache,state,manifest)
        features,bp,flp,valid=next(cache.outer_train_chunks(outer_study=outer,epoch=None))
        features,bp,flp,valid=[v[:8192] for v in (features,bp,flp,valid)]
        validate_targets(bp,flp,valid)
        pb,pf=infer(models,features,state)
        ids=np.arange(len(bp),dtype=np.int64); xy=np.zeros(len(bp))
        stats,_=record(bp.astype(np.float64),flp.astype(np.float64),valid,pb,pf,ids,xy,xy)
        atomic_json(RUN/'preflight'/f'outer_{outer}_smoke.json',
          {'rows':len(bp),'flp_rows':int(valid.sum()),'bp':score(stats['bp']),
           'flp':score_flp(stats['flp']),'checkpoint_count':len(models)})
        train_baseline(cache,outer)
        del models
    atomic_json(RUN/'preflight'/'passed.json',{'studies':5,'role':'TRAIN only','manifest_sha256':sha256(RUN/'evaluation_manifest.json')})


def evaluate(cache,manifest):
    gate=json.loads((RUN/'preflight'/'passed.json').read_text())
    if gate['studies']!=5 or gate['manifest_sha256']!=sha256(RUN/'evaluation_manifest.json'):
        raise ValueError('TRAIN-only preflight missing or manifest changed')
    canonical=ROOT/'artifacts/dataset/canonical/partitions'
    outer_dir=ROOT/'artifacts/spatial_validation/05b/partitions'
    expected=[19595079,19594646,21984912,22597733,27044861]
    for outer in range(1,6):
        state=state_for(outer,cache); models=models_for(outer,cache,state,manifest)
        baseline=train_baseline(cache,outer)
        totals={}; count=0; offsets=cache.arrays['partition_offsets']
        for tile,name in enumerate(cache.names):
            start,stop=map(int,offsets[tile:tile+2]); out=RUN/'partitions'/f'outer_{outer}'/f'{tile:04d}.json'
            if out.exists():
                saved=json.loads(out.read_text())
                check=json.loads(out.with_suffix('.integrity.json').read_text())
                fold_for_check=pq.read_table(outer_dir/name,columns=['cell_id','outer_fold'])
                ids_for_check=fold_for_check['cell_id'].to_numpy(zero_copy_only=False)
                selected_for_check=fold_for_check['outer_fold'].to_numpy(zero_copy_only=False)==outer
                id_hash=hashlib.sha256(np.ascontiguousarray(ids_for_check[selected_for_check]).tobytes()).hexdigest()
                sample_file=RUN/'samples'/f'outer_{outer}'/f'{tile:04d}.parquet'
                if (saved['test_cell_id_sha256']!=id_hash or check['record_sha256']!=sha256(out)
                    or (saved['n'] and check['sample_sha256']!=sha256(sample_file))):
                    raise ValueError('Recovered partition integrity mismatch')
                if saved['tile_name']!=name or saved['manifest_sha256']!=gate['manifest_sha256']:
                    raise ValueError('Recovery record provenance mismatch')
                if saved['n']:
                    count+=saved['n']; totals=merge(totals,saved['stats'])
                continue
            fold=pq.read_table(outer_dir/name,columns=['cell_id','outer_fold',f'train_eligible_fold_{outer}'])
            ids=fold['cell_id'].to_numpy(zero_copy_only=False)
            if not np.array_equal(ids,cache.arrays['cell_id'][start:stop]):raise ValueError('Fold/cache ID mismatch')
            selected=fold['outer_fold'].to_numpy(zero_copy_only=False)==outer
            eligible=fold[f'train_eligible_fold_{outer}'].to_numpy(zero_copy_only=False)
            if np.any(selected & eligible):raise ValueError('TRAIN/TEST overlap')
            rows=np.flatnonzero(selected)+start; n=len(rows)
            if n:
                coords=pq.read_table(canonical/name,columns=['cell_id','x_5070','y_5070'])
                if not np.array_equal(coords['cell_id'].to_numpy(zero_copy_only=False),ids):raise ValueError('Coordinate ID mismatch')
                x=coords['x_5070'].to_numpy(zero_copy_only=False)[selected]
                y=coords['y_5070'].to_numpy(zero_copy_only=False)[selected]
                bp=np.asarray(cache.arrays['bp'][rows],dtype=np.float64)
                flp=np.asarray(cache.arrays['flp'][rows],dtype=np.float64)
                valid=np.asarray(cache.arrays['flp_valid'][rows],dtype=bool)
                validate_targets(bp,flp,valid)
                chunks=[]; samples=[]
                for lo in range(0,n,65536):
                    hi=min(n,lo+65536)
                    pb,pf=infer(models,np.asarray(cache.arrays['features'][rows[lo:hi]]),state)
                    st,tab=record(bp[lo:hi],flp[lo:hi],valid[lo:hi],pb,pf,ids[selected][lo:hi],x[lo:hi],y[lo:hi])
                    chunks=merge(chunks,st) if chunks else st
                    samples.append(tab)
                sample_path=RUN/'samples'/f'outer_{outer}'/f'{tile:04d}.parquet'
                sample_path.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.concat_tables(samples),sample_path,compression='zstd')
                count+=n; totals=merge(totals,chunks)
            else:chunks={}
            atomic_json(out,{'tile_name':name,'manifest_sha256':gate['manifest_sha256'],'n':n,'stats':chunks,
                'test_cell_id_sha256':hashlib.sha256(np.ascontiguousarray(ids[selected]).tobytes()).hexdigest()})
            sample_file=RUN/'samples'/f'outer_{outer}'/f'{tile:04d}.parquet'
            atomic_json(out.with_suffix('.integrity.json'),{'record_sha256':sha256(out),
                'sample_sha256':sha256(sample_file) if n else None})
            if tile%100==0:print(f'outer {outer}: tile {tile}/{len(cache.names)} rows {count}',flush=True)
        if count!=expected[outer-1]:raise ValueError(f'outer {outer} TEST support mismatch {count}')
        atomic_json(RUN/'summaries'/f'outer_{outer}_raw.json',totals)
        report={'study':outer,'bp':score(totals['bp'],baseline),'flp':score_flp(totals['flp']),
          'members':[{'member':k,'bp':score(v['bp'],baseline),'flp':score_flp(v['flp'])} for k,v in enumerate(totals['members'])],
          'target_bins':[{'lower':float(BINS[j]),'upper':float(BINS[j+1]),**score(v,baseline)} for j,v in enumerate(totals['bins'])],
          'strata':{k:{'bp':score(v['bp'],baseline),'flp':score_flp(v['flp']),
                         'mean_bp_between_variance':v['variance_sum']/v['bp']['n'] if v['bp']['n'] else None} for k,v in totals['strata'].items()},
          'uncertainty':totals['uncertainty']}
        u=report['uncertainty']; n=u['n']; a=u['sum_v2']-u['sum_v']**2/n; b=u['sum_error2']-u['sum_error']**2/n
        report['uncertainty']={'n':n,'mean_bp_between_variance':u['sum_v']/n,
          'mean_bp_within_variance':u['within_bp_sum']/n,
          'bp_variance_error_pearson':(u['sum_v_error']-u['sum_v']*u['sum_error']/n)/np.sqrt(a*b) if a>0 and b>0 else None,
          'mean_flp_between_variance':(np.array(u['between_flp_sum'])/n).tolist(),
          'mean_flp_within_variance':(np.array(u['within_flp_sum'])/n).tolist()}
        atomic_json(RUN/'summaries'/f'outer_{outer}.json',report)
        del models
    atomic_json(RUN/'complete.json',{'studies':5,'manifest_sha256':gate['manifest_sha256']})



def export_predictions(cache, manifest):
    gate=json.loads((RUN/'complete.json').read_text())
    if gate['studies']!=5 or gate['manifest_sha256']!=sha256(RUN/'evaluation_manifest.json'):
        raise ValueError('Completed held-out evaluation missing or manifest changed')
    offsets=cache.arrays['partition_offsets']
    for outer in range(1,6):
        state=state_for(outer,cache); models=models_for(outer,cache,state,manifest)
        count=0
        for tile,name in enumerate(cache.names):
            part=json.loads((RUN/'partitions'/f'outer_{outer}'/f'{tile:04d}.json').read_text())
            if not part['n']:continue
            path=RUN/'predictions'/f'outer_{outer}'/f'{tile:04d}.parquet'
            sidecar=path.with_suffix('.sha256.json')
            if path.exists():
                if json.loads(sidecar.read_text())['sha256']!=sha256(path):
                    raise ValueError('Prediction export recovery hash mismatch')
                count+=part['n'];continue
            start,_stop=map(int,offsets[tile:tile+2])
            fold=pq.read_table(ROOT/'artifacts/spatial_validation/05b/partitions'/name,
                columns=['cell_id','outer_fold',f'train_eligible_fold_{outer}'])
            ids=fold['cell_id'].to_numpy(zero_copy_only=False)
            selected=fold['outer_fold'].to_numpy(zero_copy_only=False)==outer
            if np.any(selected&fold[f'train_eligible_fold_{outer}'].to_numpy(zero_copy_only=False)):
                raise ValueError('TRAIN/TEST overlap during export')
            chosen=ids[selected]
            if len(chosen)!=part['n'] or hashlib.sha256(np.ascontiguousarray(chosen).tobytes()).hexdigest()!=part['test_cell_id_sha256']:
                raise ValueError('Prediction export input mismatch')
            rows=np.flatnonzero(selected)+start
            batches=[]
            for lo in range(0,len(rows),65536):
                hi=min(lo+65536,len(rows));pb,pf=infer(models,np.asarray(cache.arrays['features'][rows[lo:hi]]),state)
                avg_b=pb.mean(axis=0);var_b=pb.var(axis=0)
                avg_f=pf.mean(axis=0);var_f=pf.var(axis=0)
                batches.append(pa.table({'cell_id':chosen[lo:hi],
                    'bp_mean':avg_b.astype('float32'),'bp_between_variance':var_b.astype('float32'),
                    **{f'flp{k+1}_mean':avg_f[:,k].astype('float32') for k in range(6)},
                    **{f'flp{k+1}_between_variance':var_f[:,k].astype('float32') for k in range(6)}}))
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
            pq.write_table(pa.concat_tables(batches),tmp,compression='zstd')
            tmp.replace(path)
            atomic_json(sidecar,{'sha256':sha256(path),'n':len(chosen),
                'input_cell_id_sha256':part['test_cell_id_sha256']})
            count+=len(chosen)
        if count!=json.loads((RUN/'summaries'/f'outer_{outer}.json').read_text())['bp']['n']:
            raise ValueError('Prediction export support mismatch')
        print(f'exported outer {outer}: {count} rows',flush=True)
    atomic_json(RUN/'predictions_complete.json',{'studies':5,'manifest_sha256':gate['manifest_sha256']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','evaluate','export']);args=parser.parse_args()
    cache=HpoCache(ROOT,ROOT/'artifacts/training/hpo_cache_v1',check_files=False)
    manifest=json.loads((PHASE8/'manifest.json').read_text())
    if args.action=='preflight':preflight(cache,manifest)
    elif args.action=='evaluate':evaluate(cache,manifest)
    else:export_predictions(cache,manifest)

if __name__=='__main__':main()
