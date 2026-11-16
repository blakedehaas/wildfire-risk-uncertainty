"""Rebuild Phase 9 tables and figures from completed numerical artifacts."""
import copy
import csv
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from wildfire_risk_uncertainty.training.phase9 import RUN, merge, score, score_flp


def save(fig,name):
    path=RUN/'figures'/name;path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def main():
    if not (RUN/'complete.json').exists():raise RuntimeError('All five TEST studies must finish first')
    reports=[json.loads((RUN/'summaries'/f'outer_{i}.json').read_text()) for i in range(1,6)]
    raws=[json.loads((RUN/'summaries'/f'outer_{i}_raw.json').read_text()) for i in range(1,6)]
    pooled={}
    for raw in raws:pooled=merge(pooled,copy.deepcopy(raw))
    baseline_ce=sum(r['bp']['baseline_ce']*r['bp']['n'] for r in reports)/pooled['bp']['n']
    pooled_bp=score(pooled['bp']);pooled_bp['baseline_ce']=baseline_ce
    pooled_bp['ce_skill']=1-pooled_bp['ce']/baseline_ce
    result={'pooled_pixel_descriptive':{'bp':pooled_bp,'flp':score_flp(pooled['flp'])},
      'equal_study_summary':{name:{'mean':float(np.mean([r['bp'][name] for r in reports])),
        'min':float(np.min([r['bp'][name] for r in reports])),
        'max':float(np.max([r['bp'][name] for r in reports]))} for name in ('mae','rmse','ce','ce_skill')},
      'study_support':[{'study':r['study'],'bp_n':r['bp']['n'],'flp_n':r['flp']['n']} for r in reports]}
    (RUN/'summaries'/'pooled.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    tables=RUN/'tables';tables.mkdir(exist_ok=True)
    with (tables/'performance.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=['study','model','bp_n','flp_n','bp_mae','bp_rmse','bp_bias','bp_r2','bp_ce','bp_kl','bp_ce_skill','bp_pearson','flp_ce','flp_kl','flp_tv','flp_dominant_accuracy']);w.writeheader()
        for r in reports:
            for label,v in [('ensemble',r),*((f'member_{m["member"]}',m) for m in r['members'])]:
                b=v['bp'];f=v['flp'];w.writerow({'study':r['study'],'model':label,'bp_n':b['n'],'flp_n':f['n'],'bp_mae':b['mae'],'bp_rmse':b['rmse'],'bp_bias':b['bias'],'bp_r2':b['r2'],'bp_ce':b['ce'],'bp_kl':b['excess_ce_kl'],'bp_ce_skill':b['ce_skill'],'bp_pearson':b['pearson'],'flp_ce':f['ce'],'flp_kl':f['excess_ce_kl'],'flp_tv':f['mean_tv'],'flp_dominant_accuracy':f['dominant_class_accuracy']})
        b=result['pooled_pixel_descriptive']['bp'];f=result['pooled_pixel_descriptive']['flp'];w.writerow({'study':'pooled_pixels','model':'ensemble','bp_n':b['n'],'flp_n':f['n'],'bp_mae':b['mae'],'bp_rmse':b['rmse'],'bp_bias':b['bias'],'bp_r2':b['r2'],'bp_ce':b['ce'],'bp_kl':b['excess_ce_kl'],'bp_ce_skill':b['ce_skill'],'bp_pearson':b['pearson'],'flp_ce':f['ce'],'flp_kl':f['excess_ce_kl'],'flp_tv':f['mean_tv'],'flp_dominant_accuracy':f['dominant_class_accuracy']})
    with (tables/'flp_classwise.csv').open('w',newline='') as handle:
        w=csv.writer(handle);w.writerow(['study','class','n','mae','rmse','bias'])
        for r in reports:
            for k in range(6):w.writerow([r['study'],k+1,r['flp']['n'],r['flp']['class_mae'][k],r['flp']['class_rmse'][k],r['flp']['class_bias'][k]])
    with (tables/'risk_strata.csv').open('w',newline='') as handle:
        w=csv.writer(handle);w.writerow(['study','threshold','bp_n','flp_n','bp_mae','bp_bias','flp_tv','bp_between_variance'])
        for r in reports:
            for t,v in r['strata'].items():w.writerow([r['study'],t,v['bp']['n'],v['flp']['n'],v['bp']['mae'],v['bp']['bias'],v['flp'].get('mean_tv'),v['mean_bp_between_variance']])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for r in reports:
        bins=r['target_bins'];x=[(v['lower']+v['upper'])/2 for v in bins]
        axes[0].plot(x,[v.get('prediction_mean',np.nan) for v in bins],marker='o',label=f"Study {r['study']}")
        axes[1].plot(x,[v.get('mae',np.nan) for v in bins],marker='o')
    axes[0].plot([0,.1],[0,.1],'k--',linewidth=1);axes[0].set(xlim=(0,.1),xlabel='Target BP bin center',ylabel='Mean predicted BP',title='Target-binned BP reconstruction');axes[0].legend(fontsize=7)
    axes[1].set(xlim=(0,.1),xlabel='Target BP bin center',ylabel='MAE',title='BP error by target bin')
    save(fig,'bp_reconstruction.png')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(5)
    ax.plot(x,[r['bp']['mae'] for r in reports],'o-',label='Ensemble')
    for m in range(5):ax.scatter(x,[r['members'][m]['bp']['mae'] for r in reports],alpha=.45,label='Member' if m==0 else None)
    ax.set(xticks=x,xticklabels=[f'Study {i}' for i in range(1,6)],ylabel='BP MAE',title='Held-out member versus ensemble');ax.legend();save(fig,'member_vs_ensemble.png')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(6);w=.16
    for j,r in enumerate(reports):ax.bar(x+(j-2)*w,r['flp']['class_mae'],width=w,label=f"Study {r['study']}")
    ax.set(xticks=x,xticklabels=[f'FLP{i}' for i in range(1,7)],ylabel='Class MAE',title='Conditional FLP class reconstruction');ax.legend(fontsize=7);save(fig,'flp_classwise.png')
    fig,axes=plt.subplots(1,5,figsize=(16,3.5))
    for ax,r in zip(axes,reports):
        c=np.array(r['flp']['modal_confusion']);im=ax.imshow(c/c.sum(axis=1,keepdims=True).clip(min=1),vmin=0,vmax=1,cmap='Blues');ax.set(title=f"Study {r['study']}",xlabel='Predicted mode',ylabel='Target mode',xticks=range(6),yticks=range(6))
    fig.colorbar(im,ax=axes.tolist(),shrink=.7);save(fig,'flp_modal_confusion.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(np.arange(5)-.16,[r['uncertainty']['mean_bp_within_variance'] for r in reports],width=.32,label='Within member')
    axes[0].bar(np.arange(5)+.16,[r['uncertainty']['mean_bp_between_variance'] for r in reports],width=.32,label='Between members')
    axes[0].set(xticks=range(5),xticklabels=range(1,6),xlabel='Study',ylabel='Mean BP variance component',title='Predictive mixture decomposition');axes[0].legend(fontsize=7)
    axes[1].bar(range(1,6),[r['uncertainty']['bp_variance_error_pearson'] for r in reports]);axes[1].axhline(0,color='k',lw=1)
    axes[1].set(xlabel='Study',ylabel='Pearson correlation',title='Disagreement versus absolute BP error');save(fig,'uncertainty_error.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,threshold in enumerate(('0.01','0.02')):
        axes[j].bar(range(1,6),[r['strata'][threshold]['bp']['mae'] for r in reports])
        axes[j].set(xlabel='Geographic study',ylabel='BP MAE',title=f'Target BP >= {threshold}')
        for x,r in enumerate(reports,1):
            axes[j].text(x,r['strata'][threshold]['bp']['mae'],f"n={r['strata'][threshold]['bp']['n']:,}",rotation=90,ha='center',va='bottom',fontsize=7)
    save(fig,'elevated_risk_bp.png')
    sample=[]
    study_associations=[]
    for study in range(1,6):
        study_sample=[]
        for path in sorted((RUN/'samples'/f'outer_{study}').glob('*.parquet')):
            t=pq.read_table(path,columns=['bp_target','bp_prediction','bp_between_variance'])
            if t.num_rows:
                values_one=np.column_stack([t[c].to_numpy() for c in t.column_names])
                sample.append(values_one);study_sample.append(values_one)
        one=np.concatenate(study_sample)
        study_associations.append({'study':study,'n':len(one),
            'bp_prediction_target_spearman':float(spearmanr(one[:,0],one[:,1]).statistic),
            'variance_absolute_bp_error_spearman':float(spearmanr(one[:,2],np.abs(one[:,1]-one[:,0])).statistic)})
    values=np.concatenate(sample)
    association=spearmanr(values[:,2],np.abs(values[:,1]-values[:,0]))
    quantiles=np.quantile(values[:,2],[0,.2,.4,.6,.8,1])
    error=np.abs(values[:,1]-values[:,0])
    disagreement_bins=[{'lower':float(quantiles[j]),'upper':float(quantiles[j+1]),
        'n':int(((values[:,2]>=quantiles[j])&(values[:,2]<(quantiles[j+1]) if j<4 else values[:,2]<=quantiles[j+1])).sum()),
        'mean_absolute_bp_error':float(error[(values[:,2]>=quantiles[j])&(values[:,2]<(quantiles[j+1]) if j<4 else values[:,2]<=quantiles[j+1])].mean())} for j in range(5)]
    (RUN/'summaries'/'sample_association.json').write_text(json.dumps({'n':len(values),'sample_rule':'cell_id % 101 == 0',
        'bp_prediction_target_spearman':float(spearmanr(values[:,0],values[:,1]).statistic),
        'spearman_variance_absolute_bp_error':float(association.statistic),
        'per_study':study_associations,'disagreement_quintiles':disagreement_bins},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(5,4));ax.hexbin(values[:,2],np.abs(values[:,1]-values[:,0]),gridsize=65,mincnt=1,bins='log');ax.set(xlabel='Between-member BP variance',ylabel='Absolute BP error',title='Deterministic 1/101 sample');save(fig,'sample_uncertainty_error.png')
    for study,raw in enumerate(raws,1):
        geo=raw['geography'];points=np.array([[*map(int,k.split(',')),v['n'],v['target_bp']/v['n'],v['prediction_bp']/v['n'],v['variance']/v['n']] for k,v in geo.items() if v['n']>=10])
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,col,title in zip(axes,[3,4,5],['Target BP','Ensemble BP','Between-member variance']):
            im=ax.scatter(points[:,0]*100,points[:,1]*100,c=points[:,col],s=20,marker='s',cmap='viridis');fig.colorbar(im,ax=ax,shrink=.7);ax.set(title=title,xlabel='EPSG:5070 x (km)',ylabel='EPSG:5070 y (km)',aspect='equal')
        save(fig,f'outer_{study}_geographic_maps.png')
    print('Figures and tables complete')

if __name__=='__main__':main()
