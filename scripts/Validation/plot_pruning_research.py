"""Standalone ATE versus actual query-reduction figures from saved summaries."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/coslam_pruning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_pruning_research import collect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    rows=json.loads((args.root/'aggregates.json').read_text())
    colors={'dense':'black','legacy':'#d62728','tail':'#2ca02c','margin':'#1f77b4',
            'ht2':'#ff7f0e','ht4':'#e377c2','adaptive':'#9467bd','features':'#17becf','refine':'#a65628','refine_dense':'#f781bf',
            'features_dense':'#7f7f7f','tail_ba':'#bcbd22','tail_tracking':'#8c564b'}
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.8))
    markers={'dense':'s','legacy':'X','tail':'o','margin':'v','ht2':'^',
             'adaptive':'P','features':'D','refine':'*','refine_dense':'h',
             'tail_ba':'<','tail_tracking':'>'}
    for column,dataset in enumerate(('Tum','Replica','Synthetic')):
        items=[r for r in rows if r['dataset']==dataset]
        for row,ax in enumerate(axes[:,column]):
            for r in items:
                if row==1 and r['preset']=='legacy':
                    continue
                ax.errorbar(r['all_stages_reduction_pct'],r['ate_cm'],yerr=r['ate_cm_sd'] if r['n']>1 else None,
                            fmt=markers.get(r['preset'],'o'),label=f"{r['preset']} (n={r['n']})",
                            color=colors.get(r['preset']),capsize=3,markersize=7 if r['preset']=='refine' else 5)
            ax.set_title(dataset+(' — all policies' if row==0 else ' — detail, legacy excluded'))
            ax.set_xlabel('Joint RGB/SDF queries saved, all stages (%)')
            ax.set_ylabel('Full-scene ATE RMSE (cm)')
            ax.grid(alpha=.2)
            if row==0:
                ax.set_ylim(bottom=0)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=6,fontsize=8)
    fig.suptitle('Accuracy versus actual sample pruning; full scenes, initialization included')
    fig.tight_layout(rect=(0,.11,1,.96))
    for suffix in ('png','pdf'):
        fig.savefig(args.root/f'ate_vs_pruning.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    # Paired seed differences expose variation hidden by aggregate means.
    runs=collect(args.root)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for ax,dataset in zip(axes,('Tum','Replica','Synthetic')):
        expected={'Tum':592,'Replica':2000,'Synthetic':395}[dataset]
        dense={r['seed']:r['ate_cm'] for r in runs if r['dataset']==dataset and r['preset']=='dense' and r['frames']==expected}
        for preset in ('tail','refine'):
            items=sorted([r for r in runs if r['dataset']==dataset and r['preset']==preset and r['frames']==expected and r['seed'] in dense],key=lambda r:r['seed'])
            ax.plot([r['seed'] for r in items],[10*(r['ate_cm']-dense[r['seed']]) for r in items],
                    'o-',color=colors[preset],label=preset)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_title(dataset)
        ax.set_xticks(sorted(dense))
        ax.set_xlabel('Paired seed')
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel('ATE change versus dense (mm)\nPositive = degradation')
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(args.root/f'paired_ate_changes.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    # Time-resolved errors after one rigid alignment of each FINAL trajectory.
    # These include later BA pose updates and must not be described as online ATE.
    fig,axes=plt.subplots(3,1,figsize=(11,8.5))
    for ax,dataset in zip(axes,('Tum','Replica','Synthetic')):
        for preset in ('dense','legacy','tail','margin','ht2','adaptive','features','refine'):
            expected={'Tum':592,'Replica':2000,'Synthetic':395}[dataset]
            data=None
            for path in args.root.glob(f'*/{dataset}_{preset}_seed0/trajectories.npz'):
                candidate=np.load(path)
                if len(candidate['ground_truth'])==expected:
                    data=candidate
                    break
            if data is None:
                continue
            truth=data['ground_truth'][:,:3,3].astype(np.float64)
            pred=data['estimated'][:,:3,3].astype(np.float64)
            x,y=pred-pred.mean(0),truth-truth.mean(0)
            u,_,v=np.linalg.svd(x.T@y)
            correction=np.eye(3); correction[-1,-1]=np.linalg.det(v.T@u.T)
            rotation=v.T@correction@u.T
            errors=100*np.linalg.norm(x@rotation.T-y,axis=1)
            size=20
            positions=np.arange(0,len(errors),size)
            rms=[np.sqrt(np.mean(errors[i:i+size]**2)) for i in positions]
            ax.plot(positions,rms,label=preset,color=colors[preset],linewidth=1.1,alpha=.85)
        ax.set_title(dataset+' — seed 0, final globally aligned poses')
        ax.set_ylabel('Translation error (cm, 20-frame RMS)')
        ax.grid(alpha=.2)
        ax.legend(ncol=7,fontsize=7)
    axes[-1].set_xlabel('Frame')
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(args.root/f'trajectory_errors_seed0.{suffix}',dpi=180,bbox_inches='tight')


if __name__=='__main__':
    main()
