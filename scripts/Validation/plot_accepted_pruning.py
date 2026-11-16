"""Plot only a pruning study whose independent acceptance checks passed."""
import argparse
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/coslam_accepted_pruning_plots')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    result=json.loads((args.root/'acceptance.json').read_text())
    if not result['passed'] or not all(r['passed'] for r in result['scenes']):
        raise ValueError('Only accepted data may be plotted')
    datasets=['Tum','Replica','Synthetic']
    groups={d:[r for r in result['cases'] if r['dataset']==d] for d in datasets}
    mean=lambda d,k:statistics.mean(r[k] for r in groups[d])
    positions=np.arange(3)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    colors={'tracking':'#247ba0','ba':'#f29536'}
    for label,offset in [('tracking',-.18),('ba',.18)]:
        axes[0].bar(positions+offset,[mean(d,label+'_pruning_pct') for d in datasets],width=.32,color=colors[label],label='Tracking' if label=='tracking' else 'BA')
        savings=[100*(1-mean(d,'selected_'+label+'_ms')/mean(d,'dense_'+label+'_ms')) for d in datasets]
        axes[2].bar(positions+offset,savings,width=.32,color=colors[label],label='Tracking' if label=='tracking' else 'BA',alpha=.7)
        for i,d in enumerate(datasets):
            x=positions[i]+offset+np.linspace(-.07,.07,len(groups[d]))
            axes[2].scatter(x,[r[label+'_speedup_pct'] for r in groups[d]],s=14,color=colors[label],edgecolors='black',linewidths=.4,zorder=3)
    for i,d in enumerate(datasets):
        values=[r['ate_change_pct'] for r in groups[d]]
        axes[1].scatter(i+np.linspace(-.13,.13,len(values)),values,s=30,color='#379b50',zorder=3)
        axes[1].scatter(i,100*(mean(d,'selected_ate_cm')/mean(d,'dense_ate_cm')-1),s=40,color='black',marker='D',zorder=4)
    axes[1].axhline(10,color='#c72c41',linestyle='--',label='10% scene-mean limit')
    axes[1].axhline(0,color='black',linewidth=.7)
    axes[0].set_title('Actual network sample pruning')
    axes[1].set_title('ATE change versus paired dense seed')
    axes[2].set_title('Complete stage wall-time savings')
    axes[0].set_ylabel('Queries saved (%)')
    axes[1].set_ylabel('ATE change (%); negative = improvement')
    axes[2].set_ylabel('Time saved (%)')
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(top=max(12,axes[1].get_ylim()[1]))
    for ax in axes:
        ax.set_xticks(positions,['TUM','Replica','Synthetic'])
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    fig.suptitle(f"Buffered interval-support pruning — three development scenes, {len(result['seeds'])} paired seeds")
    fig.text(.5,.01,'ATE: dots = paired seeds, diamond = ratio of scene means. Runtime: bars = ratio of mean times, dots = paired seeds. Added sampling costs included.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94))
    for suffix in ('png','pdf'):
        fig.savefig(args.root/f'accepted_pruning.{suffix}',dpi=200,bbox_inches='tight')


if __name__=='__main__':
    main()
