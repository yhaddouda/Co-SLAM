"""Plot a qualified three-scene sample-pruning study; export PNG and PDF."""
import argparse,json,os,statistics
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/coslam_ba30_plots')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args()
    r=json.loads((args.root/'acceptance.json').read_text());assert r['passed']
    datasets=['Tum','Replica','Synthetic'];x=np.arange(3)
    groups={d:[v for v in r['cases'] if v['dataset']==d] for d in datasets}
    mean=lambda d,k:statistics.mean(v[k] for v in groups[d])
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    prior=r['prior_reference_guard']
    for i,d in enumerate(datasets):
        den=mean(d,'prior_ate_cm' if prior else 'dense_ate_cm')
        a=[v['selected_ate_cm'] for v in groups[d]]
        axs[0].bar(i,100*(statistics.mean(a)/den-1),color='#299b80',width=.55)
        axs[0].errorbar(i,100*(statistics.mean(a)/den-1),yerr=100*statistics.stdev(a)/den,color='black',capsize=4)
        for stage,offset,color in [('tracking',-.18,'#247ba0'),('ba',.18,'#e49a32')]:
            value=100*(1-mean(d,'selected_'+stage+'_ms')/mean(d,'dense_'+stage+'_ms'))
            axs[1].bar(i+offset,value,width=.32,color=color,label=stage.upper() if i==0 else None)
            axs[1].scatter(i+offset+np.linspace(-.06,.06,len(groups[d])),[v[stage+'_time_saved_pct'] for v in groups[d]],s=12,color='black')
    for stage,offset,color in [('tracking',-.24,'#247ba0'),('ba',0,'#e49a32'),('feature',.24,'#9163b0')]:
        axs[2].bar(x+offset,[mean(d,stage+'_pruning_pct') for d in datasets],width=.22,color=color,label={'tracking':'Tracking RGB/SDF','ba':'BA RGB/SDF','feature':'BA feature-only'}[stage])
    axs[0].axhline(17,color='#c32f45',linestyle='--',label='17% mean limit')
    axs[0].set_title('ATE versus '+('previous dense mean' if prior else 'fresh dense mean'))
    axs[0].set_ylabel('Mean ATE increase (%)')
    axs[1].axhline(30,color='#c32f45',linestyle='--',label='30% BA target (any scene)')
    axs[1].set_title('Complete stage wall-time reduction')
    axs[1].set_ylabel('Time saved (%)')
    axs[2].set_title('Actual neural sample reduction')
    axs[2].set_ylabel('Queries pruned (%)')
    for ax in axs:
        ax.axhline(0,color='black',linewidth=.6);ax.set_xticks(x,['TUM','Replica','Synthetic'])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=7)
    axs[2].set_ylim(0,110)
    fig.suptitle('Buffered ray support + sampled feature smoothness — three seeds per scene')
    fig.text(.5,.01,'ATE error bars: candidate sample SD / reference mean. Timing dots: individual paired seeds. All sampling costs included.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    for ext in ('png','pdf'):fig.savefig(args.root/f'pruning_ba30.{ext}',dpi=200,bbox_inches='tight')

if __name__=='__main__':main()
