"""Summarize compare_ba_bucket.py runs and plot losses and clean timings."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
root = parser.parse_args().directory
summary={}
fig, axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for mode in ('baseline','bucket'):
    path=root/f'{mode}-diagnostic'
    df=pd.read_csv(path/'losses.csv')
    ba=df[df.phase=='ba'].reset_index(drop=True)
    initial=df[df.phase=='initial'].reset_index(drop=True)
    probe=pd.read_csv(path/'probe.csv')
    updates=pd.read_csv(path/'updates.csv')
    empty=ba.valid_depths.eq(0)
    cols=['rgb_loss','depth_loss','sdf_loss','fs_loss','total_loss']
    summary[mode]={
      'first100_ba_means':ba.head(100)[cols].mean().to_dict(),
      'first100_initial_means':initial.head(100)[cols].mean().to_dict(),
      'ba_batches':len(ba),'empty_depth_batches':int(empty.sum()),
      'first100_empty_depth_batches':int(empty.head(100).sum()),
      'valid_ray_fraction':float(ba.valid_depths.sum()/ba.rays.sum()),
      'mean_samples_per_ray':float(np.average(ba.samples,weights=ba.rays)),
      'pruned_batches':int(ba.pruned.sum()),
      'empty_batch_rgb_mean':float(ba.loc[empty,'rgb_loss'].mean()) if empty.any() else None,
      'empty_batch_gradient_min':float(updates.loc[empty,'grad_norm'].min()) if empty.any() else None,
      'empty_batch_update_min':float(updates.loc[empty,updates.columns[-1]].min()) if empty.any() else None,
      'probe_before':probe.iloc[0].to_dict(),
      'probe_after_initial':probe.iloc[1].to_dict(),
      'probe_final':probe.iloc[-1].to_dict(),
      'probe_ba_mean':probe[probe.label=='after_ba'][['rgb_loss','depth_loss','sdf_loss','fs_loss']].mean().to_dict(),
    }
    axes[0,0].plot(np.arange(1,101),ba.total_loss.head(100),label=mode,alpha=.8)
    axes[0,1].plot(np.arange(1,101),ba.valid_depths.head(100)/ba.rays.head(100),label=mode)
    axes[1,0].plot(probe.frame.iloc[1:],probe.depth_loss.iloc[1:],label=mode,marker='.')
    timing=[]
    for p in sorted(root.glob(f'{mode}-timing-*/frames.csv')):
        t=pd.read_csv(p); timing.append(t)
    if timing:
        summary[mode]['timing_runs']=[dict(mean_ms=float(t.wall_ms.mean()),
               median_ms=float(t.wall_ms.median()),ba_frame_ms=float(t[t.has_ba==1].wall_ms.mean()),
               tracking_only_ms=float(t[t.has_ba==0].wall_ms.mean())) for t in timing]
        joined=pd.concat(timing)
        summary[mode]['timing_pooled']=dict(mean_ms=float(joined.wall_ms.mean()),
             median_ms=float(joined.wall_ms.median()),ba_frame_ms=float(joined[joined.has_ba==1].wall_ms.mean()),
             tracking_only_ms=float(joined[joined.has_ba==0].wall_ms.mean()))
        axes[1,1].plot(joined.groupby('frame').wall_ms.mean(),label=mode,alpha=.75)
axes[0,0].set(title='First 100 BA iterations: training objective',xlabel='BA iteration',ylabel='Weighted loss')
axes[0,1].set(title='Depth supervision per BA batch',xlabel='BA iteration',ylabel='Valid-depth ray fraction')
axes[1,0].set(title='Same fixed rays, dense rendering (frame-zero probe)',xlabel='Last BA frame',ylabel='Depth MSE')
axes[1,1].set(title='Clean frame timings (mean across repetitions)',xlabel='Frame',ylabel='Wall time (ms)')
for ax in axes.flat:
    if ax.lines:
        ax.legend()
    ax.grid(alpha=.25)
fig.savefig(root/'comparison.png',dpi=160)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
