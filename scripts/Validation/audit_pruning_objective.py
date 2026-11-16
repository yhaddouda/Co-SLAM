"""Actual trained-CUDA-map audit of tail-pruning loss/gradient preservation.

Train a short dense sequence, then compare paired frozen forwards. This is a
diagnostic supplement; full-scene ATE is measured separately by the research suite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import config
import coslam
import scripts.Validation.check_guarded_objective as checker
from optimization.sample_pruning import forward_pruned
from scripts.Validation.rank_replica_ba_scenes import trajectory_rmse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('Tum','Replica','Synthetic'),required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--frames',type=int,default=30)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh audit directory')
    args.output.mkdir(parents=True)
    (args.output/'artifacts').mkdir()
    cfg=config.load_config(f'configs/{args.dataset}/{args.scene}.yaml')
    for name in ('ba_bucket','tracking_bucket','tracking_naive_pruning','tracking_depth_spatial'):
        cfg.setdefault(name,{})['enabled']=False
    cfg['tracking_experiment']=dict(variant='baseline')
    cfg['ba_experiment']=dict(variant='baseline')
    cfg['sample_pruning']=dict(enabled=False,policy='tail',margin=0.,free_stride=1)
    cfg['grid'].update(hash='CoherentPrime',hash_size=19,morton_sort=False)
    cfg.setdefault('timing',{}).update(mode='none',max_frames=args.frames,disable_eval=True)
    cfg.setdefault('profiling',{})['enabled']=False
    cfg['mapping']['first_mesh']=False
    cfg['mesh'].update(vis=1000000000,visualisation=False)
    cfg['data'].update(output=str(args.output),exp_name='artifacts')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    coslam.cfg=cfg
    slam=coslam.CoSLAM(cfg)
    slam.save_ckpt=lambda *a,**k:None
    slam.run()
    gt=np.stack([slam.pose_gt[i].cpu().numpy() for i in sorted(slam.est_c2w_data)])
    ate=trajectory_rmse(gt,slam.est_c2w_data)/cfg['data']['sc_factor']
    # The existing frozen-map checker performs dense forward and guarded forward
    # with identical perturbations, and reports separate map/rotation/translation
    # gradients. Replace only its guarded callback for this process-local audit.
    def pruned_callback(model,*inputs,**kwargs):
        model._pruning_stage='Tracking'
        model._pruning_frame=0
        model._pruning_iteration=0
        ret=forward_pruned(model,*inputs)
        row=model.pruning_diagnostics[-1]
        model.ba_experiment_sample_counts.append(dict(dense_samples=row['dense_samples'],
            queried_samples=row['queried_samples'],fallback_samples=row['recovery_samples']))
        return ret
    checker.forward_ba_lattice=pruned_callback
    result=checker.check_guarded_objective(slam,samples=256)
    def refined_callback(model,*inputs,**kwargs):
        original=model.config['sample_pruning']['policy']
        try:
            model.config['sample_pruning']['policy']='refine'
            return pruned_callback(model,*inputs,**kwargs)
        finally:
            model.config['sample_pruning']['policy']=original
    cfg['sample_pruning'].update(enabled=True,policy='refine_dense')
    checker.forward_ba_lattice=refined_callback
    result['refined_objective_check']=checker.check_guarded_objective(slam,samples=256)
    result.update(dataset=args.dataset,scene=args.scene,training_frames=len(slam.est_c2w_data),
        diagnostic_prefix_ate_cm=100*ate,method='tail',device=torch.cuda.get_device_name(0))
    files=('coslam.py','model/scene_rep.py','model/utils.py','optimization/sample_pruning.py',
        'optimization/ba_experiments.py','scripts/Validation/audit_pruning_objective.py',
        'scripts/Validation/check_guarded_objective.py')
    result['source_hashes']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2))
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2))
    print('AUDIT_COMPLETE',args.dataset,result['passed'],flush=True)


if __name__=='__main__':
    main()
