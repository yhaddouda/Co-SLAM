"""Independently verify saved trajectories, actual query counts and source archives."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

FULL={'Tum':592,'Replica':2000,'Synthetic':395}


def rigid_rmse(prediction, truth):
    x=prediction.astype(np.float64);y=truth.astype(np.float64)
    x=x-x.mean(0);y=y-y.mean(0)
    u,_,v=np.linalg.svd(x.T@y)
    s=np.eye(3)
    if np.linalg.det(v.T@u.T)<0:
        s[-1,-1]=-1
    rotation=v.T@s@u.T
    return np.sqrt(np.mean(np.sum((x@rotation.T-y)**2,axis=1)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--final',action='store_true')
    args=p.parse_args()
    checked=[];sources=0;seen=set();cohorts=defaultdict(set);versions=defaultdict(set)
    for path in sorted(args.root.glob('*/*/summary.json')):
        run=path.parent;s=json.loads(path.read_text());cfg=json.loads((run/'config.json').read_text())
        assert s['dataset'] in FULL,run
        assert s['finite_parameters_and_poses'],run
        trajectory=np.load(run/'trajectories.npz')
        gt=trajectory['ground_truth'];est=trajectory['estimated'];relative=trajectory['relative']
        assert len(est)==len(gt)==s['frames'],run
        valid=np.isfinite(gt).all(axis=(1,2))
        assert np.isfinite(est).all() and np.isfinite(relative).all(),run
        scale=cfg['data']['sc_factor']
        direct=rigid_rmse(est[valid,:3,3],gt[valid,:3,3])/scale
        rel=rigid_rmse(relative[valid,:3,3],gt[valid,:3,3])/scale
        assert abs(direct-s['ate_rmse_m'])<1e-8,(run,direct,s['ate_rmse_m'])
        assert abs(rel-s['ate_relative_rmse_m'])<1e-8,(run,rel,s['ate_relative_rmse_m'])
        hashes=json.loads((run/'source_hashes.json').read_text())
        for name,digest in hashes.items():
            assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest,(run,name)
            sources+=1;versions[name].add(digest)
        diagnostic=run/'pruning_diagnostics.csv'
        if diagnostic.exists():
            with diagnostic.open() as f:
                rows=list(csv.DictReader(f))
            sums=defaultdict(lambda:dict(dense_samples=0,queried_samples=0))
            for r in rows:
                for k in sums[r['stage']]:
                    sums[r['stage']][k]+=int(r[k])
            for stage,counts in sums.items():
                for key,value in counts.items():
                    assert value==s['stage_queries'][stage][key],(run,stage,key,value)
        for stage,counts in s['stage_queries'].items():
            expected=100*(1-counts['queried_samples']/counts['dense_samples'])
            assert abs(expected-counts['reduction_pct'])<1e-9,(run,stage)
        assert s['stage_queries']['First_frame_mapping']['reduction_pct']==0.,run
        preset=run.name[len(s['dataset'])+1:].rsplit('_seed',1)[0]
        full=s['frames']==FULL[s['dataset']]
        if full:
            identity=(s['dataset'],preset,s['seed'])
            assert identity not in seen,('Duplicate full-scene seed',identity)
            seen.add(identity);cohorts[(s['dataset'],preset)].add(s['seed'])
            if preset in ('tail','margin','ht2','adaptive','features','refine'):
                for stage in ('Tracking','Bundle_Adjustment'):
                    assert s['stage_queries'][stage]['reduction_pct']>0.,(run,stage)
            if preset=='dense':
                assert all(r['reduction_pct']==0. for r in s['stage_queries'].values()),run
        checked.append(dict(run=str(run.relative_to(args.root)),full_scene=full))
    audit_results=[]
    for path in args.root.glob('audits/*/audit.json'):
        s=json.loads(path.read_text());assert s['passed'],path
        assert s.get('refined_objective_check',{}).get('passed'),path
        audit_results.append(dict(dataset=s['dataset'],passed=True))
    if args.final:
        for dataset in FULL:
            for preset in ('dense','legacy','tail','refine'):
                assert cohorts[(dataset,preset)]=={0,1,2},(dataset,preset,cohorts[(dataset,preset)])
            for preset in ('margin','ht2','adaptive','features','refine_dense','tail_ba','tail_tracking'):
                assert cohorts[(dataset,preset)]=={0},(dataset,preset,cohorts[(dataset,preset)])
        assert len(audit_results)==3,audit_results
    result=dict(passed=True,final=args.final,completed_runs=len(checked),
        full_scene_runs=sum(r['full_scene'] for r in checked),
        source_files_verified=sources,source_versions={k:len(v) for k,v in versions.items()},
        trajectories_and_relative_ate_recomputed=True,query_counts_cross_checked=True,
        no_scannet_runs=True,frozen_cuda_audits=audit_results,runs=checked)
    (args.root/'integrity_checks.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k not in ('runs','source_versions')})


if __name__=='__main__':
    main()
