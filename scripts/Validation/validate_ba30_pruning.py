"""Verify 17% scene-mean ATE, faster stages, and >=30% BA savings on >=1 scene."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from validate_pruning_acceptance import inspect_run, SCENES


def evaluate(root,prior_root=None,seeds=(0,1,2),require_ablations=False):
    cases=[];reference_hashes=None;method=None;verified=0
    for dataset,(scene,frames) in SCENES.items():
        for seed in seeds:
            paths={f:root/f/f'{dataset}_seed{seed}' for f in ('controls','selected')}
            a,ca,n=inspect_run(paths['controls'],dataset,scene,frames,seed)
            b,cb,m=inspect_run(paths['selected'],dataset,scene,frames,seed);verified+=n+m
            assert a['warmup_frames']==b['warmup_frames']==20
            assert a['device']==b['device'] and a['torch_version']==b['torch_version']
            assert a['map_optimizer_eps']==b['map_optimizer_eps']
            for family in paths:
                hashes=json.loads((paths[family]/'source_hashes.json').read_text())
                if reference_hashes is None:reference_hashes=hashes
                assert hashes==reference_hashes,'Runtime source must be identical in every fresh run'
            x=json.loads(json.dumps(ca));y=json.loads(json.dumps(cb))
            x.pop('sample_pruning');y.pop('sample_pruning')
            x['data'].pop('output');y['data'].pop('output')
            assert x==y,'Only sampling and the output directory may change'
            for config in (ca,cb):
                for flag in ('ba_bucket','tracking_bucket','tracking_naive_pruning','tracking_depth_spatial'):
                    assert not config[flag]['enabled']
                assert config['ba_experiment']['variant']==config['tracking_experiment']['variant']=='baseline'
                assert not config['grid']['morton_sort']
            sampling={k:v for k,v in cb['sample_pruning'].items() if k!='seed'}
            assert cb['sample_pruning']['seed']==seed
            assert sampling['policy']=='support_tv' and sampling['pixel_strategy']=='random'
            assert sampling['stages']==['Tracking','Bundle_Adjustment']
            if method is None:method=sampling
            assert method==sampling,'The sampling parameters must be common to all scenes and seeds'
            row=dict(dataset=dataset,scene=scene,seed=seed,dense_ate_cm=100*a['ate_rmse_m'],selected_ate_cm=100*b['ate_rmse_m'],
                     ate_change_pct=100*(b['ate_rmse_m']/a['ate_rmse_m']-1),dense_run_s=a['run_wall_seconds'],selected_run_s=b['run_wall_seconds'])
            for stage,metric,label in [('Tracking','tracking_invocations','tracking'),('Bundle_Adjustment','ba_invocations','ba')]:
                aa=a['stage_queries'][stage];bb=b['stage_queries'][stage]
                assert aa['forwards']==bb['forwards'] and aa['dense_samples']==bb['dense_samples']
                assert aa['queried_samples']==aa['dense_samples']
                assert a[metric]['count']==b[metric]['count']
                iterations=cb['tracking']['iter'] if label=='tracking' else cb['mapping']['iters']
                every=sampling['tracking_anchor_every'] if label=='tracking' else sampling['anchor_every']
                anchors={0,iterations-1}
                if every:anchors.update(range(0,iterations,every))
                thin=iterations-len(anchors)
                with (paths['selected']/'pruning_diagnostics.csv').open() as f:
                    diag=[r for r in csv.DictReader(f) if r['stage']==stage]
                assert len(diag)==bb['forwards']//iterations*thin
                dense=sum(int(r['dense_samples']) for r in diag);queried=sum(int(r['queried_samples']) for r in diag)
                assert dense==bb['dense_samples']//iterations*thin
                assert all(int(r['recovery_samples'])==0 for r in diag)
                assert bb['queried_samples']==bb['dense_samples']-dense+queried
                reduction=100*(1-bb['queried_samples']/bb['dense_samples'])
                assert reduction>0 and abs(reduction-bb['reduction_pct'])<1e-9
                row[label+'_pruning_pct']=reduction
                row['dense_'+label+'_ms']=a[metric]['mean_ms'];row['selected_'+label+'_ms']=b[metric]['mean_ms']
                row[label+'_time_saved_pct']=100*(1-b[metric]['mean_ms']/a[metric]['mean_ms'])
                if label=='ba':
                    row['dense_joint_points']=aa['queried_samples'];row['selected_joint_points']=bb['queried_samples']
                    row['ba_forwards']=bb['forwards']
            vertices=(cb['training']['smooth_pts']-1)**3
            edges=max(1,int(vertices*sampling['tv_fraction']/6))
            forwards=b['stage_queries']['Bundle_Adjustment']['forwards']
            assert a['stage_feature_queries']['Bundle_Adjustment']==vertices*forwards
            with (paths['selected']/'smoothness_diagnostics.csv').open() as f:tv=list(csv.DictReader(f))
            assert len(tv)==forwards
            assert all(int(r['dense_samples'])==vertices and int(r['queried_samples'])==6*edges and int(r['sampled_edges'])==3*edges for r in tv)
            assert sum(int(r['queried_samples']) for r in tv)==b['stage_feature_queries']['Bundle_Adjustment']
            for stage,count in a['stage_feature_queries'].items():
                if stage!='Bundle_Adjustment':assert b['stage_feature_queries'].get(stage)==count
            row['dense_feature_points']=vertices*forwards
            row['selected_feature_points']=b['stage_feature_queries']['Bundle_Adjustment']
            row['feature_pruning_pct']=100*(1-row['selected_feature_points']/row['dense_feature_points'])
            if prior_root:
                old,old_cfg,n=inspect_run(prior_root/'controls'/f'{dataset}_seed{seed}',dataset,scene,frames,seed);verified+=n
                original=json.loads(json.dumps(old_cfg));fresh=json.loads(json.dumps(ca))
                for config in (original,fresh):
                    config.pop('sample_pruning');config['data'].pop('output')
                assert original==fresh,'Historical reference must use the same non-sampling configuration'
                row['prior_ate_cm']=old['ate_rmse_m']*100
                row['prior_ba_ms']=old['ba_invocations']['mean_ms'];row['prior_tracking_ms']=old['tracking_invocations']['mean_ms']
            cases.append(row)
    ablations=[]
    for dataset,(scene,frames) in SCENES.items():
        seed=seeds[0]
        selected_path=root/'selected'/f'{dataset}_seed{seed}'
        selected=json.loads((selected_path/'summary.json').read_text())
        selected_cfg=json.loads((selected_path/'config.json').read_text())
        for family in ('matched_control','interval_only'):
            path=root/family/f'{dataset}_seed{seed}'
            if not (path/'summary.json').exists():
                assert not require_ablations,('Missing ablation',family,dataset)
                continue
            control,cc,n=inspect_run(path,dataset,scene,frames,seed);verified+=n
            assert json.loads((path/'source_hashes.json').read_text())==reference_hashes
            x=json.loads(json.dumps(cc));y=json.loads(json.dumps(selected_cfg))
            x.pop('sample_pruning');y.pop('sample_pruning')
            x['data'].pop('output');y['data'].pop('output')
            assert x==y
            c=cc['sample_pruning'];b=selected_cfg['sample_pruning']
            if family=='matched_control':
                assert c==dict(b,margin=1000000.,tv_fraction=1.)
            else:
                assert c=={k:('support' if k=='policy' else v) for k,v in b.items() if k not in ('tv_fraction','tracking_anchor_every')}
            record=dict(dataset=dataset,seed=seed,family=family,ate_cm=100*control['ate_rmse_m'])
            for stage,metric,label in [('Tracking','tracking_invocations','tracking'),('Bundle_Adjustment','ba_invocations','ba')]:
                a=control['stage_queries'][stage];z=selected['stage_queries'][stage]
                assert a['forwards']==z['forwards'] and a['dense_samples']==z['dense_samples']
                if family=='matched_control':assert a['queried_samples']==a['dense_samples']
                else:assert a['queried_samples']==z['queried_samples']
                record[label+'_ms']=control[metric]['mean_ms']
                record['selected_'+label+'_ms']=selected[metric]['mean_ms']
                record[label+'_saved_pct']=100*(1-selected[metric]['mean_ms']/control[metric]['mean_ms'])
            assert control['stage_feature_queries']['Bundle_Adjustment']==vertices_for_cfg(cc)*control['stage_queries']['Bundle_Adjustment']['forwards']
            record['dense_feature_points']=control['stage_feature_queries']['Bundle_Adjustment']
            record['selected_feature_points']=selected['stage_feature_queries']['Bundle_Adjustment']
            ablations.append(record)
    for manifest,directory in [('validation_source_hashes.json','validation_source'),('supplemental_source_hashes.json','supplemental_source')]:
        if (root/manifest).exists():
            for name,digest in json.loads((root/manifest).read_text()).items():
                assert hashlib.sha256((root/directory/name).read_bytes()).hexdigest()==digest
    scenes=[]
    for dataset in SCENES:
        values=[r for r in cases if r['dataset']==dataset]
        mean=lambda k:statistics.mean(r[k] for r in values)
        scene=dict(dataset=dataset,ate_change_pct=100*(mean('selected_ate_cm')/mean('dense_ate_cm')-1))
        for label in ('tracking','ba'):
            scene[label+'_time_saved_pct']=100*(1-mean('selected_'+label+'_ms')/mean('dense_'+label+'_ms'))
        if prior_root:
            scene['ate_change_vs_prior_pct']=100*(mean('selected_ate_cm')/mean('prior_ate_cm')-1)
            scene['ba_time_saved_vs_prior_pct']=100*(1-mean('selected_ba_ms')/mean('prior_ba_ms'))
        scene['passed']=scene['ate_change_pct']<=17 and (not prior_root or scene['ate_change_vs_prior_pct']<=17) and all(scene[k+'_time_saved_pct']>0 for k in ('tracking','ba'))
        scenes.append(scene)
    qualified=sum(s['ba_time_saved_pct']>=30 and (not prior_root or s['ba_time_saved_vs_prior_pct']>=30) for s in scenes)
    causal=True
    if require_ablations:
        qualifying={s['dataset'] for s in scenes if s['ba_time_saved_pct']>=30 and (not prior_root or s['ba_time_saved_vs_prior_pct']>=30)}
        causal=any(all(a['ba_saved_pct']>0 for a in ablations if a['dataset']==d) for d in qualifying)
    return dict(passed=all(s['passed'] for s in scenes) and qualified>=1 and causal,seeds=list(seeds),cases=cases,scenes=scenes,
        pruning_benefit_in_qualifying_scene=causal if require_ablations else None,ablations=ablations,ablations_verified=len(ablations)==2*len(SCENES),qualifying_ba_scenes=qualified,source_files_verified=verified,method_config=method,
        criterion='Each scene three-seed mean ATE <=17%; tracking/BA means faster everywhere; BA wall-time reduction >=30% on at least one scene',
        prior_reference_guard=prior_root is not None,source_identical_in_fresh_pairs=True,independent_ate=True,wall_times_recomputed=True,actual_queries_verified=True)


def vertices_for_cfg(cfg):
    return (cfg['training']['smooth_pts']-1)**3


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--prior-root',type=Path);p.add_argument('--write',action='store_true');p.add_argument('--require-ablations',action='store_true')
    args=p.parse_args();result=evaluate(args.root,args.prior_root,require_ablations=args.require_ablations)
    if not result['passed']:raise SystemExit('Not qualified; no candidate results exported.')
    if args.write:(args.root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: every scene <=17% ATE, both stages faster; >=30% BA on at least one scene.')

if __name__=='__main__':main()
