"""Local orchestration only. Scientific execution stays in unchanged runtime v1."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
SCAN = ROOT.parent / 'orin_scannet_repair_v1'
EXPORT = ROOT.parent / 'orin_existing_dp_export_v1'
OLD = ROOT.parent / 'orin_after_v4_maxn_v1'
CO = ROOT.parent / 'orin_coslam_runtime_v1/runtime'
TOP = ROOT.parents[1]
sys.path.insert(0, str(OLD))
from post_v4_follower import idle, identity
import post_v4_execute as support
support.ROOT = ROOT


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp'); t.write_text(json.dumps(value, indent=2) + '\n'); t.replace(p)


def config_identity(cfg):
    c = copy.deepcopy(cfg)
    c.pop('round2', None)
    for k in ['hash', 'morton_sort']: c['grid'].pop(k, None)
    for k in ['output', 'exp_name']: c['data'].pop(k, None)
    c['data']['datadir'] = str(Path(c['data']['datadir']).resolve())
    return c


def audit(j, folder, source_hash, control_cfg=None, diagnostic=False):
    """Fresh raw-CSV audit; called only with no timed child, under ownership."""
    out = Path(folder); summary = read(out/'summary.json'); cfg = read(out/'config.json')
    n = 50 if diagnostic else j['expected_native_frames']
    assert summary['dataset'] == j['dataset'] and summary['scene'] == j['scene']
    assert summary['condition'] == j['condition'] and summary['seed'] == j['seed']
    assert summary['frames'] == n and summary['diagnostic'] == diagnostic
    assert summary['full_sequence'] == (not diagnostic) and summary['finite_parameters_and_poses']
    assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MAXN\n0'
    assert summary['warmup_frames'] == 20 and summary['quality'] is False
    assert sha(out/'source_manifest.json') == source_hash
    assert cfg['grid']['hash_size'] == j['hash_size'] and cfg['round2']['engineering'] is False
    assert cfg['mapping']['n_pixels'] == .05 and not cfg['mapping']['first_mesh'] and not cfg['mesh']['visualisation']
    prune = j['condition'] in ['DP', 'MC_DP']; morton = j['condition'] in ['MC', 'MC_DP']
    assert cfg['grid']['hash'] == ('Morton' if morton else 'CoherentPrime') and cfg['grid']['morton_sort'] == morton
    assert cfg['round2']['sampler'] == ('support' if prune else 'dense')
    assert cfg['round2']['margin_scene_units'] == .15 and cfg['round2']['anchor_every'] == 3
    assert cfg['round2']['seed_policy'] == 'once before model construction; ordinary RNG evolution'
    if control_cfg is not None: assert config_identity(cfg) == config_identity(control_cfg), 'Effective native config mismatch'
    if j.get('datadir'): assert Path(cfg['data']['datadir']).resolve() == Path(j['datadir']).resolve()
    def rows(name):
        with (out/name).open() as f: return list(csv.DictReader(f))
    calls, events, queries, frames = [rows(name) for name in ['calls.csv','iterations.csv','queries.csv','frames.csv']]
    assert len(frames) == n and {int(x['frame_id']) for x in frames} == set(range(n))
    assert set(events[0]) == {'frame_id','iteration_type','iteration_id','forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms'}
    stages = {}
    for stage, ids, steps in [('tracking', set(range(1,n)), cfg['tracking']['iter']),
                             ('bundle_adjustment', set(range(cfg['mapping']['map_every'],n,cfg['mapping']['map_every'])), cfg['mapping']['iters'])]:
        c = [x for x in calls if x['stage']==stage]; e = [x for x in events if x['iteration_type']==stage]; q = [x for x in queries if x['stage']==stage]
        expected = {(f,i) for f in ids for i in range(steps)}
        assert len(c)==len(ids) and {int(x['frame_id']) for x in c}==ids
        assert len(e)==len(expected) and {(int(x['frame_id']),int(x['iteration_id'])) for x in e}==expected
        assert len(q)==len(expected) and {(int(x['frame_id']),int(x['iteration'])) for x in q}==expected
        pruned = 0
        for x in q:
            actual,dense,i = int(x['actual_queries']),int(x['dense_queries']),int(x['iteration'])
            assert 0 < actual <= dense
            if not prune or i in [0,steps-1] or i%3==0: assert actual==dense
            pruned += actual < dense
        if prune: assert pruned > 0, 'No actual pruning in '+stage
        stages[stage] = dict(calls=len(c),iterations=len(e),anchors_full=True,pruned_updates=pruned)
    assert all(math.isfinite(float(x[k])) and float(x[k])>=0 for x in calls for k in ['wall_ms','cuda_ms'])
    assert all(math.isfinite(float(x[k])) and float(x[k])>=0 for x in events for k in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms'])
    with (out/'frame_iteration_sums.csv').open() as f: sums=list(csv.reader(f))
    assert len(sums)==n-1 and {int(x[0]) for x in sums}==set(range(1,n))
    return dict(status='PASS',summary=summary,config=cfg,config_sha256=sha(out/'config.json'),source_manifest_sha256=source_hash,
                frames=n,stages=stages,output=str(out))


def update(s):
    s['updated_utc']=now(); s['counts']=dict(Counter(j['status'] for j in s['jobs']))
    native=read(SCAN/'CURRENT_SCOPE_STATUS.json')
    n_done=sum(j['status'] in ['done','reused'] for j in native['jobs'] if j['status']!='excluded' and j.get('required_for_completion',True))
    p_done=sum(j['status'] in ['done','reused'] for j in s['jobs'])
    s.update(native_engineering_required=196,native_engineering_completed=n_done,pruning_required=96,pruning_completed=p_done,
             expanded_required=292,expanded_completed=n_done+p_done,ALL_REQUIRED_MAXN_COMPLETE=n_done==196 and p_done==96,
             next_automatic_job=next((j['id'] for j in s['jobs'] if j['status']=='queued'),None),EC_ENG='excluded')
    controls=s.get('controls',{}); ratios={}
    for j in s['jobs']:
        if j['status'] not in ['done','reused'] or not j.get('summary'): continue
        key='BASE' if j['condition']=='DP' else 'MC'; ctrl=controls.get(j['control_ids'][key])
        if ctrl and ctrl['summary']['slam_work_wall_seconds']>0:
            ratios.setdefault(j['condition'],[]).append(j['summary']['slam_work_wall_seconds']/ctrl['summary']['slam_work_wall_seconds'])
    estimates=[]
    for j in s['jobs']:
        if j['status'] not in ['queued','running']: continue
        ctrl=controls.get(j['control_ids']['BASE' if j['condition']=='DP' else 'MC'])
        if ctrl:
            estimate=ctrl['summary']['slam_work_wall_seconds']*statistics.mean(ratios.get(j['condition'],[1.0]))+10
            if j['status']=='running' and j.get('started_utc'):
                estimate=max(0,estimate-(datetime.now(timezone.utc)-datetime.fromisoformat(j['started_utc'])).total_seconds())
            estimates.append(estimate)
    if estimates:
        seconds=sum(estimates);s['forecast']=dict(remaining_ready_seconds=seconds,estimated_finish_utc=(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),
             calibrated_completed_ratios={k:statistics.mean(v) for k,v in ratios.items()},
             method='Matched local full BASE/MC elapsed work, calibrated to completed DP/MC_DP; launch allowance 10 s/job; scene variation remains uncertain; excludes blocked identities; refresh after each completed pair')
    save(ROOT/'DP_COMPLETION_STATUS.json',s)
    md=['# MAXN DP / MC_DP continuation','',s['updated_utc'],'',
        f"Phase: **{s['phase']}**. Pruning: 96 targets / 8 initial inventory-backed / 88 initial missing. Counts: {s['counts']}.",
        f"Native/ENG: {n_done}/196; pruning: {p_done}/96; expanded: {n_done+p_done}/292. **ALL_REQUIRED_MAXN_COMPLETE={str(s['ALL_REQUIRED_MAXN_COMPLETE']).lower()}**. EC ENG excluded.",'',
        f"Follower: {s.get('follower_pid')}; worker: {s.get('worker_pid')}; child: {s.get('active_child_pid')}; log: {s.get('active_log')}",
        f"Next ready job: {s.get('next_automatic_job')}. Forecast: {s.get('forecast','At idle: use local matched durations; 14 h dense non-ScanNet workload proxy alone, plus 24 ScanNet jobs; not an overnight promise.')}",'',
        '| ID | State | Evidence / reason |','|---|---|---|']
    md += [f"| {j['id']} | {j['status']} | {j.get('output',j.get('reason',''))} |" for j in s['jobs']]
    (ROOT/'ORIN_RUN_AGENDA.md').write_text('\n'.join(md)+'\n')
