import copy,csv,math
from pathlib import Path
from supervisor import read,sha

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
    assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MODE_15W\n1'
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
