"""Independently validate complete RTX dense BA observations."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.Validation.execution_benchmark_common import rigid_ate, statistics

META = ROOT / 'results/ba_experiments/rtx4070_metadata'
OUT = ROOT / 'results/ba_experiments/rtx4070_dense_fast_full'
SCENES = ('office1', 'room1', 'office0')
VARIANTS = ('baseline', 'baseline_fast')


def audit_run(path, require_snapshot=True):
    result = json.loads((path / 'summary.json').read_text())
    cfg = json.loads((path / 'config.json').read_text())
    assert result['frames'] == 2000 and result['warmup_frames'] == 20 and result['seed'] == 0
    assert not result['diagnostic'] and result['finite_parameters_and_poses']
    assert result['tracking'] == 'baseline'
    assert cfg.get('tracking_experiment', {}).get('variant', 'baseline') == 'baseline'
    assert cfg['ba_experiment']['variant'] == result['variant']
    assert not cfg['ba_bucket']['enabled']
    assert not cfg['training']['uniform_samples_until_depth']
    assert all(not cfg[name]['enabled'] for name in
               ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'))
    assert cfg['grid']['hash'] == 'CoherentPrime' and cfg['grid']['hash_size'] == 19
    assert not cfg['grid']['morton_sort'] and not cfg['profiling']['enabled']
    assert cfg['timing']['mode'] == 'none' and cfg['timing']['disable_eval']
    assert cfg['mapping']['iters'] == 10 and cfg['mapping']['sample'] == 2048
    assert cfg['mapping']['map_every'] == cfg['mapping']['keyframe_every'] == 5
    assert cfg['training']['n_samples_d'] + cfg['training']['n_range_d'] == 43
    poses = np.load(path / 'trajectories.npz')
    for name in ('ground_truth', 'estimated', 'relative'):
        assert poses[name].shape == (2000, 4, 4) and np.isfinite(poses[name]).all()
    for name, field in (('estimated', 'ate_rmse_m'), ('relative', 'ate_relative_rmse_m')):
        ate = rigid_ate(poses['ground_truth'], poses[name]) / cfg['data']['sc_factor']
        assert abs(ate - result[field]) < 1e-12, (path, field, ate, result[field])
    with (path / 'frames.csv').open() as stream:
        frames = list(csv.DictReader(stream))
    assert [int(f['frame_id']) for f in frames] == list(range(1, 2000))
    assert all(int(f['has_ba']) == (int(f['frame_id']) % 5 == 0) for f in frames)
    measured = [f for f in frames if int(f['frame_id']) >= 20]
    ba = [f for f in measured if int(f['has_ba'])]
    assert [int(f['frame_id']) for f in ba] == list(range(20, 2000, 5))
    assert len(ba) == 396 and len(measured) - len(ba) == 1584
    values = dict(ba_invocations=[float(f['ba_wall_ms']) for f in ba],
                  ba_frames=[float(f['wall_ms']) for f in ba],
                  tracking_frames=[float(f['wall_ms']) for f in measured if not int(f['has_ba'])])
    for phase, samples in values.items():
        assert np.isfinite(samples).all() and min(samples) > 0
        for metric, value in statistics(samples).items():
            assert abs(value - result[phase][metric]) < 1e-9, (path, phase, metric)
    snapshot = path / 'source_hashes.json'
    if require_snapshot:
        assert snapshot.exists()
    signature = json.loads(snapshot.read_text()) if snapshot.exists() else None
    if signature:
        for name, value in signature.items():
            assert hashlib.sha256((path / 'source' / name).read_bytes()).hexdigest() == value
    return result, cfg, frames, signature


def main():
    assert (OUT / 'SUITE_COMPLETE').exists()
    assert len(list(OUT.glob('*/*/summary.json'))) == 6
    records, signature = [], None
    for scene in SCENES:
        paired = []
        for variant in VARIANTS:
            path = OUT / scene / f'{variant}_seed0'
            result, cfg, frames, hashes = audit_run(path)
            expected = json.loads((META / f'{scene}_{variant}_expected_config.json').read_text())
            assert cfg == expected
            signature = hashes if signature is None else signature
            assert hashes == signature, 'Source changed across matched RTX runs'
            comparable = json.loads(json.dumps(cfg))
            comparable['data'].pop('output')
            comparable['ba_experiment'].pop('variant')
            paired.append(comparable)
            records.append(dict(run=str(path.relative_to(OUT)), frames=2000,
                                ba_call_samples=396, ba_frame_samples=396,
                                pure_tracking_frame_samples=1584,
                                trajectories_ate_timing_configs_source_hashes='passed'))
        assert paired[0] == paired[1]
    runtime = json.loads((META / 'runtime_source_hashes.json').read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
               for name, value in runtime.items())
    (OUT / 'AUDIT.json').write_text(json.dumps(dict(suite_complete=True, completed_runs=6,
        checked_runs=records, source_versions=1, runtime_source_unchanged=True,
        dense_work='43 samples, 2048 mapping rays plus baseline current-frame rays, 10 iterations',
        tracking_variant='baseline', ba_variants=list(VARIANTS)), indent=2) + '\n')
    print('BA_AUDIT_PASSED: six full paired trajectories, ATE/statistics independently recomputed, '
          'dense work and baseline tracking verified, snapshots unchanged.')


if __name__ == '__main__':
    main()
