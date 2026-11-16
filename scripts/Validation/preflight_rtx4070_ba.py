"""Validate the dense BA comparison against the packaged Orin evidence."""
import ast
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
import numpy as np
from datasets.dataset import get_dataset
from scripts.Validation.preflight_rtx4070_tracking import ConfigReference, differences, function

STAGE = ROOT / 'transfer/rtx4070/staging/coslam_experiments'
ORIN = STAGE / 'results/ba_experiments/full'
META = ROOT / 'results/ba_experiments/rtx4070_metadata'
OUTPUT = ROOT / 'results/ba_experiments/rtx4070_dense_fast_full'
SCENES = ('office1', 'room1', 'office0')
VARIANTS = ('baseline', 'baseline_fast')


def resolved(scene, variant):
    cfg = config.load_config(f'configs/Replica/{scene}.yaml')
    assert cfg.get('tracking_experiment', {}).get('variant', 'baseline') == 'baseline'
    for name in ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'):
        cfg.setdefault(name, {})['enabled'] = False
    cfg['tracking_depth_spatial'].update(collect_sample_count_stats=False,
                                       collect_fallback_stats=False, stats_only=False)
    cfg['training']['uniform_samples_until_depth'] = False
    cfg['ba_bucket'].update(enabled=False, debug_log=False, morton_inside_bucket=False)
    cfg['ba_experiment'] = dict(variant=variant, margin=0., collect_samples=True)
    cfg['grid'].update(hash='CoherentPrime', hash_size=19, morton_sort=False)
    cfg.setdefault('timing', {}).update(mode='none', max_frames=None, disable_eval=True)
    cfg.setdefault('profiling', {})['enabled'] = False
    cfg['mapping']['first_mesh'] = False
    cfg['mesh'].update(visualisation=False, vis=1000000000)
    cfg['data'].update(output=str(OUTPUT / scene / f'{variant}_seed0'), exp_name='artifacts')
    return cfg


def main():
    META.mkdir(parents=True, exist_ok=True)
    reference = ORIN / 'office1/baseline_fast_seed0/source/coslam.py'
    assert ast.dump(ConfigReference().visit(function(reference, 'CoSLAM', 'global_BA')),
                    include_attributes=False) == ast.dump(
        ConfigReference().visit(function(ROOT / 'coslam.py', 'CoSLAM', 'global_BA')),
        include_attributes=False)
    allowed = {'.data.datadir', '.data.output', '.profiling.output_dir',
               '.timing.coslam_output_csv', '.timing.scene_output_csv',
               '.timing.frame_total_output_csv', '.timing.frame_total_write_header',
               '.timing.warmup_frames', '.ba_experiment.collect_samples'}
    records, hashes, unavailable = [], {}, []
    for scene in SCENES:
        for variant in VARIANTS:
            reference_dir = ORIN / scene / f'{variant}_seed0'
            cfg = resolved(scene, variant)
            expected = json.loads((reference_dir / 'config.json').read_text())
            raw_changes = differences(cfg, expected)
            normalized = json.loads(json.dumps(cfg))
            normalized.pop('tracking_experiment', None)
            expected.pop('tracking_experiment', None)
            changes = differences(normalized, expected)
            assert all(change['path'] in allowed for change in changes), changes
            assert cfg['mapping']['iters'] % cfg['mapping']['pose_accum_step'] == 0
            (META / f'{scene}_{variant}_expected_config.json').write_text(json.dumps(cfg, indent=2))
            records.append(dict(scene=scene, variant=variant, config_differences=raw_changes,
                                active_config_matches_orin=True, tracking_variant='baseline'))
            signature_path = reference_dir / 'source_hashes.json'
            if not signature_path.exists():
                unavailable.append(str(reference_dir.relative_to(STAGE)))
                continue
            signature = json.loads(signature_path.read_text())
            for name, value in signature.items():
                local = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                hashes.setdefault(name, []).append(dict(scene=scene, variant=variant, orin=value,
                                                       rtx=local, identical=local == value))
                if local != value:
                    patch = ''.join(difflib.unified_diff(
                        (reference_dir / 'source' / name).read_text().splitlines(True),
                        (ROOT / name).read_text().splitlines(True), fromfile='Orin/' + name,
                        tofile='RTX/' + name))
                    (META / f'{scene}_{variant}_{name.replace("/", "__")}.diff').write_text(patch)
        dataset = get_dataset(resolved(scene, 'baseline'))
        assert len(dataset) == len(dataset.depth_paths) == len(dataset.poses) == 2000
        truth = np.stack([p.numpy() for p in dataset.poses])
        reference_truth = np.load(ORIN / scene / 'baseline_seed0/trajectories.npz')['ground_truth']
        assert np.array_equal(truth, reference_truth)
        assert all(Path(p).stat().st_size > 0 for p in dataset.img_files + dataset.depth_paths)
    same = {name for name, groups in hashes.items() if all(group['identical'] for group in groups)}
    fast_same = {name for name, groups in hashes.items()
                 if all(group['identical'] for group in groups if group['variant'] == 'baseline_fast')}
    assert {'optimization/ba_experiments.py', 'scripts/Validation/benchmark_replica_ba.py',
            'scripts/Validation/rank_replica_ba_scenes.py', 'model/utils.py',
            'model/keyframe.py', 'optimization/utils.py'} <= fast_same
    commit = subprocess.check_output(['git', '-C', 'tiny-cuda-nn', 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == json.loads((STAGE / 'RTX_TRANSFER_MANIFEST.json').read_text())['tiny_cuda_nn_source_commit']
    result = dict(configurations=records, source_comparison=hashes,
                  ba_body_matches_orin_after_normalizing_local_config_references=True,
                  matching_source_files=sorted(same), tiny_cuda_nn_source_commit=commit,
                  matching_dense_fast_source_files=sorted(fast_same),
                  orin_runs_without_source_snapshots=unavailable,
                  runtime_differences='coslam.py has later optional tracking paths and retained local guards; '
                  'scene_rep.py has an optional depth cache. Tracking execution flags and depth cache stay off.',
                  datasets='Three complete 2000-frame sequences; loaded ground truth exactly matches Orin',
                  planned_order=[f'{scene}/{variant}' for variant in VARIANTS for scene in SCENES],
                  order_difference='Matches baseline-then-fast grouping; intervening Orin bucket/lattice runs omitted.')
    (META / 'PREFLIGHT.json').write_text(json.dumps(result, indent=2) + '\n')
    print('BA_PREFLIGHT_PASSED: exact helper/harness match, normalized BA AST match, '
          'active configs matched, baseline tracking, full datasets.')


if __name__ == '__main__':
    main()
