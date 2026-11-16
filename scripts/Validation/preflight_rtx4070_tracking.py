"""Record the local handoff reconciliation before the RTX tracking suite."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
import cv2
import numpy as np
from datasets.dataset import get_dataset

STAGE = ROOT / 'transfer/rtx4070/staging/coslam_experiments'
OUT = ROOT / 'results/execution_experiments/rtx4070_metadata'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((differences(a.get(k, '<absent>'), b.get(k, '<absent>'),
                                f'{prefix}.{k}') for k in sorted(a.keys() | b.keys())), [])
    return [] if a == b else [dict(path=prefix, rtx=a, orin=b)]


def resolved(scene):
    # These overrides mirror the unchanged packaged tracking harness.
    cfg = config.load_config(f'configs/Replica/{scene}.yaml')
    for name in ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'):
        cfg.setdefault(name, {})['enabled'] = False
    cfg['tracking_depth_spatial'].update(collect_sample_count_stats=False,
                                       collect_fallback_stats=False, stats_only=False)
    cfg['training']['uniform_samples_until_depth'] = False
    cfg['ba_bucket'].update(enabled=False, debug_log=False, morton_inside_bucket=False)
    cfg['ba_experiment'] = dict(variant='baseline', margin=0., collect_samples=False)
    cfg['tracking_experiment'] = dict(variant='baseline')
    cfg['grid'].update(hash='CoherentPrime', hash_size=19, morton_sort=False)
    cfg.setdefault('timing', {}).update(mode='none', max_frames=None, disable_eval=True)
    cfg.setdefault('profiling', {})['enabled'] = False
    cfg['mapping']['first_mesh'] = False
    cfg['mesh'].update(visualisation=False, vis=1000000000)
    cfg['data'].update(output=str(ROOT / 'results/execution_experiments/coslam/rtx4070_full'
                                  / scene / 'baseline_seed0'), exp_name='artifacts')
    return cfg


def function(source, cls, name):
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)


class ConfigReference(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'self' and node.attr == 'config':
            return ast.copy_location(ast.Name(id='cfg', ctx=ast.Load()), node)
        return self.generic_visit(node)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = []
    allowed = {'.data.datadir', '.data.output', '.profiling.output_dir',
               '.timing.coslam_output_csv', '.timing.scene_output_csv',
               '.timing.frame_total_output_csv', '.timing.frame_total_write_header',
               '.timing.warmup_frames'}
    for scene in ('office1', 'room1', 'office0'):
        cfg = resolved(scene)
        reference = ROOT / f'results/execution_experiments/coslam/full/{scene}/baseline_seed0'
        delta = differences(cfg, json.loads((reference / 'config.json').read_text()))
        assert all(d['path'] in allowed for d in delta), delta
        assert cfg['timing']['mode'] == 'none' and not cfg['profiling']['enabled']
        dataset = get_dataset(cfg)
        assert len(dataset) == len(dataset.depth_paths) == len(dataset.poses) == 2000
        directory = Path(cfg['data']['datadir']).resolve()
        inventory = []
        for i, (rgb, depth) in enumerate(zip(dataset.img_files, dataset.depth_paths)):
            assert Path(rgb).stem == f'frame{i:06d}'
            assert Path(depth).stem == f'depth{i:06d}'
            for file in (Path(rgb).resolve(), Path(depth).resolve()):
                assert file.stat().st_size > 0
                inventory.append(dict(path=str(file.relative_to(directory)), bytes=file.stat().st_size))
        shapes = {}
        for i in (0, 999, 1999):
            rgb = cv2.imread(dataset.img_files[i])
            depth = cv2.imread(dataset.depth_paths[i], cv2.IMREAD_UNCHANGED)
            assert rgb is not None and depth is not None and rgb.shape[:2] == depth.shape
            sample = dataset[i]
            shapes[str(i)] = dict(raw_rgb=list(rgb.shape), raw_depth=list(depth.shape),
                                 loaded_rgb=list(sample['rgb'].shape))
        truth = np.stack([pose.numpy() for pose in dataset.poses])
        orin_truth = np.load(reference / 'trajectories.npz')['ground_truth']
        assert np.isfinite(truth).all() and np.array_equal(truth, orin_truth)
        (OUT / f'{scene}_dataset_inventory.json').write_text(json.dumps(inventory, indent=2))
        (OUT / f'{scene}_expected_config.json').write_text(json.dumps(cfg, indent=2))
        scenes.append(dict(scene=scene, directory=str(directory), rgb_frames=2000,
                           depth_frames=2000, poses=2000, decoded_sample_shapes=shapes,
                           trajectory_sha256=digest(directory / 'traj.txt'),
                           ground_truth_matches_orin_exactly=True, config_differences=delta,
                           image_content_identity_scope='All filenames/sizes validated; three frames decoded. '
                           'The Orin archive contains no image hashes, so RGB/depth byte identity is unverified.'))
    source_hashes = json.loads((ROOT / 'results/execution_experiments/coslam/full/office1'
                                '/baseline_seed0/source_hashes.json').read_text())
    source_comparison = {name: dict(orin=value, rtx=digest(ROOT / name),
                                   identical=value == digest(ROOT / name))
                         for name, value in source_hashes.items()}
    assert [name for name, value in source_comparison.items() if not value['identical']] == ['coslam.py']
    ba_local = function(ROOT / 'coslam.py', 'CoSLAM', 'global_BA')
    ba_orin = function(STAGE / 'coslam.py', 'CoSLAM', 'global_BA')
    assert ast.dump(ConfigReference().visit(ba_local), include_attributes=False) == ast.dump(
        ConfigReference().visit(ba_orin), include_attributes=False)
    replica_local = function(ROOT / 'datasets/dataset.py', 'ReplicaDataset', '__getitem__')
    replica_orin = function(STAGE / 'datasets/dataset.py', 'ReplicaDataset', '__getitem__')
    assert ast.dump(replica_local, include_attributes=False) == ast.dump(replica_orin, include_attributes=False)
    commit = subprocess.check_output(['git', '-C', 'tiny-cuda-nn', 'rev-parse', 'HEAD'], text=True).strip()
    expected = json.loads((STAGE / 'RTX_TRANSFER_MANIFEST.json').read_text())['tiny_cuda_nn_source_commit']
    assert commit == expected
    result = dict(scenes=scenes, source_comparison=source_comparison,
                  dense_ba_ast_matches_orin_after_normalizing_retained_config_reference=True,
                  replica_loading_ast_matches_orin=True, tiny_cuda_nn_source_commit=commit,
                  timing_warmup='20 frames in the benchmark harness; inactive inner timer settings differ',
                  local_configurations_preserved=True)
    (OUT / 'PREFLIGHT.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PREFLIGHT_PASSED: 3 x 2000 RGB/depth/pose sequences; ground truth matches Orin; '
          'only inactive configuration fields and preserved runtime guards differ.')


if __name__ == '__main__':
    main()
