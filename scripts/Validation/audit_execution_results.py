"""Audit saved trajectories, timing samples, source snapshots and paired configs."""
import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.Validation.execution_benchmark_common import rigid_ate


def comparable_config(config):
    result = json.loads(json.dumps(config))
    result.pop('tracking_experiment', None)
    result.pop('mapping_experiment', None)
    result['data'].pop('output', None)
    return result


def ba_body(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CoSLAM')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'global_BA')
    return ast.dump(node, include_attributes=False)


def audit(root, require_complete):
    completed = root / 'SUITE_COMPLETE'
    if require_complete and not completed.exists():
        raise ValueError(f'Suite incomplete: {root}')
    summaries = {path.parent: json.loads(path.read_text()) for path in root.glob('*/*/summary.json')}
    if completed.exists():
        expected = json.loads(completed.read_text())
        for scene in expected['scenes']:
            for variant in expected['variants']:
                for seed in expected['seeds']:
                    path = root / scene / f'{variant}_seed{seed}'
                    assert path in summaries, path
                    assert summaries[path]['frames'] == expected['frames'], path
    baselines = {(s['scene'], s['seed']): path for path, s in summaries.items() if s['variant'] == 'baseline'}
    rows, sources = [], {}
    for path, summary in sorted(summaries.items()):
        assert not summary['diagnostic'], path
        assert summary['finite_parameters_and_poses'], path
        config = json.loads((path / 'config.json').read_text())
        baseline = baselines.get((summary['scene'], summary['seed']))
        if baseline:
            reference = json.loads((baseline / 'config.json').read_text())
            assert comparable_config(config) == comparable_config(reference), path
        if summary['system'] == 'coslam':
            assert not config['ba_bucket']['enabled'], path
            assert config['ba_experiment']['variant'] == 'baseline', path
            assert all(not config[name]['enabled'] for name in
                       ('tracking_bucket', 'tracking_depth_spatial', 'tracking_naive_pruning')), path
            assert not config['grid']['morton_sort'], path
            assert summary['mapping_variant'] == 'baseline', path
            assert ba_body(path / 'source/coslam.py') == ba_body(ROOT / 'coslam.py'), path
            scale = config['data']['sc_factor']
        else:
            assert not config['HashGrid']['morton_sort'], path
            scale = config['scale']
        trajectories = np.load(path / 'trajectories.npz')
        for name in ('ground_truth', 'estimated', 'relative'):
            assert trajectories[name].shape == (summary['frames'], 4, 4), (path, name)
            assert np.isfinite(trajectories[name]).all(), (path, name)
        for name, key in (('estimated', 'ate_rmse_m'), ('relative', 'ate_relative_rmse_m')):
            value = rigid_ate(trajectories['ground_truth'], trajectories[name]) / scale
            assert abs(value - summary[key]) < 1e-12, (path, name, value, summary[key])
        with (path / 'frames.csv').open() as stream:
            records = list(csv.DictReader(stream))
        warmup = summary['warmup_frames']
        measured = [r for r in records if int(r['frame_id']) >= warmup]
        frame_ids = [int(r['frame_id']) for r in measured]
        assert frame_ids == list(range(warmup, summary['frames'])), path
        tracking_count = summary['tracking_invocations']['count']
        assert tracking_count == len(measured), path
        assert all(float(r['tracking_wall_ms']) > 0 for r in measured), path
        if summary['system'] == 'coslam':
            count = sum(not int(r['has_mapping']) for r in measured)
            assert summary['tracking_frames']['count'] == count, path
            timing_values = dict(
                tracking_invocations=[float(r['tracking_wall_ms']) for r in measured],
                tracking_frames=[float(r['wall_ms']) for r in measured if not int(r['has_mapping'])])
        else:
            count = sum(int(r['has_mapping']) for r in measured)
            assert summary['tracking_frames']['count'] == len(measured), path
            assert summary['mapping_invocations']['count'] == count, path
            assert summary['mapping_frames']['count'] == count, path
            assert all(float(r['mapping_wall_ms']) > 0 for r in measured if int(r['has_mapping'])), path
            timing_values = dict(
                tracking_invocations=[float(r['tracking_wall_ms']) for r in measured],
                tracking_frames=[float(r['tracking_frame_wall_ms']) for r in measured],
                mapping_invocations=[float(r['mapping_wall_ms']) for r in measured if int(r['has_mapping'])],
                mapping_frames=[float(r['mapping_frame_wall_ms']) for r in measured if int(r['has_mapping'])])
        for phase, values in timing_values.items():
            data = np.asarray(values, dtype=np.float64)
            assert np.isfinite(data).all() and (data > 0).all(), (path, phase)
            recomputed = dict(mean_ms=float(data.mean()), median_ms=float(np.median(data)),
                              p95_ms=float(np.percentile(data, 95)), std_ms=float(data.std()))
            assert len(data) == summary[phase]['count'], (path, phase)
            for metric, value in recomputed.items():
                assert abs(value - summary[phase][metric]) < 1e-9, (path, phase, metric)
        manifest = json.loads((path / 'source_hashes.json').read_text())
        for name, value in manifest.items():
            actual = hashlib.sha256((path / 'source' / name).read_bytes()).hexdigest()
            assert actual == value, (path, name)
            sources.setdefault(name, {}).setdefault(value, []).append(str(path.relative_to(root)))
        rows.append(dict(run=str(path.relative_to(root)), frames=summary['frames'],
                         paired_config_checked=baseline is not None,
                         trajectories_ate_timing_statistics_configs_and_source_hashes='passed'))
    source_changes = {name: groups for name, groups in sources.items() if len(groups) > 1}
    result = dict(root=str(root), suite_complete=completed.exists(), completed_runs=len(rows),
                  checked_runs=rows, differing_source_snapshots=source_changes)
    (root / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    return dict(root=str(root), completed_runs=len(rows), source_files_with_multiple_versions=list(source_changes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    for root in args.roots:
        print(json.dumps(audit(root.resolve(), args.require_complete)), flush=True)


if __name__ == '__main__':
    main()
