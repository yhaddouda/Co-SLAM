#!/usr/bin/env python3
"""Read-only Round 2 inventory. Does not import SLAM, run CUDA, or change modes."""
import argparse
import datetime
import hashlib
try:
    from importlib import metadata
except ImportError:
    metadata = None
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def command(argv, timeout=20):
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=timeout)
        return dict(argv=argv, returncode=result.returncode,
                    stdout=result.stdout[:20000], stderr=result.stderr[:4000])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=argv, error=str(error))


def inspect_repo(root, kind):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError('Repository directory does not exist: ' + str(root))
    required = 'coslam.py' if kind == 'Co-SLAM' else 'src/Model.py'
    if not (root / required).is_file():
        raise ValueError('Not the expected ' + kind + ' checkout: ' + str(root))
    common = ['configs/Replica/replica.yaml', 'configs/ScanNet/scannet.yaml',
              'configs/Tum/tum.yaml', 'configs/Synthetic/synthetic.yaml']
    files = (['coslam.py', 'model/encodings.py', 'model/scene_rep.py',
              'model/keyframe.py', 'model/utils.py', 'model/layer_timing.py',
              'optimization/ba_experiments.py', 'optimization/sample_pruning.py']
             if kind == 'Co-SLAM' else
             ['run.py', 'src/Model.py', 'src/system.py', 'src/Tracker.py',
              'src/Mapper.py', 'configs/EC_SLAM.yaml']) + common
    records = {}
    pattern = re.compile(r'dtype\s*=|hash_size\s*:|n_pixels\s*:|'
                         r'pixels_bg\s*:|keyframe_every\s*:|'
                         r'keyframe.*(?:ratio|fraction)|(?:first_)?iters\s*:')
    for name in files:
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        lines = data.decode('utf-8', errors='replace').splitlines()
        records[name] = dict(sha256=hashlib.sha256(data).hexdigest(),
                            settings_to_audit=[dict(line=i, text=line.strip())
                              for i, line in enumerate(lines, 1)
                              if pattern.search(line)])
    git = lambda args: command(['git', '-C', str(root)] + args)
    return dict(root=str(root), revision=git(['rev-parse', 'HEAD']),
                status=git(['status', '--porcelain', '--untracked-files=no']),
                tracked_delta=git(['diff', '--stat', 'HEAD']), files=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coslam-root', type=Path, required=True)
    parser.add_argument('--ecslam-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh filename to preserve the inventory')
    try:
        repos = {'Co-SLAM': inspect_repo(args.coslam_root, 'Co-SLAM'),
                 'EC-SLAM': inspect_repo(args.ecslam_root, 'EC-SLAM')}
    except ValueError as error:
        parser.error(str(error))
    packages = {}
    for name in ('torch', 'numpy', 'tinycudann', 'open3d', 'PyYAML'):
        if metadata is None:
            packages[name] = 'Package metadata unavailable in this Python; audit separately'
            continue
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = 'No distribution metadata found; runtime/build audit required'
    commands = {}
    for name, options in [('nvpmodel', ['-q']), ('jetson_clocks', ['--show']),
                          ('nvcc', ['--version']), ('nvidia-smi', ['-L'])]:
        path = shutil.which(name)
        if not path and Path('/usr/sbin', name).is_file():
            path = str(Path('/usr/sbin', name))
        commands[name] = command([path] + options) if path else dict(available=False)
    release = Path('/etc/nv_tegra_release')
    payload = dict(schema=1, purpose='Read-only preflight inventory; not an accepted baseline',
                   utc_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   python=sys.version, executable=sys.executable,
                   platform=platform.platform(), machine=platform.machine(),
                   jetson_release=release.read_text() if release.is_file() else None,
                   packages=packages, commands=commands, repositories=repos,
                   limitations=['Static dtype settings do not prove runtime/internal precision.',
                                'No SLAM imports, benchmarks or power-mode changes were performed.',
                                'Native merged configs and full dirty-source snapshots require the next audit.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print('Wrote read-only inventory:', args.output)


if __name__ == '__main__':
    main()
