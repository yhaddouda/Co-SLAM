"""Copy EC-SLAM's current code into a writable experimental workspace.

The adjacent repository and its uncommitted edits are preserved. Dataset files
are referenced through an absolute symlink, not copied or modified.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'EC-SLAM')
    parser.add_argument('--destination', type=Path, default=ROOT / 'experiments/ec_slam_execution')
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination.exists():
        raise ValueError(f'Refusing to overwrite an existing workspace: {destination}')
    destination.mkdir(parents=True)
    for name in ('src', 'configs'):
        shutil.copytree(source / name, destination / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so'))
    for name in ('run.py', 'readme.md', 'requirements.txt', 'requirements_safe_orin.txt'):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    (destination / 'datasets').symlink_to(source / 'datasets', target_is_directory=True)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ('src', 'configs') for p in (source / folder).rglob('*')
              if p.is_file() and p.suffix in {'.py', '.yaml', '.cu'}}
    (destination / 'UPSTREAM_SNAPSHOT.json').write_text(json.dumps(
        {'source': str(source), 'sha256': hashes}, indent=2) + '\n')
    print(f'Prepared {destination}; original source retained at {source}')


if __name__ == '__main__':
    main()
