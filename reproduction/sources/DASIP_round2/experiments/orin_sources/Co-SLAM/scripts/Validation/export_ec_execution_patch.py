"""Export and dry-run a patch against the recorded, unchanged EC source snapshot."""
import difflib
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / 'experiments/ec_slam_execution'


def main():
    snapshot = json.loads((WORKSPACE / 'UPSTREAM_SNAPSHOT.json').read_text())
    source = Path(snapshot['source'])
    names = ['src/Tracker.py', 'src/Mapper.py', 'configs/Replica/replica.yaml',
             'src/execution_experiments.py']
    changes = []
    for name in names:
        original = source / name
        if name in snapshot['sha256']:
            actual = hashlib.sha256(original.read_bytes()).hexdigest()
            if actual != snapshot['sha256'][name]:
                raise ValueError(f'Original changed since snapshot: {original}; do not overwrite it')
            before = original.read_text().splitlines(keepends=True)
            fromfile = 'a/' + name
        else:
            if original.exists():
                raise ValueError(f'New file now exists upstream: {original}')
            before, fromfile = [], '/dev/null'
        after = (WORKSPACE / name).read_text().splitlines(keepends=True)
        changes.extend(difflib.unified_diff(before, after, fromfile=fromfile, tofile='b/' + name))
    output = WORKSPACE / 'execution_variants.patch'
    output.write_text(''.join(changes))
    with tempfile.TemporaryDirectory(prefix='ec-execution-patch-', dir='/tmp') as directory:
        for name in names:
            if (source / name).exists():
                target = Path(directory) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
        subprocess.run(['git', 'apply', '--check', str(output)], cwd=directory, check=True)
    print(f'Exported and dry-run checked {output}; original EC repository untouched.')


if __name__ == '__main__':
    main()
