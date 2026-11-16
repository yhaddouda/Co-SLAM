#!/usr/bin/env python3
"""Restore versioned paper scripts to their original, locally ignored paths."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from preserve_research import safe_path


ROOT = Path(__file__).resolve().parents[2]


def restore(source, destination):
    manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_text())
    pending = []
    for row in manifest['files']:
        original = safe_path(source, row['path'])
        target = safe_path(destination, row['path'])
        if hashlib.sha256(original.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Corrupt versioned source: {row["path"]}')
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Independent local changes; refusing to overwrite {target}')
        else:
            pending.append((original, target))
    for original, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    print(f'Restored {len(pending)} files; {len(manifest["files"]) - len(pending)} already matched.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'reproduction/sources')
    parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    restore(args.source.resolve(), args.destination.resolve())
