#!/usr/bin/env python3
"""Restore a selected Orin source subset to a fresh destination; dry-run by default."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def safe_destination(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Unsafe path: {relative}')
    target = root
    for part in rel.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f'Symlink: {target}')
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError('Destination must have no symlink or parent traversal')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', choices=['Co-SLAM', 'EC-SLAM', 'coslam_profiling'], required=True)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--shared-repo', type=Path, required=True,
                        help='Git checkout or bare repository containing the fixed RTX reference')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(__file__).with_name('SOURCE_MANIFEST.json').read_text())
    revision = manifest['shared_reference']
    pending = []
    rows = manifest['unique_orin_sources'] + manifest['shared_sources']
    for row in rows:
        if row['root'] != args.root or not row['path'].startswith(args.prefix):
            continue
        if 'stored_path' in row:
            data = (source_root / row['stored_path']).read_bytes()
        else:
            data = subprocess.check_output(['git', '-C', str(args.shared_repo), 'show',
                                            revision + ':' + row['shared_path']])
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Source checksum mismatch: {row["path"]}')
        target = safe_destination(args.destination.absolute(), row['path'])
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise ValueError(f'Different existing file: {target}')
        else:
            pending.append((target, data))
    if not any(r['root'] == args.root and r['path'].startswith(args.prefix) for r in rows):
        raise ValueError('No matching sources')
    if args.apply:
        for target, data in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(data)
    print(json.dumps(dict(mode='applied' if args.apply else 'dry_run',
                          new_files=len(pending), bytes=sum(len(data) for _, data in pending),
                          reference=revision)))


if __name__ == '__main__':
    main()
