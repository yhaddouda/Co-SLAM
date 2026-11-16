#!/usr/bin/env python3
"""Compare local configurations with pinned public Git objects, without SLAM imports."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import yaml


REFERENCES = {'Co-SLAM': '7b83b26d5ae24f4baa6c1c684dd745898276816d',
              'EC-SLAM': 'c269de61d7ff4623f3d5584aa47c6261c89dbad6'}


def merge(base, extra):
    result = copy.deepcopy(base)
    for key, value in extra.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        path = prefix + '.' + key if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def audit(root, system):
    revision = REFERENCES[system]
    def public(name):
        return subprocess.check_output(['git', '-C', str(root), 'show', revision + ':' + name])
    def load(name, original, default=None, stack=()):
        if name in stack:
            raise ValueError('Cyclic config inheritance')
        raw = public(name) if original else (root / name).read_bytes()
        special = yaml.safe_load(raw)
        if not isinstance(special, dict):
            raise ValueError('Empty or non-mapping configuration: ' + name)
        parent = special.get('inherit_from')
        base = load(parent, original, default, stack + (name,)) if parent else (
            load(default, original, None, stack + (name,)) if default else {})
        return merge(base, special)
    paths = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only',
                                     revision, 'configs'], text=True).splitlines()
    rows = []
    for name in paths:
        if not name.endswith('.yaml') or not any('/' + d + '/' in name for d in (
                'Replica', 'Tum', 'TUM', 'ScanNet', 'Scannet', 'Synthetic')):
            continue
        default = 'configs/EC_SLAM.yaml' if system == 'EC-SLAM' else None
        try:
            original = load(name, True, default)
            current = load(name, False, default)
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            rows.append(dict(path=name, status='unavailable', reason=str(error)))
            continue
        before, after = flatten(original), flatten(current)
        differences = [dict(key=k, public=before.get(k), local=after.get(k))
                       for k in sorted(before) if k != 'inherit_from' and before[k] != after.get(k)]
        rows.append(dict(path=name, status='compared', differences=differences,
                         local_added_keys=sorted(set(after) - set(before)),
                         public_config=original, local_config=current))
    source = 'model/encodings.py' if system == 'Co-SLAM' else 'src/Model.py'
    return dict(reference=revision, root=str(root), configurations=rows,
                encoder_source=dict(path=source, public_sha256=hashlib.sha256(public(source)).hexdigest(),
                    local_sha256=hashlib.sha256((root / source).read_bytes()).hexdigest()),
                working_revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],text=True).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coslam-root', type=Path, required=True)
    parser.add_argument('--ecslam-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output filename')
    result = {system: audit(root.resolve(), system) for system, root in (
        ('Co-SLAM', args.coslam_root), ('EC-SLAM', args.ecslam_root))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for system, record in result.items():
        rows = record['configurations']
        print(system, sum(r['status']=='compared' for r in rows), 'configs compared;',
              sum(r['status']=='unavailable' for r in rows), 'unavailable')


if __name__ == '__main__':
    main()
