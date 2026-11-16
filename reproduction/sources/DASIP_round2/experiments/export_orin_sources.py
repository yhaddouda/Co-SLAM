#!/usr/bin/env python3
"""Export only the previously audited Orin sources; never change repositories."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh filename')
    inventory = json.loads(args.inventory.read_text())
    repos = inventory['repositories']
    locations = [('Co-SLAM', Path(repos['Co-SLAM']['root'])),
                 ('EC-SLAM', Path(repos['EC-SLAM']['root'])),
                 ('EC-execution-workspace', Path(inventory['preflight_audit']['execution_workspace']))]
    locations.sort(key=lambda pair: len(pair[1].parts), reverse=True)
    sources = inventory['preflight_audit']['source_sha256']
    allowed = {'.py', '.yaml', '.cu', '.txt'}
    payloads, changed, missing = [], [], []
    for name, expected in sorted(sources.items()):
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts or path.suffix not in allowed:
            parser.error('Unexpected source path in inventory: ' + name)
        match = next(((tag, root) for tag, root in locations if path.is_relative_to(root)), None)
        if match is None:
            parser.error('Source outside inventoried repositories: ' + name)
        tag, root = match
        if not path.is_file():
            missing.append(name)
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            parser.error('Source symlink leaves its repository: ' + name)
        if path.stat().st_size > 8 * 1024 * 1024:
            parser.error('Unexpectedly large source file: ' + name)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            changed.append(dict(path=name, audited_sha256=expected, current_sha256=actual))
        payloads.append((tag + '/' + path.relative_to(root).as_posix(), data, actual))
    if missing or changed:
        print(json.dumps(dict(missing=missing, changed=changed), indent=2))
        parser.error('Source changed since inventory; no archive written. Return this change report first.')
    manifest = dict(purpose='Audited source handoff only; not an experiment or runtime patch',
                    inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                    files={name: digest for name, _, digest in payloads})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing archive even if another process creates it.
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data, _ in payloads:
            archive.writestr(name, data)
        archive.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name, _, digest in payloads:
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print('Exported', len(payloads), 'audited text source/config files to', args.output)
    print('No repository source, power setting, dependency or result was modified.')


if __name__ == '__main__':
    main()
