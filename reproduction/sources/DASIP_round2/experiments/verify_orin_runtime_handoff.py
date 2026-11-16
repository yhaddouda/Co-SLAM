#!/usr/bin/env python3
"""Verify the extracted package, optionally its first four Orin diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    hashes = json.loads((root / 'SHA256SUMS.json').read_text())
    for name, digest in hashes.items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise RuntimeError('Unsafe package path: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Package file differs: ' + name)
    print('PACKAGE_HASHES_OK', len(hashes))
    if args.diagnostics:
        manifest = json.loads((root / 'manifests/orin_coslam_diagnostic.json').read_text())
        directory = args.diagnostics.resolve()
        for job in manifest['jobs']:
            result = json.loads((directory / job['id'] / 'summary.json').read_text())
            assert (directory / (job['id'] + '.done.json')).is_file(), job['id']
            assert result['condition'] == job['condition'] and result['seed'] == job['seed']
            assert result['diagnostic'] and result['frames'] == 50
            assert result['finite_parameters_and_poses'] and not result['quality']
            for stage in ('tracking', 'bundle_adjustment'):
                counts = result[stage]
                assert counts['count'] > 0 and counts['actual_queries'] > 0
                if job['condition'] in ('BASE', 'MC'):
                    assert counts['actual_queries'] == counts['dense_queries']
                else:
                    assert counts['actual_queries'] < counts['dense_queries']
            assert (directory / job['id'] / 'source_manifest.json').read_bytes() == (root / 'runtime/RUNTIME_MANIFEST.json').read_bytes()
            print('DIAGNOSTIC_OK', job['id'])
        print('FIRST_ORIN_DIAGNOSTICS_OK; full-scene quality/speed acceptance remains pending')


if __name__ == '__main__':
    main()
