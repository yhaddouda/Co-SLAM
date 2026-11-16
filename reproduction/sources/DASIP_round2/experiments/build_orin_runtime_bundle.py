#!/usr/bin/env python3
"""Package the frozen, validated runtime without binaries or local environments."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    base = Path(__file__).resolve().parent
    runtime = base / 'coslam_reference_runtime_v1'
    manifest = json.loads((runtime / 'RUNTIME_MANIFEST.json').read_text())
    entries = {}
    for name, digest in manifest['runtime_hashes'].items():
        data = (runtime / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        entries['runtime/' + name] = data
    entries['runtime/RUNTIME_MANIFEST.json'] = (runtime / 'RUNTIME_MANIFEST.json').read_bytes()
    for source, target in [
        ('ORIN_RUNTIME_V1_README.md', 'README.md'),
        ('verify_orin_runtime_handoff.py', 'verify_handoff.py'),
        ('run_manifest_queue.py', 'run_manifest_queue.py'),
        ('manifests/orin_coslam_diagnostic.json', 'manifests/orin_coslam_diagnostic.json'),
        ('manifests/orin_coslam_first_full.json', 'manifests/orin_coslam_first_full.json'),
        ('validation_reference_runtime_v1/VALIDATION.json', 'rtx_validation/VALIDATION.json'),
        ('validation_reference_runtime_v1/support_cpu_regression.log', 'rtx_validation/support_cpu_regression.log'),
    ]:
        entries[target] = (base / source).read_bytes()
    for path in sorted((base / 'validation_reference_runtime_v1').glob('*/summary.json')):
        entries['rtx_validation/' + path.parent.name + '/summary.json'] = path.read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items())}
    entries['SHA256SUMS.json'] = (json.dumps(hashes, indent=2) + '\n').encode()
    archive = base.parent / 'bundles/to_orin/orin_coslam_runtime_v1.zip'
    if archive.exists():
        raise FileExistsError('Preserve existing bundle: ' + str(archive))
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for name, data in sorted(entries.items()):
            output.writestr(name, data)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(check.read(name)).hexdigest() == digest
    print(json.dumps(dict(archive=str(archive), files=len(entries), bytes=archive.stat().st_size,
                          sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
