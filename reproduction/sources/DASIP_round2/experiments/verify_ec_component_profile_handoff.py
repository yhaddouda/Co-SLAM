#!/usr/bin/env python3
"""CPU-only EC package/hash/native-computation verification; never loads torch."""
import hashlib
import json
from pathlib import Path
from build_ec_component_profile_runtime import verify_ast


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    hashes=json.loads((root/'SHA256SUMS.json').read_text())
    for name,expected in hashes.items():
        assert sha(root/name)==expected,name
    runtime=root/'runtime'
    manifest=json.loads((runtime/'RUNTIME_MANIFEST.json').read_text())
    for name,expected in manifest['runtime_hashes'].items():
        assert sha(runtime/name)==expected,name
    reference=json.loads((runtime/'REFERENCE_RUNTIME_MANIFEST.json').read_text())
    assert sha(runtime/'REFERENCE_RUNTIME_MANIFEST.json')==manifest['reference_manifest_sha256']
    changed={f'src/{name}.py' for name in ('Tracker','Mapper','Model')}
    for name,expected in reference['runtime_hashes'].items():
        if name in changed:
            assert sha(runtime/'reference'/name)==expected,name
            verify_ast((runtime/'reference'/name).read_text(),(runtime/name).read_text())
        else:
            assert sha(runtime/name)==expected,name
    for mode in ('maxn','15w'):
        for prefix,count in (('orin_component_profile',3),('orin_ec_component_profile',2)):
            spec=json.loads((root/'manifests'/f'{prefix}_{mode}_single_seed_v1.json').read_text())
            assert len(spec['jobs'])==count
            assert all(j['seed']==0 and j['condition']=='BASE' for j in spec['jobs'])
    print('EC_HANDOFF_VERIFIED: package/runtime hashes, native-file parity, three computational ASTs, single-seed Co/EC execution templates; no CUDA initialized')


if __name__=='__main__':main()
