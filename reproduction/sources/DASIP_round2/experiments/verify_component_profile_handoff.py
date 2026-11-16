#!/usr/bin/env python3
"""Verify a profiling handoff on CPU; no CUDA initialization or dataset reads."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root=Path(__file__).resolve().parent
    for name,expected in json.loads((root/'SHA256SUMS.json').read_text()).items():
        path=(root/name).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    runtime=root/'runtime'
    manifest=json.loads((runtime/'RUNTIME_MANIFEST.json').read_text())
    for name,expected in manifest['runtime_hashes'].items():
        assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==expected,name
    sys.path.insert(0,str(root))
    from build_component_profile_runtime import verify_ast
    for name in ('coslam.py','model/scene_rep.py'):
        verify_ast((runtime/'reference'/name).read_text(),(runtime/name).read_text())
    reference=json.loads((runtime/'REFERENCE_RUNTIME_MANIFEST.json').read_text())
    assert hashlib.sha256((runtime/'REFERENCE_RUNTIME_MANIFEST.json').read_bytes()).hexdigest()==manifest['reference_manifest_sha256']
    for name,expected in reference['runtime_hashes'].items():
        path=runtime/('reference/'+name if name in ('coslam.py','model/scene_rep.py') else name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    print('PACKAGE_RUNTIME_HASHES_AND_COMPUTATIONAL_AST_OK; GPU validation remains local')


if __name__=='__main__':main()
