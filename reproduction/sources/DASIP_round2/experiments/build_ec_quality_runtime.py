#!/usr/bin/env python3
"""Freeze an isolated, byte-identical public EC source plus a quality-only driver."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    here = Path(__file__).resolve().parent
    source = here/'ec_public_reference_source_v1'
    target = here/'ec_quality_runtime_v1'
    if target.exists():
        raise RuntimeError('Do not overwrite a frozen runtime')
    shutil.copytree(source, target)
    shutil.copy2(here/'run_ec_quality_round2.py', target/'run_ec_quality_round2.py')
    shutil.copy2(here/'coslam_reference_runtime_v1/model/round2_morton/morton_cuda.cu', target/'morton_cuda.cu')
    for p in source.rglob('*'):
        if p.is_file():
            assert p.read_bytes() == (target/p.relative_to(source)).read_bytes()
    hashes = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob('*')) if p.is_file()}
    manifest = dict(public_revision='c269de61d7ff4623f3d5584aa47c6261c89dbad6', runtime_hashes=hashes, native_files_byte_identical=True, purpose='quality only; no whole-call tracking/BA timing claim', declared_changes=['seed Python, NumPy and Torch once before construction', 'explicit data/output paths and labelled H16/H19 capacity', 'BASE uses unchanged native source and implicit native default hash', 'MC replaces encoder hash with Morton and reorders decoder points with inverse permutation; FP32 request retained', 'diagnostic only: cap dataset length to 50; full runs untruncated', 'fail process on worker exception instead of leaving native peer hung', 'post-SLAM raw/corrected poses, independent rigid ATE, field checkpoint and mesh export'], preserves=['native mapping/tracking threads and handshakes', 'native ray/lattice/feature/keyframe sampling, losses and pose updates', 'native dataset and all data-loader worker counts', 'native dense feature smoothness', 'native periodic and final pose evaluation'])
    (target/'RUNTIME_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
