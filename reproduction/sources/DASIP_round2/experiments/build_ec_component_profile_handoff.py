#!/usr/bin/env python3
"""Package EC profiler plus the author-approved single-seed Orin override."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from round2_paths import local_path


def main():
    base=Path(__file__).resolve().parents[1]
    runtime=base/'experiments/ec_component_profile_runtime_v2'
    archive=base/'bundles/to_orin/orin_ec_component_profile_runtime_v2_single_seed.zip'
    assert not archive.exists(), 'Preserve transferred archives'
    source=json.loads((runtime/'RUNTIME_MANIFEST.json').read_text())
    entries={}
    for name,expected in source['runtime_hashes'].items():
        raw=(runtime/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected,name
        entries['runtime/'+name]=raw
    entries['runtime/RUNTIME_MANIFEST.json']=(runtime/'RUNTIME_MANIFEST.json').read_bytes()
    for name in ('run_ec_component_profile_campaign.py','verify_ec_component_profile_handoff.py',
                 'build_ec_component_profile_runtime.py','report_component_profiles.py','report_ec_component_profiles.py'):
        entries[name]=(base/'experiments'/name).read_bytes()
    for mode in ('maxn','15w'):
        for prefix in ('orin_component_profile','orin_ec_component_profile'):
            name=f'{prefix}_{mode}_single_seed_v1.json'
            entries['manifests/'+name]=(base/'experiments/manifests'/name).read_bytes()
    entries['README.md']=(base/'experiments/ec_component_profile_handoff/README.md').read_bytes()
    for name in ('EC_COMPONENT_PROFILING_PROTOCOL.md','COMPONENT_PROFILING_PROTOCOL.md','NOTIFICATION_POLICY.md'):
        entries[name]=local_path(base,name).read_bytes()
    validation=base/'results/ec_component_profile_rtx_v2'
    diagnostics=sorted(validation.glob('*_diagnostic/summary.json'))
    assert len(diagnostics)==2
    for path in diagnostics:
        summary=json.loads(path.read_text())
        assert summary['diagnostic'] and summary['frames']==50 and summary['finite_parameters_and_poses']
        assert summary['source_manifest_sha256']==hashlib.sha256(entries['runtime/RUNTIME_MANIFEST.json']).hexdigest()
        proof=path.parent/'fixed_state_validation/PASSED.json'
        assert json.loads(proof.read_text())['status']=='PASSED'
        prefix='RTX_VALIDATION/'+path.parent.name+'/'
        entries[prefix+'summary.json']=path.read_bytes()
        entries[prefix+'fixed_state.json']=proof.read_bytes()
    entries['RTX_VALIDATION/INITIAL_HARNESS_FAILURE.txt']=(base/'results/ec_component_profile_rtx_v1/EC_Replica_office0_H16_BASE_profile_seed0_diagnostic.log').read_bytes()
    entries['BUILD_MANIFEST.json']=(json.dumps(dict(author_date='2026-10-21',system='EC-SLAM',runtime_version=2,
        native_reference_manifest_sha256=source['reference_manifest_sha256'],
        actual_runtime_manifest_sha256=hashlib.sha256(entries['runtime/RUNTIME_MANIFEST.json']).hexdigest(),
        computational_AST_match=True,rtx_diagnostics_passed=2,orin_diagnostics_passed=False,
        orin_remote_installed=False,orin_remote_launched=False,orin_profile_seeds=[0],
        maxn_co_full_profiles=3,maxn_ec_full_profiles=2,power15w_co_full_profiles=3,power15w_ec_full_profiles=2,
        platform_repetitions_override='Latest author instruction: one Orin profiling run/workload; frozen protocol frame lists unchanged',
        original_co_runtime_required='orin_component_profile_runtime_v2.zip',
        active_benchmark_queues_modified=False,mode_switch_automatic=False,
        initial_failure='Validator omitted native post-camera-conversion CUDA transfer; v2 restores this harness transfer only'),indent=2)+'\n').encode()
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(entries.items())}
    entries['SHA256SUMS.json']=(json.dumps(hashes,indent=2)+'\n').encode()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as saved:
        for name,raw in sorted(entries.items()):saved.writestr(name,raw)
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        with tempfile.TemporaryDirectory(prefix='dasip_ec_profile_verify_') as temporary:
            saved.extractall(temporary)
            subprocess.run([sys.executable,str(Path(temporary)/'verify_ec_component_profile_handoff.py')],check=True)
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 rtx_diagnostics_passed=2,remote_installed=False,remote_launched=False,orin_profiles_per_mode=5,orin_profile_seeds=[0])
    (base/'results/EC_COMPONENT_PROFILE_HANDOFF.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
