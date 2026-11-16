#!/usr/bin/env python3
"""Package the validated profiling runtime and deferred Orin continuation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    base=Path(__file__).resolve().parents[1]
    runtime=base/'experiments/component_profile_runtime_v2'
    archive=base/'bundles/to_orin/orin_component_profile_runtime_v2.zip'
    assert not archive.exists(), 'Preserve an already transferred archive'
    source=json.loads((runtime/'RUNTIME_MANIFEST.json').read_text())
    entries={}
    for name,expected in source['runtime_hashes'].items():
        content=(runtime/name).read_bytes()
        assert hashlib.sha256(content).hexdigest()==expected,name
        entries['runtime/'+name]=content
    entries['runtime/RUNTIME_MANIFEST.json']=(runtime/'RUNTIME_MANIFEST.json').read_bytes()
    for name in ('run_component_profile_campaign.py','verify_component_profile_handoff.py',
                 'build_component_profile_runtime.py','report_component_profiles.py'):
        entries[name]=(base/'experiments'/name).read_bytes()
    for name in ('orin_component_profile_maxn_v1.json','orin_component_profile_15w_v1.json'):
        entries['manifests/'+name]=(base/'experiments/manifests'/name).read_bytes()
    entries['README.md']=(base/'experiments/component_profile_handoff/README.md').read_bytes()
    entries['COMPONENT_PROFILING_PROTOCOL.md']=(base/'docs/protocols/COMPONENT_PROFILING_PROTOCOL.md').read_bytes()
    entries['NOTIFICATION_POLICY.md']=(base/'docs/protocols/NOTIFICATION_POLICY.md').read_bytes()
    validation=base/'results/component_profile_rtx_v2'
    diagnostics=sorted(validation.glob('*_diagnostic/summary.json'))
    assert len(diagnostics)==3
    for path in diagnostics:
        summary=json.loads(path.read_text())
        assert summary['diagnostic'] and summary['frames']==50 and summary['finite_parameters_and_poses']
        assert summary['source_manifest_sha256']==hashlib.sha256(entries['runtime/RUNTIME_MANIFEST.json']).hexdigest()
        evidence=path.parent/'fixed_state_validation/PASSED.json'
        assert json.loads(evidence.read_text())['status']=='PASSED'
        prefix='RTX_VALIDATION/'+path.parent.name+'/'
        entries[prefix+'summary.json']=path.read_bytes()
        entries[prefix+'fixed_state.json']=evidence.read_bytes()
    failed=base/'results/component_profile_rtx_v1'
    entries['RTX_VALIDATION/INITIAL_HARNESS_FAILURE.txt']=(failed/'Replica_office0_H16_BASE_profile_seed0_diagnostic.log').read_bytes()
    entries['BUILD_MANIFEST.json']=(json.dumps(dict(
        author_date='2026-10-21',runtime_version=2,protocol_version=1,
        native_reference_manifest_sha256=source['reference_manifest_sha256'],
        actual_runtime_manifest_sha256=hashlib.sha256(entries['runtime/RUNTIME_MANIFEST.json']).hexdigest(),
        computational_AST_match=True,rtx_diagnostics_passed=3,
        orin_diagnostics_passed=False,orin_remote_installed=False,orin_remote_launched=False,
        maxn_full_profiles=6,power15w_full_profiles=6,
        active_large_queues_modified=False,mode_switch_automatic=False,
        initial_failure='V1 diagnostic harness reduced max over an empty parameter; V2 skips empty tensors without changing estimator/timing computations'
    ),indent=2)+'\n').encode()
    hashes={name:hashlib.sha256(content).hexdigest() for name,content in sorted(entries.items())}
    entries['SHA256SUMS.json']=(json.dumps(hashes,indent=2)+'\n').encode()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as saved:
        for name,content in sorted(entries.items()):saved.writestr(name,content)
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        assert all(hashlib.sha256(saved.read(name)).hexdigest()==expected for name,expected in hashes.items())
        with tempfile.TemporaryDirectory(prefix='dasip_profile_verify_') as temporary:
            saved.extractall(temporary)
            subprocess.run([sys.executable,str(Path(temporary)/'verify_component_profile_handoff.py')],check=True)
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 runtime_version=2,rtx_diagnostics_passed=3,remote_installed=False,remote_launched=False)
    (base/'results/COMPONENT_PROFILE_HANDOFF.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
