#!/usr/bin/env python3
"""Supply an explicit large continuation list and the unchanged required runtime."""
import hashlib
import json
from pathlib import Path
import zipfile
from round2_paths import local_path


def main():
    root = Path(__file__).resolve().parents[1]
    replica = ['office0', 'room0', 'office1', 'office2', 'office3', 'office4', 'room1', 'room2']
    co, ec, eng = [], [], []
    for capacity in (16, 19):
        for scene in replica:
            for seed in (0, 1):
                for condition in (['BASE', 'MC'] if seed==0 else ['MC', 'BASE']):
                    co.append(dict(id=f'Replica_{scene}_{condition}_H{capacity}_seed{seed}', dataset='Replica', scene=scene, datadir=f'/home/yh279050/work/data/Replica/{scene}', condition=condition, seed=seed, hash_size=capacity, quality=False, export_mesh=False, frames=0))
                    data_scene = scene[:-1]+'_'+scene[-1]
                    ec.append(dict(id=f'EC_Replica_{scene}_{condition}_H{capacity}_seed{seed}', system='EC-SLAM', dataset='Replica', scene=scene, datadir=f'/home/yh279050/work/EC-SLAM/datasets/vmap/{data_scene}/imap/00', condition=condition, seed=seed, hash_size=capacity, quality=False, export_mesh=False, frames=0))
    inventory=json.loads((root/'experiments/ALL_CONFIGURED_NONREPLICA_INVENTORY.json').read_text())
    for item in inventory['scenes']:
        dataset,scene,capacity=item['dataset'],item['scene'],item['native_hash_size']
        folder={'Synthetic':'neural_rgbd_data','Tum':'TUM','ScanNet':'scans'}[dataset]
        subdir=Path(item['datadir']).name
        for seed in (0, 1):
            for condition in (['BASE','MC'] if seed==0 else ['MC','BASE']):
                co.append(dict(id=f'{dataset}_{scene}_{condition}_H{capacity}_seed{seed}', dataset=dataset, scene=scene, datadir=f'/home/yh279050/work/data/{folder}/{subdir}', condition=condition, seed=seed, hash_size=capacity, quality=False, export_mesh=False, frames=0, expected_native_frames=item['frames']))
    for system in ('Co-SLAM',):
        for seed in (0, 1):
            eng.append(dict(id=f'{system}_office0_ENG_H16_seed{seed}',system=system,dataset='Replica',scene='office0',condition='ENG',hash_size=16,seed=seed,requires='Validated isolated engineering implementation and equivalent two-seed BASE controls; do not send ENG to Co scientific runner'))
    assert len(co)==128 and len(ec)==64 and len(eng)==2
    entries = {}
    def add(name, content):
        entries[name] = content if isinstance(content, bytes) else content.encode()
    for name, jobs in [('orin_continuous_coslam', co), ('orin_continuous_ec', ec), ('orin_pending_engineering_coslam_only', eng)]:
        content = json.dumps(dict(purpose='Full-sequence timing only, every ready block in current author-selected verified mode; reconcile completed identities before launch', data_paths='Hints only: verify actual local contents and materialize resolved manifests', warmup_frames=20, seeds=[0,1], jobs=jobs),indent=2)+'\n'
        (root/'experiments/manifests'/f'{name}.json').write_text(content)
        add('manifests/'+name+'.json',content)
    for name in ('ORIN_CONTINUOUS_PROMPT.md','CONTINUOUS_RUN_AGENDA.md','NOTIFICATION_POLICY.md'):
        add(name,local_path(root,name).read_bytes())
    add('ALL_CONFIGURED_NONREPLICA_INVENTORY.json',(root/'experiments/ALL_CONFIGURED_NONREPLICA_INVENTORY.json').read_bytes())
    add('orin_coslam_runtime_v1.zip',(root/'bundles/to_orin/orin_coslam_runtime_v1.zip').read_bytes())
    source = root/'experiments/ec_public_reference_source_v1'
    native = json.loads((source/'NATIVE_SOURCE_MANIFEST.json').read_text())
    for name, expected in native['file_hashes'].items():
        content = (source/name).read_bytes()
        assert hashlib.sha256(content).hexdigest()==expected
        add('ec_public_reference/'+name,content)
    add('ec_public_reference/NATIVE_SOURCE_MANIFEST.json',(source/'NATIVE_SOURCE_MANIFEST.json').read_bytes())
    add('README.md','''# Orin continuous continuation, 18 October 2026

Read and execute ORIN_CONTINUOUS_PROMPT.md. This supersedes the eight-hour cap
and small finite night list in v2. The exact list is supplied: up to 192
BASE/MC jobs plus two conditional Co-SLAM ENG jobs PER VERIFIED MODE, minus matching
jobs already done locally. New lists are not evidence of launch or completion.

The required unchanged Co runtime zip is INCLUDED here; do not lose it when
extracting. Use the already installed exact runtime when it matches. Preserve
both original repos and existing Orin-specific timing scripts. Never enable
quality or mesh export on Orin. Query power mode and preserve the author's
manual selection. Do not automatically switch modes.

Co manifests use its supplied queue schema after actual data paths are verified
and complete matching jobs are reconciled. EC manifest describes jobs for the
separately validated local EC timing runner; Co's runner cannot execute EC.
ec_public_reference/ is a pinned source snapshot, NOT an instrumented EC runtime.
ENG is a conditional independent block, not a blocker for ready native jobs.

Freeze runnable local manifests, start a detached serial parent, and verify a
real GPU child plus the next executable block before telling the author it is
running. Missing datasets/runtimes must remain explicit. No meaningless repeats
to fill a GPU, no unchanged rejected-pruning expansion, no outcome filtering.
Return results and the validated EC timing wrapper. No routine emails.
''')
    hashes = {name:hashlib.sha256(data).hexdigest() for name,data in sorted(entries.items())}
    add('SHA256SUMS.json',json.dumps(hashes,indent=2)+'\n')
    archive=root/'bundles/to_orin/orin_continuous_v5.zip'
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as saved:
        for name,data in sorted(entries.items()): saved.writestr(name,data)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        for name,expected in hashes.items(): assert hashlib.sha256(check.read(name)).hexdigest()==expected
        assert check.read('orin_coslam_runtime_v1.zip')==(root/'bundles/to_orin/orin_coslam_runtime_v1.zip').read_bytes()
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),native_co_jobs=128,native_ec_jobs=64,conditional_eng_jobs=2,runtime_zip_included=True),indent=2))


if __name__=='__main__':main()
