#!/usr/bin/env python3
"""Portable stdlib-only verification of the final self-contained 15 W ZIP."""
import hashlib,json,sys,zipfile
from pathlib import PurePosixPath
from collections import Counter

def verify(path):
 with zipfile.ZipFile(path)as z:
  names=z.namelist();assert len(names)==len(set(names)),'Duplicate ZIP members';ns=set(names)
  assert all(not PurePosixPath(n).is_absolute()and'..'not in PurePosixPath(n).parts for n in names),'Unsafe archive paths'
  assert z.testzip()is None,'ZIP CRC/integrity failure'
  hashes=json.loads(z.read('SHA256SUMS.json'));assert ns==set(hashes)|{'SHA256SUMS.json'},'Unhashed or missing member'
  for n,h in hashes.items():
   digest=hashlib.sha256()
   with z.open(n)as f:
    for b in iter(lambda:f.read(1048576),b''):digest.update(b)
   assert digest.hexdigest()==h,('Payload SHA256 mismatch',n)
  master=json.loads(z.read('MASTER_RUN_INDEX.json'));rows=master['runs'];assert len(rows)==135 and len({r['canonical_id']for r in rows})==135
  assert Counter(r['execution_kind']for r in rows)=={'benchmark':130,'component_profile':5}
  byid={r['canonical_id']:r for r in rows};refs=0
  def check_ref(rec):
   nonlocal refs
   name=rec['archive_path'];assert name in ns,('Missing indexed payload',name);assert hashes[name]==rec['sha256'];assert z.getinfo(name).file_size==rec['bytes'];refs+=1
  sources=json.loads(z.read('SOURCE_INDEX.json'))
  for family,src in sources.items():
   assert src['manifest']in ns and hashes[src['manifest']]==src['manifest_sha256']
   for rec in src['files'].values():check_ref(rec)
   frozen=json.loads(z.read(src['manifest']))
   for rel,h in frozen.get('runtime_hashes',frozen.get('files',{})).items():assert src['files'][rel]['sha256']==h,(family,rel)
  successes=0
  for r in rows:
   assert r['seed']==0
   for rec in r['archive_files'].values():check_ref(rec)
   for _,cid in r['paired_control_ids'].items():
    assert cid in byid
    assert all(r[k]==byid[cid][k]for k in ['system','dataset','scene','hash_size','seed','runtime_family'])
   if r['method']=='MC':assert r['dataset']=='Replica'
   if r['method']=='MC_DP'and r['dataset']!='Replica':assert set(r['paired_control_ids'])=={'BASE','DP'}and'MC'in r['unavailable_comparisons']
   if r['status']!='successful':continue
   successes+=1;prefix=r['archive_directory']
   for name,key in [('summary.json','summary_sha256'),('config.json','config_sha256'),('source_manifest.json','source_manifest_sha256')]:assert hashes[prefix+'/'+name]==r[key]
   summary=json.loads(z.read(prefix+'/summary.json'));assert summary['seed']==0 and summary['frames']==r['full_frame_count']==r['expected_full_frame_count'];assert summary['full_sequence']and not summary['diagnostic']and summary['finite_parameters_and_poses']and not summary['quality'];assert summary['initial_power']==summary['final_power']=='NV Power Mode: MODE_15W\n1'
   assert r['source_snapshot_manifest']==sources[r['runtime_family']]['manifest'];assert r['source_manifest_sha256']==sources[r['runtime_family']]['manifest_sha256']
   required=['summary.json','config.json','source_manifest.json','completion.json','controller/launch.json','controller/process_receipt.json','controller/controller.log']
   if r['execution_kind']=='benchmark':required+=['calls.csv','iterations.csv','queries.csv']
   elif r['system']=='Co-SLAM':required+=['parents.csv','components.csv','calls.csv','query_calls.csv','selection.json']
   else:required+=['selection.json']+[stage+'/'+f for stage in ['tracking','bundle_adjustment']for f in ['parents.csv','components.csv','segments.csv','calls.csv','query_calls.csv']]
   for name in required:assert prefix+'/'+name in ns,(r['id'],name)
   completion=json.loads(z.read(prefix+'/completion.json'));assert completion['status']=='PASS'and completion['summary_sha256']==r['summary_sha256'];assert json.loads(z.read(prefix+'/controller/process_receipt.json'))['returncode']==0
   if r['execution_kind']=='component_profile':assert summary['selected_calls']=={'tracking':300,'bundle_adjustment':60}and r['protocol_archive_path']in ns and hashes[r['protocol_archive_path']]==r['protocol_sha256']
  exclusions=json.loads(z.read('AUTHOR_EXCLUSIONS.json'));assert len(exclusions['jobs'])==162;assert not set(byid)&{x['id']for x in exclusions['jobs']}
  assert Counter(x['reason']for x in exclusions['jobs'])=={'second_seed_removed_by_author':146,'mc_only_replica_scope':16}
  report=json.loads(z.read('RECONCILIATION.json'));assert [p['expected']for p in report['stages']]==[48,56,26,5];assert successes==report['counts']['successful']
  mapping=json.loads(z.read('PATH_MAPPING.json'));assert all(n in ns for n in mapping)
  banned_dirs={'ckpts','checkpoints','__pycache__','.git','.aws','.codex','.cache','build','dist','envs','venv','conda-meta','prepared_data','artifacts'}
  banned_suffix={'.so','.pt','.pth','.tar','.sens','.ply','.pcd','.npz','.mp4','.jpg','.png'}
  assert all(not(set(PurePosixPath(n).parts)&banned_dirs)and PurePosixPath(n).suffix.lower()not in banned_suffix for n in names),'Forbidden dataset/checkpoint/build/environment/mesh payload'
  return dict(status='PASS',zip_crc='PASS',payload_SHA256='PASS',hashed_members=len(hashes),zip_members=len(names),index_entries=len(rows),successful_entries=successes,indexed_file_references_verified=refs,benchmark_entries=130,profile_entries=5,excluded=162,unique_run_ids=True,unique_zip_members=True,all_indexed_files_and_sources_present=True)
if __name__=='__main__':
 assert len(sys.argv)==2,'usage: python verify_export.py archive.zip'
 print(json.dumps(verify(sys.argv[1]),indent=2))
