"""CPU-only local binding; supplied manifests and scientific runtimes remain frozen."""
from pathlib import Path
import ast, copy, hashlib, json, os
R=Path(__file__).resolve().parent; E=R.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(a,indent=2)+'\n')
C=E/'orin_coslam_runtime_v1/runtime'; ENG=E/'orin_after_v4_maxn_v1/coslam_engineering_local'; EC=E/'orin_continuous_v4'; P=E/'orin_component_profile_metadata_recovery_v1'
master=read(R/'manifests/ALL_15W_TARGETS.json'); old=read(E/'orin_maxn_missing_export_v1/MAXN_MASTER_INDEX.json'); lookup={j['id']:j for j in old['benchmarks']}
assert len(lookup)==292
files={}; resolved=[]
def freeze(p):
 p=Path(p).resolve(); h=sha(p); assert str(p)not in files or files[str(p)]==h; files[str(p)]=h
for rt in [C,ENG,P/'co_bundle/runtime',P/'bundle/runtime']:
 m=read(rt/'RUNTIME_MANIFEST.json');freeze(rt/'RUNTIME_MANIFEST.json')
 for n,h in m['runtime_hashes'].items():assert sha(rt/n)==h,(rt,n);freeze(rt/n)
for n,h in read(EC/'EC_RUNTIME_MANIFEST.json')['files'].items():assert sha(EC/n)==h,n;freeze(EC/n)
freeze(EC/'EC_RUNTIME_MANIFEST.json'); freeze(EC/'run_ec_round2.py')
adapter=E/'orin_dp_completion_maxn_v1/run_queue_inherited_lock.py';freeze(adapter)
for j0 in master['jobs']:
 j=copy.deepcopy(j0); family=j['runtime_family']; isprofile=j['execution_kind']=='component_profile'
 if isprofile:
  control=next(x for x in old['benchmarks'] if (x['system'],x['dataset'],x['scene'],x['hash_size'],x['seed'],x['condition'])==(j['system'],j['dataset'],j['scene'],j['hash_size'],0,'BASE'))
 else:control=lookup[j['id']]
 out=Path(control['output']);cfg=read(out/'config.json')
 assert sha(out/'config.json')==control['config_sha256']
 assert sha(out/'summary.json')==control['summary_sha256']
 j['matching_maxn_identity']={k:v for k,v in control.items() if k!='raw_files'}
 j['datadir']=cfg['data']['input_folder' if j['system']=='EC-SLAM' else 'datadir'];assert Path(j['datadir']).is_dir()
 j['power_mode']='MODE_15W';j['status']='queued'
 save(R/'resolved/maxn_configs'/(j['id']+'.json'),cfg);freeze(R/'resolved/maxn_configs'/(j['id']+'.json'))
 if family in ['co_scientific','co_engineering_validated']:
  rt=ENG if family=='co_engineering_validated' else C
  execution={k:j[k]for k in ['id','dataset','scene','datadir','seed','hash_size','frames','quality','export_mesh']}
  execution['condition']='BASE' if family=='co_engineering_validated' else j['condition'];execution['engineering']=j['condition']=='ENG'
  mp=R/'resolved/jobs'/(j['id']+'.json');save(mp,dict(warmup_frames=20,jobs=[execution]));freeze(mp)
  j['output']=str(R/'results'/(j['id']+'_queue')/j['id']);j['runtime']=str(rt);j['driver_condition']=execution['condition'];j['source_sha256']=sha(rt/'RUNTIME_MANIFEST.json')
  j['argv']=['@PYTHON@',str(adapter),'--inherited-workload-lock-fd','@FD@','--runtime',str(rt),'--manifest',str(mp),'--output',str(Path(j['output']).parent),'--power-mode','MODE_15W']
 elif family=='ec_native_timing':
  j['output']=str(R/'results'/j['id']);j['runtime']=str(EC);j['source_sha256']=sha(EC/'EC_RUNTIME_MANIFEST.json')
  j['argv']=['@PYTHON@',str(EC/'run_ec_round2.py'),'--condition',j['condition'],'--seed',str(j['seed']),'--scene',j['scene'],'--hash-size',str(j['hash_size']),'--expected-power-mode','MODE_15W','--frames','0','--datadir',j['datadir'],'--output',j['output'],'--co-runtime',str(C)]
 else:
  co=j['system']=='Co-SLAM';rt=P/('co_bundle' if co else 'bundle')/'runtime';j['runtime']=str(rt);j['source_sha256']=sha(rt/'RUNTIME_MANIFEST.json');j['output']=str(R/'results/profiles'/j['id'])
  j['argv']=['@PYTHON@',str(rt/('run_component_profile.py'if co else'run_ec_component_profile.py')),'--workload',j['workload'],'--seed','0','--datadir',j['datadir'],'--output',j['output'],'--power-mode','MODE_15W']
 resolved.append(j)
# Keep the established scientific CSV auditor, changing only its explicit mode expectation.
a=ast.parse((E/'orin_dp_completion_maxn_v1/common.py').read_text()); fs=[ast.get_source_segment((E/'orin_dp_completion_maxn_v1/common.py').read_text(),n)for n in a.body if isinstance(n,ast.FunctionDef)and n.name in ['config_identity','audit']]
assert len(fs)==2
text='import copy,csv,math\nfrom pathlib import Path\nfrom supervisor import read,sha\n\n'+'\n\n'.join(fs)
assert text.count("'NV Power Mode: MAXN\\n0'")==1
(R/'native_audit.py').write_text(text.replace("'NV Power Mode: MAXN\\n0'","'NV Power Mode: MODE_15W\\n1'")+'\n')
for p in [P/'local_support.py',P/'idle_check.py',E/'orin_after_v4_maxn_v1/post_v4_follower.py',E/'orin_after_v4_maxn_v1/validate_rtx_reference_fixed_state.py',P/'bundle/report_component_profiles.py',P/'bundle/report_ec_component_profiles.py',E/'orin_after_v4_maxn_v1/CO_ENGINEERING_LOCAL_VALIDATION.json']:
 freeze(p)
for n in read(R/'SHA256SUMS.json'):freeze(R/n)
for p in [R/'native_audit.py',Path(__file__)]:freeze(p)
profiles=[j for j in resolved if j['execution_kind']=='component_profile']
for co in [True,False]:
 jobs=[j for j in profiles if (j['system']=='Co-SLAM')==co];tag='CO'if co else'EC';mp=R/'resolved'/(tag+'_PROFILE_MANIFEST.json');save(mp,dict(platform='ORIN_15W',power_mode='MODE_15W',jobs=jobs));freeze(mp)
 save(R/'resolved'/(tag+'_PROFILE_ACTIVE.json'),dict(queue=str(R/'results/profiles'),runtime=jobs[0]['runtime'],manifest=str(mp)));freeze(R/'resolved'/(tag+'_PROFILE_ACTIVE.json'))
# Original successful MAXN records plus exact process start identities, never a circular 15 W flag.
profile=read(P/'PROFILING_COMPLETE.json');export=read(E/'orin_maxn_missing_export_v1/EXPORT_COMPLETE.json')
assert profile['status']=='COMPLETE'and profile['full_profiles_completed']==5 and profile['ALL_REQUIRED_MAXN_COMPLETE'] is True
assert export['status']=='EXPORTED_AND_VERIFIED' and export['master_benchmarks']==292 and export['profiles']==5
assert sha(export['master_index'])==export['master_index_sha256']
processes=[]
for root in [P,E/'orin_dp_completion_maxn_v1',E/'orin_scannet_repair_v1',E/'orin_maxn_missing_export_v1']:
 for p in root.glob('*IDENTITY.json'):
  d=read(p)
  if isinstance(d,dict)and all(k in d for k in ['pid','start_ticks','boot_id']):processes.append(dict(record=str(p),identity=d))
save(R/'UPSTREAM_GATES.json',dict(profile_complete=str(P/'PROFILING_COMPLETE.json'),export_complete=str(E/'orin_maxn_missing_export_v1/EXPORT_COMPLETE.json'),export=export,processes=processes,notification=read(P/'INSTALLATION_CONFIG.json')['notification'],not_gated_on_15W_global_completion=True))
save(R/'resolved/ALL_LOCAL_TARGETS.json',dict(total=297,phases=master['phases'],jobs=resolved))
for p in [R/'UPSTREAM_GATES.json',R/'resolved/ALL_LOCAL_TARGETS.json']:freeze(p)
# Metadata inventory only; never treat MAXN as a 15 W completion.
candidates=[]
for root in E.iterdir():
 if not root.is_dir()or root==R:continue
 results=root/'results'
 if not results.is_dir():continue
 for p in results.rglob('summary.json'):
  try:
   d=read(p)
   if d.get('initial_power')=='NV Power Mode: MODE_15W\n1'and d.get('full_sequence')and not d.get('diagnostic'):candidates.append(str(p.parent))
  except (OSError,ValueError):pass
save(R/'EXISTING_15W_CANDIDATES.json',dict(outputs=candidates,policy='Require full local source/config/frame/timing/mode and exact clock-policy completion proof before reuse; MAXN never reused'))
save(R/'BINDING_FREEZE.json',dict(files=files,source_runtimes_unchanged=True,ec_office0_wrapper='Validated generalized v4 runner extends the MAXN office0-only wrapper with exact power/scene/capacity flags; native EC source, effective config, threads and timing unchanged; compare every resulting effective config to its matching MAXN execution.'))
print('LOCAL_BINDING_VERIFIED',len(resolved),'identities; existing 15W full candidates',len(candidates))
