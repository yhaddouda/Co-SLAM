#!/usr/bin/env python3
"""Read-only reconciliation and self-contained CPU export; never launches SLAM."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,copy,fcntl,hashlib,importlib.util,io,json,math,os,platform,statistics,subprocess,sys,traceback,zipfile
B=Path(__file__).resolve().parent;E=B.parent;TOP=E.parent
R=E/'orin_15w_complete_priority_v1';A=R/'amendments/single_seed_replica_mc_v1';P=E/'orin_component_profile_metadata_recovery_v1'
TARGET=TOP/'ORIN_15W_FINAL_RESULTS_20261027.zip';MODE='NV Power Mode: MODE_15W\n1'
for p in [R,P/'bundle']:
 if str(p)not in sys.path:sys.path.insert(0,str(p))
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p);return p

def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def csvrows(p):
 with Path(p).open()as f:return list(csv.DictReader(f))

def csvreport(name,rows):
 p=B/name;p.parent.mkdir(parents=True,exist_ok=True)
 if rows:
  fields=list(dict.fromkeys(k for r in rows for k in r))
  with p.open('w',newline='')as f:
   w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows)
 else:p.write_text('')
 return p

payload={};path_mapping={};omitted=[];missing_evidence=[]
ALLOW={'.py','.json','.yaml','.yml','.md','.csv','.log','.txt','.cu','.cuh','.h','.hpp','.cpp','.c','.sh','.ini','.toml'}
BAD_DIRS={'__pycache__','.git','.aws','.codex','.cache','build','dist','envs','venv','conda-meta','ckpts','checkpoints','artifacts','prepared_data'}
def allowed(p):
 p=Path(p)
 return p.is_file()and not p.is_symlink()and p.suffix.lower()in ALLOW and not any(x in BAD_DIRS for x in p.parts)and not any(x in p.name.lower()for x in ['credential','id_rsa','.env','access_token','private_key'])
def add(p,name):
 p=Path(p).resolve();assert p.is_file()and not p.is_symlink(),p
 n=Path(name).as_posix();assert not Path(n).is_absolute()and'..'not in Path(n).parts
 if n in payload:assert payload[n]['original_path']==str(p);return payload[n]
 rec=dict(original_path=str(p),archive_path=n,sha256=sha(p),bytes=p.stat().st_size)
 payload[n]=rec;path_mapping[n]=str(p);return rec

def addtree(root,prefix):
 root=Path(root)
 if not root.exists():return
 for p in sorted(root.rglob('*')):
  if allowed(p):add(p,prefix+'/'+p.relative_to(root).as_posix())

def gen(name,value):return add(save(B/name,value),name)

def controls(job,alljobs):
 d=dict(job.get('control_ids',{}));role=job['condition']
 if role in ['MC','ENG']and not d:
  target='BASE_CONTROL'if role=='ENG'else'BASE'
  match=[j for j in alljobs if all(j[k]==job[k]for k in ['system','dataset','scene','hash_size','seed','runtime_family'])and j['condition']==target]
  assert len(match)==1,(job['id'],target);d[target]=match[0]['id']
 return d

def regime(j):
 if j['runtime_family']=='co_engineering_validated':return 'Co_engineering_dedicated_control_BA_only'
 return 'Replica_H'+str(j['hash_size'])if j['dataset']=='Replica'else'native_scene_capacities'

call_stats=[];iteration_stats=[];query_stats=[];schemas={};profile_audits=[];analysis={}
def benchmark_stats(j):
 out=Path(j['output']);summary=read(out/'summary.json');cfg=read(out/'config.json');calls=csvrows(out/'calls.csv');events=csvrows(out/'iterations.csv');queries=csvrows(out/'queries.csv');n=j['expected_native_frames']
 for name in ['calls.csv','iterations.csv','queries.csv','frames.csv']:
  p=out/name
  if p.exists():
   with p.open()as f:header=next(csv.reader(f))
   schemas.setdefault(name,{})[j['runtime_family']]=header
 schemas.setdefault('frame_iteration_sums.csv',{})['contract']='Headerless: frame_id,total_iteration_cuda_ms,tracking_iteration_cuda_ms,ba_iteration_cuda_ms; intervals, not host wall time'
 def exact(rows,framekey,iterkey,expected):
  got=[(int(x[framekey]),int(x[iterkey]))for x in rows];assert len(got)==len(expected)and set(got)==expected
 if j['system']=='EC-SLAM':
  tr={(f,i)for f in range(1,n)for i in range(cfg['tracking']['iters'])};ba_frames=set(range(cfg['mapping']['every_frame'],n,cfg['mapping']['every_frame']))|{n-1};ba={(f,i)for f in ba_frames for i in range(cfg['mapping']['iters'])}
  exact([e for e in events if e['iteration_type']=='tracking'],'frame_id','iteration_id',tr);exact([e for e in events if e['iteration_type']=='bundle_adjustment'],'frame_id','iteration_id',ba)
  assert len(events)==len(tr)+len(ba)
  # Native render_rays optionally calls decoder.forward for missing-depth
  # preliminary sampling, then performs its main field query. Counters are
  # per hash-grid forward, not per optimizer update; event contexts stay unique.
  for stage,expected in [('tracking',tr),('bundle_adjustment',ba)]:
   contexts=Counter((int(q['frame_id']),int(q['iteration']))for q in queries if q['stage']==stage)
   assert set(contexts)==expected and all(n in [1,2]for n in contexts.values()),(j['id'],stage,'Native EC query context/multiplicity mismatch')
 for stage in sorted({r['stage']for r in calls}):
  rows=[r for r in calls if r['stage']==stage and int(r['frame_id'])>=20]
  if not rows:continue
  for col in ['wall_ms','cuda_ms']:
   vals=[float(r[col])for r in rows];assert all(math.isfinite(v)and v>=0 for v in vals);mean=statistics.mean(vals)
   key='mean_'+col
   if isinstance(summary.get(stage),dict)and key in summary[stage]:assert math.isclose(mean,summary[stage][key],rel_tol=1e-8,abs_tol=1e-6),(j['id'],stage,key,mean,summary[stage][key])
   if j['runtime_family']=='co_engineering_validated'and stage!='bundle_adjustment':continue
   call_stats.append(dict(id=j['id'],system=j['system'],dataset=j['dataset'],scene=j['scene'],method=j['condition'],hash_size=j['hash_size'],seed=0,regime=regime(j),stage=stage,scope='complete_call_host_wall'if col=='wall_ms'else'complete_call_CUDA_interval',metric='mean_'+col,value_ms=mean,measured_calls=len(rows),warmup_frames_excluded=20,across_seed_SD=None))
 for stage in sorted({r['iteration_type']for r in events}):
  rows=[r for r in events if r['iteration_type']==stage and int(r['frame_id'])>=20]
  if not rows:continue
  if j['runtime_family']=='co_engineering_validated'and stage!='bundle_adjustment':continue
  byframe=defaultdict(lambda:defaultdict(float))
  for col in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms']:
   vals=[float(r[col])for r in rows];assert all(math.isfinite(v)and v>=0 for v in vals)
   iteration_stats.append(dict(id=j['id'],system=j['system'],dataset=j['dataset'],scene=j['scene'],method=j['condition'],hash_size=j['hash_size'],seed=0,regime=regime(j),stage=stage,scope='optimizer_iteration_CUDA_interval',metric=col,value_ms=statistics.mean(vals),sample_count=len(vals),warmup_frames_excluded=20,across_seed_SD=None))
   for r in rows:byframe[int(r['frame_id'])][col]+=float(r[col])
  for col in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms']:
   iteration_stats.append(dict(id=j['id'],system=j['system'],dataset=j['dataset'],scene=j['scene'],method=j['condition'],hash_size=j['hash_size'],seed=0,regime=regime(j),stage=stage,scope='per_worker_frame_sum_of_optimizer_CUDA_intervals',metric=col,value_ms=statistics.mean(v[col]for v in byframe.values()),sample_count=len(byframe),warmup_frames_excluded=20,across_seed_SD=None))
 for stage in sorted({q['stage']for q in queries}):
  selected=[q for q in queries if q['stage']==stage and int(q['frame_id'])>=20]
  if not selected:continue
  dense=sum(int(q['dense_queries'])for q in selected);actual=sum(int(q['actual_queries'])for q in selected)
  query_stats.append(dict(id=j['id'],stage=stage,scope='query_rows_frames_ge_20',rows=len(selected),dense_queries=dense,actual_queries=actual,pruned_pct=100*(1-actual/dense)if dense else None))
 return dict(raw_frame_and_iteration_coverage='PASS',summary_means_vs_raw_warmup20='PASS',EC_worker_overlap='preserved, non-additive'if j['system']=='EC-SLAM'else None)


def source_snapshots(jobs):
 runtime_ids={'co_scientific':'co_scientific_runtime_v1','ec_native_timing':'ec_native_timing_v4','co_engineering_validated':'co_engineering_rtx_reference','co_component_profile_repaired':'co_component_profile_metadata_recovery','ec_component_profile_validated':'ec_component_profile_metadata_recovery'}
 result={}
 for family,sid in runtime_ids.items():
  j=next(j for j in jobs if j['runtime_family']==family);root=Path(j['runtime']);mfile=root/('EC_RUNTIME_MANIFEST.json'if family=='ec_native_timing'else'RUNTIME_MANIFEST.json');m=read(mfile);assert sha(mfile)==j['source_sha256']
  for n,h in m.get('runtime_hashes',m.get('files',{})).items():assert sha(root/n)==h,(family,n)
  if family=='ec_native_timing':paths=[p for p in(root/'ec_runtime').rglob('*')if allowed(p)]+[root/'run_ec_round2.py',root/'ec_timing.py',mfile]
  else:paths=[p for p in root.rglob('*')if allowed(p)]
  entries={}
  for p in sorted(set(paths)):
   rel=p.relative_to(root).as_posix();entries[rel]=add(p,'source_snapshots/'+sid+'/'+rel)
  declared=m.get('runtime_hashes',m.get('files',{}))
  for n,h in declared.items():assert n in entries and entries[n]['sha256']==h,(sid,n)
  precision=[]
  for name,rec in entries.items():
   if Path(name).name in ['encodings.py','Model.py']:
    lines=Path(rec['original_path']).read_text().splitlines()
    for ln,line in enumerate(lines,1):
     if 'dtype=torch.float' in line.replace(' ',''):precision.append(dict(archive_file=rec['archive_path'],line=ln,source=line.strip(),evidence_kind='verified_frozen_source_setting_not_new_runtime_probe',dtype='torch.float (float32)'))
  result[family]=dict(snapshot_id=sid,original_root=str(root),manifest=entries[mfile.relative_to(root).as_posix()]['archive_path'],manifest_sha256=sha(mfile),public_revision=m.get('public_revision'),files=entries,declared_hashes_verified=True,precision_evidence=precision)
 return result


def evidence_for(j,prefix):
 out=Path(j['output']);records={};missing=[]
 for p in sorted(out.rglob('*')):
  if allowed(p):
   rel=p.relative_to(out).as_posix();records['output/'+rel]=add(p,prefix+'/'+rel)
 required=['summary.json','config.json','source_manifest.json']
 if j['execution_kind']=='benchmark':required+=['calls.csv','iterations.csv','queries.csv']
 elif j['system']=='Co-SLAM':required+=['parents.csv','components.csv','calls.csv','query_calls.csv','selection.json']
 else:required+=['selection.json']+[f'{stage}/{name}'for stage in ['tracking','bundle_adjustment']for name in ['parents.csv','components.csv','calls.csv','query_calls.csv','segments.csv']]
 for name in required:
  if not(out/name).is_file():missing.append(str(out/name))
 proof=[(Path(j.get('completion_record',str(R/'completion_records'/(j['id']+'.json')))),'completion.json'),(R/'launches'/(j['id']+'.json'),'controller/launch.json'),(R/'process_receipts'/(j['id']+'.json'),'controller/process_receipt.json'),(R/'logs'/(j['id']+'.log'),'controller/controller.log')]
 if j['runtime_family']in ['co_scientific','co_engineering_validated']:
  for p in out.parent.iterdir():
   if p.is_file()and allowed(p):proof.append((p,'controller/queue/'+p.name))
  proof.append((R/'resolved/jobs'/(j['id']+'.json'),'controller/resolved_job_manifest.json'))
 elif j['execution_kind']=='component_profile':proof.append((out.parent/(j['id']+'.done.json'),'controller/profile_done.json'))
 for p,name in proof:
  if p.is_file():records[name]=add(p,prefix+'/'+name)
  else:missing.append(str(p))
 return records,missing


def metadata_and_state():
 # Copy only controller metadata, code and validation. Never its datasets,
 # artifact/checkpoint directories, installed environments or build caches.
 for p in sorted(R.iterdir()):
  if allowed(p)and p.name not in ['DECISION_WAKE.json']:add(p,'queue/'+p.name)
 for sub in ['manifests','resolved','amendments','readiness','completion_records','stage_completions','failures','diagnostic_receipts','process_receipts','launches','exports']:
  addtree(R/sub,'queue/'+sub)
 addtree(R/'diagnostics','validation/engineering_15W')
 for p in sorted((R/'logs').glob('*')):
  if allowed(p):add(p,'queue/logs/'+p.name)
 for p in sorted((R/'results/profiles').glob('*_diagnostic')):
  if p.is_dir():addtree(p,'validation/profiling_15W/'+p.name)
 for p in sorted((R/'results/profiles').glob('*_diagnostic.*')):
  if allowed(p):add(p,'validation/profiling_15W/'+p.name)
 for p in (R/'results').glob('*component_profile_analysis_v1'):addtree(p,'analysis/existing/'+p.name)
 add(R/'TIMING_SUMMARIES.json','analysis/existing/queue_TIMING_SUMMARIES.json')
 old=E/'orin_after_v4_maxn_v1'
 for name in ['validate_rtx_reference_fixed_state.py','CO_ENGINEERING_LOCAL_VALIDATION.json','CO_ENGINEERING_RUNTIME_FREEZE.json','ENGINEERING_FAILURE_REPORT.json','ENGINEERING_FAILURE_REPORT.md','MINIMAL_ORIGINAL_ASSERTION_CASES.json']:
  p=old/name
  if p.is_file():add(p,('source_support/'if p.suffix=='.py'else'validation/historical_engineering/')+name)
 p=E/'orin_continuous_v4/ENGINEERING_PREFLIGHT_VALIDATION.json'
 if p.is_file():add(p,'validation/historical_engineering/ORIGINAL_ENGINEERING_PREFLIGHT_VALIDATION.json')
 for name in ['RECOVERY_DIAGNOSIS.json','RECOVERY_NOTES.md','RECOVERY_RUNTIME_IDENTITIES.json','RECOVERED_PACKAGE_VERIFICATION.json','local_support.py','idle_check.py']:
  p=P/name
  if p.is_file():add(p,'source_support/profile_metadata_recovery/'+name)
 for p in (P/'preserved_failure').rglob('*'):
  if allowed(p):add(p,'validation/historical_profile_metadata_failure/'+p.relative_to(P/'preserved_failure').as_posix())
 for name in ['report_component_profiles.py','report_ec_component_profiles.py']:
  add(P/'bundle'/name,'source_support/profile_reports/'+name)
 for p in P.glob('*/COMPONENT_PROFILING_PROTOCOL.md'):add(p,'source_support/profile_protocols/'+p.parent.name+'_'+p.name)
 add(E/'orin_dp_completion_maxn_v1/run_queue_inherited_lock.py','source_support/run_queue_inherited_lock.py')
 add(E/'orin_after_v4_maxn_v1/post_v4_follower.py','source_support/validated_orin_idle_checker.py')


def main():
 assert not TARGET.exists(),'Preserve existing target; verify it instead of overwriting'
 with open('/tmp/dasip_round2_gpu0.lock','a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  support=load('export_orin_idle',P/'local_support.py');s=read(R/'CONTINUATION_STATUS.json');end=read(R/'CONTINUATION_COMPLETE.json')
  assert s['state']in ['COMPLETE','EXHAUSTED_WITH_FAILURES']and end['status']in ['COMPLETE','EXHAUSTED_WITH_FAILURES']
  assert not s.get('active_child_identity')and not s.get('current')and not s.get('next_job')
  assert not support.still_alive(s['supervisor_identity'])
  fresh=[support.idle_sample('MODE_15W')for _ in range(3)]
  save(B/'EXPORT_GATES.json',dict(utc=now(),controller_terminal=True,recorded_controller=s['supervisor_identity'],controller_identity_exited=True,exclusive_gpu_lock_inode=os.fstat(lock.fileno()).st_ino,fresh_idle=fresh,current_power_query=support.require_mode('MODE_15W'),current_query_is_not_historical_run_clock_evidence=True,no_experiment_launched=True))
  validate=load('author_135_plan_verifier',A/'bundle/validate_plan.py');validate.validate(A/'bundle')
  plan=read(A/'bundle/manifests/ALL_15W_TARGETS.json');exclusions=read(A/'bundle/EXCLUSIONS.json');byid={j['id']:j for j in s['jobs']};assert len(byid)==135
  assert [j['id']for j in plan['jobs']]==[j['id']for j in s['jobs']]
  assert len(exclusions['jobs'])==162 and not set(byid)&{j['id']for j in exclusions['jobs']}
  for filename in ['BINDING_FREEZE.json','INSTALLATION_FREEZE.json']:
   for p,h in read(R/filename)['files'].items():assert sha(p)==h,'Frozen provenance changed: '+p
  sources=source_snapshots(s['jobs']);validator=load('original_15W_cpu_validators',A/'supervisor_amended.py');index=[];statuses=defaultdict(list);profile_weighted=[];profile_windows=[]
  for declared in plan['jobs']:
   j=byid[declared['id']];out=Path(j['output']);kind=j['execution_kind'];prefix=('benchmarks/'if kind=='benchmark'else'profiles/')+j['id'];proof,missing=evidence_for(j,prefix)
   row=dict(canonical_id=j['id'],id=j['id'],local_id=out.name,system=j['system'],dataset=j['dataset'],scene=j['scene'],method=j['condition'],driver_condition=j.get('driver_condition',j['condition']),hash_size=j['hash_size'],hash_capacity_label='H'+str(j['hash_size']),nominal_max_hash_entries_per_level=2**j['hash_size'],seed=j['seed'],execution_kind=kind,runtime_family=j['runtime_family'],stage=j['phase'],ordinal=declared['ordinal'],expected_full_frame_count=j['expected_native_frames'],original_output_path=str(out),archive_directory=prefix,controller_status=j['status'],source_snapshot_id=sources[j['runtime_family']]['snapshot_id'],source_manifest_archive_path=prefix+'/source_manifest.json',source_manifest_sha256=j['source_sha256'],paired_control_ids=controls(j,s['jobs'])if kind=='benchmark'else{},unavailable_comparisons=j.get('unavailable_comparisons',{}),engineering_role=('dedicated_engineering_BASE_control'if j['condition']=='BASE_CONTROL'else'engineering_variant_BA_only')if j['runtime_family']=='co_engineering_validated'else None,reported_engineering_stage='bundle_adjustment_only'if j['runtime_family']=='co_engineering_validated'else None,archive_files=proof,missing_evidence=missing,across_seed_SD=None,started_utc=j.get('started_utc'),finished_utc=j.get('finished_utc'))
   try:
    for k in ['system','dataset','scene','condition','hash_size','seed','execution_kind','runtime_family','expected_native_frames']:assert j[k]==declared[k],(j['id'],k)
    assert j['seed']==0
    if j['status']in ['blocked','failed']:
     row.update(status=j['status'],reason=j.get('reason'));raise RuntimeError('Controller reported '+j['status'])
    if missing:raise FileNotFoundError('Required evidence absent: '+repr(missing))
    completion=read(j['completion_record']);assert completion['status']=='PASS'and completion['full']and completion['power']==MODE
    for name,key in [('summary.json','summary_sha256'),('config.json','config_sha256'),('source_manifest.json','source_manifest_sha256')]:assert sha(out/name)==completion[key]
    assert read(R/'process_receipts'/(j['id']+'.json'))['returncode']==0
    checked=validator.audit(j);a=checked['summary'];assert a['initial_power']==a['final_power']==MODE
    row.update(full_frame_count=a['frames'],full_sequence=a['full_sequence'],config_sha256=sha(out/'config.json'),summary_sha256=sha(out/'summary.json'),historical_initial_power=a['initial_power'],historical_final_power=a['final_power'],measured_slam_work_seconds=a.get('slam_work_wall_seconds',a.get('measured_run_seconds')),launch_to_exit_seconds=j.get('elapsed_seconds'),run_environment={k:a.get(k)for k in ['device','torch_version','cuda_runtime']},source_snapshot_manifest=sources[j['runtime_family']]['manifest'])
    if kind=='benchmark':row['raw_validation']=benchmark_stats(j)
    else:
     co=j['system']=='Co-SLAM';rep=load('export_profile_'+('co'if co else'ec'),P/'bundle'/('report_component_profiles.py'if co else'report_ec_component_profiles.py'));protocol=read(Path(j['runtime'])/'PROTOCOL.json');work=next(w for w in protocol['workloads']if w['id']==j['workload']);parts=rep.audit_run(out,j,work,j['source_sha256'])
     profile_weighted.extend(dict(id=j['id'],system=j['system'],**v)for v in parts[0]);profile_windows.extend(dict(id=j['id'],system=j['system'],**v)for v in parts[1]);row['profiling_selected_calls']=a['selected_calls'];row['protocol_archive_path']=sources[j['runtime_family']]['files']['PROTOCOL.json']['archive_path'];row['protocol_sha256']=sha(Path(j['runtime'])/'PROTOCOL.json');row['raw_validation']=dict(status='PASS',fixed_selection_and_component_closure='PASS',decompositions=['forward','optimizer_iteration','tracking_or_BA_call'])
     if co:
      calls=csvrows(out/'calls.csv');n=j['expected_native_frames'];cfg=read(out/'config.json')
      for stage,ids in [('tracking',set(range(1,n))),('bundle_adjustment',set(range(cfg['mapping']['map_every'],n,cfg['mapping']['map_every'])))]:
       rows=[x for x in calls if x['stage']==stage];assert len(rows)==len(ids)and{int(x['frame_id'])for x in rows}==ids
     else:row['full_coverage_evidence']='Native full-sequence summary and driver end-state assertions; worker CSVs intentionally contain selected calls, not every frame'
    row['status']='successful';save(B/'run_audits'/(j['id']+'.json'),checked)
   except Exception as error:
    row.setdefault('status','missing'if isinstance(error,FileNotFoundError)else'failed');row.update(error=repr(error),traceback=traceback.format_exc());save(B/'run_audits'/(j['id']+'.json'),row)
   statuses[row['status']].append(j['id']);index.append(row)
   print('RECONCILED',declared['ordinal'],j['id'],row['status'],flush=True)
  # Check numerical-preflight evidence, never rerun validation or GPU work.
  validations=[]
  for label,count in [('CO',3),('EC',2)]:
   rec=read(R/'readiness'/(label+'_PROFILE_PREFLIGHTS.json'));assert rec['status']=='PASS'and len(rec['checks'])==count
   jobs=[j for j in s['jobs']if j['execution_kind']=='component_profile'and(j['system']=='Co-SLAM')==(label=='CO')]
   for j in jobs:
    d=Path(j['output']+'_diagnostic');receipt=read(R/'diagnostic_receipts'/(j['id']+'_diagnostic.json'));assert receipt['status']=='PASS'and receipt['summary_sha256']==sha(d/'summary.json');assert read(d/'fixed_state_validation/PASSED.json')['status']=='PASSED';a=read(d/'summary.json');assert a['diagnostic']and a['frames']==50 and not a['full_sequence']and a['initial_power']==a['final_power']==MODE;validations.append(dict(id=j['id']+'_diagnostic',status='PASS',frames=50,counted_as_full=False))
  eng=read(R/'readiness/ENGINEERING_15W_VALIDATION.json');assert eng['status']=='PASS'and eng['fixed_state']['status']=='PASS';assert eng['fixed_state']['atol']==eng['fixed_state']['rtol']==1e-6
  for tag in ['BASE_CONTROL','ENG']:
   out=R/'diagnostics'/('ENGINEERING_'+tag+'_50');a=read(out/'summary.json');assert a['diagnostic']and a['frames']==50 and not a['full_sequence']and a['finite_parameters_and_poses']and a['initial_power']==a['final_power']==MODE
  recovery=read(A/'AUDIT_LOADER_RECOVERY.json');assert recovery['status']=='PASS'and recovery['no_experiment_repeated']and recovery['no_check_or_tolerance_changed']
  stages=[]
  for p in plan['phases']:
   rows=[j for j in index if j['stage']==p['id']];stages.append(dict(id=p['id'],expected=p['count'],counts={st:sum(j['status']==st for j in rows)for st in ['successful','missing','failed','blocked']}))
  reconcile=dict(utc=now(),expected=135,benchmarks_expected=130,profiles_expected=5,counts={st:len(statuses[st])for st in ['successful','missing','failed','blocked']},identities={st:statuses[st]for st in ['successful','missing','failed','blocked']},stages=stages,author_excluded=162,exclusion_counts=dict(Counter(x['reason']for x in exclusions['jobs'])),export_checker_schema_revision='Native EC query rows are per field forward: one main plus optional missing-depth preliminary query; full iteration contexts and multiplicity 1 or 2 checked, unique optimizer event rows unchanged',historical_resolved_audit_incident=dict(job=recovery['job'],exact_error=recovery['exact_error'],status='resolved; full raw audit PASS; no rerun',receipt='queue/amendments/single_seed_replica_mc_v1/AUDIT_LOADER_RECOVERY.json'),numerical_profile_preflights=validations,engineering_15W_validation='PASS',ALL_REQUIRED_15W_COMPLETE=len(statuses['successful'])==135)
  save(B/'RECONCILIATION.json',reconcile)
  print('RECONCILIATION_COUNTS',reconcile['counts'],flush=True)
  # Derived reports preserve per-scene means, scope separation and negative comparisons.
  allstats=call_stats+iteration_stats;lookup={j['canonical_id']:j for j in index};comparisons=[];available={(r['id'],r['stage'],r['scope'],r['metric']):r for r in allstats}
  for r in allstats:
   entry=lookup[r['id']]
   for role,cid in entry['paired_control_ids'].items():
    control=lookup[cid];assert all(entry[k]==control[k]for k in ['system','dataset','scene','hash_size','seed','runtime_family'])
    other=available.get((cid,r['stage'],r['scope'],r['metric']))
    if other and other['value_ms']>0:comparisons.append(dict(**{k:r[k]for k in ['id','system','dataset','scene','hash_size','seed','regime','stage','scope','metric']},method=entry['method'],control_method=role,control_id=cid,method_ms=r['value_ms'],control_ms=other['value_ms'],reduction_pct=100*(1-r['value_ms']/other['value_ms']),across_seed_SD=None))
  groups=defaultdict(list)
  for r in allstats:groups[tuple(r[k]for k in ['system','dataset','regime','method','stage','scope','metric'])].append(r)
  equal=[dict(zip(['system','dataset','regime','method','stage','scope','metric'],k),scene_count=len(v),scenes=[r['scene']for r in v],equal_scene_mean_ms=statistics.mean(r['value_ms']for r in v),seed=0,across_seed_SD=None)for k,v in groups.items()]
  pairs=defaultdict(list)
  for r in comparisons:pairs[tuple(r[k]for k in ['system','dataset','regime','method','control_method','stage','scope','metric'])].append(r)
  equal_pairs=[dict(zip(['system','dataset','regime','method','control_method','stage','scope','metric'],k),scene_count=len(v),scenes=[r['scene']for r in v],equal_scene_mean_method_ms=statistics.mean(r['method_ms']for r in v),equal_scene_mean_control_ms=statistics.mean(r['control_ms']for r in v),equal_scene_mean_reduction_pct=statistics.mean(r['reduction_pct']for r in v),reduction_of_equal_scene_means_pct=100*(1-statistics.mean(r['method_ms']for r in v)/statistics.mean(r['control_ms']for r in v)),seed=0,across_seed_SD=None)for k,v in pairs.items()]
  metadata_and_state()
  gen('RECONCILIATION.json',reconcile);gen('AUTHOR_EXCLUSIONS.json',exclusions);gen('SOURCE_INDEX.json',sources)
  add(B/'EC_QUERY_CHECK_REVISION.json','validation/export_checker_revision/EC_QUERY_CHECK_REVISION.json')
  add(B/'attempt1_rejected_query_assumption/run_audits/EC_Replica_office0_BASE_H16_seed0.json','validation/export_checker_revision/REJECTED_ONE_ROW_ASSUMPTION_TRACE.json')
  for p in (B/'run_audits').glob('*.json'):add(p,'validation/final_raw_reaudit/'+p.name)
  for name,rows in [('analysis/complete_call_per_run.csv',call_stats),('analysis/optimizer_iteration_and_frame_sums_per_run.csv',iteration_stats),('analysis/query_counts_per_run.csv',query_stats),('analysis/paired_per_scene_comparisons.csv',comparisons),('analysis/equal_scene_dataset_means.csv',equal),('analysis/equal_scene_paired_dataset_comparisons.csv',equal_pairs),('analysis/profile_weighted_thirds_per_run.csv',profile_weighted),('analysis/profile_per_third.csv',profile_windows)]:add(csvreport(name,rows),name)
  gen('analysis/CSV_SCHEMAS_AND_UNITS.json',dict(original_headers=schemas,timings='milliseconds except explicitly named seconds fields in summary/completion metadata; raw columns and byte contents unchanged',derived_report_value_ms='ms',primary_warmup='Exclude frame IDs 0..19 from summary means',aggregation='Each scene contributes one per-run mean with equal weight within system/dataset/capacity-regime/method/scope. ScanNet native scene-specific capacities are not pooled with a sensitivity regime.',single_seed_SD=None,EC='Native concurrent workers, shared stream/global synchronization; intervals can overlap/include peer work/host idle; never sum workers as scene latency or FPS',engineering='Co BA only; dedicated engineering BASE_CONTROL is distinct from scientific BASE',unavailable_comparisons='MC_DP versus MC outside Replica intentionally unavailable in 15 W; BASE and DP controls retained; no MAXN/RTX value substitution'))
  historical=dict(utc=now(),historical_mode_source='Per-run summary.json initial_power/final_power plus profile environment snapshots and queue readiness; all successful rows verified MODE_15W / ID 1',historical_clock_policy_record='queue/ENVIRONMENT_15W.json',historical_clock_baseline=read(R/'ENVIRONMENT_15W.json'),recorded_GPU_frequency_bounds_hz={k:read(R/'ENVIRONMENT_15W.json')['sysfs'].get('/sys/class/devfreq/17000000.gpu/'+k)for k in ['min_freq','max_freq']},limits='These are recorded policy bounds, not continuous achieved clock measurements. Full jetson_clocks lock status and unrecorded historical clocks are unknown. Optional sysfs EAGAIN/null errors remain unchanged; no current reading is backfilled into history.',precision='Frozen HashGrid/OneBlob constructor source explicitly uses torch.float (float32); precision evidence in SOURCE_INDEX.json. This export runs no new dtype probe. Engineering fixed-state receipts independently record torch.float32 input dtype.',current_export_observation=read(B/'EXPORT_GATES.json'),historical_environment='Device/Torch/CUDA versions from each original summary, optional profile environment_initial/final snapshots; no installed environment/build cache is included')
  gen('metadata/POWER_CLOCK_PRECISION_AND_ENVIRONMENT.json',historical);add(B/'EXPORT_GATES.json','metadata/EXPORT_TIME_HOST_OBSERVATION.json')
  master=dict(utc=now(),mode='MODE_15W',platform='ORIN_15W',benchmark_count=130,profile_count=5,required_count=135,counts=reconcile['counts'],single_seed=True,seeds=[0],across_seed_SD=None,author_excluded_count=162,source_index='SOURCE_INDEX.json',reconciliation='RECONCILIATION.json',runs=index,benchmarks=[j for j in index if j['execution_kind']=='benchmark'],profiles=[j for j in index if j['execution_kind']=='component_profile'])
  gen('MASTER_RUN_INDEX.json',master)
  rows=[]
  for j in index:
   rows.append({k:(json.dumps(v)if isinstance(v,(dict,list))else v)for k,v in j.items()if k not in ['archive_files','traceback','raw_validation']})
  add(csvreport('MASTER_RUN_INDEX.csv',rows),'MASTER_RUN_INDEX.csv')
  limitations=["Single seed 0: across-seed SD unavailable, not zero.","Historical full jetson_clocks lock state/continuous achieved clocks were not recorded; baseline policy bounds and available snapshots are included without inference.","EC worker timing intervals overlap and include synchronization/contention; they cannot be summed into pipeline latency or FPS.","EC profiles record selected calls; full scene execution is supported by native summary/end-state checks, not an every-frame profiling CSV.","Installed Python/conda environments and compiled extension binaries/build caches are excluded; exact available source/config/manifest files and recorded library versions are included.","Native EC profile ckpts/ and their auxiliary reconstruction outputs are excluded with all checkpoints/plots; original timing logs and worker measurements are preserved.","Non-Replica MC_DP-versus-MC is intentionally unavailable; no MAXN/RTX timing substitution.","Historical queue import failure was recovered without repeating the completed DP run; original failure and successful recovery evidence are both included."]
  gen('LIMITATIONS_AND_MISSING_EVIDENCE.json',dict(limitations=limitations,missing_required_run_evidence=[dict(id=j['id'],files=j['missing_evidence'])for j in index if j['missing_evidence']],failed_or_missing_identities={k:reconcile['identities'][k]for k in ['failed','missing','blocked']},optional_unrecorded_benchmark_files='EC native runner does not write separate frames.csv/frame_iteration_sums.csv; frame-level tracking/mapping calls are retained in calls.csv. Profiling decompositions use parents/components/segments rather than benchmark iterations.csv.'))
  text='''# Orin 15 W final programme — RTX import, 27 October 2026

This is a self-contained export, not an incremental ZIP. Raw file contents,
CSV column names and units are preserved. The authoritative index is
`MASTER_RUN_INDEX.json` (also CSV); `RECONCILIATION.json` lists successful,
missing, failed and blocked identities separately. Queue termination alone
was not used as proof: every claimed success was checked against the amended
ledger, real completion/process receipts, actual source/config hashes, full
native frame/iteration/query evidence, mode records and the existing raw
CPU validators. The controller was terminal/exited; the validated Orin
checker verified zero GPU load, no device users and no timed programs while
this exporter held the shared exclusive GPU lock. No experiment was rerun.

Scope: seed 0 only, 130 benchmarks plus five component profiles =135 full
slots. Stages are 48 BASE/DP across all four datasets, 56 Replica MC/BASE
(H16/H19, Co/EC, all eight scenes), 26 native MC_DP plus Co engineering/control,
and five profiles. The 162 author exclusions (146 seed-1 benchmarks and 16
non-Replica MC-only jobs) are separate in `AUTHOR_EXCLUSIONS.json`; they are
neither failures nor missing requirements. Original 297-slot manifests under
`queue/` are provenance, superseded by the amendment under `queue/amendments/`.

`benchmarks/<canonical_id>/` and `profiles/<canonical_id>/` contain actual
outputs, including any reused output resolved from its recorded original
location. `controller/` contains original launch commands, logs, completion
and process receipts, queue markers and resolved one-job manifests. Every
indexed archive path resolves to a real member. `PATH_MAPPING.json` maps
archive paths to original local paths. No prior MAXN archive is needed to
recover the 15 W raw data. Matching MAXN source/config references in historical
queue metadata are provenance only; no MAXN/RTX measurements replace 15 W data.

`source_snapshots/` stores each exact local scientific/engineering/profiling
runtime once, with mappings and verified manifest hashes in `SOURCE_INDEX.json`.
`source_support/` and `queue/` preserve Orin wrappers, the inherited-lock adapter,
CPU validators, metadata-reader fixes, controller amendments and numerical
validation/protocol evidence. `validation/` contains full numerical-preflight
receipts and raw 50-frame diagnostic measurements separately from full slots.
Existing profiling reports and both decompositions are under `analysis/existing/`;
raw parents/components/calls/query_calls and EC worker segments remain alongside
the full profile outputs. `analysis/profile_*` independently rechecks exact
frozen chronological-thirds selections after warmup and the unchanged closure
criterion. No numerical tolerance was changed.

Complete-call host wall means and CUDA call intervals are separate from
optimizer forward/backward/full-iteration CUDA intervals and per-worker
frame sums. All derived timing means exclude frame IDs 0–19. Dataset means
first compute each scene's run mean, then give scenes equal weight within
system/dataset/capacity regime/method/stage/scope. Native ScanNet keeps its
scene-specific capacities; H16/H19 Replica regimes remain separate. Raw all-call
means including warmup remain in original summaries and are explicitly named.
Negative speed reductions are retained. Paired reports distinguish the mean
of scene-wise percentage reductions from reductions of equal-scene mean times.
These are single-seed results: across-seed SD is unavailable/null, never zero.

EC tracker and mapper retain native concurrent workers. Stream/global-sync
intervals may include peer work, contention and host idle; they are non-additive
and must not be summed into whole-scene latency/FPS. EC profiling explicitly
separates call envelopes, actual handshake wait and active-body host wall;
CUDA parent/component partitions are different quantities. Profile aggregation
weights each chronological third by eligible calls as frozen in PROTOCOL.json.
Co engineering is BA only, with its separate dedicated BASE_CONTROL; scientific
BASE is never substituted for that control. Outside Replica, MC_DP retains
BASE and DP controls; its MC comparison is intentionally unavailable in 15 W.

Historical mode/clock/precision/environment records are in `metadata/` and
original per-run/queue files. The recorded initial 15 W GPU min/max policy
bounds were 306000000/408000000 Hz; these are not a claim of fixed achieved
clocks. Full historical jetson_clocks locking and unrecorded clocks are unknown.
Current export-time observations are separately labelled and never backfilled.
FP32 HashGrid/positional constructor evidence is frozen source evidence, not
a new runtime dtype probe. Original optional sysfs read errors/nulls are retained.

One historical controller handoff audit import error was preserved and resolved:
`ModuleNotFoundError: No module named supervisor`. The native DP execution
finished normally; its strict full raw audit subsequently passed with no rerun.
See the original failure and `AUDIT_LOADER_RECOVERY.json` under queue/amendments.
Historical engineering/metadata failures are separately labelled and do not
count as current failed required slots. See `LIMITATIONS_AND_MISSING_EVIDENCE.json`.

Datasets, meshes, checkpoints, reconstruction plots, installed environments,
build caches, credentials and compiled binaries are excluded; originals remain
untouched. Native EC auxiliary ckpts outputs are omitted; timing logs are intact.
`SHA256SUMS.json` covers every archive member except itself. Run
`python verify_export.py ORIN_15W_FINAL_RESULTS_20261027.zip` to verify ZIP CRC,
every payload hash, unique identities, controls, and all indexed file/source refs.
The external export receipt records the archive's final SHA-256 and verified counts.
'''
  text+='\nFresh reconciliation: '+json.dumps(reconcile['counts'])+'; stage counts '+json.dumps(stages)+'.\n'
  p=B/'README.md';p.write_text(text);add(p,'README.md')
  add(Path(__file__),'export_programme.py');add(B/'verify_export.py','verify_export.py')
  gen('PATH_MAPPING.json',path_mapping.copy())
  gen('RETURN_MANIFEST.json',dict(utc=now(),format='Self-contained MAXN-style benchmark/profile directories with explicit source and original-path mapping',scope='ORIN_15W seed0',required_full_slots=135,benchmarks=130,profiles=5,counts=reconcile['counts'],stage_counts=stages,excluded_author_slots=162,master_index='MASTER_RUN_INDEX.json',source_index='SOURCE_INDEX.json',files={n:r['sha256']for n,r in payload.items()},no_experiments_repeated=True,raw_bytes_preserved=True,checksum_manifest='SHA256SUMS.json',verification='ZIP CRC plus SHA256 per member plus all index and source refs; external EXPORT_COMPLETE.json is written only after actual success'))
  hashes={n:r['sha256']for n,r in payload.items()};checks=B/'SHA256SUMS.json';save(checks,hashes)
  tmp=TARGET.with_suffix('.zip.tmp');assert not tmp.exists()
  with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True)as z:
   for n,record in sorted(payload.items()):
    assert sha(record['original_path'])==record['sha256'],'Source changed during export: '+n
    z.write(record['original_path'],n)
   z.write(checks,'SHA256SUMS.json')
  verifier=load('portable_export_verifier',B/'verify_export.py');verified=verifier.verify(tmp)
  tmp.replace(TARGET);digest=sha(TARGET);TARGET.with_suffix('.zip.sha256').write_text(digest+'  '+TARGET.name+'\n')
  final_idle=support.idle_sample('MODE_15W')
  receipt=dict(utc=now(),status='EXPORTED_AND_VERIFIED',path=str(TARGET),bytes=TARGET.stat().st_size,sha256=digest,counts=reconcile['counts'],stage_counts=stages,benchmark_successful=sum(j['status']=='successful'and j['execution_kind']=='benchmark'for j in index),profile_successful=sum(j['status']=='successful'and j['execution_kind']=='component_profile'for j in index),author_excluded=162,verification=verified,missing_required_evidence=[j['id']for j in index if j['missing_evidence']],limitations_file='LIMITATIONS_AND_MISSING_EVIDENCE.json',originals_preserved=True,no_experiment_launched=True,no_power_or_clock_change=True,fresh_idle_after_export=final_idle)
  save(B/'EXPORT_COMPLETE.json',receipt);print('FINAL_EXPORT',json.dumps({k:v for k,v in receipt.items()if k!='fresh_idle_after_export'}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:save(B/'EXPORT_FAILURE.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()));raise
