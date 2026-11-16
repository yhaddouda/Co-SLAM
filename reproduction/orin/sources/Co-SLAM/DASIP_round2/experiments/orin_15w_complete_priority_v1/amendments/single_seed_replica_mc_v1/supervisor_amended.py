#!/usr/bin/env python3
"""Detached heterogeneous four-stage 15 W dispatcher; scientific runtimes unchanged."""
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
import copy,csv,fcntl,hashlib,importlib.util,json,math,os,statistics,subprocess,sys,time,traceback,zipfile
A=Path(__file__).resolve().parent
R=A.parents[1];E=R.parent;TOP=E.parent
PLAN=A/'resolved/ALL_LOCAL_TARGETS.json'
ADOPT_FD=None
NORMAL_CHILD=None
PYTHON=Path('/home/yh279050/miniforge3/envs/coslam/bin/python')
PROFILE=E/'orin_component_profile_metadata_recovery_v1';CO=E/'orin_coslam_runtime_v1/runtime'
LOCK=Path('/tmp/dasip_round2_gpu0.lock');MODE='NV Power Mode: MODE_15W\n1'
GOOD={'done','reused'};TERMINAL=GOOD|{'failed','blocked'}
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
support=load('validated_profile_local_support',PROFILE/'local_support.py')
def identity(pid):return support.process_identity(pid)
class GlobalGateError(RuntimeError):pass
class LocalJobError(RuntimeError):pass
def mode():
 try:
  actual=support.require_mode('MODE_15W');assert actual==MODE;return actual
 except Exception as e:raise GlobalGateError('Manual 15 W mode gate failed: '+repr(e))from e
def hardware():
 helper=load('optional_telemetry',PROFILE/'co_bundle/runtime/optional_sysfs_metadata.py');values={};errors={}
 for pattern in ['/sys/class/devfreq/17000000.gpu/*freq','/sys/class/thermal/thermal_zone*/temp','/sys/class/thermal/thermal_zone*/type','/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor']:
  for p in Path('/').glob(pattern.lstrip('/')):
   v,err=helper.read_optional_sysfs(p);values[str(p)]=v
   if err:errors[str(p)]=err
 return dict(utc=now(),power=mode(),sysfs=values,read_errors=errors,clock_policy='Retain actual manually selected 15 W clock settings; never set power or clocks; jetson_clocks lock state not inferred')
def clocks():
 h=hardware();baseline=read(R/'ENVIRONMENT_15W.json')
 for name in ['min_freq','max_freq']:
  key='/sys/class/devfreq/17000000.gpu/'+name
  if not isinstance(h['sysfs'].get(key),str)or h['sysfs'][key]!=baseline['sysfs'][key]:raise GlobalGateError('Retained GPU clock-policy mismatch: '+key)
 return h
def idle(three=False):
 samples=[]
 try:
  for _ in range(3 if three else 1):
   samples.append(support.idle_sample('MODE_15W'))
   if three:time.sleep(2)
 except Exception as e:raise GlobalGateError('Exclusive host GPU idle gate failed: '+repr(e))from e
 return samples

def freeze():
 for filename in ['BINDING_FREEZE.json','INSTALLATION_FREEZE.json']:
  for p,h in read(R/filename)['files'].items():
   if sha(p)!=h:raise GlobalGateError('Frozen runtime/plan/config integrity mismatch: '+p)

def upstream(verify_archive=False):
 g=read(R/'UPSTREAM_GATES.json');p=read(g['profile_complete']);x=read(g['export_complete'])
 if not(p['status']=='COMPLETE'and p['full_profiles_completed']==5 and p['ALL_REQUIRED_MAXN_COMPLETE']and x['status']=='EXPORTED_AND_VERIFIED'and x['master_benchmarks']==292 and x['profiles']==5):return False
 if x!=g['export']:raise GlobalGateError('Successful MAXN export identity changed')
 for row in g['processes']:
  if support.still_alive(row['identity']):return False
 if verify_archive:
  assert sha(x['path'])==x['sha256'];assert sha(x['master_index'])==x['master_index_sha256']
 return True

def update(s):
 s['updated_utc']=now();s['counts']=dict(Counter(j['status']for j in s['jobs']));s['completed_full_slots']=sum(j['status']in GOOD for j in s['jobs'])
 s['ALL_REQUIRED_15W_COMPLETE']=s['completed_full_slots']==135;s['EC_ENG']='excluded';s['required_full_slots']=135;s['excluded_by_author_scope_change']=162
 s['next_job']=next((j['id']for j in s['jobs']if j['status']=='queued'),None)
 s['stages']=[dict(id=p['id'],target=p['count'],counts=dict(Counter(j['status']for j in s['jobs']if j['phase']==p['id'])))for p in read(PLAN)['phases']]
 groups=defaultdict(Counter)
 for j in s['jobs']:groups[(j['seed'],j['system'],j['dataset'],j['hash_size'],j['condition'],j['execution_kind'])][j['status']]+=1
 s['coverage']=[dict(seed=k[0],system=k[1],dataset=k[2],hash_size=k[3],condition=k[4],kind=k[5],counts=dict(v))for k,v in groups.items()]
 # Forecast only from measured 15 W jobs, with explicit proxies for unmeasured families.
 measured=defaultdict(list)
 for j in s['jobs']:
  if j['status']in GOOD and j.get('elapsed_seconds'):measured[(j['system'],j['dataset'],j['hash_size'],j['condition'],j['execution_kind'])].append(j['elapsed_seconds']/j['expected_native_frames'])
 forecasts={}
 for end in [48,104,130,135]:
  lo=hi=0.;unknown=[];proxies=[]
  for j in s['jobs'][:end]:
   if j['status']not in ['queued','running']:continue
   k=(j['system'],j['dataset'],j['hash_size'],j['condition'],j['execution_kind']);v=measured.get(k)
   if not v:
    v=[x for key,vs in measured.items()if key[0]==k[0]and key[1]==k[1]and key[4]==k[4]for x in vs]
    if v:proxies.append(j['id'])
   if not v:unknown.append(j['id']);continue
   estimate=statistics.mean(v)*j['expected_native_frames'];elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(j['started_utc'])).total_seconds()if j['status']=='running'else 0
   lo+=max(0,estimate*.75-elapsed);hi+=max(0,estimate*1.5+30-elapsed)
  forecasts[str(end)]=dict(remaining_known_seconds_range=[lo,hi],unknown_jobs=unknown,proxy_jobs=proxies,estimated_finish_utc_range=None if unknown else[(datetime.now(timezone.utc)+timedelta(seconds=x)).isoformat()for x in [lo,hi]],excludes_blocked=True,excludes_unmeasured_preflights_exports=True)
 s['forecast']=dict(method='Measured 15 W job elapsed incl loading; matched family or explicit within-system/dataset proxy, 0.75–1.5 range; unknown never silently zero',milestones=forecasts)
 save(R/'CONTINUATION_STATUS.json',s)
 lines=['# 15 W amended single-seed continuation','',s['updated_utc'],f"State: {s['state']}; completed {s['completed_full_slots']}/135. EC ENG excluded.",f"Supervisor {s.get('supervisor_identity')}; child {s.get('active_child_identity')}; log {s.get('active_log')}",f"Next: {s['next_job']}",'','| Stage | Targets | Counts |','|---|---:|---|']
 lines +=[f"| {p['id']} | {p['target']} | {p['counts']} |"for p in s['stages']]
 lines+=['','Seed 0 only: 162 pending slots excluded by author; no seed-1 dispatch. Warmup 20, full native sequences, separate complete-call wall/CUDA and iteration CUDA scopes. Retained manual 15 W clocks. Native Co MC controls precede MC; ordering is not fully randomized. No EC engineering, quality, meshes, or manual kernel profiling.','','| Ordinal / Identity | State | Evidence / reason |','|---|---|---|']
 lines +=[f"| {j['ordinal']} {j['id']} | {j['status']} | {j.get('reason',j.get('output',''))} |"for j in s['jobs']]
 (R/'ORIN_RUN_AGENDA.md').write_text('\n'.join(lines)+'\n')

def readiness(s,label):
 s['state']='READINESS_SETTLING';update(s);samples=[];deadline=time.monotonic()+900
 while time.monotonic()<deadline:
  idle();samples.append(clocks())
  if len(samples)>=3:
   keys=[k for k,v in samples[-1]['sysfs'].items()if k.endswith('/temp')and isinstance(v,str)and v.isdigit()and all(isinstance(x['sysfs'].get(k),str)and x['sysfs'][k].isdigit()for x in samples[-3:])]
   if keys and all(max(int(x['sysfs'][k])for x in samples[-3:])-min(int(x['sysfs'][k])for x in samples[-3:])<=2000 for k in keys):
    save(R/'readiness'/(label+'.json'),dict(status='READY',criterion='Existing Orin criterion: three idle samples 30 s apart, <=2 C range on readable thermal channels, unchanged GPU min/max, exact mode; optional unreadable fields preserved',samples=samples));return
  time.sleep(30)
 raise GlobalGateError('Existing 15 W thermal/clock readiness failed after 15 minutes')

def run(s,argv,log,fd,label):
 freeze();mode();clocks();idle();assert os.fstat(fd).st_ino==LOCK.stat().st_ino
 argv=[str(PYTHON)if x=='@PYTHON@'else str(fd)if x=='@FD@'else str(x)for x in argv]
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('x')as f:
  began=time.perf_counter();p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',DASIP_POST_V4_EXCLUSIVE='1'))
  ident=identity(p.pid);s.update(active_child_identity=ident,active_log=str(log),active_command=argv,current=label,state='RUNNING'if label in {j['id']for j in s['jobs']}else'PREPARING_OR_VALIDATING');update(s)
  save(R/'launches'/(label+'.json'),dict(utc=now(),argv=argv,child_identity=ident,log=str(log),power=MODE,ownership_fd=fd,lock_inode=LOCK.stat().st_ino))
  fault=None
  while p.poll()is None:
   time.sleep(5)
   try:mode();clocks()
   except GlobalGateError as e:
    fault=e;p.terminate();break
   update(s)
  if fault:
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   raise fault
  rc=p.wait();elapsed=time.perf_counter()-began
 s.update(active_child_identity=None,current=None);mode();clocks()
 for i in range(15):
  try:idle();break
  except GlobalGateError:
   if i==14:raise
   time.sleep(2)
 update(s)
 save(R/'process_receipts'/(label+'.json'),dict(returncode=rc,elapsed_seconds=elapsed,utc=now(),child_identity=ident,log=str(log)))
 if rc:raise LocalJobError(f'{label} exited {rc}; original log: {log}')
 return elapsed

def normalized_config(cfg):
 c=copy.deepcopy(cfg)
 for key in ['output','exp_name']:c.get('data',{}).pop(key,None)
 return c

def audit(j,diagnostic=False):
 out=Path(j['output']);a=read(out/'summary.json');cfg=read(out/'config.json');n=50 if diagnostic else j['expected_native_frames']
 assert a['frames']==n and a['finite_parameters_and_poses']and a['diagnostic']==diagnostic and a['full_sequence']==(not diagnostic)
 assert a['initial_power']==a['final_power']==MODE and a['seed']==j['seed']and not a['quality']
 assert sha(out/'source_manifest.json')==j['source_sha256']
 assert (out/'queries.csv').is_file()or (out/'query_calls.csv').is_file()or j['runtime_family']=='ec_component_profile_validated'
 if j['execution_kind']=='component_profile':
  assert a['condition']=='BASE'and a['workload']==j['workload']
  if diagnostic:assert read(out/'fixed_state_validation/PASSED.json')['status']=='PASSED'
  else:
   assert a['selected_calls']=={'tracking':300,'bundle_adjustment':60}
   co=j['system']=='Co-SLAM';reporter=load('profile_audit_'+('co'if co else'ec'),PROFILE/'bundle'/('report_component_profiles.py'if co else'report_ec_component_profiles.py'))
   work=next(w for w in read(Path(j['runtime'])/'PROTOCOL.json')['workloads']if w['id']==j['workload']);reporter.audit_run(out,j,work,j['source_sha256'])
 else:
  assert a['condition']==j.get('driver_condition',j['condition'])and a['scene']==j['scene']and a['warmup_frames']==20
  assert normalized_config(cfg)==normalized_config(read(R/'resolved/maxn_configs'/(j['id']+'.json'))),'Effective scientific/engineering configuration changed'
  if j['runtime_family']=='co_scientific':
   native=load('frozen_native_auditor',R/'native_audit.py');native.audit(j,out,j['source_sha256'],diagnostic=diagnostic)
  elif j['runtime_family']=='co_engineering_validated':
   assert cfg['round2']['engineering']==(j['condition']=='ENG')
   verifier=load('engineering_frozen_audit',E/'orin_after_v4_maxn_v1/post_v4_execute.py')
   # Use exact inherited verifier AST, replacing only the explicit mode constant.
   source=(E/'orin_after_v4_maxn_v1/post_v4_execute.py').read_text();import ast
   node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='verify_output')
   fn=ast.get_source_segment(source,node).replace("'NV Power Mode: MAXN\\n0'",repr(MODE));scope=dict(read=read,csv=csv,math=math)
   exec(compile(fn,'<unchanged engineering verifier, 15 W expectation>','exec'),scope)
   scope['verify_output'](dict(j,condition='BASE'),out,engineering=j['condition']=='ENG',diagnostic=diagnostic)
  else:
   assert a['native_thread_schedule_preserved']and cfg['round2']['engineering']is False
   assert cfg['HashGrid']['hash_size']==j['hash_size']
   calls=list(csv.DictReader((out/'calls.csv').open()));events=list(csv.DictReader((out/'iterations.csv').open()));queries=list(csv.DictReader((out/'queries.csv').open()))
   for stage,ids in [('tracking_frame',set(range(n))),('bundle_adjustment',set(range(cfg['mapping']['every_frame'],n,cfg['mapping']['every_frame']))|{n-1})]:
    rows=[x for x in calls if x['stage']==stage];assert len(rows)==len(ids)and{int(x['frame_id'])for x in rows}==ids
   assert events and queries and all(int(q['actual_queries'])==int(q['dense_queries'])for q in queries)
   assert set(events[0])=={'frame_id','iteration_type','iteration_id','forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms'}
   assert all(math.isfinite(float(x[k]))and float(x[k])>=0 for x in events for k in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms'])
   assert all(math.isfinite(float(x[k]))and float(x[k])>=0 for x in calls for k in ['wall_ms','cuda_ms'])
 return dict(status='PASS',summary=a,summary_sha256=sha(out/'summary.json'),config_sha256=sha(out/'config.json'),source_manifest_sha256=j['source_sha256'],power=MODE,full=not diagnostic)

def mark_failed(s,j,error,failed=True):
 j.update(status='failed'if failed else'blocked',reason=repr(error));save(R/'failures'/(j['id']+'.json'),dict(utc=now(),error=repr(error),traceback=traceback.format_exc(),job=j));update(s)

def preflight_group(s,fd,system):
 tag='CO'if system=='Co-SLAM'else'EC';receipt=R/'readiness'/(tag+'_PROFILE_PREFLIGHTS.json')
 if receipt.exists():return read(receipt)['status']=='PASS'
 jobs=[j for j in s['jobs']if j['execution_kind']=='component_profile'and j['system']==system];records=[]
 try:
  for j in jobs:
   d=copy.deepcopy(j);d['output']=j['output']+'_diagnostic';name=j['id']+'_diagnostic'
   success=R/'diagnostic_receipts'/(name+'.json')
   if success.exists():assert read(success)['summary_sha256']==sha(Path(d['output'])/'summary.json');records.append(read(success));continue
   assert not Path(d['output']).exists(),'Preserving partial diagnostic; no unchanged retry'
   argv=list(j['argv']);argv[argv.index('--output')+1]=d['output'];argv+=['--frames','50','--validate-state']
   run(s,argv,R/'logs'/(name+'.log'),fd,name);result=audit(d,True);save(success,result);records.append(result)
  save(receipt,dict(status='PASS',utc=now(),checks=records));readiness(s,tag+'_after_profile_preflights');return True
 except GlobalGateError:raise
 except Exception as e:
  save(receipt,dict(status='FAIL',utc=now(),error=repr(e),traceback=traceback.format_exc(),passed=records))
  for j in jobs:
   if j['status']=='queued':mark_failed(s,j,LocalJobError('Affected profiling series preflight failed; exact evidence: '+str(receipt)),False)
  return False

def engineering_ready(s,fd):
 receipt=R/'readiness/ENGINEERING_15W_VALIDATION.json'
 if receipt.exists():return read(receipt)['status']=='PASS'
 try:
  old=read(E/'orin_after_v4_maxn_v1/CO_ENGINEERING_LOCAL_VALIDATION.json');assert old['status']=='PASS'
  rt=E/'orin_after_v4_maxn_v1/coslam_engineering_local';fixed=R/'diagnostics/ENGINEERING_FIXED_STATE_15W.json';fixed.parent.mkdir(exist_ok=True)
  run(s,['@PYTHON@',str(E/'orin_after_v4_maxn_v1/validate_rtx_reference_fixed_state.py'),'--runtime',str(rt),'--output',str(fixed)],R/'logs/ENGINEERING_FIXED_STATE_15W.log',fd,'ENGINEERING_FIXED_STATE_15W');assert read(fixed)['status']=='PASS'
  for condition in ['BASE_CONTROL','ENG']:
   j=copy.deepcopy(next(j for j in s['jobs']if j['runtime_family']=='co_engineering_validated'and j['seed']==0 and j['condition']==condition));j['output']=str(R/'diagnostics'/('ENGINEERING_'+condition+'_50'))
   assert not Path(j['output']).exists()
   argv=['@PYTHON@',str(rt/'run_coslam_round2.py'),'--dataset','Replica','--scene','office0','--datadir',j['datadir'],'--condition','BASE','--seed','0','--hash-size','16','--frames','50','--warmup','20','--output',j['output'],'--expected-power-mode','MODE_15W']
   if condition=='ENG':argv+=['--engineering']
   run(s,argv,R/'logs'/('ENGINEERING_'+condition+'_50.log'),fd,'ENGINEERING_'+condition+'_50');audit(j,True)
  save(receipt,dict(status='PASS',fixed_state=read(fixed),original_failed_online_checks_unchanged=str(E/'orin_after_v4_maxn_v1/ENGINEERING_FAILURE_REPORT.json'),reported_stage='bundle_adjustment_only'))
  readiness(s,'after_15W_engineering_preflight');return True
 except GlobalGateError:raise
 except Exception as e:
  save(receipt,dict(status='FAIL',error=repr(e),traceback=traceback.format_exc()))
  for j in s['jobs']:
   if j['runtime_family']=='co_engineering_validated'and j['status']=='queued':mark_failed(s,j,e,False)
  return False

def summaries(s):
 # Preserve both scopes; comparisons require like-for-like runtime/config/capacity/seed.
 rows=[];callgroups=defaultdict(list);byid={j['id']:j for j in s['jobs']};ratios=[]
 for j in s['jobs']:
  if j['status']not in GOOD or j['execution_kind']!='benchmark':continue
  a=read(Path(j['output'])/'summary.json')
  for stage in (['bundle_adjustment']if j['runtime_family']=='co_engineering_validated'else ['tracking_frame','bundle_adjustment']if j['system']=='EC-SLAM'else ['tracking','bundle_adjustment']):
   for metric in ['mean_wall_ms','mean_cuda_ms']:
    row=dict(id=j['id'],seed=j['seed'],system=j['system'],dataset=j['dataset'],scene=j['scene'],hash_size=j['hash_size'],condition=j['condition'],stage=stage,scope='complete_call_'+metric,ms=a[stage][metric],sd=None);rows.append(row);callgroups[(j['system'],j['dataset'],j['hash_size'],j['condition'],stage,metric,j['seed'])].append(row['ms'])
  events=list(csv.DictReader((Path(j['output'])/'iterations.csv').open()))
  for stage in sorted({x['iteration_type']for x in events if int(x['frame_id'])>=20}):
   if j['runtime_family']=='co_engineering_validated'and stage!='bundle_adjustment':continue
   for metric in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms']:
    vals=[float(x[metric])for x in events if x['iteration_type']==stage and int(x['frame_id'])>=20];rows.append(dict(id=j['id'],seed=j['seed'],system=j['system'],dataset=j['dataset'],scene=j['scene'],hash_size=j['hash_size'],condition=j['condition'],stage=stage,scope='optimizer_iteration_'+metric,ms=statistics.mean(vals),sd=None))
 for j in s['jobs']:
  if j['status']not in GOOD or j['execution_kind']!='benchmark':continue
  controls=dict(j.get('control_ids',{}))
  if j['condition']in ['MC','ENG']and not controls:
   matches=[c for c in s['jobs']if c['system']==j['system']and c['dataset']==j['dataset']and c['scene']==j['scene']and c['hash_size']==j['hash_size']and c['seed']==j['seed']and c['runtime_family']==j['runtime_family']and c['condition']==('BASE_CONTROL'if j['condition']=='ENG'else'BASE')]
   if matches:controls['BASE_CONTROL'if j['condition']=='ENG'else'BASE']=matches[0]['id']
  for name,cid in controls.items():
   c=byid[cid]
   if c['status']not in GOOD:continue
   for row in [r for r in rows if r['id']==j['id']]:
    other=next((r for r in rows if r['id']==cid and r['stage']==row['stage']and r['scope']==row['scope']),None)
    if other and other['ms']>0:ratios.append(dict(id=j['id'],control_id=cid,seed=j['seed'],dataset=j['dataset'],scene=j['scene'],system=j['system'],hash_size=j['hash_size'],condition=j['condition'],stage=row['stage'],scope=row['scope'],reduction_pct=100*(1-row['ms']/other['ms'])))
 equal=[dict(system=k[0],dataset=k[1],hash_size=k[2],condition=k[3],stage=k[4],scope=k[5],seed=k[6],equal_scene_mean_ms=statistics.mean(v),scene_count=len(v),across_seed_sd=None)for k,v in callgroups.items()]
 paired=defaultdict(list)
 for row in ratios:paired[(row['system'],row['dataset'],row['scene'],row['hash_size'],row['condition'],row['stage'],row['scope'],row['control_id'].split('_seed')[0])].append(row)
 two=[dict(system=k[0],dataset=k[1],scene=k[2],hash_size=k[3],condition=k[4],stage=k[5],scope=k[6],seeds=[x['seed']for x in v],mean_reduction_pct=statistics.mean(x['reduction_pct']for x in v),sample_sd_reduction_pct=statistics.stdev(x['reduction_pct']for x in v)if len(v)>1 else None)for k,v in paired.items()]
 save(R/'TIMING_SUMMARIES.json',dict(per_run=rows,paired_comparisons=ratios,equal_scene_means=equal,paired_seed_statistics=two,single_seed_SD=None,unavailable_comparisons=[dict(id=j['id'],comparisons=j['unavailable_comparisons'])for j in s['jobs']if j.get('unavailable_comparisons')],units='ms; reductions percent',order_disclosure='Frozen seed-first chronology; native Co BASE precedes MC in separate stages, not fully randomized; negative reductions preserved; EC concurrent intervals non-additive; Co engineering BA only'))

def export(s,tag):
 receipt=R/'exports'/(tag+'.json')
 if receipt.exists():
  x=read(receipt);assert sha(x['path'])==x['sha256'];return
 s['state']='EXPORTING_AT_EXCLUSIVE_IDLE';update(s);idle(three=True);freeze();summaries(s)
 for co in [True,False]:
  jobs=[j for j in s['jobs']if j['execution_kind']=='component_profile'and(j['system']=='Co-SLAM')==co]
  if all(j['status']in GOOD for j in jobs):
   reporter=load('report_'+str(co),PROFILE/'bundle'/('report_component_profiles.py'if co else'report_ec_component_profiles.py'));reporter.report(R,R/'resolved'/('CO_PROFILE_ACTIVE.json'if co else'EC_PROFILE_ACTIVE.json'))
 target=TOP/('ORIN_15W_'+tag+'_RESULTS.zip');temp=target.with_suffix('.zip.tmp');assert not target.exists()and not temp.exists();hashes={}
 prior={}
 for p in (R/'exports').glob('*.json'):
  for i,h in read(p).get('exported_identities',{}).items():prior[i]=h
 newly=[j for j in s['jobs']if j['status']in GOOD and j['id']not in prior]
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED)as z:
  for p in R.rglob('*'):
   if not p.is_file()or '__pycache__'in p.parts or p.suffix in ['.zip','.tmp','.so','.sens','.pt','.ply','.pth','.npz']:continue
   # Raw data is incremental; reports, evidence, diagnostics and frozen local bindings accompany every checkpoint.
   if 'results'in p.relative_to(R).parts:
    allowed=any(p.is_relative_to(Path(j['output']))for j in newly)
    if not allowed and '_diagnostic'not in str(p)and 'component_profile_analysis_v1'not in str(p)and 'ec_component_profile_analysis_v1'not in str(p):continue
   name=p.relative_to(R).as_posix();data=p.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
  master=[dict(id=j['id'],ordinal=j['ordinal'],status=j['status'],output=j['output'],source_sha256=j['source_sha256'],summary_sha256=sha(Path(j['output'])/'summary.json')if j['status']in GOOD else None,previous_increment=j['id']in prior)for j in s['jobs']]
  z.writestr('RETURN_MANIFEST.json',json.dumps(dict(utc=now(),mode=MODE,total135=True,completed=s['completed_full_slots'],EC_ENG='excluded',files=hashes,master_index=master,excluded_identities=read(A/'bundle/EXCLUSIONS.json')['jobs'],single_seed_SD=None,incremental_identities=[j['id']for j in newly]),indent=2))
 with zipfile.ZipFile(temp)as z:
  assert z.testzip()is None
  for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 temp.replace(target);digest=sha(target);target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n');save(receipt,dict(status='EXPORTED_AND_VERIFIED',utc=now(),path=str(target),sha256=digest,exported_identities={j['id']:sha(Path(j['output'])/'summary.json')for j in newly},files_verified=len(hashes),completed=s['completed_full_slots']))
 readiness(s,'after_export_'+tag)

def prepare_data():
 import numpy as np
 sys.path.insert(0,str(CO));from datasets.dataset import get_dataset
 done={};errors={}
 for j in read(PLAN)['jobs']:
  key=(j['system'],j['dataset'],j['scene'])
  if key in done or key in errors:continue
  try:
   cfg=read(R/'resolved/maxn_configs'/(j['id']+'.json'));n=j['expected_native_frames'];probes=[]
   if j['system']=='Co-SLAM':
    ds=get_dataset(cfg);assert len(ds)==n
    for i in sorted({0,n//2,n-1}):
     b=ds[i];assert np.isfinite(b['rgb'].numpy()).all()and np.isfinite(b['depth'].numpy()).all();probes.append(dict(frame=i,rgb_shape=list(b['rgb'].shape),depth_shape=list(b['depth'].shape),native_pose_finite=bool(np.isfinite(b['c2w'].numpy()).all())))
   else:
    d=Path(j['datadir']);assert len(list((d/'depth').glob('depth_*.png')))==n
    poses=np.loadtxt(d/'traj_w_c.txt').reshape(-1,4,4);assert len(poses)==n
    import cv2
    for i in sorted({0,n//2,n-1}):
     dep=cv2.imread(str(d/'depth'/f'depth_{i}.png'),-1);rgb=cv2.imread(str(d/'rgb'/f'rgb_{i}.png'));assert dep is not None and rgb is not None;probes.append(dict(frame=i,depth_shape=list(dep.shape),rgb_shape=list(rgb.shape)))
   done[key]=dict(status='PASS',frames=n,datadir=j['datadir'],native_full_sequence=True,probes=probes)
  except Exception as e:errors[key]=dict(status='FAIL',error=repr(e),traceback=traceback.format_exc())
 save(R/'DATA_READINESS.json',dict(verified={'/'.join(k):v for k,v in done.items()},failures={'/'.join(k):v for k,v in errors.items()}))
 print('DATA_READINESS',len(done),'passed',len(errors),'failed',flush=True)

def prepare_gpu():
 mode();import torch
 from torch.utils.cpp_extension import load as extload
 from torch.utils.cpp_extension import _get_build_directory
 major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}';name=f'dasip_round2_morton_{major}{minor}'
 cache=Path(_get_build_directory(name,False));binary=cache/(name+'.so');assert binary.is_file(),'Prepared Morton binary missing; no automatic install/compilation'
 before=sha(binary);ext=extload(name=name,sources=[str(CO/'model/round2_morton/morton_cuda.cu')],extra_cuda_cflags=['-O3'],verbose=True);assert sha(binary)==before,'Unexpected cache rebuild'
 pts=torch.tensor([[0.,0.,0.],[.5,.5,.5],[1.,1.,1.]],device='cuda',dtype=torch.float32);assert ext.morton3d_keys(pts.contiguous(),128).numel()==3;torch.cuda.synchronize()
 save(R/'GPU_READINESS.json',dict(status='PASS',mode=mode(),device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,capability=[major,minor],binary=str(binary),sha256=before,compiled=False,performed_before_timings=True))
 print('GPU_CACHE_READY',str(binary),flush=True)

def reconcile(s):
 # Finished receipts win over stale running status after a controller restart; never rerun a partial or failed output.
 for j in s['jobs']:
  receipt=R/'completion_records'/(j['id']+'.json')
  if receipt.exists():
   a=read(receipt);assert a['status']=='PASS'and a['summary_sha256']==sha(Path(j['output'])/'summary.json');assert a['source_manifest_sha256']==j['source_sha256']and a['config_sha256']==sha(Path(j['output'])/'config.json');j.update(status='done',elapsed_seconds=a['elapsed_seconds']);continue
  if j['status']=='running':
   launch=read(R/'launches'/(j['id']+'.json'))
   if support.still_alive(launch['child_identity']):raise GlobalGateError('Prior child still alive; preserving ownership, no duplicate: '+j['id'])
  if j['status']not in TERMINAL and(Path(j['output']).exists()or (R/'launches'/(j['id']+'.json')).exists()):mark_failed(s,j,LocalJobError('Prior incomplete launch preserved; no unchanged retry'),False)
 candidates=read(R/'EXISTING_15W_CANDIDATES.json')['outputs']
 if candidates:
  # Never duplicate an unverified existing 15 W full identity; fail closed for its compatibility proof.
  for p in candidates:
   a=read(Path(p)/'summary.json');cfg=read(Path(p)/'config.json')
   for j in s['jobs']:
    if j['status']=='queued'and j['execution_kind']=='benchmark'and(a.get('dataset'),a.get('scene'),a.get('condition'),a.get('seed'),cfg.get('grid',cfg.get('HashGrid',{})).get('hash_size'))==(j['dataset'],j['scene'],j.get('driver_condition',j['condition']),j['seed'],j['hash_size']):mark_failed(s,j,LocalJobError('Existing 15 W full result requires exact clock-policy/completion proof before reuse: '+p),False)

def apply_amendment_state(s):
 old={j['id']:j for j in s['jobs']};plan=read(PLAN);retained=[];removed=[]
 for template in plan['jobs']:
  j=copy.deepcopy(old.get(template['id'],template))
  for k in ['ordinal','control_ids','unavailable_comparisons']:
   if k in template:j[k]=template[k]
   elif k in j and k=='unavailable_comparisons':j.pop(k)
  assert j['seed']==0
  retained.append(j)
 keep={j['id']for j in retained}
 for item in read(A/'bundle/EXCLUSIONS.json')['jobs']:
  j=old.get(item['id'])
  if j:
   assert j['status']!='running','Active excluded child must finish normally before amendment'
   removed.append(dict(item,status='excluded_by_author_scope_change',previous_status=j['status'],output=j.get('output'),required_for_completion=False))
  else:removed.append(dict(item,status='excluded_by_author_scope_change',required_for_completion=False))
 assert len(retained)==135 and len(removed)==162 and all(x['id']not in keep for x in removed)
 save(A/'EXCLUDED_LOCAL_IDENTITIES.json',dict(utc=now(),count=162,jobs=removed))
 s['jobs']=retained;s['amendment']=dict(path=str(A),required=135,excluded=162,scope='seed0_only; MC_only_Replica; MC_DP_all_four_datasets; Co_ENG_BA_only; five_profiles')
 s['state']='AMENDMENT_LOADED_AT_NORMAL_DISPATCH_BOUNDARY'
 return s

def adopt_normal_completion(s):
 if not NORMAL_CHILD:return
 j=next(j for j in s['jobs']if j['id']==NORMAL_CHILD)
 receipt=R/'completion_records'/(j['id']+'.json')
 if receipt.exists():return
 child=read(R/'process_receipts'/(j['id']+'.json'));assert child['returncode']==0
 assert not support.still_alive(child['child_identity'])
 try:
  result=audit(j);result.update(elapsed_seconds=child['elapsed_seconds'],utc=now(),clock_policy_record=str(R/'ENVIRONMENT_15W.json'),adopted_normal_dispatch_child=True)
  save(receipt,result);j.update(status='done',elapsed_seconds=child['elapsed_seconds'],finished_utc=now(),summary=result['summary'],completion_record=str(receipt))
 except GlobalGateError:raise
 except Exception as e:mark_failed(s,j,e)
 update(s)

def self_test():
 a=read(PLAN);assert len(a['jobs'])==135 and len({j['id']for j in a['jobs']})==135
 assert [p['count']for p in a['phases']]==[48,56,26,5]
 assert all(j['seed']==0 for j in a['jobs'])
 for j in a['jobs']:
  assert Path(j['datadir']).is_dir();assert '--frames'not in j['argv']or j['argv'][j['argv'].index('--frames')+1]=='0'
  assert j['source_sha256']==sha(Path(j['runtime'])/('EC_RUNTIME_MANIFEST.json'if j['runtime_family']=='ec_native_timing'else'RUNTIME_MANIFEST.json'))
  assert str(j['runtime'])in ' '.join(j['argv']);assert j['output']in j['argv']or str(Path(j['output']).parent)in j['argv']
 assert all(j['dataset']=='Replica' for j in a['jobs'] if j['condition']=='MC')
 assert len([j for j in a['jobs'] if j['condition']=='MC'])==32
 assert all('MC'not in j.get('control_ids',{}) and 'MC'in j.get('unavailable_comparisons',{})for j in a['jobs']if j['condition']=='MC_DP'and j['dataset']!='Replica')
 assert len([j for j in a['jobs']if j['execution_kind']=='component_profile'])==5
 print('SUPERVISOR_SELF_TEST_PASS: 135 resolved dispatches, all 4 stages 48/56/26/5; seed 0 only; no obsolete non-Replica MC dependency; original numerical validators unchanged')

def main():
 with (R/'supervisor.lock').open('a')as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status=R/'CONTINUATION_STATUS.json';s=read(status)if status.exists()else dict(jobs=read(PLAN)['jobs'],state='WAITING_FOR_NORMAL_MAXN_END',active_child_identity=None)
  s=apply_amendment_state(s)
  s.update(supervisor_identity=identity(os.getpid()),session_id=os.getsid(0),active_log=None);save(R/'SUPERVISOR_IDENTITY.json',s['supervisor_identity']);update(s)
  while not upstream():s['state']='WAITING_FOR_NORMAL_MAXN_END';update(s);time.sleep(30)
  while True:
   try:mode();break
   except GlobalGateError:s['state']='WAITING_FOR_MANUAL_15W';update(s);time.sleep(30)
  with (os.fdopen(os.dup(ADOPT_FD),'a')if ADOPT_FD is not None else LOCK.open('a'))as lock:
   while True:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:s['state']='WAITING_FOR_EXCLUSIVE_GPU';update(s);time.sleep(15)
   fd=lock.fileno();os.set_inheritable(fd,True);assert os.fstat(fd).st_ino==LOCK.stat().st_ino;freeze();idle(three=True);assert upstream(False)
   if ADOPT_FD is not None and ADOPT_FD!=fd:os.close(ADOPT_FD)
   save(A/'LAUNCH_GATES_PASSED.json',dict(utc=now(),mode=mode(),normal_maxn_profile_and_export=True,prior_processes_exited=True,verified_final_maxn_archive=True,exclusive_lock_inode=LOCK.stat().st_ino,fresh_idle=idle(three=True)))
   if not(R/'ENVIRONMENT_15W.json').exists():save(R/'ENVIRONMENT_15W.json',hardware())
   clocks();adopt_normal_completion(s);reconcile(s)
   save(A/'AMENDMENT_CONSUMED.json',dict(utc=now(),status='CONSUMED_BY_LIVE_SUPERVISOR',supervisor_identity=identity(os.getpid()),session_id=os.getsid(0),required=135,stages=[p['count']for p in read(PLAN)['phases']],loaded_ids=[j['id']for j in s['jobs']],loaded_ids_sha256=hashlib.sha256(json.dumps([j['id']for j in s['jobs']],separators=(',',':')).encode()).hexdigest(),excluded=162,normal_job_adopted=NORMAL_CHILD,gpu_lock_inode=LOCK.stat().st_ino,source_runtimes_unchanged=True));update(s)
   if not(R/'DATA_READINESS.json').exists():run(s,['@PYTHON@',str(Path(__file__).resolve()),'--prepare-data'],R/'logs/data_readiness.log',fd,'data_readiness')
   data=read(R/'DATA_READINESS.json')
   for j in s['jobs']:
    if '/'.join([j['system'],j['dataset'],j['scene']])in data['failures']and j['status']=='queued':mark_failed(s,j,LocalJobError('Affected native scene failed data readiness; exact DATA_READINESS.json preserved'),False)
   if not(R/'GPU_READINESS.json').exists():run(s,['@PYTHON@',str(Path(__file__).resolve()),'--prepare-gpu'],R/'logs/gpu_cache_readiness.log',fd,'gpu_cache_readiness')
   readiness(s,'after_author_queue_amendment_before_next_required_full_run');update(s)
   for phase in read(PLAN)['phases']:
    s['phase']=phase['id'];update(s)
    for j in [x for x in s['jobs']if x['phase']==phase['id']]:
     if j['status']in TERMINAL:continue
     # A control queued later in the same frozen phase (seed-1 reversed pair) does not gate execution.
     lookup={x['id']:x for x in s['jobs']}
     bad=[cid for cid in j.get('control_ids',{}).values()if lookup[cid]['status']in ['failed','blocked']]
     if bad:mark_failed(s,j,LocalJobError('Affected comparison control failed: '+repr(bad)),False);continue
     if j['runtime_family']=='co_engineering_validated'and not engineering_ready(s,fd):continue
     if j['execution_kind']=='component_profile'and not preflight_group(s,fd,j['system']):continue
     try:
      assert not Path(j['output']).exists(),'Preserving pre-existing partial output'
      j.update(status='running',started_utc=now());update(s)
      elapsed=run(s,j['argv'],R/'logs'/(j['id']+'.log'),fd,j['id']);result=audit(j);result.update(elapsed_seconds=elapsed,utc=now(),clock_policy_record=str(R/'ENVIRONMENT_15W.json'));save(R/'completion_records'/(j['id']+'.json'),result)
      j.update(status='done',elapsed_seconds=elapsed,finished_utc=now(),summary=result['summary'],completion_record=str(R/'completion_records'/(j['id']+'.json')));update(s)
      if j['execution_kind']=='component_profile':save(Path(j['output']).parent/(j['id']+'.done.json'),dict(summary_sha256=result['summary_sha256'],elapsed_seconds=elapsed))
     except GlobalGateError:raise
     except Exception as error:mark_failed(s,j,error);mode();clocks();idle()
    save(R/'stage_completions'/(phase['id']+'.json'),dict(utc=now(),stage=phase['id'],counts=dict(Counter(j['status']for j in s['jobs']if j['phase']==phase['id'])),normal_ready_backlog_exhausted=True))
    if phase['priority']in [1,3,4]:
     tag={1:'S0_DP_48',3:'S0_BENCHMARKS_130',4:'S0_PROFILES_135'}[phase['priority']]
     try:export(s,tag)
     except GlobalGateError:raise
     except Exception as error:
      save(R/'failures'/('EXPORT_'+tag+'.json'),dict(utc=now(),error=repr(error),traceback=traceback.format_exc(),raw_completed_results_preserved=True,independent_ready_backlog_continues=True));readiness(s,'after_preserved_export_failure_'+tag)
   assert not any(j['status']in ['queued','running']for j in s['jobs']);s['state']='READY_BACKLOG_EXHAUSTED';update(s);export(s,'FINAL_135');fresh=idle(three=True)
   save(R/'CONTINUATION_COMPLETE.json',dict(utc=now(),status='COMPLETE'if s['ALL_REQUIRED_15W_COMPLETE']else'EXHAUSTED_WITH_FAILURES',ALL_REQUIRED_15W_COMPLETE=s['ALL_REQUIRED_15W_COMPLETE'],full_required=135,counts=s['counts'],pending=[dict(id=j['id'],status=j['status'],reason=j.get('reason'))for j in s['jobs']if j['status']not in GOOD],EC_ENG='excluded',exports=[read(p)for p in (R/'exports').glob('*.json')],fresh_idle=fresh,power=mode(),no_ready_continuation=True))
  s['state']='COMPLETE'if s['ALL_REQUIRED_15W_COMPLETE']else'EXHAUSTED_WITH_FAILURES';update(s)
  notification=read(R/'UPSTREAM_GATES.json')['notification'];receipt=R/'DECISION_WAKE.json'
  if not receipt.exists():
   message=f"The authorized four-stage Orin 15 W programme has exhausted its admitted backlog. Read {R/'CONTINUATION_COMPLETE.json'} and export receipts. Recheck fresh GPU/program idle; check EMAIL_SENT.json and Gmail Sent first. Only if this is a completed major series/export and no ready continuation remains, send the authorized concise decision email to {notification['recipient']} via connected Gmail, state exact 135-slot counts and failures truthfully, EC ENG excluded, and save a real sent-message receipt. Keep 15 W; no repeats, mode/clock changes or new series. Wake is not email delivery."
   try:
    r=subprocess.run([notification['codex_executable'],'queue','--thread',notification['thread_id'],'--message',message],capture_output=True,text=True,timeout=55);save(receipt,dict(utc=now(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,email_sent=False))
   except Exception as e:save(receipt,dict(utc=now(),error=repr(e),email_sent=False))

if __name__=='__main__':
 if '--self-test'in sys.argv:self_test()
 elif '--prepare-data'in sys.argv:prepare_data()
 elif '--prepare-gpu'in sys.argv:prepare_gpu()
 else:
  import argparse
  parser=argparse.ArgumentParser();parser.add_argument('--adopt-owned-lock-fd',type=int);parser.add_argument('--completed-normal-child');args=parser.parse_args()
  ADOPT_FD=args.adopt_owned_lock_fd;NORMAL_CHILD=args.completed_normal_child
  try:main()
  except BaseException as e:
   save(R/'SUPERVISOR_STOPPED.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),not_normal_completion=True,power_or_clocks_changed=False));print(traceback.format_exc(),flush=True)
   if (R/'CONTINUATION_STATUS.json').exists():
    s=read(R/'CONTINUATION_STATUS.json');s['state']='BLOCKED_GLOBAL_GATE';s['stop_reason']=repr(e);update(s)
   raise
