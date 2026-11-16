#!/usr/bin/env python3
"""Serial, detached NCU first-500 grid-kernel captures; no power/clock changes."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,fcntl,hashlib,importlib.util,io,json,os,platform,pwd,re,signal,subprocess,sys,time,traceback,zipfile
ROOT=Path(__file__).resolve().parent
VARIANTS={'A':dict(hash='CoherentPrime',morton_sort=False,label='A_baseline'),'B':dict(hash='Morton',morton_sort=False,label='B_morton_hash'),'C':dict(hash='CoherentPrime',morton_sort=True,label='C_morton_order'),'D':dict(hash='Morton',morton_sort=True,label='D_memory_coherent')}
SECTIONS=['MemoryWorkloadAnalysis','SourceCounters','SpeedOfLight'];LIMIT=500

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def ident(pid):
 try:
  p=Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),ppid=int(s[1]),session_id=int(s[3]),start_ticks=int(s[19]),state=s[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1])
 except OSError:return None

def mode(config):
 if config['platform']=='RTX':return dict(platform='RTX',power_mode_not_applicable=True)
 raw=subprocess.check_output(['/usr/sbin/nvpmodel','-q'],text=True,timeout=10).strip();expected='MODE_15W'if config['mode']=='15W'else'MAXN'
 name=next((s.split(':',1)[1].strip()for s in raw.splitlines()if s.startswith('NV Power Mode:')),None)
 assert name==expected,'Manual '+expected+' required; no settings changed: '+raw
 return dict(raw=raw,name=name)
def optional_read(p):
 try:
  fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC)
  try:return os.read(fd,4096).decode().strip()
  finally:os.close(fd)
 except Exception as e:return dict(unavailable=True,error=repr(e))
def hardware(config):
 values={}
 if config['platform']=='ORIN':
  for pattern in ['class/devfreq/17000000.gpu/*freq','class/thermal/thermal_zone*/temp']:
   for p in Path('/sys').glob(pattern):values[str(p)]=optional_read(p)
 return dict(utc=now(),power=mode(config),sysfs=values,clock_policy='Retain actual settings; NCU --clock-control none; no power/clock setters',cache_policy='NCU --cache-control all (original default; caches invalidated for replay)',profiler='Nsight Compute kernel replay, profiling startup/first-frame work, not representative full-scene latency')
def idle(config):
 if config['platform']=='ORIN':
  spec=importlib.util.spec_from_file_location('validated_orin_idle',ROOT/'support/validated_orin_idle_checker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);sample=m.idle()
 else:
  q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert q.returncode==0,q.stderr;assert not q.stdout.strip(),'Other GPU compute users: '+q.stdout;sample=dict(utc=now(),compute_users=q.stdout)
 # The validated Orin checker already covers ncu/nsys and native runners;
 # also reject standalone component-profile children on either platform.
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except FileNotFoundError:continue
  if any(Path(a).name in ['run_component_profile.py','run_ec_component_profile.py']for a in args if a):raise RuntimeError('Active component profile: '+p.name)
 sample['power']=mode(config);return sample

def profiler_env():return dict(os.environ,PYTHONDONTWRITEBYTECODE='1')

def env_for(config):
 prefix=Path(config['python']).parent.parent;e=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PATH=str(prefix/'bin')+':'+os.environ.get('PATH',''),LD_LIBRARY_PATH=str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),TORCH_EXTENSIONS_DIR=str(ROOT/'prepared_extensions'/config['platform']))
 if config.get('preload_libstdcxx')and(prefix/'lib/libstdc++.so.6').is_file():e['LD_PRELOAD']=str(prefix/'lib/libstdc++.so.6')
 return e

def cases(config):
 jobs=[]
 for system in ['CoSLAM','ECSLAM']:
  for capacity in [16,19]:
   for variant,spec in VARIANTS.items():
    tag=f"{config['platform']}_{config['mode']}_{system}_office0_H{capacity}_{spec['label']}_seed0_grid_first500";folder=ROOT/'results'/config['platform']/config['mode']/system/'office0'/('H'+str(capacity))/spec['label']
    jobs.append(dict(id=tag,system=system,scene='office0',hash_size=capacity,seed=0,variant=variant,hash=spec['hash'],morton_sort=spec['morton_sort'],folder=str(folder),report=str(folder/(tag+'.ncu-rep')),status='queued'))
 return jobs

def driver_command(j,config):
 out=Path(j['folder'])/'app_output';co=ROOT/'runtime/co';ec=ROOT/'runtime/ec';args=[config['python'],str(co/'run_coslam_round2.py'if j['system']=='CoSLAM'else ec/'run_ec_round2.py')]
 if j['system']=='CoSLAM':args+=['--dataset','Replica','--scene','office0','--datadir',config['co_datadir'],'--warmup','20']
 else:args+=['--scene','office0','--datadir',config['ec_datadir'],'--co-runtime',str(co)]
 args+=['--condition','BASE','--ncu-variant',j['variant'],'--hash-size',str(j['hash_size']),'--seed','0','--frames','0','--output',str(out)]
 if config['platform']=='ORIN':args+=['--expected-power-mode','MODE_15W'if config['mode']=='15W'else'MAXN']
 return args

def ncu_command(j,config):
 args=[config['ncu'],'--target-processes','all','--kernel-name-base','function','--kernel-name','regex:.*grid.*','--filter-mode','global','--launch-count',str(LIMIT),'--launch-skip','0','--replay-mode','kernel','--cache-control','all','--clock-control','none','--kill','yes','--check-exit-code','no']
 for section in SECTIONS:args+=['--section',section]
 return args+['-f','--export',str(Path(j['report']).with_suffix('')),'/bin/bash',str(ROOT/'run_target.sh'),config['python'],str(ROOT/'prepared_extensions'/config['platform']),'yes'if config.get('preload_libstdcxx')else'no']+driver_command(j,config)[1:]

def verify_kit():
 manifest=read(ROOT/'SHA256SUMS.json')
 for name,h in manifest.items():assert sha(ROOT/name)==h,'Kit file changed: '+name
 for rt,name,key in [('co','RUNTIME_MANIFEST.json','runtime_hashes'),('ec','EC_RUNTIME_MANIFEST.json','files')]:
  p=ROOT/'runtime'/rt
  for name,h in read(p/name)[key].items():assert sha(p/name)==h,'Runtime file changed: '+str(p/name)

def check_config(config):
 assert config['platform']in ['ORIN','RTX'];assert config['mode']in (['15W','MAXN']if config['platform']=='ORIN'else['RTX'])
 assert Path(config['python']).is_file()and Path(config['ncu']).is_file()
 assert Path(config['co_datadir']).is_dir()and Path(config['ec_datadir']).is_dir(),'Resolve native dataset paths before launching'
 assert len(cases(config))==16

def report_count(path):
 text=Path(path).read_text(errors='replace');lines=text.splitlines();header=next((i for i,l in enumerate(lines)if {'ID','Kernel Name'}.issubset(set(next(csv.reader([l]))))),None)
 if header is None:raise ValueError('No NCU raw CSV kernel table; preserve import log')
 rows=list(csv.DictReader(io.StringIO('\n'.join(lines[header:]))));records=[r for r in rows if r.get('ID')and r.get('Kernel Name')];keys={(r.get('Process ID',''),r.get('Context',''),r['ID'])for r in records};names=sorted({r['Kernel Name']for r in records});assert len(keys)==LIMIT,f'Expected {LIMIT} matched launches, got {len(keys)}'
 assert all('grid'in n for n in names),'Report contains a nonmatching kernel name'
 section_names=sorted({r.get('Section Name','')for r in records})
 return dict(matched_kernel_launches=len(keys),kernel_names=names,recorded_sections=section_names,raw_metric_rows=len(records),filter='regex:.*grid.*',kernel_name_base='function',no_warmup_skip=True,first_500_matching_grid_launches=True)

def verify_sections(path):
 lines=Path(path).read_text(errors='replace').splitlines();header=next((i for i,l in enumerate(lines)if {'Section Name','Kernel Name'}.issubset(set(next(csv.reader([l]))))),None)
 assert header is not None,'No section table in details CSV'
 rows=list(csv.DictReader(io.StringIO('\n'.join(lines[header:]))));names=sorted({r.get('Section Name','')for r in rows if r.get('Kernel Name')});norm=[''.join(c.lower()for c in s if c.isalnum())for s in names]
 for section in ['memoryworkloadanalysis','sourcecounters','speedoflight']:assert any(section in s for s in norm),'Requested section evidence absent: '+section+'; names='+repr(names)
 return names

def fix_ownership(folder,config):
 uid=config.get('owner_uid',os.getuid());gid=config.get('owner_gid',os.getgid())
 if os.geteuid()!=0:return
 for p in [Path(folder),*Path(folder).rglob('*')]:
  if not p.is_symlink():os.chown(p,uid,gid)

def status_save(state,base,config):
 state['updated_utc']=now();state['counts']={k:sum(j['status']==k for j in state['jobs'])for k in ['done','running','queued','failed','blocked']};state['next_job']=next((j['id']for j in state['jobs']if j['status']=='queued'),None);save(base/'STATUS.json',state)
 text='# NCU first-500 grid captures\n\n'+state['updated_utc']+'\n\n'+state['state']+'; '+str(state['counts'])+'; next '+str(state['next_job'])+'\n\n| Capture | State | GUI report |\n|---|---|---|\n'
 text+=''.join('| '+j['id']+' | '+j['status']+' | '+j['report']+' |\n'for j in state['jobs']);(base/'STATUS.md').write_text(text);fix_ownership(base,config)

def execute(j,config,state,base,fd,baseline):
 folder=Path(j['folder']);folder.mkdir(parents=True,exist_ok=False);verify_kit();idle(config);start_hw=hardware(config)
 if config['platform']=='ORIN':
  for name in ['min_freq','max_freq']:
   k='/sys/class/devfreq/17000000.gpu/'+name;assert start_hw['sysfs'][k]==baseline['sysfs'][k],'Clock policy changed'
 argv=ncu_command(j,config);save(folder/'command.json',dict(argv=argv,driver=driver_command(j,config),variant=VARIANTS[j['variant']],mode=config['mode'],launch_limit=LIMIT,sections=SECTIONS,pruning=False,engineering=False,clock_control='none',cache_control='all',replay='kernel',report_not_a_completed_SLAM_benchmark=True));save(folder/'environment_initial.json',start_hw)
 with (folder/'capture.log').open('x')as log:
  child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=profiler_env(),pass_fds=(fd,),start_new_session=True)
  state.update(state='RUNNING',active_child=ident(child.pid),active_log=str(folder/'capture.log'));j['status']='running';status_save(state,base,config)
  fault=None
  while child.poll()is None:
   time.sleep(5)
   try:mode(config)
   except Exception as e:fault=e;os.killpg(child.pid,signal.SIGTERM);break
   status_save(state,base,config)
  try:rc=child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
 state['active_child']=None
 for i in range(15):
  try:idle(config);break
  except Exception:
   if i==14:raise
   time.sleep(2)
 if fault:raise RuntimeError('Power mode changed during capture; exact evidence preserved: '+repr(fault))
 assert rc==0,'NCU failed with exit '+str(rc)+'; inspect capture.log'
 report=Path(j['report']);assert report.is_file()and report.stat().st_size>0,'Missing GUI report'
 commands=[(['--page','details','--csv','--print-details','all','--print-units','base'],folder/'section_details.csv'),(['--page','raw','--csv','--print-units','base'],folder/'raw_metrics.csv'),(['--print-summary','per-kernel'],folder/'kernel_summary.txt'),(['--print-summary','per-kernel','--csv','--print-units','base'],folder/'kernel_summary.csv')]
 for opts,out in commands:
  with out.open('x')as stream:
   q=subprocess.run([config['ncu'],'--import',str(report),*opts],stdout=stream,stderr=subprocess.PIPE,text=True,env=profiler_env());assert q.returncode==0,q.stderr
 checked=report_count(folder/'raw_metrics.csv');checked['requested_sections_verified']=verify_sections(folder/'section_details.csv');end_hw=hardware(config);save(folder/'environment_final.json',end_hw)
 if config['platform']=='ORIN':
  for name in ['min_freq','max_freq']:
   key='/sys/class/devfreq/17000000.gpu/'+name;assert end_hw['sysfs'][key]==baseline['sysfs'][key],'Clock policy changed'
 receipt=dict(status='CAPTURED_AND_VERIFIED',utc=now(),id=j['id'],report=str(report),sha256=sha(report),bytes=report.stat().st_size,NCU_returncode=rc,target_stopped_by_profiler_after_limit=True,validation=checked,config_snapshot=str(folder/'app_output/config.json'),full_benchmark_not_executed=True)
 assert read(folder/'app_output/config.json')['round2']['ncu_variant']==j['variant'];assert read(folder/'app_output/config.json')['round2']['engineering']is False
 save(folder/'CAPTURE_COMPLETE.json',receipt);j.update(status='done',receipt=receipt);status_save(state,base,config)

def package_results(base,config,state):
 assert not state.get('active_child');idle(config);name=f"{config['platform']}_{config['mode']}_NCU_office0_ABCD_H16_H19_first500_RESULTS.zip";target=ROOT/name;assert not target.exists();hashes={};tmp=target.with_suffix('.zip.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True)as z:
  for p in sorted(base.rglob('*')):
   if not p.is_file()or any(x in p.parts for x in ['__pycache__','ckpts','artifacts']):continue
   rel='results/'+p.relative_to(base).as_posix();z.write(p,rel);hashes[rel]=sha(p)
  for name in ['README.md','PROTOCOL.json','SHA256SUMS.json','CO_SOURCE_DERIVATION.json','EC_SOURCE_DERIVATION.json']:
   p=ROOT/name;z.write(p,'kit/'+name);hashes['kit/'+name]=sha(p)
  manifest=json.dumps(dict(utc=now(),platform=config['platform'],power_mode=config['mode'],counts=state['counts'],jobs=state['jobs'],files=dict(hashes)),indent=2).encode();z.writestr('REPORT_MANIFEST.json',manifest);hashes['REPORT_MANIFEST.json']=hashlib.sha256(manifest).hexdigest();z.writestr('SHA256SUMS.json',json.dumps(hashes,indent=2))
 with zipfile.ZipFile(tmp)as z:
  assert z.testzip()is None
  for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
 tmp.replace(target);digest=sha(target);target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n');save(base/'EXPORT_COMPLETE.json',dict(utc=now(),path=str(target),bytes=target.stat().st_size,sha256=digest,counts=state['counts'],zip_integrity_and_payloads_verified=True));fix_ownership(base,config)
 if os.geteuid()==0:os.chown(target,config['owner_uid'],config['owner_gid']);os.chown(target.with_suffix('.zip.sha256'),config['owner_uid'],config['owner_gid'])

def worker(config):
 check_config(config);verify_kit()
 if config['platform']=='ORIN'and os.geteuid()!=0:raise PermissionError('Orin NCU requires root. Run the prepared start_orin_15w.sh with sudo in your own terminal; do not send passwords to chat.')
 base=ROOT/'results'/config['platform']/config['mode'];base.mkdir(parents=True,exist_ok=True)
 with (base/'controller.lock').open('a')as own,open('/tmp/dasip_round2_gpu0.lock','a')as lock:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fd=lock.fileno();os.set_inheritable(fd,True);idle(config)
  if config['platform']=='ORIN':
   # Full prior paper programme/export already complete; never treat a job gap as a gate.
   bound=config.get('paper_completion')
   if bound:
    r=read(bound);assert r['status']=='COMPLETE'and r['ALL_REQUIRED_15W_COMPLETE']and r['full_required']==135
  freeze=read(ROOT/'SHA256SUMS.json');save(base/'KIT_IDENTITY.json',dict(utc=now(),files=freeze,config=config,profiler_version=subprocess.check_output([config['ncu'],'--version'],text=True),profile_from_start=True,first_launch_skip=0))
  prep=ROOT/'preparation'/config['platform']/config['mode']/'PREPARED.json';assert prep.is_file()and read(prep)['status']=='PASS','Prepare native Morton extension before counters; run --prepare first'
  ready=read(prep);assert ready['files_sha256']==read(ROOT/'SHA256SUMS.json'),'Prepared source identity differs'
  sections=subprocess.run([config['ncu'],'--list-sections'],capture_output=True,text=True,env=profiler_env());save(base/'NCU_SECTIONS.json',dict(returncode=sections.returncode,stdout=sections.stdout,stderr=sections.stderr));assert sections.returncode==0,sections.stderr
  for name in SECTIONS:assert name in sections.stdout,'NCU section unavailable: '+name
  previous=read(base/'STATUS.json')if(base/'STATUS.json').exists()else None
  if previous:assert previous['config']==config;state=previous
  else:state=dict(state='READY',config=config,jobs=cases(config),parent=ident(os.getpid()),active_child=None)
  state['parent']=ident(os.getpid());save(base/'PARENT_IDENTITY.json',state['parent']);baseline=hardware(config);save(base/'CLOCK_POLICY.json',baseline);status_save(state,base,config)
  for j in state['jobs']:
   if j['status']=='done':
    receipt=read(Path(j['folder'])/'CAPTURE_COMPLETE.json');assert sha(j['report'])==receipt['sha256']and receipt['validation']['matched_kernel_launches']==LIMIT;continue
   if j['status']in ['failed','blocked']:continue
   if Path(j['folder']).exists():j.update(status='blocked',reason='Existing partial capture preserved; no automatic unchanged retry');status_save(state,base,config);continue
   try:execute(j,config,state,base,fd,baseline)
   except Exception as e:
    j.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(Path(j['folder'])/'CAPTURE_FAILED.json',j);state['active_child']=None;status_save(state,base,config);mode(config);idle(config)
  state['state']='COMPLETE'if all(j['status']=='done'for j in state['jobs'])else'EXHAUSTED_WITH_FAILURES';state['active_child']=None;status_save(state,base,config);save(base/'SERIES_COMPLETE.json',dict(utc=now(),state=state['state'],counts=state['counts'],fresh_idle=idle(config),no_power_or_clock_change=True));package_results(base,config,state)

def prepare(config):
 check_config(config);verify_kit();folder=ROOT/'preparation'/config['platform']/config['mode'];folder.mkdir(parents=True,exist_ok=True)
 with open('/tmp/dasip_round2_gpu0.lock','a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle(config);script=ROOT/'prepare_extensions.py';log=folder/('prepare_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.log')
  with log.open('x')as stream:r=subprocess.run([config['python'],'-B',str(script)],stdout=stream,stderr=subprocess.STDOUT,env=env_for(config))
  assert r.returncode==0,'Preparation failed; see '+str(log);idle(config);save(folder/'GPU_PREPARATION_INFO.json',read(ROOT/'prepared_extensions'/config['platform']/'GPU_PREPARATION_INFO.json'));save(folder/'PREPARED.json',dict(status='PASS',utc=now(),files_sha256=read(ROOT/'SHA256SUMS.json'),log=str(log),power=mode(config),compiled_or_loaded_before_NCU=True));fix_ownership(folder,config)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--machine',type=Path,default=ROOT/'machine_orin.json');p.add_argument('--mode',choices=['15W','MAXN','RTX']);p.add_argument('--prepare',action='store_true');p.add_argument('--start',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args();config=read(a.machine)
 if a.mode:config['mode']=a.mode
 if a.dry_run:
  for j in cases(config):print(json.dumps(dict(job=j,command=ncu_command(j,config))))
  return
 if a.prepare:prepare(config);return
 if a.start:
  check_config(config);verify_kit();mode(config)
  if config['platform']=='ORIN'and os.geteuid()!=0:raise PermissionError('Run start_orin_15w.sh with sudo in your terminal; NCU requires root on this Jetson')
  base=ROOT/'results'/config['platform']/config['mode'];base.mkdir(parents=True,exist_ok=True)
  marker=base/'PARENT_IDENTITY.json'
  if marker.exists():
   expected=read(marker);current=ident(expected['pid'])
   assert not(current and current['state']!='Z'and current['start_ticks']==expected['start_ticks']and current['boot_id']==expected['boot_id']),'Controller already active'
  effective=base/'EFFECTIVE_MACHINE.json';save(effective,config);log=base/('controller_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.log')
  with log.open('x')as stream:child=subprocess.Popen([config['python'],'-B','-u',str(Path(__file__).resolve()),'--machine',str(effective),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=profiler_env(),start_new_session=True)
  time.sleep(2);assert child.poll()is None,'Detached worker exited; see '+str(log);save(base/'DETACHED_LAUNCH.json',dict(utc=now(),pid=child.pid,session_id=os.getsid(child.pid),log=str(log),command=[config['python'],str(Path(__file__).resolve()),'--worker'],config=config));fix_ownership(base,config);print(json.dumps(read(base/'DETACHED_LAUNCH.json'),indent=2));return
 if a.worker:
  try:worker(config)
  except Exception as e:
   base=ROOT/'results'/config['platform']/config['mode'];save(base/'CONTROLLER_STOPPED.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()));fix_ownership(base,config);raise
 else:p.error('Choose --prepare, --start, --worker or --dry-run')
if __name__=='__main__':main()
