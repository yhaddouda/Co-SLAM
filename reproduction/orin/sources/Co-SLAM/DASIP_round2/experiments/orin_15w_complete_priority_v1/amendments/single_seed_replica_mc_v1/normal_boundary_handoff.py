"""One-shot exec handoff through the existing CPU-only post-child audit load."""
from pathlib import Path
import hashlib,inspect,json,os,sys
A=Path(__file__).resolve().parent;R=A.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(a,indent=2)+'\n');t.replace(p)
def at_normal_boundary():
 main=sys.modules.get('__main__')
 if not main or Path(getattr(main,'__file__','/')).resolve()!=R/'supervisor.py':return
 request=read(A/'AMENDMENT_REQUEST.json')
 if (A/'AMENDMENT_CONSUMED.json').exists():return
 expected=request['supervisor_identity'];actual=main.identity(os.getpid())
 assert actual and all(actual[k]==expected[k]for k in ['pid','start_ticks','boot_id']),'Amendment parent identity mismatch'
 frame=inspect.currentframe().f_back
 while frame is not None and not(frame.f_code.co_name=='main'and frame.f_globals is main.__dict__):frame=frame.f_back
 assert frame is not None,'No authorized live controller dispatch frame'
 s=frame.f_locals['s'];j=frame.f_locals['j'];fd=frame.f_locals['fd']
 assert s.get('active_child_identity')is None,'GPU child has not released ownership'
 done=read(R/'process_receipts'/(j['id']+'.json'));assert done['returncode']==0
 assert not main.support.still_alive(done['child_identity'])
 assert j['id']in request['retained_ids']
 assert os.fstat(fd).st_ino==Path('/tmp/dasip_round2_gpu0.lock').stat().st_ino
 assert os.get_inheritable(fd),'Inherited GPU ownership descriptor would close across exec'
 for p,h in request['files'].items():assert sha(p)==h,'Amendment integrity mismatch: '+p
 main.freeze();main.mode();main.clocks();fresh=main.idle(three=True)
 save(A/'NORMAL_BOUNDARY_HANDOFF.json',dict(utc=main.now(),status='NORMAL_CHILD_COMPLETED_EXEC_HANDOFF',supervisor_identity=actual,session_id=os.getsid(0),normal_job=j['id'],normal_process_receipt=done,gpu_lock_fd=fd,gpu_lock_inode=os.fstat(fd).st_ino,required=135,excluded=162,stages=[48,56,26,5],fresh_idle=fresh,no_signal_or_child_interruption=True))
 argv=[str(main.PYTHON),'-B','-u',str(A/'supervisor_amended.py'),'--adopt-owned-lock-fd',str(fd),'--completed-normal-child',j['id']]
 print('AUTHOR_QUEUE_AMENDMENT_NORMAL_BOUNDARY_EXEC',j['id'],'135 slots; same PID/session/ownership',flush=True)
 os.execve(str(main.PYTHON),argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
