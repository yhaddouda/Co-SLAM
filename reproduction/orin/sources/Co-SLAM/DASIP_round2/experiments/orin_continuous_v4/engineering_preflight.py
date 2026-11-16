"""Bounded serial BASE/ENG equivalence gate, called only by the GPU-lock owner."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
def command(system,seed,out,eng=False,frames=0):
    if system=='Co-SLAM':
        argv=[sys.executable,str(ROOT/'engineering/co_runtime/run_coslam_round2.py'),'--dataset','Replica','--scene','office0','--condition','BASE','--seed',str(seed),'--hash-size','16','--datadir','/home/yh279050/work/data/Replica/office0','--frames',str(frames),'--warmup','20','--output',str(out),'--expected-power-mode','MAXN']
    else:
        argv=[sys.executable,str(ROOT/'engineering/ec/run_ec_round2.py'),'--condition','BASE','--seed',str(seed),'--scene','office0','--hash-size','16','--datadir','/home/yh279050/work/EC-SLAM/datasets/vmap/office_0/imap/00','--frames',str(frames),'--output',str(out),'--co-runtime',str(ROOT.parent/'orin_coslam_runtime_v1/runtime'),'--expected-power-mode','MAXN']
    if eng:argv+=['--engineering']
    if frames:argv+=['--equivalence-probe']
    return argv
def check(system,out,eng):
    summary=read(out/'summary.json');cfg=read(out/'config.json')
    assert summary['frames']==50 and summary['finite_parameters_and_poses'] and not summary['full_sequence'] and not summary['quality']
    assert summary['initial_power']==summary['final_power']=='NV Power Mode: MAXN\n0'
    assert summary['condition']==('ENG' if eng else 'BASE') and cfg['round2']['engineering']==eng
    ec=system=='EC-SLAM';assert cfg['HashGrid' if ec else 'grid']['hash_size']==16
    with (out/'iterations.csv').open() as f:rows=list(csv.DictReader(f))
    for stage,iters in [('tracking',cfg['tracking']['iters' if ec else 'iter']),('bundle_adjustment',cfg['mapping']['iters'])]:
        frame_ids=set(range(1,50)) if stage=='tracking' else set(range(5,50,5))|({49} if ec else set())
        selected=[x for x in rows if x['iteration_type']==stage]
        assert len(selected)==len(frame_ids)*iters
        assert {(int(x['frame_id']),int(x['iteration_id'])) for x in selected}=={(i,j) for i in frame_ids for j in range(iters)}
        assert all(math.isfinite(float(x[k])) and float(x[k])>=0 for x in selected for k in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms'])
    with (out/'queries.csv').open() as f:assert all(int(x['actual_queries'])==int(x['dense_queries']) for x in csv.DictReader(f))
    return summary
def main():
    from equivalence_probe import compare
    request=read(ROOT/'ENGINEERING_READY_REQUEST.json')
    for p,h in request['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    records=[];failures=[]
    for system in ['Co-SLAM','EC-SLAM']:
        try:
            outputs=[]
            for eng in [False,True]:
                out=ROOT/'engineering_diagnostics'/(system+('_ENG' if eng else '_BASE'));out.parent.mkdir(exist_ok=True)
                assert not out.exists(),'Preserve prior partial diagnostic'
                argv=command(system,0,out,eng,50);save(out.with_suffix('.launch.json'),argv)
                with out.with_suffix('.log').open('x') as log:
                    result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
                assert result.returncode==0,'Diagnostic child failed '+str(out.with_suffix('.log'))
                check(system,out,eng);outputs.append(out)
            comparison=compare(outputs[0]/'equivalence.pt',outputs[1]/'equivalence.pt')
            records.append(dict(system=system,status='PASS',equivalence=comparison,native_base_compatible=True,quality_metrics=False))
        except Exception as e:
            failures.append(dict(system=system,error=repr(e),traceback=traceback.format_exc()));records.append(dict(system=system,status='FAIL',error=repr(e)))
    save(ROOT/'ENGINEERING_PREFLIGHT_VALIDATION.json',dict(status='FAIL' if failures else 'PASS',comparisons=records,failures=failures,scope='50 native frames including first-frame mapping, tracking and multiple BA invocations; once-only seed 0; fixed numerical tolerances declared before execution; no ATE or meshes'))
    if failures:raise SystemExit('Engineering not admitted; native queue continues; '+repr(failures))
    print('ENGINEERING_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':main()
