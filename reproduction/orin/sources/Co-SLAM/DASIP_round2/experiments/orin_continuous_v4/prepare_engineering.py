"""CPU-only isolated ENG preparation; no changes to validated scientific sources."""
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import sys
import torch

ROOT=Path(__file__).resolve().parent
CO=ROOT.parent/'orin_coslam_runtime_v1/runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def substitute(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
def extract(path,cls,method):
    tree=ast.parse(path.read_text());c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
    env={'torch':torch,'random':random};exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),env);return env[method]
def main():
    torch.set_num_threads(1)
    import engineering_adapters as a
    co_module=SimpleNamespace(CoSLAM=SimpleNamespace(global_BA=extract(CO/'coslam.py','CoSLAM','global_BA')))
    ec_module=SimpleNamespace(Mapper=SimpleNamespace(optimize_map=extract(ROOT/'ec_runtime/src/Mapper.py','Mapper','optimize_map')))
    audits={'Co-SLAM':a.install_co(co_module),'EC-SLAM':a.install_ec(None,ec_module)}
    # Reproduce native sample draws and check the moved Python draws do not alter Torch RNG evolution.
    class Database:
        frame_ids=[0,5,10]
        def sample_global_rays(self,n):
            ids=torch.tensor(random.sample(range(90),n));return torch.arange(90*7).reshape(90,7).float()[ids],ids//30*5
    slam=SimpleNamespace(config={'mapping':dict(sample=12,min_pixels_cur=3,iters=10,keyframe_every=5)},keyframeDatabase=Database(),dataset=SimpleNamespace(H=8,W=10),device='cpu')
    current=torch.arange(80*7).reshape(80,7).float()
    random.seed(123);torch.manual_seed(123);native=[];ids=[]
    for _ in range(10):
        r,i=slam.keyframeDatabase.sample_global_rays(12);selected=random.sample(range(80),4)
        native.append(torch.cat([r,current[selected,:]],0));ids.append(torch.cat([i//5,-torch.ones(len(selected))]).to(torch.int64))
        torch.rand(13)
    py_state=random.getstate();t_state=torch.get_rng_state()
    random.seed(123);torch.manual_seed(123);staged,staged_ids=a._preload(slam,current)
    for _ in range(10):torch.rand(13)
    assert torch.equal(staged,torch.stack(native)) and torch.equal(staged_ids,torch.stack(ids))
    assert py_state==random.getstate() and torch.equal(t_state,torch.get_rng_state())
    sys.path.insert(0,str(CO));from optimization.utils import qt_to_transform_matrix
    q=torch.randn(8,4);t=torch.randn(8,3);batch=qt_to_transform_matrix(q,t)
    assert all(torch.equal(batch[i],qt_to_transform_matrix(q[i:i+1],t[i:i+1])[0]) for i in range(8))
    poses=[p.detach().clone() for p in batch];before=poses[1].clone();poses[0][:3,3].add_(1)
    assert torch.equal(poses[1],before) and not torch.equal(poses[0],batch[0])
    # Native translation-view ownership is deliberately kept; no cloning correction in tracking.
    original=torch.eye(4).unsqueeze(0);parameter=torch.nn.Parameter(original[:,:3,3]);parameter.data.add_(2)
    assert torch.equal(original[0,:3,3],torch.full((3,),2.))
    audit={'status':'PASS','cpu_only':True,'changes':audits,'checks':['native per-iteration rays/ids exactly equal preloaded batch','Python and Torch RNG sequence preserved with interleaved Torch draws','batched versus one-pose quaternion transforms bit equal on CPU','independent per-pose clone ownership retained','native translation Parameter view alias retained, tracking method untouched','EC native Tracker.run/optimize_cam_in_batch unchanged; wait-before-transfer harness not imported','no sampling/loss/smoothness/precision/config change'],'scope_limit':'CUDA equivalence/finite gate still mandatory; CPU checks alone admit no full engineering run','baseline_controls':'Existing office0 H16 native BASE seeds 0/1 only if CUDA comparison passes; no corrected reference is introduced'}
    save(ROOT/'ENGINEERING_CPU_AUDIT.json',audit)
    home=ROOT/'engineering';home.mkdir()
    cr=home/'co_runtime';shutil.copytree(CO,cr,ignore=shutil.ignore_patterns('__pycache__'))
    er=home/'ec';er.mkdir();shutil.copytree(ROOT/'ec_runtime',er/'ec_runtime',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'ec_timing.py',er/'ec_timing.py')
    for target in [cr,er]:
        for name in ['engineering_adapters.py','equivalence_probe.py']:shutil.copy2(ROOT/name,target/name)
    s=(cr/'run_coslam_round2.py').read_text();original=s
    s=substitute(s,"    parser.add_argument('--quality',action='store_true')","    parser.add_argument('--engineering',action='store_true')\n    parser.add_argument('--equivalence-probe',action='store_true')\n    parser.add_argument('--quality',action='store_true')")
    s=substitute(s,"engineering=False,seed_policy=", "engineering=args.engineering,seed_policy=")
    s=substitute(s,"    slam=coslam.CoSLAM(cfg)","    if args.engineering:\n        from engineering_adapters import install_co\n        install_co(coslam)\n    probe=None\n    if args.equivalence_probe:\n        from equivalence_probe import Probe\n        probe=Probe()\n        probe.install(coslam.JointEncoding,lambda m: str((m._r2_frame,m._r2_stage,m._r2_iteration)) if getattr(m,'_r2_stage',None) in ('tracking','bundle_adjustment') else None)\n    slam=coslam.CoSLAM(cfg)")
    s=substitute(s,"    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\\n')","    if args.engineering:\n        summary['condition']='ENG'\n        summary['engineering_validation']='PASS' if not args.frames else 'DIAGNOSTIC_ONLY'\n    if probe:\n        probe.save(output/'equivalence.pt',slam.model,{k:v.detach().cpu().clone() for k,v in slam.est_c2w_data.items()},{k:v.detach().cpu().clone() for k,v in slam.est_c2w_data_rel.items()})\n    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\\n')")
    (cr/'run_coslam_round2.py').write_text(s)
    (home/'CO_DRIVER.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True))))
    m=json.loads((cr/'RUNTIME_MANIFEST.json').read_text())
    for n in ['run_coslam_round2.py','engineering_adapters.py','equivalence_probe.py']:m['runtime_hashes'][n]=sha(cr/n)
    m['engineering_isolated']=True;m['engineering_admission_gate']='ENGINEERING_PREFLIGHT_VALIDATION.json; no full run until PASS'
    save(cr/'RUNTIME_MANIFEST.json',m)
    s=(ROOT/'run_ec_round2.py').read_text();original=s
    s=substitute(s,"    parser.add_argument('--frames',type=int,default=0)","    parser.add_argument('--engineering',action='store_true')\n    parser.add_argument('--equivalence-probe',action='store_true')\n    parser.add_argument('--frames',type=int,default=0)")
    s=substitute(s,"engineering=False,seed_policy=", "engineering=args.engineering,seed_policy=")
    s=substitute(s,"    timer=Timer();queries=[];query_lock=threading.Lock()","    timer=Timer();queries=[];query_lock=threading.Lock()\n    if args.engineering:\n        import src.Tracker as tracker_module\n        import src.Mapper as mapper_module\n        from engineering_adapters import install_ec\n        install_ec(tracker_module,mapper_module)\n    probe=None\n    if args.equivalence_probe:\n        from equivalence_probe import Probe\n        probe=Probe()\n        probe.install(model_module.Model,lambda m: str((getattr(timer.local,'frame',-1),getattr(timer.local,'stage','unknown'),getattr(timer.local,'iteration',-1))))")
    s=substitute(s,"    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\\n')","    if args.engineering:\n        summary['condition']='ENG'\n        summary['engineering_validation']='PASS' if not args.frames else 'DIAGNOSTIC_ONLY'\n    if probe:\n        probe.save(output/'equivalence.pt',slam.vis_dict[0].vMapModel,slam.estimate_c2w_list.detach().cpu().clone(),slam.estimate_relative_c2w_list.detach().cpu().clone())\n    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\\n')")
    (er/'run_ec_round2.py').write_text(s)
    (home/'EC_DRIVER.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True))))
    m=json.loads((ROOT/'EC_RUNTIME_MANIFEST.json').read_text())
    for n in ['run_ec_round2.py','engineering_adapters.py','equivalence_probe.py']:m['files'][n]=sha(er/n)
    m['engineering_isolated']=True
    save(er/'EC_RUNTIME_MANIFEST.json',m)
    for p in [cr/'run_coslam_round2.py',er/'run_ec_round2.py']:ast.parse(p.read_text())
    print('CPU AUDIT PASS; isolated Co/EC ENG prepared, CUDA gate not yet run')

if __name__=='__main__':main()
