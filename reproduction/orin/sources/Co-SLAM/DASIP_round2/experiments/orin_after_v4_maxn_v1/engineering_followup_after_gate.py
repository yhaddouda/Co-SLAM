"""Deferred failure investigation and independent, bounded Co RTX-reference gate."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
V4=ROOT.parent/'orin_continuous_v4'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def permitted():
    assert os.environ.get('DASIP_POST_V4_EXCLUSIVE')=='1' and (ROOT/'NORMAL_EXHAUSTION_TRANSITION.json').is_file()
def failure_report():
    permitted()
    import torch
    torch.set_num_threads(1);systems=[]
    for system in ['Co-SLAM']:
        base=V4/'engineering_diagnostics'/(system+'_BASE');eng=V4/'engineering_diagnostics'/(system+'_ENG')
        a=torch.load(base/'equivalence.pt',map_location='cpu');b=torch.load(eng/'equivalence.pt',map_location='cpu');mismatches=[]
        for key in a['calls']:
            for metric,x in a['calls'][key]['losses'].items():
                y=b['calls'][key]['losses'][metric];absolute=float((x-y).abs().max());relative=absolute/max(abs(float(x)),1e-30)
                if not torch.allclose(x,y,atol=1e-5,rtol=1e-4):mismatches.append(dict(context=key,metric=metric,base=float(x),eng=float(y),absolute_error=absolute,relative_error_to_base=relative,allowed_absolute_at_base=1e-5+1e-4*abs(float(x))))
        config=read(base/'config.json')
        records={}
        for name,folder in [('BASE',base),('ENG',eng)]:
            records[name]=dict(command=read(folder.with_suffix('.launch.json')),source_manifest=read(folder/'source_manifest.json'),source_manifest_sha256=sha(folder/'source_manifest.json'),effective_config=read(folder/'config.json'),config_sha256=sha(folder/'config.json'),snapshot_sha256=sha(folder/'equivalence.pt'),summary=read(folder/'summary.json'))
        systems.append(dict(system=system,classification='Two independently evolving online CUDA trajectories; not identical fixed inputs/one optimizer update',declared_atol=1e-5,declared_rtol=1e-4,rotation_representation=config.get('training',{}).get('rot_rep','EC native quaternion'),seed=0,power='MAXN',persisted_parameter_dtypes=sorted({str(v.dtype) for v in a['model'].values()}),persisted_loss_dtypes=sorted({str(v.dtype) for c in a['calls'].values() for v in c['losses'].values()}),input_dtype_at_failure='Not explicitly persisted in old online probe; loader FP32 is source evidence, not a recorded runtime input-dtype observation',gradient_equivalence_checked_by_old_gate=False,alias_semantics='Co native tracking translation view kept in old port; final committed poses cloned individually; reference uses nonoverlapping views of one cloned batch. EC native tracker unchanged; mapping pose/inverse cache differs.',identical_observation_ids=all(a['calls'][k]['observations_sha256']==b['calls'][k]['observations_sha256'] for k in a['calls']),final_python_rng_equal=a['python_rng']==b['python_rng'],final_cpu_rng_equal=torch.equal(a['torch_cpu_rng'],b['torch_cpu_rng']),final_cuda_rng_equal=torch.equal(a['torch_cuda_rng'],b['torch_cuda_rng']),mismatch_count=len(mismatches),first_assertion_case=mismatches[0] if mismatches else None,max_absolute_error=max((x['absolute_error'] for x in mismatches),default=0),max_relative_error=max((x['relative_error_to_base'] for x in mismatches),default=0),original_records=records))
    failure=read(V4/'ENGINEERING_PREFLIGHT_VALIDATION.json')
    report=dict(original_failure=failure,original_trace_and_commands_preserved=True,old_failed_source_hashes={str(p):sha(p) for p in [V4/'engineering_adapters.py',V4/'equivalence_probe.py',V4/'engineering_preflight.py']},systems=systems,comparison_to_supplied_reference='Old Co port preloaded every iteration draw before BA and reused the last pose batch; RTX reference retains draw placement, groups each ray transfer, caches fixed tracking targets and recomputes batched pose commit. Native online nondeterminism versus a semantic defect is not established by the failed online checks.',new_validation_definition='Independent RTX-reference fixed-state/ray/gradient/one-update/storage checks at atol=rtol=1e-6, plus separate finite 50-frame BASE/ENG diagnostics. This does not reinterpret or pass the old 1e-5/1e-4 online failure.',EC_status='EXCLUDED FROM SCOPE: historical errors preserved verbatim; no new EC engineering analysis, diagnostics or completion requirement',input_recovery_limit='Original ray/model inputs for each failing step were not saved; the minimal scalar assertion reproduces the recorded failure only, not its causal online trajectory. New fixed-state code supplies a bounded executable numerical case.')
    save(ROOT/'ENGINEERING_FAILURE_REPORT.json',report)
    save(ROOT/'MINIMAL_ORIGINAL_ASSERTION_CASES.json',[dict(system=x['system'],case=x['first_assertion_case'],atol=1e-5,rtol=1e-4) for x in systems])
    text='# Engineering failure investigation\n\nOriginal failures remain FAIL. No old tolerance is widened and no online identity is claimed. Both checks compared independently evolving trajectories; observed mismatches begin during early tracking before the first post-initialization BA. Nondeterminism is a hypothesis, not an established cause.\n\n'
    for x in systems:
        error=next(v for v in failure['failures'] if v['system']==x['system']);text+='## '+x['system']+'\n\nExact error: `'+error['error']+'`.\n\n```text\n'+error['traceback']+'```\n\nMeasured first assertion: '+json.dumps(x['first_assertion_case'])+'. Maximum scalar absolute/relative errors: '+str(x['max_absolute_error'])+' / '+str(x['max_relative_error'])+'.\n\nCommands, exact source/config/snapshot hashes, rotation, recorded dtypes, RNG/observation evidence and limitations are in ENGINEERING_FAILURE_REPORT.json.\n\n'
    text+='The supplied Co reference is a different isolated execution port, not a patch to v4. Its stricter fixed-state test compares identical inputs and operations. Matched two-seed BASE controls are added explicitly before full ENG interpretation. EC ENG is excluded from scope and completion requirements; its historical errors remain preserved. ScanNet measurements run before the new Co engineering investigation. No ATE, meshes or reconstruction metrics are collected.\n'
    (ROOT/'ENGINEERING_FAILURE_REPORT.md').write_text(text)
def prepare_and_validate(fd,state,run,baseline,readiness):
    permitted();reference=ROOT/'coslam_engineering_reference';runtime=ROOT/'coslam_engineering_local'
    assert not runtime.exists(),'Prior local ENG preparation preserved; no silent retry'
    # Full source/config audit is deferred until after native ScanNet measurements.
    manifest=read(reference/'RUNTIME_MANIFEST.json')
    for name,h in manifest['runtime_hashes'].items():assert sha(reference/name)==h,name
    from round2_audit_after_gate import audit_reference
    audit_reference(reference)
    shutil.copytree(reference,runtime,ignore=shutil.ignore_patterns('__pycache__'))
    fixed=ROOT/'diagnostics/CO_RTXREF_FIXED_STATE.json';fixed.parent.mkdir(exist_ok=True)
    argv=[sys.executable,'-B',str(ROOT/'validate_rtx_reference_fixed_state.py'),'--runtime',str(runtime),'--output',str(fixed)]
    assert run(argv,ROOT/'logs/CO_RTXREF_fixed_state.log',fd,state)==0,'Co fixed-state/ray/gradient/alias gate failed; native results preserved'
    assert read(fixed)['status']=='PASS'
    finite=[]
    for eng in [False,True]:
        out=ROOT/'diagnostics'/('CO_RTXREF_ENG' if eng else 'CO_RTXREF_BASE')
        argv=[sys.executable,str(runtime/'run_coslam_round2.py'),'--dataset','Replica','--scene','office0','--datadir','/home/yh279050/work/data/Replica/office0','--condition','BASE','--seed','0','--hash-size','16','--frames','50','--warmup','20','--output',str(out),'--expected-power-mode','MAXN']
        if eng:argv+=['--engineering']
        assert run(argv,ROOT/'logs'/(out.name+'.log'),fd,state)==0,'Co finite diagnostic failed'
        from post_v4_execute import verify_output
        summary=verify_output(dict(condition='BASE',seed=0,scene='office0',hash_size=16,expected_native_frames=2000),out,engineering=eng,diagnostic=True);finite.append(summary)
    save(ROOT/'CO_ENGINEERING_LOCAL_VALIDATION.json',dict(status='PASS',reference_manifest_sha256=sha(runtime/'RUNTIME_MANIFEST.json'),fixed_state=read(fixed),finite_diagnostics=finite,online_bitwise_identity_claimed=False,old_online_failure_reclassified=False,paired_extra_BASE_controls=2,EC_status='Excluded from scope and completion requirements'))
    save(ROOT/'CO_ENGINEERING_RUNTIME_FREEZE.json',dict(files={str(p):sha(p) for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts}))
    readiness('after_engineering_validation_before_matched_timings',baseline)
    return runtime
