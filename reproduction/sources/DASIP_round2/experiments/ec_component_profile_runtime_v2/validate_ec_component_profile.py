"""Paired native-versus-timed EC loss, render, gradient and Adam-step checks."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import statistics
import time


def gpu_fixed_state(cfg, reader, output):
    import numpy as np
    import torch
    import src.Model as timed_module
    from src.Scene import cameraInfo
    from src.utils import get_camera_from_tensor, get_tensor_from_camera
    from ec_component_profile_timer import WorkerProfiler
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('ec_native_validation_model',root/'reference/src/Model.py')
    native_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_module)
    directory = Path(output)/'fixed_state_validation'
    directory.mkdir()
    profiler = WorkerProfiler(directory, {'tracking':[20], 'bundle_adjustment':[20]})
    timed_module.R2_PROFILE = profiler
    models = {False:native_module.Model(cfg).to(cfg['device']), True:timed_module.Model(cfg).to(cfg['device'])}
    state = copy.deepcopy(models[False].state_dict())
    sample = reader[0]
    depth = torch.as_tensor(sample['depth']).reshape(-1)
    valid = torch.nonzero(depth>0).flatten()
    size = 128
    assert len(valid)>=size
    indices = valid[torch.linspace(0,len(valid)-1,size).long()]
    cam = cfg['cam']
    dirs = cameraInfo(cfg['device'],cam['H'],cam['W'],cam['fx'],cam['fy'],cam['cx'],cam['cy']).rays_dir_cache
    directions = dirs.reshape(-1,3)[indices].detach()
    rgb = torch.as_tensor(sample['image'],device=cfg['device'],dtype=torch.float32).reshape(-1,3)[indices]
    target_d = depth[indices,None].to(cfg['device'])
    # Native get_tensor_from_camera builds a CPU tensor; Tracker explicitly
    # transfers its result before calling get_camera_from_tensor.
    pose = get_tensor_from_camera(torch.as_tensor(sample['T'],device=cfg['device'],dtype=torch.float32)).to(cfg['device']).detach()
    rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    captured = {}
    for enabled,model in models.items():
        render = model.render_rays
        def wrapped(*args,_render=render,_enabled=enabled,**kwargs):
            result = _render(*args,**kwargs)
            captured[_enabled] = result
            return result
        model.render_rays = wrapped
    def step(stage, enabled, capture=False):
        model = models[enabled]
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        camera = torch.nn.Parameter(pose.clone())
        optimizer = torch.optim.Adam([{'params':[camera],'lr':.001}]) if stage=='tracking' else torch.optim.Adam(model.decoder.get_optParameter(),betas=(.9,.99))
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        profiler.enabled = enabled
        torch.cuda.synchronize()
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();begin.record()
        with profiler.call(stage,20):
            with profiler.iteration(0):
                with profiler.scope('iteration_component','ray_pose_preparation'):
                    c2w = get_camera_from_tensor(camera)
                    rays_o = c2w[:,3].repeat(size,1)
                    rays_d = (c2w[None,:,:3]@directions[:,:,None]).squeeze(-1)
                loss = model.forward(rays_o,rays_d,rgb,target_d,tracker=stage=='tracking',smooth=stage=='bundle_adjustment')
                with profiler.scope('iteration_component','backward'):
                    loss.backward()
                with profiler.scope('iteration_component','optimizer_update'):
                    optimizer.step()
        end.record();end.synchronize()
        elapsed=dict(wall_ms=(time.perf_counter()-wall)*1000,cuda_ms=begin.elapsed_time(end))
        if not capture:
            return elapsed,None
        values={'loss':loss.detach().clone()}
        for key,value in zip(('rgb','depth','sdf','z_vals'),captured[enabled]):
            values[key]=value.detach().clone()
        params=[camera,*model.parameters()]
        return elapsed,(values,[p.grad.detach().clone() if p.grad is not None else None for p in params],[p.detach().clone() for p in params])
    report=dict(output_loss_atol=1e-6,output_loss_rtol=1e-6,gradient_step_atol=1e-5,gradient_step_rtol=1e-5,
                diagnostic_rays=size,warmup_pairs=5,measured_pairs=20,stages={},
                overhead_scope='Native AST model versus instrumented AST model/events, fixed-state single-thread bounded diagnostic; not a correction of concurrent full-run timing',
                limitations='Native workers intentionally retained online; fixed-state equality does not establish identical online thread interleaving')
    for stage in ('tracking','bundle_adjustment'):
        _,native=step(stage,False,True)
        _,timed=step(stage,True,True)
        errors={}
        for key in native[0]:
            torch.testing.assert_close(native[0][key],timed[0][key],atol=1e-6,rtol=1e-6)
            errors[key]=float((native[0][key]-timed[0][key]).abs().max())
        for label,left,right in (('gradient',native[1],timed[1]),('one_step_parameter',native[2],timed[2])):
            maximum=0.
            for a,b in zip(left,right):
                assert (a is None)==(b is None)
                if a is None or not a.numel():continue
                torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
                maximum=max(maximum,float((a-b).abs().max()))
            errors[label]=maximum
        times={False:[],True:[]}
        for pair in range(25):
            for enabled in ((False,True) if pair%2==0 else (True,False)):
                elapsed,_=step(stage,enabled)
                if pair>=5:times[enabled].append(elapsed)
        means={str(k):{field:statistics.mean(r[field] for r in rows) for field in ('wall_ms','cuda_ms')} for k,rows in times.items()}
        report['stages'][stage]=dict(max_absolute_errors=errors,timer_off_on_means=means,
                                    overhead_wall_pct=100*(means['True']['wall_ms']/means['False']['wall_ms']-1),
                                    smoothness_computed=False)
    report.update(status='PASSED',closure_max_ms=profiler.summary()['closure_max_ms'])
    (directory/'PASSED.json').write_text(json.dumps(report,indent=2)+'\n')
