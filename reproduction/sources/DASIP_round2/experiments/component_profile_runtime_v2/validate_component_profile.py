"""AST, event-accounting and fixed-state checks for timing-only instrumentation."""
import copy
import json
from pathlib import Path
import random
import statistics
import time


def gpu_fixed_state(slam, timer, cfg, output):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    # Validation has separate outputs and does not fill selected-frame slots.
    from component_profile_timer import ComponentTimer
    validation = Path(output) / 'fixed_state_validation'
    validation.mkdir()
    engine = ComponentTimer(validation, {'tracking': [20], 'bundle_adjustment': [20]})
    slam._r2_component = slam.model._r2_component = engine
    engine.model = slam.model
    # Existing dynamic module wrappers refer to timer: activate neither here;
    # only the computation/parent wrappers are needed for fixed-state checks.
    batch = next(iter(DataLoader(slam.dataset, num_workers=0)))
    size = min(cfg['tracking']['sample'], 128)
    flat_depth = batch['depth'].reshape(-1)
    valid = torch.nonzero((flat_depth > 0) & (flat_depth < cfg['cam']['depth_trunc'])).flatten()
    assert len(valid) >= size
    indices = valid[torch.linspace(0, len(valid)-1, size).long()]
    directions = batch['direction'].reshape(-1, 3)[indices].to(slam.device)
    target_rgb = batch['rgb'].reshape(-1, 3)[indices].to(slam.device)
    target_depth = batch['depth'].reshape(-1, 1)[indices].to(slam.device)
    c2w = batch['c2w'][0].to(slam.device)
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[None, :3, -1].repeat(size, 1)
    model_state = copy.deepcopy(slam.model.state_dict())
    optimizer_state = copy.deepcopy(slam.map_optimizer.state_dict())
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    report = dict(output_loss_atol=1e-6, output_loss_rtol=1e-6,
                  gradient_step_atol=1e-5, gradient_step_rtol=1e-5,
                  warmup_pairs=5, measured_pairs=20, stages={},
                  overhead_scope='Incremental events enabled/disabled with timing wrappers present; bounded diagnostic, not full benchmark overhead or corrected primary performance')
    def restore():
        slam.model.load_state_dict(model_state)
        slam.map_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        slam.map_optimizer.zero_grad()
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
    def step(stage, enabled, capture):
        restore()
        engine.enabled = enabled
        timer.enabled = enabled
        # Encoder/decoder wrappers use timer, model/AST wrappers use engine.
        # Temporarily route dynamic module closure contexts to engine as well.
        original_scope = timer.scope
        original_active = timer.active
        timer.scope = engine.scope
        timer.active = False  # Disable only its optional query-counter writes.
        slam.model._r2_iteration = 0
        slam.model._r2_stage = None
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter(); start.record()
        engine.begin(stage, 20)
        with engine.scope('iteration'):
            ret = slam.model.forward(rays_o, rays_d, target_rgb, target_depth)
            with engine.scope('iteration_component', 'external_loss'):
                loss = slam.get_loss_from_ret(ret, smooth=stage=='bundle_adjustment')
            with engine.scope('iteration_component', 'backward'):
                loss.backward()
            with engine.scope('iteration_component', 'optimizer_update'):
                slam.map_optimizer.step()
        engine.finish(); end.record(); torch.cuda.synchronize()
        elapsed = dict(cuda_ms=start.elapsed_time(end), wall_ms=(time.perf_counter()-wall)*1000)
        engine.resolve()
        timer.scope, timer.active = original_scope, original_active
        if capture:
            values = dict(loss=loss.detach().clone(), rgb=ret['rgb'].detach().clone(), depth=ret['depth'].detach().clone())
            grads = [p.grad.detach().clone() if p.grad is not None else None for p in slam.model.parameters()]
            params = [p.detach().clone() for p in slam.model.parameters()]
            return elapsed, (values, grads, params)
        return elapsed, None
    for stage in ('tracking', 'bundle_adjustment'):
        _, plain = step(stage, False, True)
        _, timed = step(stage, True, True)
        errors = {}
        for key in plain[0]:
            errors[key] = float((plain[0][key]-timed[0][key]).abs().max())
            torch.testing.assert_close(plain[0][key], timed[0][key], atol=1e-6, rtol=1e-6)
        for label, left, right in [('gradient', plain[1], timed[1]), ('one_step_parameter', plain[2], timed[2])]:
            maximum = 0.
            for a, b in zip(left, right):
                assert (a is None) == (b is None)
                if a is None:
                    continue
                assert a.shape == b.shape
                if a.numel() == 0:
                    continue
                maximum = max(maximum, float((a-b).abs().max()))
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            errors[label] = maximum
        times = {False: [], True: []}
        for pair in range(25):
            order = (False, True) if pair % 2 == 0 else (True, False)
            for enabled in order:
                elapsed, _ = step(stage, enabled, False)
                if pair >= 5:
                    times[enabled].append(elapsed)
        means = {str(enabled): {key: statistics.mean(row[key] for row in values) for key in ('wall_ms', 'cuda_ms')} for enabled, values in times.items()}
        report['stages'][stage] = dict(max_absolute_errors=errors, timer_off_on_means=means,
                                       overhead_wall_pct=100*(means['True']['wall_ms']/means['False']['wall_ms']-1),
                                       diagnostic_rays=size, native_smoothness=stage=='bundle_adjustment')
    report['status'] = 'PASSED'
    report['closure_max_ms'] = engine.closure_max_ms
    (validation / 'PASSED.json').write_text(json.dumps(report, indent=2)+'\n')
    timer.enabled = True


if __name__ == '__main__':
    print('Use --validate-state on the diagnostic driver; this module launches no experiment alone.')
