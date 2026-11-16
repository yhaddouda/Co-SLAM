"""Compare complete EC tracking loops on the same short-fitted frozen field.

This is a numerical diagnostic, not a timed benchmark or a trained SLAM scene.
It calls the actual Tracker optimizer method with one identical fixed ray batch.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import torch


def differences(reference, candidate):
    delta = (candidate - reference).double()
    norm = reference.double().norm().item()
    return dict(max_absolute=delta.abs().max().item(),
                relative_l2=delta.norm().item() / max(norm, 1e-30))


def iterative_tracking_checks(cfg, model, colors, depths, directions, gt_camera):
    from src.Tracker import Tracker
    from src.execution_experiments import tracking_flags
    from src.utils import get_camera_from_tensor

    matrix = get_camera_from_tensor(gt_camera.detach())
    rays_d = (matrix[None, :, :3] @ directions[..., None]).squeeze(-1)
    rays_o = matrix[:, 3].repeat(len(directions), 1)
    # Fit a nontrivial field before checking repeated pose updates. This keeps
    # validation independent of disabled full-run checkpoint output. The fit is
    # local to this diagnostic and is not a Replica reconstruction benchmark.
    torch.manual_seed(431)
    torch.cuda.manual_seed_all(431)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit_losses = []
    for _ in range(32):
        optimizer.zero_grad(set_to_none=True)
        loss = model.forward(rays_o, rays_d, colors, depths, tracker=False, smooth=False)
        loss.backward()
        optimizer.step()
        fit_losses.append(loss.detach().clone())
    torch.cuda.synchronize()
    assert torch.isfinite(torch.stack(fit_losses)).all()

    records = []
    actual_forward = model.forward
    scene = SimpleNamespace(vMapModel=model,
        get_tracking_samples=lambda *args: (colors, directions, depths.reshape(-1)))
    offset = gt_camera.new_tensor([.001, .002, -.001, .001, .004, -.003, .002])
    for multiplier in (1., -1., 2.):
        initial = gt_camera.detach() + multiplier * offset
        pairs = []
        for variant in ('baseline', 'baseline_fast'):
            torch.manual_seed(733)
            torch.cuda.manual_seed_all(733)
            model.zero_grad(set_to_none=True)
            quad = initial[:4].clone().requires_grad_(True)
            translation = initial[4:].clone().requires_grad_(True)
            losses, states, gradients_q, gradients_t = [], [], [], []
            handles = [quad.register_hook(lambda grad: gradients_q.append(grad.detach().clone())),
                       translation.register_hook(lambda grad: gradients_t.append(grad.detach().clone()))]
            def observed_forward(*args, **kwargs):
                states.append(torch.cat([quad.detach(), translation.detach()]).clone())
                loss = actual_forward(*args, **kwargs)
                losses.append(loss.detach().clone())
                return loss
            model.forward = observed_forward
            worker = Tracker.__new__(Tracker)
            settings = {**cfg, 'verbose': False,
                'debug': {**cfg['debug'], 'flag': False, 'showTrackerLoss': False},
                'tracking_experiment': {'variant': variant}}
            worker.execution = tracking_flags(settings)
            worker.device = gt_camera.device
            worker.num_cam_iters = cfg['tracking']['iters']
            worker.tracking_pixels_bg = len(directions)
            worker.vis_dict = {0: scene}
            worker.last_best = cfg['tracking']['last_best']
            worker.verbose = worker.debug = worker.showTrackerLoss = False
            worker.iteration_timer = SimpleNamespace(stage=lambda *args: nullcontext())
            pose_optimizer = torch.optim.Adam([
                {'params': [translation], 'lr': cfg['tracking']['lr_T']},
                {'params': [quad], 'lr': cfg['tracking']['lr_R']}])
            try:
                torch.cuda.synchronize()
                selected = worker.optimize_cam_in_batch(None, None, None,
                    pose_optimizer, gt_camera.detach().cpu(), quad, translation, 10)
                final_matrix = (worker._execution_candidate_matrix
                    if worker._execution_candidate_matrix is not None
                    else get_camera_from_tensor(selected))
                torch.cuda.synchronize()
            finally:
                model.forward = actual_forward
                for handle in handles:
                    handle.remove()
            history = torch.stack(states)
            values = torch.stack(losses)
            assert len(values) == worker.num_cam_iters
            assert len(gradients_q) == len(gradients_t) == worker.num_cam_iters
            assert torch.isfinite(values).all() and torch.isfinite(history).all()
            index = len(values) - 1 if worker.last_best else int(values.argmin().item())
            torch.testing.assert_close(selected, history[index], rtol=0, atol=0)
            torch.testing.assert_close(final_matrix, get_camera_from_tensor(selected), rtol=0, atol=0)
            pairs.append(dict(losses=values, states=history,
                gradients=torch.cat([torch.stack(gradients_q), torch.stack(gradients_t)], -1),
                candidate=selected.detach().clone(), matrix=final_matrix.detach().clone(),
                selected_iteration=index, cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state()))
        baseline, fast = pairs
        gradient_norm = baseline['gradients'].double().norm().item()
        movement = (baseline['states'] - baseline['states'][0]).abs().max().item()
        assert gradient_norm > 0 and movement > 0, 'Trivial tracking probe: no pose gradient or update'
        assert torch.equal(baseline['cpu_rng'], fast['cpu_rng'])
        assert torch.equal(baseline['cuda_rng'], fast['cuda_rng'])
        errors = {key: differences(baseline[key], fast[key])
                  for key in ('losses', 'states', 'gradients', 'candidate', 'matrix')}
        assert errors['losses']['relative_l2'] < 1e-3, errors
        assert errors['gradients']['relative_l2'] < 1e-3 or errors['gradients']['max_absolute'] < 1e-7, errors
        assert errors['states']['max_absolute'] < 1e-5, errors
        assert errors['candidate']['max_absolute'] < 1e-5, errors
        assert baseline['selected_iteration'] == fast['selected_iteration'], errors
        records.append(dict(stage='complete_tracking_invocation', rays=len(directions),
            iterations=cfg['tracking']['iters'], initial_pose_offset_multiplier=multiplier,
            frozen_map='Real EC model fitted for 32 updates to one frame; not a SLAM checkpoint',
            baseline_selected_iteration=baseline['selected_iteration'],
            fast_selected_iteration=fast['selected_iteration'],
            cpu_and_cuda_rng='exact match', numerical_errors=errors,
            candidate_and_matrix_association='exact within each invocation',
            baseline_pose_gradient_l2=gradient_norm,
            baseline_max_camera_change=movement,
            field_fit_initial_loss=float(fit_losses[0]), field_fit_last_loss=float(fit_losses[-1])))
    return records
