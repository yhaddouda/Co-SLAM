#!/usr/bin/env python3
"""Full native online BASE run with selected-call component profiling."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--datadir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--power-mode')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--validate-state', action='store_true')
    args = parser.parse_args()
    if os.environ.get('PYTHONHASHSEED') != str(args.seed):
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], dict(os.environ, PYTHONHASHSEED=str(args.seed)))
    root = Path(__file__).resolve().parent
    datadir, output = args.datadir.resolve(), args.output.resolve()
    assert datadir.is_dir() and not output.exists()
    output.mkdir(parents=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    manifest = json.loads((root / 'RUNTIME_MANIFEST.json').read_text())
    for name, expected in manifest['runtime_hashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    work = next(w for w in protocol['workloads'] if w['id'] == args.workload)
    assert hashlib.sha256((root / work['native_scene_config']).read_bytes()).hexdigest() == work['native_scene_config_sha256']
    import numpy as np
    import torch
    import config
    import coslam
    from torch.utils.data import DataLoader
    from component_profile_timer import ComponentTimer
    from round2_sampling import install
    assert torch.cuda.is_available()
    def seed():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    seed()
    cfg = config.load_config(work['native_scene_config'])
    cfg['data'].update(datadir=str(datadir), output=str(output), exp_name='artifacts')
    cfg['mapping']['first_mesh'] = False
    cfg['mesh']['visualisation'] = False
    cfg['grid'].update(hash='CoherentPrime', morton_sort=False, hash_size=work['hash_size'])
    cfg['round2'] = dict(condition='BASE', seed=args.seed, sampler='dense', margin_scene_units=.15,
                         anchor_every=3, engineering=False, component_profile=True)
    assert cfg['tracking']['iter_point'] == 0
    assert cfg['mapping']['map_every'] == work['map_every']
    coslam.cfg = cfg
    (output / 'artifacts').mkdir()
    save(output / 'config.json', cfg)
    save(output / 'source_manifest.json', manifest)
    save(output / 'selection.json', work)
    def power():
        if not args.power_mode:
            return None
        import shutil, subprocess
        command = shutil.which('nvpmodel') or '/usr/sbin/nvpmodel'
        raw = subprocess.check_output([command, '-q'], text=True, timeout=10).strip()
        name = next((line.split(':', 1)[1].strip() for line in raw.splitlines() if line.startswith('NV Power Mode:')), None)
        assert name == args.power_mode, raw
        return raw
    initial_power = power()
    def environment():
        import platform
        readings = {}
        for pattern in ('/sys/class/devfreq/*/cur_freq', '/sys/class/thermal/thermal_zone*/temp'):
            import glob
            for name in glob.glob(pattern):
                try:
                    readings[name] = Path(name).read_text().strip()
                except (OSError, PermissionError):
                    readings[name] = 'unavailable'
        return dict(machine=platform.machine(), python=sys.version, device=torch.cuda.get_device_name(),
                    torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                    initial_power=initial_power, sysfs_snapshot=readings,
                    clock_lock_state='Not inferred; use the local campaign clock-policy record')
    save(output/'environment_initial.json', environment())
    slam = coslam.CoSLAM(cfg)
    assert len(slam.dataset) == work['expected_native_frames']
    selected = {stage: [frame for third in work['thirds'] for frame in third[stage]['selected_frames']]
                for stage in ('tracking', 'bundle_adjustment')}
    timer = ComponentTimer(output, selected)
    slam._r2_component = timer
    slam.model._r2_component = timer
    slam.coslam_timing = timer
    # Preserve the same dense BASE adapter/counters as the benchmark runtime.
    install(slam.model, prune=False, morton=False)
    timer.wrap_modules(slam.model)
    slam.create_optimizer()
    if args.validate_state:
        from validate_component_profile import gpu_fixed_state
        gpu_fixed_state(slam, timer, cfg, output)
        # Validation uses this instance only; reconstruct the primary diagnostic
        # model from the ordinary seed so diagnostics preserve online RNG policy.
        seed()
        slam = coslam.CoSLAM(cfg)
        timer = ComponentTimer(output, selected)
        slam._r2_component = slam.model._r2_component = timer
        slam.coslam_timing = timer
        install(slam.model, prune=False, morton=False)
        timer.wrap_modules(slam.model)
        slam.create_optimizer()
    calls = []
    def invoke(method, stage, batch, frame):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        timer.begin(stage, frame)
        slam.model._r2_stage = stage
        slam.model._r2_frame = frame
        slam.model._r2_support = None
        method(batch, frame)
        timer.finish()
        end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall)*1000
        was_profiled = timer.active
        timer.resolve()
        calls.append(dict(frame_id=frame, stage=stage, wall_ms=wall_ms,
                          cuda_ms=start.elapsed_time(end), component_profiled=int(was_profiled)))
        slam.model._r2_stage = None
    torch.cuda.synchronize()
    began = time.perf_counter()
    limit = args.frames or len(slam.dataset)
    for frame, batch in enumerate(DataLoader(slam.dataset, num_workers=cfg['data']['num_workers'])):
        if frame >= limit:
            break
        if frame % 100 == 0:
            power()
            save(output / 'progress.json', dict(frame=frame, native_frames=len(slam.dataset),
                                               profiled_calls=dict(timer.call_count), diagnostic=bool(args.frames)))
            print('PROFILE_FRAME', frame, '/', len(slam.dataset), flush=True)
        if frame == 0:
            slam.first_frame_mapping(batch, cfg['mapping']['first_iters'])
        else:
            invoke(slam.tracking_render, 'tracking', batch, frame)
            if frame % cfg['mapping']['map_every'] == 0:
                invoke(slam.global_BA, 'bundle_adjustment', batch, frame)
            if frame % cfg['mapping']['keyframe_every'] == 0:
                slam.keyframeDatabase.add_keyframe(batch, filter_depth=cfg['mapping']['filter_depth'])
        torch.cuda.synchronize()
    final_power = power()
    save(output/'environment_final.json', environment())
    finite = all(bool(torch.isfinite(p).all()) for p in slam.model.parameters())
    finite = finite and all(bool(torch.isfinite(p).all()) for p in slam.est_c2w_data.values())
    assert finite
    timer.append_csv('calls.csv', calls)
    timer.append_csv('queries.csv', slam.model._r2_queries)
    frames = min(limit, len(slam.dataset))
    for stage in selected:
        expected = {f for f in selected[stage] if f < frames}
        observed = {c['frame_id'] for c in calls if c['stage'] == stage and c['component_profiled']}
        assert expected == observed
    summary = dict(workload=args.workload, seed=args.seed, condition='BASE', frames=frames,
                   full_sequence=not args.frames, diagnostic=bool(args.frames),
                   native_frames=len(slam.dataset), finite_parameters_and_poses=finite,
                   selected_calls=dict(timer.call_count), closure_max_ms=timer.closure_max_ms,
                   closure_tolerance='abs_error <= 0.02ms + 0.005 * parent_cuda_ms; no normalization/clamping',
                   measured_run_seconds=time.perf_counter()-began,
                   initial_power=initial_power, final_power=final_power,
                   device=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                   source_manifest_sha256=hashlib.sha256((root / 'RUNTIME_MANIFEST.json').read_bytes()).hexdigest(),
                   protocol_sha256=hashlib.sha256((root / 'PROTOCOL.json').read_bytes()).hexdigest(),
                   quality=False, export_mesh=False, timer_resolution='deferred CUDA events; segments resolved outside selected call wall interval')
    save(output / 'summary.json', summary)
    print('COMPONENT_PROFILE_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
