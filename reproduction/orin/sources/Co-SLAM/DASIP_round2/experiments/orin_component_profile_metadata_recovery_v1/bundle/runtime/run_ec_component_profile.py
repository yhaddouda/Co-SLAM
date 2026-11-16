#!/usr/bin/env python3
"""Complete native EC-SLAM Replica BASE sequence with selected worker profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback


def save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--datadir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--validate-state', action='store_true')
    parser.add_argument('--power-mode')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if os.environ.get('PYTHONHASHSEED') != str(args.seed):
        os.execve(sys.executable, [sys.executable, str(root/'run_ec_component_profile.py'), *sys.argv[1:]], dict(os.environ, PYTHONHASHSEED=str(args.seed)))
    output, datadir = args.output.resolve(), args.datadir.resolve()
    assert datadir.is_dir()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    manifest = json.loads((root/'RUNTIME_MANIFEST.json').read_text())
    for name, expected in manifest['runtime_hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    work = next(w for w in protocol['workloads'] if w['id']==args.workload)
    import numpy as np
    import torch
    import yaml
    from src.utils import load_config
    from ec_component_profile_timer import WorkerProfiler
    assert torch.cuda.is_available()
    def seed():
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    seed()
    cfg = load_config(work['native_scene_config'], 'configs/EC_SLAM.yaml')
    cfg['data'].update(input_folder=str(datadir), output=str(output))
    cfg['HashGrid']['hash_size'] = work['hash_size']
    assert cfg['dataset']=='replica'
    assert cfg['mapping']['every_frame']==5
    assert cfg['tracking']['iters']==work['max_tracking_updates']
    assert cfg['mapping']['iters']==work['native_ba_updates']
    save(output/'config.json', cfg)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    save(output/'source_manifest.json', manifest)
    save(output/'selection.json', work)
    def power():
        if not args.power_mode:
            return None
        import shutil, subprocess
        command = shutil.which('nvpmodel') or '/usr/sbin/nvpmodel'
        raw = subprocess.check_output([command, '-q'], text=True, timeout=10).strip()
        name = next((line.split(':',1)[1].strip() for line in raw.splitlines() if line.startswith('NV Power Mode:')), None)
        assert name==args.power_mode, raw
        return raw
    initial_power = power()
    def environment():
        import glob, platform
        from optional_sysfs_metadata import read_optional_sysfs
        readings, read_errors = {}, {}
        for pattern in ('/sys/class/devfreq/*/cur_freq', '/sys/class/thermal/thermal_zone*/temp'):
            import glob
            for name in glob.glob(pattern):
                value, error = read_optional_sysfs(name)
                readings[name] = value
                if error is not None:
                    read_errors[name] = error
        return dict(machine=platform.machine(), python=sys.version, torch_version=torch.__version__,
                    cuda_runtime=torch.version.cuda, device=torch.cuda.get_device_name(),
                    sysfs_snapshot=readings, sysfs_read_errors=read_errors, power=power(), clock_policy='Unchanged; use local controller clock-policy record')
    save(output/'environment_initial.json', environment())
    selected = {stage:[f for third in work['thirds'] for f in third[stage]['selected_frames']]
                for stage in ('tracking','bundle_adjustment')}
    def progress(frame):
        power()
        save(output/'progress.json', dict(frame=frame, native_frames=work['expected_native_frames'], diagnostic=bool(args.frames)))
        print('EC_PROFILE_FRAME',frame,'/',work['expected_native_frames'],flush=True)
    profiler = WorkerProfiler(output, selected, progress)
    import importlib
    for name in ('Tracker','Mapper','Model'):
        setattr(importlib.import_module('src.'+name), 'R2_PROFILE', profiler)
    import src.Datasets as datasets
    reader = datasets.Replica(cfg)
    native_frames = len(reader)
    assert native_frames==work['expected_native_frames'], native_frames
    if args.validate_state:
        from validate_ec_component_profile import gpu_fixed_state
        gpu_fixed_state(cfg, reader, output)
        seed()
        # Validator uses a separate proxy; restore the online proxy explicitly.
        for name in ('Tracker','Mapper','Model'):
            setattr(importlib.import_module('src.'+name), 'R2_PROFILE', profiler)
    if args.frames:
        original_len = datasets.Replica.__len__
        datasets.Replica.__len__ = lambda self:min(args.frames,original_len(self))
    from src.system import EC_SLAM
    def worker_failure(info):
        traceback.print_exception(info.exc_type, info.exc_value, info.exc_traceback)
        sys.stderr.flush()
        os._exit(2)
    threading.excepthook = worker_failure
    slam = EC_SLAM(cfg, argparse.Namespace(output=str(output)))
    torch.cuda.synchronize()
    begin = time.perf_counter()
    slam.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-begin
    frames = len(slam.estimate_c2w_list)
    field = slam.vis_dict[0].vMapModel
    finite = all(bool(torch.isfinite(p).all()) for p in field.parameters())
    finite = finite and all(bool(torch.isfinite(a).all()) for a in (slam.estimate_c2w_list,slam.estimate_relative_c2w_list,slam.gt_c2w_list))
    assert finite and int(slam.mapping_idx.item())==frames-1
    assert bool(torch.all((slam.estimate_c2w_list[:,3,3]-1).abs()<1e-5))
    for stage in selected:
        expected = {f for f in selected[stage] if f<frames}
        observed = {r['frame_id'] for r in profiler.rows if r['stage']==stage}
        assert expected==observed, (stage,expected-observed,observed-expected)
        assert len(observed)==sum(1 for r in profiler.rows if r['stage']==stage)
    final_power = power()
    save(output/'environment_final.json', environment())
    summary = dict(system='EC-SLAM', workload=args.workload, condition='BASE', hash_size=work['hash_size'],
                   seed=args.seed, frames=frames, native_frames=native_frames, full_sequence=frames==native_frames and not args.frames,
                   diagnostic=bool(args.frames), finite_parameters_and_poses=finite,
                   measured_run_seconds=elapsed, quality=False, export_mesh=False, engineering=False,
                   source_manifest_sha256=hashlib.sha256((root/'RUNTIME_MANIFEST.json').read_bytes()).hexdigest(),
                   protocol_sha256=hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest(),
                   initial_power=initial_power, final_power=final_power,
                   native_smoothness='No computation in pinned Model.forward; smooth is an unused argument',
                   timer_resolution='Deferred event resolution after selected worker body; no component-boundary/device-wide sync',
                   **profiler.summary())
    save(output/'summary.json', summary)
    print('EC_COMPONENT_PROFILE_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
