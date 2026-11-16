#!/usr/bin/env python3
"""Freeze deterministic temporal selections for deferred Co-SLAM profiling."""
import hashlib
import json
from pathlib import Path


def select_uniform(values, target):
    count = min(target, len(values))
    if not count:
        return []
    if count == 1:
        return [values[0]]
    # Integer nearest rounding with ties upward, without floating ambiguity.
    indices = [(2 * j * (len(values) - 1) + count - 1) // (2 * (count - 1))
               for j in range(count)]
    return [values[index] for index in indices]


def temporal_plan(frames, warmup=20, map_every=5):
    assert frames > warmup and map_every > 0
    bounds = [warmup + k * (frames - warmup) // 3 for k in range(4)]
    windows = []
    for name, start, end in zip(('early', 'middle', 'late'), bounds, bounds[1:]):
        tracking = list(range(start, end))
        ba = [frame for frame in tracking if frame % map_every == 0]
        window = dict(name=name, start_inclusive=start, end_exclusive=end,
                      tracking=dict(eligible_calls=len(tracking), selected_frames=select_uniform(tracking, 100)),
                      bundle_adjustment=dict(eligible_calls=len(ba), selected_frames=select_uniform(ba, 20)))
        windows.append(window)
    for stage in ('tracking', 'bundle_adjustment'):
        total = sum(window[stage]['eligible_calls'] for window in windows)
        for window in windows:
            window[stage]['scene_weight'] = window[stage]['eligible_calls'] / total
        selected = [frame for window in windows for frame in window[stage]['selected_frames']]
        expected = list(range(warmup, frames))
        if stage == 'bundle_adjustment':
            expected = [frame for frame in expected if frame % map_every == 0]
        assert selected == sorted(set(selected))
        assert set(selected) <= set(expected)
        assert selected[0] == expected[0] and selected[-1] == expected[-1]
        assert sum(window[stage]['eligible_calls'] for window in windows) == len(expected)
        assert abs(sum(window[stage]['scene_weight'] for window in windows) - 1) < 1e-12
    return windows


def main():
    base = Path(__file__).resolve().parents[1]
    reference = base / 'experiments/coslam_reference_runtime_v1'
    source = reference / 'RUNTIME_MANIFEST.json'
    workloads = []
    for dataset, scene, capacity, frames, iterations in [
        ('Replica', 'office0', 16, 2000, 10),
        ('Replica', 'office0', 19, 2000, 10),
        ('Tum', 'fr1_desk', 16, 592, 20),
    ]:
        windows = temporal_plan(frames)
        assert all(len(window['tracking']['selected_frames']) == 100 for window in windows)
        assert all(len(window['bundle_adjustment']['selected_frames']) == 20 for window in windows)
        selection = {stage: [f for window in windows for f in window[stage]['selected_frames']]
                     for stage in ('tracking', 'bundle_adjustment')}
        workloads.append(dict(id=f'{dataset}_{scene}_H{capacity}', dataset=dataset, scene=scene,
                              hash_size=capacity, condition='BASE', expected_native_frames=frames,
                              native_scene_config=f'configs/{dataset}/{scene}.yaml',
                              native_scene_config_sha256=hashlib.sha256((reference / f'configs/{dataset}/{scene}.yaml').read_bytes()).hexdigest(),
                              warmup_frames=20, map_every=5, max_tracking_updates=10,
                              native_ba_updates=iterations, quality=False, export_mesh=False,
                              engineering=False, full_online_sequence_required=True,
                              selected_calls_per_run={stage: len(values) for stage, values in selection.items()},
                              selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
                              thirds=windows))
    platforms = [dict(id='RTX_4070', seeds=[0, 1, 2], full_profile_passes=9),
                 dict(id='ORIN_MAXN', seeds=[0, 1], full_profile_passes=6),
                 dict(id='ORIN_15W', seeds=[0, 1], full_profile_passes=6)]
    assert sum(p['full_profile_passes'] for p in platforms) == 21
    result = dict(protocol_version=1, author_date='2026-10-21', system='Co-SLAM',
                  native_reference_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  public_reference=json.loads(source.read_text())['public_revision'],
                  status='PLANNED_DEFERRED_NOT_A_RUNNABLE_QUEUE',
                  protocol='DASIP_round2/COMPONENT_PROFILING_PROTOCOL.md',
                  requested_outputs=['forward_decomposition_table1', 'iteration_decomposition_optional', 'call_envelope_supporting'],
                  execution='Separate end-of-benchmark profiling block; no active queue mutation, no automatic mode change',
                  profiling_driver_status='To be implemented and validated in an isolated snapshot',
                  current_benchmark_runtimes_unchanged=True, new_gpu_passes_launched=False,
                  aggregation='Call means within each third, weighted by eligible native calls; seed mean and sample SD; shares are ratios of matched aggregated times',
                  platforms=platforms, total_full_profile_passes=21, workloads=workloads)
    output = base / 'experiments/manifests/component_profile_protocol_v1.json'
    content = json.dumps(result, indent=2) + '\n'
    if output.exists() and output.read_text() != content:
        raise FileExistsError('Preserve the existing frozen selection; create a new version for a change')
    output.write_text(content)
    print('PROTOCOL_SELECTION_VERIFIED: 3 workloads; 300 tracking + 60 BA calls per full pass; 21 deferred full passes')
    print(output)


if __name__ == '__main__':
    main()
