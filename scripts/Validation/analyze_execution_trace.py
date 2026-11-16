"""Attribute Nsight SQLite activity to serialized, fenced experiment calls.

Requires ExperimentTrackingCall / ExperimentMappingCall scopes from the trace
entrypoint. Uses all host threads in the same process (autograd can launch on
another thread). GPU busy time is an interval union, not a sum of overlaps.
"""
import argparse
import collections
import csv
import json
import sqlite3
from pathlib import Path
from statistics import mean, pstdev


def interval_union(intervals):
    total, left, right = 0, None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + (right - left if left is not None else 0)


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--drop-tracking-calls', type=int, default=3)
    parser.add_argument('--drop-mapping-calls', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect('file:' + str(args.sqlite.resolve()) + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
    required = {'StringIds','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_KERNEL'}
    if not required <= tables:
        raise ValueError(f'Missing CUDA/NVTX tables: {required - tables}')
    strings = dict(connection.execute('select id,value from StringIds'))
    scopes = list(connection.execute('''
        select n.start,n.end,n.globalTid,coalesce(n.text,s.value) as label
        from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where n.end is not null and coalesce(n.text,s.value) in
        ('ExperimentTrackingCall','ExperimentMappingCall') order by n.start
    '''))
    if not scopes:
        raise ValueError('No complete experiment-call scopes; use trace_replica_execution.py')
    for previous, current in zip(scopes, scopes[1:]):
        if current['start'] < previous['end']:
            raise ValueError('Overlapping experiment calls: time containment cannot isolate stages')
    copies = dict(connection.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')) if 'ENUM_CUDA_MEMCPY_OPER' in tables else {}
    calls = []
    stage_indices = collections.Counter()
    totals = {label:dict(api=collections.defaultdict(lambda:[0,0]),
                        kernel=collections.defaultdict(lambda:[0,0]),
                        copy=collections.defaultdict(lambda:[0,0,0]),
                        nvtx=collections.defaultdict(lambda:[0,0]))
              for label in ('ExperimentTrackingCall','ExperimentMappingCall')}
    for scope in scopes:
        label = scope['label']
        stage_indices[label] += 1
        skip = args.drop_tracking_calls if label == 'ExperimentTrackingCall' else args.drop_mapping_calls
        if stage_indices[label] <= skip:
            continue
        start,end = scope['start'],scope['end']
        # Nsight globalTid packs PID in the upper bits and TID in 24 lower bits.
        process = scope['globalTid'] & ~((1 << 24) - 1)
        runtime = list(connection.execute('''select * from CUPTI_ACTIVITY_KIND_RUNTIME
            where start>=? and end<=? and globalTid>=? and globalTid<?''',
            (start,end,process,process+(1<<24))))
        kernels = list(connection.execute('''select * from CUPTI_ACTIVITY_KIND_KERNEL
            where start>=? and end<=? and globalPid=?''', (start,end,process)))
        memcopy = list(connection.execute('''select * from CUPTI_ACTIVITY_KIND_MEMCPY
            where start>=? and end<=? and globalPid=?''', (start,end,process))) if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
        memset = list(connection.execute('''select * from CUPTI_ACTIVITY_KIND_MEMSET
            where start>=? and end<=? and globalPid=?''', (start,end,process))) if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else []
        if not kernels or not runtime:
            raise ValueError(f'Missing attributed CUDA activity for {label} call {stage_indices[label]}')
        api_names = [strings[r['nameId']] for r in runtime]
        intervals = [(r['start'],r['end']) for r in [*kernels,*memcopy,*memset]]
        row = dict(stage=label, call_ordinal=stage_indices[label],
                   profiled_wall_ms=(end-start)/1e6,
                   gpu_busy_union_ms=interval_union(intervals)/1e6,
                   kernel_sum_ms=sum(r['end']-r['start'] for r in kernels)/1e6,
                   kernel_count=len(kernels), runtime_count=len(runtime),
                   runtime_sum_ms=sum(r['end']-r['start'] for r in runtime)/1e6,
                   synchronize_api_count=sum('Synchronize' in name for name in api_names),
                   synchronize_api_sum_ms=sum(r['end']-r['start'] for r,name in zip(runtime,api_names) if 'Synchronize' in name)/1e6,
                   memcpy_count=len(memcopy), memcpy_bytes=sum(r['bytes'] for r in memcopy),
                   h2d_count=sum(r['copyKind']==1 for r in memcopy),
                   h2d_bytes=sum(r['bytes'] for r in memcopy if r['copyKind']==1),
                   d2h_count=sum(r['copyKind']==2 for r in memcopy),
                   d2h_bytes=sum(r['bytes'] for r in memcopy if r['copyKind']==2),
                   memset_count=len(memset), host_cuda_threads=len({r['globalTid'] for r in runtime}))
        calls.append(row)
        component_labels = {'choose_sliding_window','setup_optimizer_BA','reconstruct_poses_BA',
                            'update_pose_BA','pos_matrix_from_tensor_tr','c2w_from_matrix_tr',
                            'initialize_tracking_metrics_tr','select_best_pose_tr','finalize_best_pose_tr'}
        components = connection.execute('''select n.start,n.end,coalesce(n.text,s.value) as label
            from NVTX_EVENTS n left join StringIds s on n.textId=s.id
            where n.start>=? and n.end<=? and n.globalTid=?''',
            (start,end,scope['globalTid']))
        for component in components:
            if component['label'] in component_labels:
                entry = totals[label]['nvtx'][component['label']]
                entry[0] += 1; entry[1] += component['end']-component['start']
        for r,name in zip(runtime,api_names):
            entry = totals[label]['api'][name]
            entry[0] += 1; entry[1] += r['end']-r['start']
        for r in kernels:
            entry = totals[label]['kernel'][strings[r['demangledName']]]
            entry[0] += 1; entry[1] += r['end']-r['start']
        for r in memcopy:
            entry = totals[label]['copy'][copies.get(r['copyKind'], str(r['copyKind']))]
            entry[0] += 1; entry[1] += r['bytes']; entry[2] += r['end']-r['start']
    if not calls:
        raise ValueError('No calls remain after warmup exclusion')
    csv_write(args.output/'calls.csv',calls)
    summary = dict(source=str(args.sqlite.resolve()),
                   scope='Profiled diagnostics only; complete calls fenced; same-process CUDA activity included across host threads.',
                   interpretation='API durations include blocking waits and can overlap GPU work. They are not exclusive CPU overhead. These wall times are not headline performance results.',
                   dropped_initial_calls=dict(tracking=args.drop_tracking_calls,mapping=args.drop_mapping_calls), stages={})
    for label in totals:
        selected = [row for row in calls if row['stage']==label]
        if not selected:
            continue
        count = len(selected)
        metrics = {key:dict(mean=mean(row[key] for row in selected),std=pstdev(row[key] for row in selected))
                   for key in selected[0] if key not in ('stage','call_ordinal')}
        summary['stages'][label] = dict(calls=count,metrics=metrics)
        for kind in ('api','kernel','copy','nvtx'):
            rows = []
            for name,values in totals[label][kind].items():
                row = dict(name=name,total_count=values[0],mean_count=values[0]/count)
                if kind == 'copy':
                    row.update(total_bytes=values[1],mean_bytes=values[1]/count,total_ms=values[2]/1e6,mean_ms=values[2]/1e6/count)
                else:
                    row.update(total_ms=values[1]/1e6,mean_ms=values[1]/1e6/count)
                rows.append(row)
            csv_write(args.output/(label+'_'+kind+'.csv'),sorted(rows,key=lambda r:r['total_ms'],reverse=True))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),stages={k:v['calls'] for k,v in summary['stages'].items()})))


if __name__ == '__main__':
    main()
