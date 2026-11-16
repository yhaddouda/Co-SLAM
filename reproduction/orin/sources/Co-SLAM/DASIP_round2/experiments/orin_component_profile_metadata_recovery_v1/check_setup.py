"""CPU-only gate rejection checks against synthetic in-memory records."""
import ast
import copy
import math
from local_support import *


def assign(value, tokens, expected):
    key = int(tokens[0]) if isinstance(value, list) else tokens[0]
    if isinstance(value, list):
        while len(value) <= key:
            value.append({})
    if len(tokens) == 1:
        value[key] = expected
        return
    if (isinstance(value, dict) and key not in value) or (isinstance(value, list) and value[key] == {}):
        value[key] = [] if tokens[1].isdigit() else {}
    assign(value[key], tokens[1:], expected)


def main():
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    frozen_sources_ok()
    gate = read(ROOT / 'gates/UPSTREAM_SUCCESS_AND_EXPORTS.json')
    records = {}
    for condition in gate['conditions']:
        value = {}
        for check in condition['checks']:
            assign(value, check['pointer'].strip('/').split('/'), check['equals'])
        records[condition['file']] = value
    def evaluate(data, alive=False, failures=()):
        return gate_ready(gate, loader=lambda path: data[path],
                          exists=lambda path: path in data or path in failures,
                          alive=lambda identity: alive)[0]
    tests = {}
    tests['exact_successful_records_and_parent_exits_admitted'] = evaluate(records)
    partial = copy.deepcopy(records)
    partial.pop(str(DP / 'DP_COMPLETION_COMPLETE.json'))
    tests['disappearing_parents_without_terminal_marker_rejected'] = not evaluate(partial)
    partial = copy.deepcopy(records)
    partial.pop(str(DP / 'RETURN_BUNDLE_VERIFIED.json'))
    tests['missing_successful_final_export_rejected'] = not evaluate(partial)
    partial = copy.deepcopy(records)
    partial[str(DP / 'DP_COMPLETION_STATUS.json')]['phase'] = 'RUNNING_MISSING_DP_MAXN'
    tests['idle_job_to_job_gap_rejected'] = not evaluate(partial)
    partial = copy.deepcopy(records)
    partial[str(DP / 'DP_COMPLETION_STATUS.json')]['jobs'][0]['id'] = 'UNEXPECTED_SAME_COUNTS'
    tests['changed_identity_with_same_counts_rejected'] = not evaluate(partial)
    partial = copy.deepcopy(records)
    partial[str(DP / 'DP_COMPLETION_STATUS.json')]['jobs'][0]['status'] = 'blocked'
    tests['blocked_target_rejected'] = not evaluate(partial)
    partial = copy.deepcopy(records)
    partial[str(DP / 'DP_COMPLETION_STATUS.json')]['jobs'][0]['source_manifest_sha256'] = 'DIFFERENT'
    tests['changed_runtime_rejected'] = not evaluate(partial)
    tests['live_original_parent_rejected'] = not evaluate(records, alive=True)
    try:
        evaluate(records, failures=[gate['failure_files'][0]])
        tests['abnormal_upstream_failure_rejected'] = False
    except RuntimeError:
        tests['abnormal_upstream_failure_rejected'] = True
    protocols = [read(ROOT / folder / 'runtime/PROTOCOL.json') for folder in ['co_bundle', 'bundle']]
    for protocol in protocols:
        for workload in protocol['workloads']:
            n = workload['expected_native_frames']
            for index, third in enumerate(workload['thirds']):
                lo, hi = 20 + index * (n - 20) // 3, 20 + (index + 1) * (n - 20) // 3
                for stage, count in [('tracking', 100), ('bundle_adjustment', 20)]:
                    ids = [f for f in range(lo, hi) if stage == 'tracking' or f % 5 == 0]
                    expected = [ids[math.floor(j * (len(ids) - 1) / (count - 1) + .5)] for j in range(count)]
                    assert expected == third[stage]['selected_frames']
    for workload in protocols[1]['workloads']:
        co = next(w for w in protocols[0]['workloads'] if w['id'] == workload['id'])
        for ec_third, co_third in zip(workload['thirds'], co['thirds']):
            for stage in ['tracking', 'bundle_adjustment']:
                assert ec_third[stage]['selected_frames'] == co_third[stage]['selected_frames']
        assert workload['max_tracking_updates'] == 8 and workload['native_ba_updates'] == 10
    for system, count in [('CO', 3), ('EC', 2)]:
        jobs = read(ROOT / 'manifests' / f'{system}_MAXN_SINGLE_SEED.json')['jobs']
        assert len(jobs) == count and all(j['seed'] == 0 and j['condition'] == 'BASE' for j in jobs)
    tests['new_execution_plan_is_three_Co_two_EC_seed0_only'] = True
    tests['EC_preserves_native_updates_and_Co_frame_lists'] = True
    ecgate = read(ROOT / 'gates/EC_AFTER_CO_SUCCESS.json')
    ecrecords = {}
    for condition in ecgate['conditions']:
        value = {}
        for check in condition['checks']:
            assign(value, check['pointer'].strip('/').split('/'), check['equals'])
        ecrecords[condition['file']] = value
    def ec_ready(data):
        return gate_ready(ecgate, loader=lambda path: data[path], exists=lambda path: path in data,
                          alive=lambda identity: False)[0]
    assert ec_ready(ecrecords)
    missing_exit = copy.deepcopy(ecrecords)
    missing_exit.pop(str(ROOT / 'CO_PARENT_EXIT.json'))
    tests['EC_rejects_Co_completion_without_normal_parent_exit'] = not ec_ready(missing_exit)
    incomplete = copy.deepcopy(ecrecords)
    incomplete[str(ROOT / 'results/co_maxn/series_complete.json')]['completed'] = 2
    tests['EC_rejects_partial_Co_profiles'] = not ec_ready(incomplete)
    tests['frozen_exact_uniform_thirds_and_warmup_selection_verified'] = True
    tests['completed_benchmark_and_export_gates_ready_without_reruns'] = gate_ready(gate)[0] is True
    assert all(tests.values()), tests
    save(ROOT / 'LIGHTWEIGHT_SETUP_VALIDATION.json', dict(utc=now(), status='PASS', tests=tests,
          CUDA_initialized=False, datasets_read=False, bulk_results_audited=False,
          note='Synthetic success records existed only in memory; no machine completion marker was fabricated'))
    print(json.dumps(tests, indent=2))


if __name__ == '__main__':
    main()
