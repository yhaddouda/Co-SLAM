#!/usr/bin/env python3
"""Import existing Orin reports for Table 2; never profile or set power/clocks.

Uses only Python's standard library and the installed Nsight Compute CLI.
The RTX extraction script was not present on this host. This implements the
user-specified, verified RTX selection protocol without RTX example assertions.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

CYCLES = 'gpc__cycles_elapsed.max'
SECTORS = 'memory_l2_theoretical_sectors_global'
IDEAL = 'memory_l2_theoretical_sectors_global_ideal'
EXCESS = 'derived__memory_l2_theoretical_sectors_global_excessive'
THROUGHPUT = 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'
METRICS = [CYCLES, SECTORS, IDEAL, EXCESS, THROUGHPUT]
PROTOCOL = 'coslam_first_ecslam_second_exact_kernel_grid_v2'
OCCURRENCE = {'CoSLAM': 1, 'ECSLAM': 2}
CONDITIONS = dict(A='A_baseline', B='B_morton_hash', C='C_morton_order',
                  D='D_memory_coherent')
IDENTITY = ['ID', 'Process ID', 'Process Name', 'Host Name', 'Context',
            'Stream', 'Block Size', 'Grid Size', 'Device', 'CC']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def integer(value):
    d = Decimal(value.replace(',', ''))
    require(d.is_finite() and d == d.to_integral_value(), f'Not an integer: {value}')
    return int(d)


def rows(path):
    with path.open(newline='') as f:
        data = list(csv.DictReader(f))
    return [r for r in data if r.get('ID') not in ('', None)]


def one(items, label):
    require(len(items) == 1, f'Expected one {label}; found {len(items)}')
    return items[0]


def arg_value(argv, flag):
    require(argv.count(flag) == 1, f'Missing/ambiguous capture argument {flag}')
    return argv[argv.index(flag) + 1]


def validate_session(path, report, mode, system, h, condition):
    # Session text wraps long attributes at fixed columns, including mid-word.
    # Remove whitespace to recover the embedded command and device identity.
    text = re.sub(r'\s+', '', path.read_text())
    expected_power = 'MAXN' if mode == 'MAXN' else 'MODE_15W'
    driver = 'run_coslam_round2.py' if system == 'CoSLAM' else 'run_ec_round2.py'
    for token in [report.stem, driver, '--sceneoffice0', f'--hash-size{h}',
                  f'--ncu-variant{condition}', '--seed0',
                  f'--expected-power-mode{expected_power}', 'display_nameOrin']:
        require(token in text, f'Embedded report session missing expected identity: {token}')
    return True


def export(ncu, report, evidence, filename, options, commands):
    argv = [ncu, '--import', str(report), *options]
    require(argv[1] == '--import', 'Only report imports are permitted')
    output = evidence / filename
    stderr = evidence / (filename + '.stderr.txt')
    record = dict(argv=argv, stdout=str(output), stderr=str(stderr),
                  shell='LC_ALL=C ' + shlex.join(argv) + ' > ' + shlex.quote(str(output)))
    commands.append(record)
    with output.open('w') as out, stderr.open('w') as err:
        result = subprocess.run(argv, stdout=out, stderr=err,
                                env={**os.environ, 'LC_ALL': 'C'}, timeout=300)
    record['returncode'] = result.returncode
    write_json(evidence / 'export_commands.json', commands)
    require(result.returncode == 0, f'Import failed: {filename}; see {stderr}')
    require('==ERROR==' not in output.read_text(), f'NCU error in {output}')
    return output


def extract(ncu, report, evidence, mode, system, h, condition):
    evidence.mkdir(parents=True, exist_ok=True)
    source = report.parent
    metadata = {}
    for name in ['environment_initial.json', 'command.json', 'CAPTURE_FAILED.json',
                 'CAPTURE_COMPLETE.json', 'app_output/config.json',
                 'app_output/command.json', 'app_output/source_manifest.json', 'capture.log']:
        path = source / name
        if path.is_file():
            dest = evidence / 'capture_metadata' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            metadata[name] = dict(source=str(path), evidence=str(dest), sha256=sha256(path))
    env = json.loads((source / 'environment_initial.json').read_text())
    capture = json.loads((source / 'command.json').read_text())
    config = json.loads((source / 'app_output/config.json').read_text())
    actual_mode = {'MAXN': 'MAXN', '15W': 'MODE_15W'}[mode]
    require(env['power']['name'] == actual_mode, 'Power mode differs from path')
    require(capture['mode'] == mode, 'Capture command mode differs from path')
    argv = capture['driver']
    for flag, expected in [('--scene', 'office0'), ('--hash-size', str(h)),
                           ('--seed', '0'), ('--ncu-variant', condition),
                           ('--expected-power-mode', actual_mode)]:
        require(arg_value(argv, flag) == expected, f'{flag} differs from path')
    driver_name = 'run_coslam_round2.py' if system == 'CoSLAM' else 'run_ec_round2.py'
    require(any(Path(a).name == driver_name for a in argv), 'System driver differs from path')
    require(config['dataset'].lower() == 'replica', 'Dataset is not Replica')
    require(config['round2']['seed'] == 0, 'Config seed mismatch')
    require(config['round2']['ncu_variant'] == condition, 'Config condition mismatch')
    require(capture['variant']['label'] == CONDITIONS[condition], 'Variant label mismatch')
    expected_name = f'ORIN_{mode}_{system}_office0_H{h}_{CONDITIONS[condition]}_seed0_grid_first500.ncu-rep'
    require(report.name == expected_name, 'Unexpected report basename')
    require(Path(arg_value(capture['argv'], '--export')).name == report.stem,
            'Capture export basename differs from report')
    commands = []
    occurrence = OCCURRENCE[system]
    skip = occurrence - 1
    selection = ['--kernel-name-base', 'function', '--kernel-name',
                 'regex:^kernel_grid$', '--launch-skip', str(skip), '--launch-count', '1']
    details_path = export(ncu, report, evidence, 'selected_kernel_details.csv',
                          selection + ['--page', 'details', '--print-units', 'base',
                                       '--print-metric-name', 'label-name', '--csv'], commands)
    raw_path = export(ncu, report, evidence, 'selected_kernel_raw.csv',
                      selection + ['--page', 'raw', '--print-units', 'base', '--csv'], commands)
    # No kernel or launch filter: all invocations in the original report order.
    order_path = export(ncu, report, evidence, 'original_order_raw.csv',
                        ['--page', 'raw', '--metrics', ','.join(METRICS),
                         '--print-kernel-base', 'function', '--print-units', 'base', '--csv'], commands)
    export(ncu, report, evidence, 'selected_kernel_details.txt',
           selection + ['--page', 'details', '--print-units', 'base',
                        '--print-metric-name', 'label-name'], commands)
    session_path = export(ncu, report, evidence, 'session.txt', ['--page', 'session'], commands)
    validate_session(session_path, report, mode, system, h, condition)
    details = rows(details_path)
    raw = one(rows(raw_path), 'selected raw invocation')
    order = rows(order_path)
    matching = [(i, row) for i, row in enumerate(order) if row['Kernel Name'] == 'kernel_grid']
    require(len(matching) >= occurrence, f'Missing exact forward kernel_grid occurrence {occurrence}')
    ordinal, selected = matching[skip]
    require(all(r['Kernel Name'] == raw['Kernel Name'] for r in details), 'Mixed detail kernels')
    require(re.match(r'^(?:void )?(?:\w+::)*kernel_grid(?:<|\()', raw['Kernel Name']),
            'Selected demangled name is not exact kernel_grid')
    for key in IDENTITY:
        require(selected[key] == raw[key], f'Original-order vs selected raw identity mismatch: {key}')
        require(all(r[key] == raw[key] for r in details), f'Details identity mismatch: {key}')
    for metric in METRICS:
        require(Decimal(selected[metric]) == Decimal(raw[metric]),
                f'Original-order vs selected raw metric mismatch: {metric}')

    def sol(metric, label, unit):
        r = one([r for r in details if r['Section Name'] == 'GPU Speed Of Light Throughput'
                 and r['Metric Name'] == f'{label} ({metric})'], metric)
        require(r['Metric Unit'] == unit, f'Incorrect unit for {metric}')
        require(Decimal(r['Metric Value']) == Decimal(raw[metric]), f'Details/raw mismatch: {metric}')
        return r['Metric Value']

    cycles = integer(sol(CYCLES, 'Elapsed Cycles', 'cycle'))
    throughput = sol(THROUGHPUT, 'Memory Throughput', '%')
    require(re.fullmatch(r'\d+\.\d{2}', throughput), 'Expected two-decimal memory throughput')
    warning = one([r for r in details if r['Rule Name'] == 'UncoalescedGlobalAccess'],
                  'UncoalescedGlobalAccess warning')
    match = re.search(r'total of ([\d,]+) excessive sectors \((\d+)% of the total ([\d,]+) sectors\)',
                      warning['Rule Description'])
    require(match is not None, 'Could not parse warning counts/whole-number percentage')
    excessive, percentage, total = map(integer, match.groups())
    require(total == integer(raw[SECTORS]), 'Warning total vs raw sectors mismatch')
    require(excessive == integer(raw[EXCESS]), 'Warning excess vs raw derived metric mismatch')
    require(excessive == total - integer(raw[IDEAL]), 'Total-minus-ideal cross-check failed')
    require(f'{100.0 * excessive / total:.0f}' == str(percentage), 'Warning percentage cross-check failed')
    state = json.loads((source / 'CAPTURE_FAILED.json').read_text()) if (source / 'CAPTURE_FAILED.json').exists() else {}
    caveats = []
    if state:
        caveats.append(f'Capture marked failed; retained report imports successfully and exact forward occurrence {occurrence} passes all metric checks.')
    order_proof = dict(original_report_invocations=len(order), exact_forward_invocations=len(matching),
                       selection_protocol=PROTOCOL, selected_exact_forward_occurrence=occurrence,
                       selected_original_order_ordinal_zero_based=ordinal,
                       preceding_invocations=order[:ordinal], selected_invocation=selected,
                       note='CSV ID is the NCU exported result ID; it is not assumed to be an API call ID.',
                       all_identity_fields_match=True, all_five_metrics_match=True)
    write_json(evidence / 'selection_validation.json', order_proof)
    result = dict(power_mode=mode, system=system, dataset='Replica', scene='office0',
                  selection_protocol=PROTOCOL, selected_exact_forward_occurrence=occurrence,
                  hash_size=h, condition=condition, seed=0, cycles=cycles,
                  total_global_sectors=total, l2_sectors_millions=f'{Decimal(total) / 1000000:.6f}',
                  excessive_sector_pct=percentage, memory_throughput_pct=throughput,
                  excessive_global_sectors=excessive, ideal_global_sectors=integer(raw[IDEAL]),
                  warning_estimated_speedup_pct_NOT_USED=warning['Estimated Speedup'],
                  warning_text=warning['Rule Description'], report_path=str(report),
                  report_sha256=sha256(report), report_bytes=report.stat().st_size,
                  kernel_function='kernel_grid', kernel_demangled=raw['Kernel Name'],
                  ncu_exported_result_id=raw['ID'], api_call_id=None,
                  api_call_id_note='Not exposed by these CLI exports; no value inferred from result ID.',
                  kernel_identity={k: raw[k] for k in IDENTITY},
                  original_report_ordinal_zero_based=ordinal, original_report_invocations=len(order),
                  exact_forward_invocations=len(matching),
                  capture_power_name=env['power']['name'], capture_power_raw=env['power']['raw'],
                  capture_utc=env['utc'], capture_command_mode=capture['mode'],
                  capture_status=state.get('status', 'not_marked_failed'),
                  capture_error=state.get('error'), caveats=caveats,
                  evidence_directory=str(evidence), metadata=metadata, export_commands=commands,
                  validation=dict(original_order=True, power_metadata=True, capture_identity=True,
                                  cycles_details_vs_raw=True, sectors_warning_vs_raw=True,
                                  excessive_percentage_vs_raw=True, throughput_details_vs_raw=True))
    write_json(evidence / 'extraction.json', result)
    (evidence / 'validation_failure.json').unlink(missing_ok=True)
    # Remove superseded exports so the handoff cannot mix selection protocols.
    for obsolete in evidence.glob('first_kernel_*'):
        obsolete.unlink()
    return result


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-root', type=Path,
                        default=base.parent / 'experiments/orin_ncu_abcd_v1/results/ORIN')
    parser.add_argument('--output', type=Path, default=base / 'table2_orin')
    parser.add_argument('--ncu', default=shutil.which('ncu'))
    parser.add_argument('--jobs', type=int, default=2, help='Concurrent read-only report imports')
    parser.add_argument('--resume', action='store_true', help='Reuse completed, hash-verified extractions')
    args = parser.parse_args()
    require(args.ncu, 'Nsight Compute CLI not found')
    require(args.jobs > 0, '--jobs must be positive')
    args.output = args.output.resolve()
    args.reports_root = args.reports_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([args.ncu, '--version'], text=True)
    (args.output / 'ncu_version.txt').write_text(version)
    expected = list(itertools.product(['MAXN', '15W'], ['CoSLAM', 'ECSLAM'], [16, 19], 'ABCD'))
    inventory, problems, results = [], [], []
    for mode, system, h, condition in expected:
        folder = args.reports_root / mode / system / 'office0' / f'H{h}' / CONDITIONS[condition]
        reports = sorted(folder.glob('*seed0*.ncu-rep'))
        key = f'{mode}/{system}/H{h}/{condition}'
        inventory.append(dict(combination=key, reports=[str(r) for r in reports]))
        if len(reports) != 1:
            problems.append(dict(combination=key, problem='missing' if not reports else 'ambiguous',
                                 reports=[str(r) for r in reports]))
    write_json(args.output / 'inventory.json', inventory)
    def process(item, combo):
        mode, system, h, condition = combo
        evidence = args.output / 'evidence' / mode / system / f'H{h}' / condition
        print(f'Importing {item["combination"]}', flush=True)
        report = Path(item['reports'][0])
        if args.resume and (evidence / 'extraction.json').exists():
            result = json.loads((evidence / 'extraction.json').read_text())
            require(result.get('selection_protocol') == PROTOCOL
                    and result.get('selected_exact_forward_occurrence') == OCCURRENCE[system],
                    'Selection protocol changed; rerun without --resume')
            require(result['report_sha256'] == sha256(report), 'Report changed since previous extraction')
            require(result['report_path'] == str(report), 'Report path changed since previous extraction')
            require(all(result['validation'].values()), 'Previous validation failed')
            validate_session(evidence / 'session.txt', report, *combo)
            return result
        return extract(args.ncu, report, evidence, *combo)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process, item, combo): (item, combo)
                   for item, combo in zip(inventory, expected) if len(item['reports']) == 1}
        for future in as_completed(futures):
            item, combo = futures[future]
            mode, system, h, condition = combo
            evidence = args.output / 'evidence' / mode / system / f'H{h}' / condition
            try:
                result = future.result()
                results.append(result)
                print(f'  verified {item["combination"]}: {result["cycles"]} {result["total_global_sectors"]} '
                      f'{result["excessive_sector_pct"]}% {result["memory_throughput_pct"]}%', flush=True)
            except Exception as error:
                problem = dict(combination=item['combination'], problem=str(error), evidence=str(evidence))
                problems.append(problem)
                evidence.mkdir(parents=True, exist_ok=True)
                write_json(evidence / 'validation_failure.json', problem)
                print(f'  FAILED {item["combination"]}: {error}', flush=True)
            results.sort(key=lambda r: expected.index((r['power_mode'], r['system'], r['hash_size'], r['condition'])))
            write_json(args.output / 'table2_orin.json', dict(expected_combinations=32,
                       completed_combinations=len(results), problems=problems, rows=results))
    fields = ['power_mode', 'system', 'dataset', 'scene', 'hash_size', 'condition', 'seed',
              'selection_protocol', 'selected_exact_forward_occurrence',
              'cycles', 'total_global_sectors', 'l2_sectors_millions', 'excessive_sector_pct',
              'memory_throughput_pct', 'report_path', 'report_sha256', 'kernel_function',
              'kernel_demangled', 'ncu_exported_result_id', 'original_report_ordinal_zero_based',
              'original_report_invocations', 'exact_forward_invocations', 'capture_power_name',
              'capture_utc', 'capture_status', 'evidence_directory', 'export_commands']
    with (args.output / 'table2_orin.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            writer.writerow({**row, 'export_commands': json.dumps(row['export_commands'])})
    lines = ['% Verified Orin rows only. Columns: system, capacity, condition, raw cycles,',
             '% Exact kernel_grid occurrence: CoSLAM first; ECSLAM second.',
             '% total global sectors in millions, excessive-sector %, memory throughput %. ',
             '% Adapt placement to the existing Table 2 on RTX; preserve its RTX panel.']
    for mode in ['MAXN', '15W']:
        lines.append(r'\multicolumn{7}{c}{Orin ' + mode + r'} \\')
        for row in results:
            if row['power_mode'] == mode:
                lines.append(f'{row["system"]} & H{row["hash_size"]} & {row["condition"]} & '
                             f'{row["cycles"]} & {row["l2_sectors_millions"]} & '
                             f'{row["excessive_sector_pct"]}\\% & {row["memory_throughput_pct"]}\\% ' + r'\\')
    (args.output / 'table2_orin_rows.tex').write_text('\n'.join(lines) + '\n')
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), ncu_version=version.strip(),
                   expected_combinations=32, completed_combinations=len(results), problems=problems,
                   captures_marked_failed=sum(r['capture_status'] == 'failed' for r in results),
                   manuscript_updated=False, manuscript_pdf_rebuilt=False,
                   manuscript_blocker='Manuscript and verified RTX script are on RTX only, absent from this workspace.',
                   selected_api_call_ids='Not exposed by CLI; exported result IDs and full launch identities saved.',
                   selection_protocol=PROTOCOL,
                   protocol='CoSLAM: first exact kernel_grid (--launch-skip 0). ECSLAM: second exact kernel_grid (--launch-skip 1). Each verified in unfiltered original report order. No averaging, profiling, or power changes.',
                   provenance='RTX script unavailable locally; parser implements user-specified protocol.',
                   metrics=dict(cycles=CYCLES, sectors=SECTORS, throughput=THROUGHPUT,
                                excessive_pct='Whole-number percentage in UncoalescedGlobalAccess warning'))
    write_json(args.output / 'validation_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['expected_combinations', 'completed_combinations',
                                            'problems', 'captures_marked_failed']}, indent=2))
    return 1 if problems or len(results) != 32 else 0


if __name__ == '__main__':
    sys.exit(main())
