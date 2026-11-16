#!/usr/bin/env python3
"""Independently audit saved Table 2 artifacts; no profiler invocation required."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import re

from extract_table2_orin import (CYCLES, SECTORS, IDEAL, EXCESS, THROUGHPUT,
                                IDENTITY, rows, one, require, integer,
                                validate_session, sha256, write_json, PROTOCOL, OCCURRENCE,
                                arg_value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path,
                        default=Path(__file__).resolve().parent / 'table2_orin')
    args = parser.parse_args()
    root = args.directory.resolve()
    data = json.loads((root / 'table2_orin.json').read_text())
    with (root / 'table2_orin.csv').open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    require(not data['problems'], 'Extraction has unresolved problems')
    require(len(data['rows']) == len(csv_rows) == 32, 'Expected 32 combinations')
    seen = set()
    proofs = []
    tex = (root / 'table2_orin_rows.tex').read_text()
    for r, c in zip(data['rows'], csv_rows):
        key = (r['power_mode'], r['system'], r['hash_size'], r['condition'])
        require(key not in seen, 'Duplicate combination')
        seen.add(key)
        evidence = root / 'evidence' / r['power_mode'] / r['system'] / f'H{r["hash_size"]}' / r['condition']
        validate_session(evidence / 'session.txt', Path(r['report_path']), *key)
        order = rows(evidence / 'original_order_raw.csv')
        occurrence = OCCURRENCE[r['system']]
        matching = [(i, row) for i, row in enumerate(order) if row['Kernel Name'] == 'kernel_grid']
        require(len(matching) >= occurrence, 'Required exact forward occurrence absent')
        selected_index, selected = matching[occurrence - 1]
        require(selected_index == r['original_report_ordinal_zero_based'], 'Original order mismatch')
        require(r['selection_protocol'] == PROTOCOL and r['selected_exact_forward_occurrence'] == occurrence,
                'Selection protocol mismatch')
        raw = one(rows(evidence / 'selected_kernel_raw.csv'), 'selected invocation')
        details = rows(evidence / 'selected_kernel_details.csv')
        require(all(selected[k] == raw[k] == r['kernel_identity'][k] for k in IDENTITY), 'Launch identity mismatch')
        require(all(all(d[k] == raw[k] for k in IDENTITY) for d in details), 'Details identity mismatch')
        require(raw['Kernel Name'] == r['kernel_demangled'], 'Demangled identity mismatch')
        cycles = one([d for d in details if d['Metric Name'] == f'Elapsed Cycles ({CYCLES})'
                      and d['Section Name'] == 'GPU Speed Of Light Throughput'], 'SOL cycles')
        memory = one([d for d in details if d['Metric Name'] == f'Memory Throughput ({THROUGHPUT})'
                      and d['Section Name'] == 'GPU Speed Of Light Throughput'], 'SOL memory percentage')
        require(memory['Metric Unit'] == '%', 'Memory throughput is not percent')
        warning = one([d for d in details if d['Rule Name'] == 'UncoalescedGlobalAccess'], 'warning')
        w = re.search(r'total of (\d+) excessive sectors \((\d+)% of the total (\d+) sectors\)',
                      warning['Rule Description'])
        require(w is not None, 'Warning unavailable')
        excessive, percentage, sectors = map(int, w.groups())
        require(integer(cycles['Metric Value']) == integer(raw[CYCLES]) == integer(selected[CYCLES]) == r['cycles'], 'Cycles differ')
        require(sectors == integer(raw[SECTORS]) == integer(selected[SECTORS]) == r['total_global_sectors'], 'Sectors differ')
        require(excessive == integer(raw[EXCESS]) == sectors - integer(raw[IDEAL]), 'Excessive sectors differ')
        require(percentage == r['excessive_sector_pct'] == int(f'{100.0 * excessive / sectors:.0f}'), 'Percentage differs')
        require(memory['Metric Value'] == raw[THROUGHPUT] == selected[THROUGHPUT] == r['memory_throughput_pct'], 'Throughput differs')
        for field in ['cycles', 'total_global_sectors', 'excessive_sector_pct', 'memory_throughput_pct',
                      'power_mode', 'system', 'hash_size', 'condition', 'report_path', 'report_sha256',
                      'selection_protocol', 'selected_exact_forward_occurrence']:
            require(str(r[field]) == c[field], f'CSV/JSON differ: {field}')
        millions = f'{Decimal(sectors) / 1000000:.6f}'
        require(millions == r['l2_sectors_millions'] == c['l2_sectors_millions'], 'Million-sector formatting differs')
        tex_row = f'{r["system"]} & H{r["hash_size"]} & {r["condition"]} & {r["cycles"]} & {millions} & {percentage}\\% & {r["memory_throughput_pct"]}\\%'
        require(tex_row in tex, 'LaTeX row does not match evidence')
        commands = json.loads((evidence / 'export_commands.json').read_text())
        require(len(commands) == 5, 'Expected five import exports per report')
        require(all(cmd['argv'][1] == '--import' and cmd['returncode'] == 0 for cmd in commands), 'Non-import/failed command')
        for cmd in commands:
            if '--kernel-name' in cmd['argv']:
                for flag, value in [('--kernel-name-base', 'function'), ('--kernel-name', 'regex:^kernel_grid$'),
                                    ('--launch-skip', str(occurrence - 1)), ('--launch-count', '1')]:
                    require(arg_value(cmd['argv'], flag) == value, f'Incorrect selection flag: {flag}')
        proofs.append(dict(combination=list(key), original_order_ordinal=selected_index,
                           selected_exact_forward_occurrence=occurrence, selection_protocol=PROTOCOL,
                           exported_result_id=raw['ID'], embedded_session_identity=True,
                           displayed_vs_raw_metrics=True, csv_json_tex_agree=True))
    write_json(root / 'artifact_validation.json', dict(checked_combinations=len(proofs), problems=[], rows=proofs))
    files = [p for p in root.rglob('*') if p.is_file() and p.name != 'evidence_sha256.json']
    write_json(root / 'evidence_sha256.json', {str(p.relative_to(root)): sha256(p) for p in sorted(files)})
    print('PASS: all 32 combinations; original order, embedded sessions, displayed/raw metrics, CSV, JSON, and LaTeX rows agree.')


if __name__ == '__main__':
    main()
