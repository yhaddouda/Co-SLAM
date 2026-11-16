#!/usr/bin/env python3
"""Inventory research and create verified, local archives without deleting inputs.

Uses only the Python standard library. Archives retain repository-relative paths.
Run ``plan`` first; inspect inventory.json before ``build``. Large omitted artifacts
are inventory entries, not deletion candidates. Run the same tool locally on Orin.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOTS = {
    'configs', 'datasets', 'docs', 'external', 'model', 'mp_slam',
    'neural_slam_eval', 'optimization', 'scripts', 'tests', 'tools', 'tiny-cuda-nn',
    'reproduction',
}
EVIDENCE_ROOTS = {
    'DASIP_round2', 'DASIP paper', 'Neuslam paper', 'DASIP_suite', 'CSI_2A',
    'Quality', 'Orin tutos', 'results', 'plots', 'experiment_data', 'profiling',
    'output', 'temp_configs', 'media',
}
SKIP_DIRS = {
    '.git', '.aws', '.agents', '.venv', 'venv', 'env', 'ENV', '__pycache__',
    '.pytest_cache', '.mypy_cache', '.torch_extensions_ada', 'node_modules',
    '.eggs', '*.egg-info',
}
NATIVE_SUFFIXES = {'.pyc', '.pyo', '.so', '.o', '.a', '.whl'}
LARGE_SUFFIXES = {'.pt', '.pth', '.ply', '.npy'}
FINAL_NAMES = {
    'paper_round2.tex', 'source_final_dasip_rev2.tex', 'main.bib', 'llncs.cls',
    'splncs04.bst', 'fig1.eps', 'fig1-eps-converted-to.pdf', 'paper_round2.bbl',
    'paper_round2.pdf', 'paper_round2_highlighted.pdf', 'paper_round2_overleaf.zip',
    'rebuttal_letter.tex', 'rebuttal_letter.txt', 'rebuttal_letter.pdf',
    'rebuttal_letter_overleaf.zip',
}


def digest(handle):
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def file_digest(path):
    with path.open('rb') as handle:
        return digest(handle)


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    return result.stdout.decode(errors='replace').strip() if result.returncode == 0 else None


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root.joinpath(*rel.parts)
    # Reject symlinks in every path component, including a replaced parent.
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Symlink in source path: {relative}')
    return path


def classify(rel, size):
    p = PurePosixPath(rel)
    if p.name.startswith('.env') or p.suffix in {'.pem', '.key'} or p.name in {'.codex', 'credentials'}:
        return 'excluded', 'local configuration or credential file'
    if p.suffix in NATIVE_SUFFIXES or 'build' in p.parts or 'build_live' in p.parts or 'build_round1_check' in p.parts:
        return 'excluded', 'native build or generated cache; keep installed environments locally'
    if p.suffix in LARGE_SUFFIXES or (p.suffix == '.npz' and size > 64 * 1024**2):
        return 'review_large', 'mesh, checkpoint or array: retain original until dependency and backup checks'
    if len(p.parts) == 1 and (p.suffix in {'.py', '.md', '.txt', '.ipynb'} or p.name in {'.gitignore', 'LICENSE'}):
        return 'code', 'repository source or documentation'
    if p.parts[0] in CODE_ROOTS:
        return 'code', 'runtime, dependency source, config, test or reproduction script'
    if p.parts[0] in EVIDENCE_ROOTS:
        if p.parts[:2] == ('DASIP_round2', 'experiments') and p.suffix in {'.py', '.sh', '.cu', '.cuh', '.cpp', '.h', '.yaml', '.yml', '.toml'}:
            return 'code', 'frozen paper runtime or reproduction script'
        return 'evidence', 'paper, raw measurements, poses, reports or provenance'
    return 'review_other', 'not selected automatically; original remains untouched'


def plan(root, out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    entries = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith('.egg-info')
                         and not (base / d).is_symlink()
                         and (base / d).resolve() != out.resolve().parent
                         and d != '_preservation')
        for name in sorted(files):
            path = base / name
            if path.is_symlink():
                continue
            rel = path.relative_to(root).as_posix()
            size = path.stat().st_size
            group, reason = classify(rel, size)
            entries.append(dict(path=rel, size=size, group=group, reason=reason))
    # Explicit paper evidence hashes override size/type heuristics. Never silently
    # omit an input just because it has an unusual extension.
    progress = root / 'DASIP_round2/paper_round2_placeholder/RESULTS_PROGRESS.json'
    expected = json.loads(progress.read_text()).get('input_hashes', {}) if progress.exists() else {}
    by_path = {r['path']: r for r in entries}
    missing = []
    for rel, sha in expected.items():
        name = 'DASIP_round2/' + rel
        if name not in by_path:
            missing.append(name)
            continue
        row = by_path[name]
        row.update(group='evidence', reason='explicit input hash in paper result ledger', ledger_sha256=sha)
    summary = defaultdict(lambda: dict(files=0, bytes=0))
    for row in entries:
        summary[row['group']]['files'] += 1
        summary[row['group']]['bytes'] += row['size']
    data = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                source_root=str(root), git_head=git(root, 'rev-parse', 'HEAD'),
                git_branch=git(root, 'branch', '--show-current'),
                git_status=git(root, 'status', '--porcelain=v1'),
                tiny_cuda_nn_head=git(root / 'tiny-cuda-nn', 'rev-parse', 'HEAD'),
                tiny_cuda_nn_status=git(root / 'tiny-cuda-nn', 'status', '--porcelain=v1'),
                ledger_missing=missing, summary=dict(summary), files=entries)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(inventory=str(out), summary=data['summary'], ledger_missing=missing), indent=2))


def verify(archive, expected_sha=None):
    if expected_sha is not None and file_digest(archive) != expected_sha:
        raise ValueError('Archive checksum mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        member = tar.next()
        if member is None or member.name != 'PRESERVATION_MANIFEST.json' or not member.isfile():
            raise ValueError('Missing archive manifest')
        manifest = json.load(tar.extractfile(member))
        expected = {r['path']: r for r in manifest['files']}
        if len(expected) != len(manifest['files']):
            raise ValueError('Duplicate manifest paths')
        seen = set()
        for member in iter(tar.next, None):
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts or not member.isfile():
                raise ValueError(f'Unsafe archive member {member.name}')
            row = expected.get(member.name)
            if row is None or member.name in seen or member.size != row['size']:
                raise ValueError(f'Unexpected archive member {member.name}')
            if digest(tar.extractfile(member)) != row['sha256']:
                raise ValueError(f'Corrupt archive member {member.name}')
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError(f'Missing {len(set(expected) - seen)} archive members')
    return dict(archive=str(archive), files=len(seen), verified=True)


def build(root, inventory, out, group):
    data = json.loads(inventory.read_text())
    if data['ledger_missing']:
        raise ValueError('Paper ledger has missing inputs; repair before archiving')
    rows = [dict(r) for r in data['files'] if r['group'] == group]
    if group == 'manuscript':
        rows = [dict(r) for r in data['files'] if PurePosixPath(r['path']).parent == PurePosixPath('DASIP_round2/paper_round2')
                and PurePosixPath(r['path']).name in FINAL_NAMES]
        names = {PurePosixPath(r['path']).name for r in rows}
        required = {'paper_round2.tex', 'source_final_dasip_rev2.tex', 'paper_round2.pdf', 'main.bib', 'llncs.cls', 'splncs04.bst'}
        if not required <= names:
            raise ValueError(f'Missing manuscript files: {required - names}')
    if not rows:
        raise ValueError(f'No files selected for {group}')
    if out.exists() or out.with_name(out.name + '.partial').exists():
        raise FileExistsError(f'Refusing to overwrite archive or partial: {out}')
    mismatches = []
    for row in rows:
        path = safe_path(root, row['path'])
        if not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError(f'Source changed since inventory: {row["path"]}')
        row['sha256'] = file_digest(path)
        if row.get('ledger_sha256') and row['sha256'] != row['ledger_sha256']:
            mismatches.append(row['path'])
    manifest = dict(schema=1, group=group, inventory_sha256=file_digest(inventory),
                    git_head=data['git_head'], ledger_mismatches=mismatches, files=rows,
                    note='Local archive only; cloud backup and full experiment reproduction are separate checks.')
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + '.partial')
    with tarfile.open(partial, 'w:gz', compresslevel=3) as tar:
        info = tarfile.TarInfo('PRESERVATION_MANIFEST.json')
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))
        for row in rows:
            tar.add(safe_path(root, row['path']), arcname=row['path'], recursive=False)
    verification = verify(partial)
    partial.rename(out)
    sha = file_digest(out)
    out.with_name(out.name + '.sha256').write_text(f'{sha}  {out.name}\n')
    out.with_name(out.name + '.manifest.json').write_bytes(encoded)
    verification.update(archive=str(out), sha256=sha, bytes=out.stat().st_size,
                        ledger_mismatches=mismatches, cloud_backup_verified=False)
    out.with_name(out.name + '.verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('plan')
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('build')
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--group', choices=['code', 'evidence', 'manuscript'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('verify')
    p.add_argument('archive', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'plan':
        plan(root, args.out.resolve())
    elif args.command == 'build':
        build(root, args.inventory.resolve(), args.out.resolve(), args.group)
    else:
        checksum = args.archive.with_name(args.archive.name + '.sha256')
        expected = checksum.read_text().split()[0] if checksum.exists() else None
        print(json.dumps(verify(args.archive, expected), indent=2))


if __name__ == '__main__':
    main()
