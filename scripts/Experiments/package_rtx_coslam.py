"""Package experiment sources and Orin references without native binaries/data."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = 'coslam_experiments'
EXCLUDED_PARTS = {'.git', '__pycache__', 'build', 'artifacts', '.pytest_cache'}
SOURCE_SUFFIXES = {'.py', '.yaml', '.yml', '.cu', '.cuh', '.c', '.cpp', '.h',
                   '.pyx', '.md', '.json', '.patch', '.pdf', '.tex', '.png', '.svg', '.txt'}
RESULT_SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.csv', '.npz', '.md', '.png', '.svg', '.log'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gather():
    files = set()
    for name in ('coslam.py', 'config.py', 'utils.py', 'README.md', 'LICENSE', 'requirements.txt'):
        files.add(ROOT / name)
    for name in ('configs/Replica', 'model', 'optimization', 'datasets', 'tools',
                 'external/NumpyMarchingCubes', 'scripts/Validation', 'scripts/Experiments',
                 'tests', 'docs', 'README', 'experiments/ec_slam_execution'):
        for path in (ROOT / name).rglob('*'):
            if path.is_symlink() or not path.is_file():
                continue
            if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
                continue
            if path.suffix in SOURCE_SUFFIXES:
                files.add(path)
    files.add(ROOT / 'results/README.md')
    for name in ('results/ba_experiments', 'results/execution_experiments'):
        for path in (ROOT / name).rglob('*'):
            if path.is_symlink() or not path.is_file():
                continue
            if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
                continue
            if path.suffix in RESULT_SUFFIXES or path.name.endswith('COMPLETE'):
                files.add(path)
    return sorted(files, key=lambda path: str(path.relative_to(ROOT)))


def revision(directory):
    completed = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                               text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.resolve().parent
    changed = []
    for name, record in manifest['files'].items():
        path = base / name
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            changed.append(name)
    print(json.dumps({'verified_files': len(manifest['files']), 'missing_or_changed': changed}, indent=2))
    if changed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'transfer/rtx4070/coslam_experiments.tar.gz')
    parser.add_argument('--verify', type=Path, help='Verify files in an extracted package; no CUDA/imports required')
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        return
    files = gather()
    records = {str(path.relative_to(ROOT)): {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files}
    manifest = dict(format_version=1, archive_directory=TOP,
        scope='Co-SLAM BA/tracking sources and tools; optional EC source snapshot/patch; Orin reference evidence.',
        coslam_base_commit=revision(ROOT), tiny_cuda_nn_source_commit=revision(ROOT / 'tiny-cuda-nn'),
        excluded=['Replica datasets', 'trained checkpoints', 'native shared libraries',
                  'conda environments', 'CUDA build caches', 'Nsight trace databases',
                  'Git metadata', 'unrelated Tum/Synthetic configs and earlier paper/profiling outputs'],
        files=records)
    manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Choose a new output path; preserving existing package: ' + str(args.output))
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=TOP + '/' + str(path.relative_to(ROOT)), recursive=False)
        info = tarfile.TarInfo(TOP + '/RTX_TRANSFER_MANIFEST.json')
        info.size = len(manifest_bytes)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))
    # Verify archived bytes rather than only trusting the input manifest.
    with tarfile.open(args.output, 'r:gz') as archive:
        for name, record in records.items():
            payload = archive.extractfile(TOP + '/' + name).read()
            assert len(payload) == record['bytes'] and hashlib.sha256(payload).hexdigest() == record['sha256'], name
    digest = sha256(args.output)
    args.output.with_name(args.output.name + '.sha256').write_text(digest + '  ' + args.output.name + '\n')
    args.output.with_name('RTX_TRANSFER_MANIFEST.json').write_bytes(manifest_bytes)
    print(json.dumps({'archive': str(args.output), 'files': len(files),
        'source_bytes': sum(record['bytes'] for record in records.values()),
        'archive_bytes': args.output.stat().st_size, 'sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
