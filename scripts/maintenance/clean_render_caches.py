#!/usr/bin/env python3
"""Remove only regenerable legacy depth renders after verified asset preservation.

Dry-run by default. Never removes meshes, checkpoints, poses, measurements or
arbitrary NPZ files. A deletion receipt is written before any unlink operation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile
import zipfile

from preserve_research import file_digest, safe_path, verify


ROOT = Path(__file__).resolve().parents[2]
ASSET_NAMES = ('sampled_poses_1000.npz', 'gt_pc_unseen.npy', 'gt_mesh_cull_virt_cams.ply')


def candidate(root, cache, assets, asset_files, hash_file=file_digest):
    if cache.is_symlink() or cache.name != 'depths_virt_cams_1000.npz':
        raise ValueError('Not a generated depth cache')
    safe_path(root, cache.relative_to(root).as_posix())
    with zipfile.ZipFile(cache) as z:
        members = z.infolist()
        if len(members) != 1 or members[0].filename != 'depths.npy':
            raise ValueError('Unexpected cache contents')
        with z.open('depths.npy') as f:
            if not f.read(8).startswith(b'\x93NUMPY'):
                raise ValueError('Invalid depth array')
        crc = members[0].CRC
    config = cache.parent / 'config.json'
    data_path = Path(json.loads(config.read_text())['data']['datadir'])
    parts = data_path.parts
    relative = Path(*parts[parts.index('data') + 1:])
    if relative.parts[0] not in ('Replica', 'neural_rgbd_data'):
        raise ValueError('Unsupported legacy dataset')
    mesh_files = sorted(cache.parent.glob('*_cull_virt_cams.ply'))
    if not mesh_files or any(m.is_symlink() or not m.is_file() for m in mesh_files):
        raise ValueError('Missing reconstructed mesh')
    dependencies = []
    for name in ASSET_NAMES:
        rel = (relative / name).as_posix()
        row = asset_files.get(rel)
        path = safe_path(assets, rel)
        if row is None or not path.is_file() or hash_file(path) != row['sha256']:
            raise ValueError(f'Unpreserved or changed evaluation asset: {path}')
        dependencies.append(rel)
    stat = cache.stat()
    return dict(path=cache.relative_to(root).as_posix(), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, inode=stat.st_ino, device=stat.st_dev,
                depth_array_crc32=crc, config_sha256=file_digest(config),
                retained_meshes=[m.relative_to(root).as_posix() for m in mesh_files],
                preserved_assets=dependencies,
                reason='Generated rendering; evaluation rerenders from retained culled mesh and saved poses.')


def clean(root, assets, asset_archive, evidence_archive, receipt, apply=False):
    if receipt.exists():
        raise FileExistsError(f'Refusing to overwrite {receipt}')
    for archive in (asset_archive, evidence_archive):
        checksum = archive.with_name(archive.name + '.sha256').read_text().split()[0]
        verify(archive, checksum)
    manifest = json.loads(asset_archive.with_name(asset_archive.name + '.manifest.json').read_text())
    # Match the sidecar to the verified embedded manifest, not merely its filename.
    with tarfile.open(asset_archive) as tar:
        if json.load(tar.extractfile('PRESERVATION_MANIFEST.json')) != manifest:
            raise ValueError('Asset manifest sidecar differs from verified archive')
    asset_files = {r['path']: r for r in manifest['files']}
    entries, skipped = [], []
    # Cache asset hashes once across repeated runs of the same scene.
    checked_assets = set()
    original_digest = file_digest
    def cached_digest(path):
        if path.is_relative_to(assets):
            key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            if key not in checked_assets:
                result = original_digest(path)
                if result != asset_files[path.relative_to(assets).as_posix()]['sha256']:
                    return result
                checked_assets.add(key)
            return asset_files[path.relative_to(assets).as_posix()]['sha256']
        return original_digest(path)
    # Hashing is cheap for configs; large common GT meshes are checked once.
    for cache in sorted((root / 'output').rglob('depths_virt_cams_1000.npz')):
        try:
            entries.append(candidate(root, cache, assets, asset_files, hash_file=cached_digest))
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            skipped.append(dict(path=str(cache), reason=str(error)))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), apply=apply,
                  asset_archive=str(asset_archive), evidence_archive=str(evidence_archive),
                  planned_files=len(entries), planned_bytes=sum(r['bytes'] for r in entries),
                  skipped=skipped, files=entries, deleted_files=0, deleted_bytes=0)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    if apply:
        try:
            for row in entries:
                path = safe_path(root, row['path'])
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) != (
                    row['bytes'], row['mtime_ns'], row['inode'], row['device']):
                    raise ValueError(f'Cache changed since inspection: {path}')
                if file_digest(path.parent / 'config.json') != row['config_sha256']:
                    raise ValueError(f'Run configuration changed: {path.parent}')
                if not all((root / m).is_file() for m in row['retained_meshes']):
                    raise ValueError(f'Reconstructed mesh disappeared: {path.parent}')
                path.unlink()
                report['deleted_files'] += 1
                report['deleted_bytes'] += row['bytes']
        finally:
            receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--asset-archive', type=Path, required=True)
    parser.add_argument('--evidence-archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    clean(args.root.resolve(), args.assets.resolve(), args.asset_archive.resolve(),
          args.evidence_archive.resolve(), args.receipt.resolve(), args.apply)
