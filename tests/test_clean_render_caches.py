import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


MAINTENANCE = Path(__file__).resolve().parents[1] / 'scripts/maintenance'
sys.path.insert(0, str(MAINTENANCE))
import clean_render_caches as cleanup
from preserve_research import file_digest


class RenderCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.run = self.root / 'output/Replica/office0/run'
        self.run.mkdir(parents=True)
        self.cache = self.run / 'depths_virt_cams_1000.npz'
        with zipfile.ZipFile(self.cache, 'w') as z:
            z.writestr('depths.npy', b'\x93NUMPY\x01\x00')
        (self.run / 'mesh_track1999_cull_virt_cams.ply').write_text('retained mesh')
        (self.run / 'config.json').write_text(json.dumps({'data': {'datadir': '../datasets/data/Replica/office0'}}))
        self.assets = Path(self.temp.name) / 'assets'
        scene = self.assets / 'Replica/office0'
        scene.mkdir(parents=True)
        self.manifest = {}
        for name in cleanup.ASSET_NAMES:
            p = scene / name
            p.write_bytes(b'preserved asset')
            self.manifest[p.relative_to(self.assets).as_posix()] = {'sha256': file_digest(p)}

    def test_candidate_requires_retained_mesh_and_matching_preserved_inputs(self):
        row = cleanup.candidate(self.root, self.cache, self.assets, self.manifest)
        self.assertEqual(row['path'], 'output/Replica/office0/run/depths_virt_cams_1000.npz')
        self.assertTrue(self.cache.exists())
        asset = self.assets / 'Replica/office0/sampled_poses_1000.npz'
        asset.write_bytes(b'changed camera poses')
        with self.assertRaisesRegex(ValueError, 'changed evaluation asset'):
            cleanup.candidate(self.root, self.cache, self.assets, self.manifest)

    def test_other_arrays_and_missing_mesh_are_rejected(self):
        other = self.run / 'trajectories.npz'
        other.write_bytes(self.cache.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Not a generated depth cache'):
            cleanup.candidate(self.root, other, self.assets, self.manifest)
        (self.run / 'mesh_track1999_cull_virt_cams.ply').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing reconstructed mesh'):
            cleanup.candidate(self.root, self.cache, self.assets, self.manifest)


if __name__ == '__main__':
    unittest.main()
