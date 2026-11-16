"""Preservation checks run without GPU, datasets or third-party packages."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'preserve_research', Path(__file__).resolve().parents[1]
    / 'scripts/maintenance/preserve_research.py')
preserve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preserve)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'repo'
        self.root.mkdir()
        (self.root / 'coslam.py').write_text('print("original")\n')
        self.inventory = self.base / 'archives/inventory.json'
        with contextlib.redirect_stdout(io.StringIO()):
            preserve.plan(self.root, self.inventory)

    def test_round_trip_and_no_overwrite(self):
        archive = self.base / 'archives/code.tar.gz'
        with contextlib.redirect_stdout(io.StringIO()):
            preserve.build(self.root, self.inventory, archive, 'code')
        self.assertEqual(preserve.verify(archive)['files'], 1)
        with tarfile.open(archive) as tar:
            self.assertEqual(tar.extractfile('coslam.py').read(), b'print("original")\n')
        with self.assertRaises(FileExistsError):
            preserve.build(self.root, self.inventory, archive, 'code')

    def test_changed_source_and_symlink_are_rejected(self):
        (self.root / 'coslam.py').write_text('changed size\n')
        with self.assertRaisesRegex(ValueError, 'changed since inventory'):
            preserve.build(self.root, self.inventory, self.base / 'changed.tar.gz', 'code')
        (self.root / 'coslam.py').unlink()
        (self.root / 'coslam.py').symlink_to(self.inventory)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            preserve.build(self.root, self.inventory, self.base / 'symlink.tar.gz', 'code')

    def test_archive_corruption_and_path_traversal_are_rejected(self):
        for member_name, contents, sha in [
            ('coslam.py', b'corrupt', '0' * 64),
            ('../escape', b'payload', preserve.hashlib.sha256(b'payload').hexdigest()),
        ]:
            archive = self.base / 'bad.tar.gz'
            manifest = json.dumps({'files': [dict(path=member_name, size=len(contents), sha256=sha)]}).encode()
            with tarfile.open(archive, 'w:gz') as tar:
                for name, value in [('PRESERVATION_MANIFEST.json', manifest), (member_name, contents)]:
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    tar.addfile(info, io.BytesIO(value))
            with self.assertRaises(ValueError):
                preserve.verify(archive)

    def test_explicit_paper_input_overrides_large_artifact_filter(self):
        paper = self.root / 'DASIP_round2'
        (paper / 'paper_round2_placeholder').mkdir(parents=True)
        (paper / 'required.npy').write_bytes(b'needed input')
        (paper / 'paper_round2_placeholder/RESULTS_PROGRESS.json').write_text(json.dumps(
            {'input_hashes': {'required.npy': preserve.file_digest(paper / 'required.npy')}}))
        inventory = self.base / 'second/inventory.json'
        with contextlib.redirect_stdout(io.StringIO()):
            preserve.plan(self.root, inventory)
        row = next(r for r in json.loads(inventory.read_text())['files'] if r['path'].endswith('required.npy'))
        self.assertEqual(row['group'], 'evidence')


if __name__ == '__main__':
    unittest.main()
