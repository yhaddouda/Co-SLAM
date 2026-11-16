"""Test timing scope, deferred synchronization and complete-window accounting."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.layer_timing import BundleAdjustmentCudaTimer
from scripts.Validation.summarize_ba_breakdown import profile_frames, windows


class FakeEvent:
    clock = 0

    def __init__(self, enable_timing):
        self.timestamp = None

    def record(self):
        FakeEvent.clock += 1
        self.timestamp = FakeEvent.clock

    def elapsed_time(self, end):
        return float(end.timestamp - self.timestamp)


class BABreakdownTests(unittest.TestCase):
    def test_deferred_sync_iteration_ids_context_and_single_header(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.Event', FakeEvent), \
                patch('torch.cuda.synchronize') as synchronize:
            timer = BundleAdjustmentCudaTimer(True, Path(tmp) / 'timing.csv', 10)
            timer.set_context(5, 'Bundle_Adjustment')
            with timer.stage('BA_TOTAL'):
                pass
            timer.set_context(10, 'Tracking')
            with timer.stage('Forward_BA'):
                pass
            self.assertEqual(timer.pending, [])
            for frame in (10, 15):
                timer.set_context(frame, 'Bundle_Adjustment')
                timer.set_ba_metadata('baseline_fast', frame // 5, frame // 5, 2)
                with timer.stage('BA_TOTAL'):
                    for _ in range(2):
                        with timer.stage('BA_ITER_PROFILE_TOTAL'):
                            with timer.stage('Forward_BA'):
                                pass
                    with timer.stage('BA_POSE_COMMIT'):
                        with timer.stage('BA_COMMIT_BATCH_COPY'):
                            pass
                self.assertEqual(synchronize.call_count, (frame - 10) // 5)
                timer.flush()
            self.assertEqual(synchronize.call_count, 2)
            with timer.output_csv.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 14)
            for frame in ('10', '15'):
                iterations = [int(r['iteration_id']) for r in rows
                              if r['frame_id'] == frame and r['stage'] == 'Forward_BA']
                self.assertEqual(iterations, [0, 1])
                commit = next(r for r in rows if r['frame_id'] == frame and r['stage'] == 'BA_POSE_COMMIT')
                self.assertEqual(commit['iteration_id'], '-1')

    def test_excludes_inclusive_and_detail_rows_from_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'timing.csv'
            stages = [('BA_TOTAL', 'inclusive', -1, 20),
                      ('BA_ITER_PROFILE_TOTAL', 'inclusive', 0, 10),
                      ('Forward_BA', 'component', 0, 8),
                      ('BA_POSE_COMMIT', 'component', -1, 5),
                      ('BA_COMMIT_REBUILD_STORE', 'detail', -1, 4)]
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(BundleAdjustmentCudaTimer.OUTPUT_COLUMNS)
                for name, scope, iteration, elapsed in stages:
                    writer.writerow([100, 'baseline', 20, 20, 1, iteration, name, scope, elapsed, 1])
            frames, _ = profile_frames(path)
            self.assertEqual(frames[100]['unattributed_cuda_ms'], 7)
            self.assertEqual(frames[100]['pose_commit_cuda_ms'], 5)
            text = path.read_text().replace('BA_ITER_PROFILE_TOTAL', 'INVALID_TOTAL')
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'Incomplete iteration'):
                profile_frames(path)

    def test_early_late_windows_are_disjoint_and_equal_sized(self):
        frames = [{'frame_id': i} for i in range(20, 80, 5)]
        selected = windows(frames, 30, 50)
        self.assertFalse(selected['early'] & selected['late'])
        self.assertEqual(len(selected['early']), len(selected['late']))
        self.assertEqual(min(selected['early']), 30)
        self.assertEqual(max(selected['late']), 75)
        with self.assertRaises(ValueError):
            windows(frames, 100, 50)


if __name__ == '__main__':
    unittest.main()
