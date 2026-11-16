"""Thread-aware EC events; whole-call walls explicitly include shared-stream work."""
from contextlib import contextmanager
import csv
import threading
import time
import torch


def write_csv(path, records, fields=None):
    if records:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]))
            writer.writeheader();writer.writerows(records)


class Timer:
    def __init__(self):
        self.local = threading.local()
        self.calls, self.pending = [], []
        self.lock = threading.Lock()

    @contextmanager
    def frame(self, stage, frame):
        self.local.frame = int(frame)
        with self.call(stage):
            yield

    @contextmanager
    def call(self, stage):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter();start.record()
        try:
            yield
        finally:
            end.record();torch.cuda.synchronize()
            row = dict(frame_id=self.local.frame, stage=stage, wall_ms=1000*(time.perf_counter()-wall), cuda_ms=start.elapsed_time(end))
            with self.lock:self.calls.append(row)

    @contextmanager
    def event(self, metric):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:yield
        finally:
            end.record()
            with self.lock:self.pending.append((self.local.frame, self.local.stage, self.local.iteration, metric, start, end))

    @contextmanager
    def iteration(self, stage, iteration):
        self.local.stage = stage;self.local.iteration = iteration
        with self.event('full_iteration_cuda_ms'):yield

    @contextmanager
    def component(self, metric):
        with self.event(metric):yield

    def save(self, output):
        torch.cuda.synchronize()
        table = {}
        for frame, stage, iteration, metric, start, end in self.pending:
            if stage == 'first_frame_mapping':continue
            table.setdefault((frame, stage, iteration), {})[metric] = start.elapsed_time(end)
        rows = [dict(frame_id=f, iteration_type=s, iteration_id=i, **v) for (f,s,i),v in sorted(table.items())]
        fields = ['frame_id','iteration_type','iteration_id','forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms']
        assert all(set(r) == set(fields) for r in rows), 'Incomplete EC iteration instrumentation'
        write_csv(output/'iterations.csv', rows, fields)
        write_csv(output/'calls.csv', sorted(self.calls, key=lambda r:(r['frame_id'],r['stage'])))
        return rows
