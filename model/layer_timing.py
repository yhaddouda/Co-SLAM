import csv
from contextlib import contextmanager
from pathlib import Path

import torch


class DeferredCudaTimer:
    def __init__(self, enabled=False, output_csv='./profiling/timing.csv', warmup_frames=0):
        self.enabled = bool(enabled)
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.warmup_frames = int(warmup_frames)
        self.current_frame_id = -1
        self.current_outer_stage = ''
        self.pending = []
        self.header_written = False

    def set_context(self, frame_id: int, outer_stage: str):
        self.current_frame_id = int(frame_id)
        self.current_outer_stage = str(outer_stage)

    @contextmanager
    def stage(self, name: str):
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        try:
            yield
        finally:
            end_evt.record()
            self.pending.append((self.current_frame_id, self.current_outer_stage, name, start_evt, end_evt))

    def flush(self):
        if not self.enabled or not self.pending:
            return
        torch.cuda.synchronize()
        write_header = not self.output_csv.exists() or not self.header_written
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['frame_id', 'outer_stage', 'stage', 'cuda_ms'])
                self.header_written = True
            for frame_id, outer_stage, stage_name, start_evt, end_evt in self.pending:
                if frame_id < self.warmup_frames:
                    continue
                writer.writerow([frame_id, outer_stage, stage_name, float(start_evt.elapsed_time(end_evt))])
        self.pending.clear()
