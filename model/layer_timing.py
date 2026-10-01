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

    def finish_frame(self, frame_id: int):
        return


class IterationBreakdownCudaTimer:
    """Record forward, backward, and total CUDA time for each Co-SLAM iteration."""

    STAGE_SPECS = {
        'forward_tr': ('tracking', 'forward_cuda_ms'),
        'backward_tr': ('tracking', 'backward_cuda_ms'),
        'TR_ITER_PROFILE_TOTAL': ('tracking', 'full_iteration_cuda_ms'),
        'Forward_BA': ('bundle_adjustment', 'forward_cuda_ms'),
        'Backward_BA': ('bundle_adjustment', 'backward_cuda_ms'),
        'BA_ITER_PROFILE_TOTAL': (
            'bundle_adjustment',
            'full_iteration_cuda_ms',
        ),
    }
    OUTPUT_COLUMNS = (
        'frame_id',
        'iteration_type',
        'iteration_id',
        'forward_cuda_ms',
        'backward_cuda_ms',
        'full_iteration_cuda_ms',
    )

    def __init__(
        self,
        enabled=False,
        output_csv='./profiling/iteration_breakdown.csv',
        warmup_frames=0,
    ):
        self.enabled = bool(enabled)
        self.output_csv = Path(output_csv)
        self.warmup_frames = int(warmup_frames)
        self.current_frame_id = -1
        self.current_outer_stage = ''
        self.pending = []
        self.next_iteration = {}
        self.active_iterations = {}
        self.orphan_occurrences = {}
        self.header_written = False

        if self.enabled:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            self.header_written = (
                self.output_csv.exists() and self.output_csv.stat().st_size > 0
            )

    def set_context(self, frame_id: int, outer_stage: str):
        self.current_frame_id = int(frame_id)
        self.current_outer_stage = str(outer_stage)

    def _iteration_id(self, frame_id, iteration_type, metric_name):
        iteration_key = (frame_id, iteration_type)
        is_total = metric_name == 'full_iteration_cuda_ms'

        if is_total:
            iteration_id = self.next_iteration.get(iteration_key, 0)
            self.next_iteration[iteration_key] = iteration_id + 1
            self.active_iterations.setdefault(iteration_key, []).append(
                iteration_id
            )
            return iteration_id, True

        active = self.active_iterations.get(iteration_key)
        if active:
            return active[-1], False

        # This fallback keeps the logger usable if a component stage is ever
        # called outside its total-stage context. The current Co-SLAM
        # instrumentation always takes the active-iteration path.
        orphan_key = (frame_id, iteration_type, metric_name)
        iteration_id = self.orphan_occurrences.get(orphan_key, 0)
        self.orphan_occurrences[orphan_key] = iteration_id + 1
        return iteration_id, False

    @contextmanager
    def stage(self, name: str):
        stage_spec = self.STAGE_SPECS.get(name)
        if (
            not self.enabled
            or stage_spec is None
            or not torch.cuda.is_available()
        ):
            yield
            return

        frame_id = self.current_frame_id
        iteration_type, metric_name = stage_spec
        iteration_id, owns_active_iteration = self._iteration_id(
            frame_id,
            iteration_type,
            metric_name,
        )

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        try:
            yield
        finally:
            end_evt.record()
            self.pending.append(
                (
                    frame_id,
                    iteration_type,
                    iteration_id,
                    metric_name,
                    start_evt,
                    end_evt,
                )
            )
            if owns_active_iteration:
                iteration_key = (frame_id, iteration_type)
                active = self.active_iterations.get(iteration_key, [])
                if active:
                    active.pop()
                if not active:
                    self.active_iterations.pop(iteration_key, None)

    def flush(self):
        if not self.enabled or not self.pending:
            return

        torch.cuda.synchronize()
        rows = {}
        row_order = []
        for (
            frame_id,
            iteration_type,
            iteration_id,
            metric_name,
            start_evt,
            end_evt,
        ) in self.pending:
            if frame_id < self.warmup_frames:
                continue

            row_key = (frame_id, iteration_type, iteration_id)
            if row_key not in rows:
                rows[row_key] = {
                    'forward_cuda_ms': None,
                    'backward_cuda_ms': None,
                    'full_iteration_cuda_ms': None,
                }
                row_order.append(row_key)
            rows[row_key][metric_name] = float(
                start_evt.elapsed_time(end_evt)
            )
        self.pending.clear()

        if not rows:
            return

        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not self.header_written:
                writer.writerow(self.OUTPUT_COLUMNS)
                self.header_written = True

            for frame_id, iteration_type, iteration_id in row_order:
                row = rows[(frame_id, iteration_type, iteration_id)]
                writer.writerow([
                    frame_id,
                    iteration_type,
                    iteration_id,
                    self._format_timing(row['forward_cuda_ms']),
                    self._format_timing(row['backward_cuda_ms']),
                    self._format_timing(row['full_iteration_cuda_ms']),
                ])

    @staticmethod
    def _format_timing(value):
        return '' if value is None else f'{value:.6f}'

    def finish_frame(self, frame_id: int):
        return


class IterationProfileFrameTimer:
    PROFILE_STAGES = ('TR_ITER_PROFILE_TOTAL', 'BA_ITER_PROFILE_TOTAL')

    def __init__(
        self,
        enabled=False,
        output_csv='./profiling/iter_profile_frame_totals.csv',
        warmup_frames=0,
        write_header=False,
    ):
        self.enabled = bool(enabled)
        self.output_csv = Path(output_csv)
        self.warmup_frames = int(warmup_frames)
        self.write_header = bool(write_header)
        self.current_frame_id = -1
        self.current_outer_stage = ''
        self.pending = []
        self.frame_totals = {}
        self.written_frames = set()
        self.header_written = False

        if self.enabled:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            self.header_written = self.output_csv.exists() and self.output_csv.stat().st_size > 0

    def set_context(self, frame_id: int, outer_stage: str):
        self.current_frame_id = int(frame_id)
        self.current_outer_stage = str(outer_stage)

    @contextmanager
    def stage(self, name: str):
        if not self.enabled or name not in self.PROFILE_STAGES or not torch.cuda.is_available():
            yield
            return

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        try:
            yield
        finally:
            end_evt.record()
            self.pending.append((self.current_frame_id, name, start_evt, end_evt))

    def flush(self):
        if not self.enabled or not self.pending:
            return

        torch.cuda.synchronize()
        for frame_id, stage_name, start_evt, end_evt in self.pending:
            if frame_id < self.warmup_frames:
                continue
            totals = self.frame_totals.setdefault(
                frame_id,
                {stage: 0.0 for stage in self.PROFILE_STAGES},
            )
            totals[stage_name] += float(start_evt.elapsed_time(end_evt))
        self.pending.clear()

    def finish_frame(self, frame_id: int):
        if not self.enabled:
            return

        self.flush()
        frame_id = int(frame_id)
        if frame_id < self.warmup_frames or frame_id in self.written_frames:
            return

        totals = self.frame_totals.setdefault(
            frame_id,
            {stage: 0.0 for stage in self.PROFILE_STAGES},
        )
        tr_ms = totals['TR_ITER_PROFILE_TOTAL']
        ba_ms = totals['BA_ITER_PROFILE_TOTAL']
        total_ms = tr_ms + ba_ms

        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if self.write_header and not self.header_written:
                writer.writerow([
                    'frame_id',
                    'total_iter_profile_ms',
                    'tr_iter_profile_total_ms',
                    'ba_iter_profile_total_ms',
                ])
                self.header_written = True
            writer.writerow([
                frame_id,
                f'{total_ms:.6f}',
                f'{tr_ms:.6f}',
                f'{ba_ms:.6f}',
            ])

        self.written_frames.add(frame_id)
