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
        pass


class FrameTotalCudaTimer:
    """Sum complete tracking and BA iteration CUDA ranges per frame."""

    COLUMNS = (
        'frame_id',
        'total_cuda_ms',
        'tracking_cuda_ms',
        'bundle_adjustment_cuda_ms',
    )
    FULL_STAGES = {
        'Tracking': ('tracking', 'TR_ITER_PROFILE_TOTAL'),
        'Bundle_Adjustment': (
            'bundle_adjustment',
            'BA_ITER_PROFILE_TOTAL',
        ),
    }

    def __init__(
        self,
        enabled=False,
        output_csv='./profiling/frame_total.csv',
        warmup_frames=0,
        write_header=False,
    ):
        self.enabled = bool(enabled)
        self.output_csv = Path(output_csv)
        if self.enabled:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.warmup_frames = int(warmup_frames)
        self.write_header = bool(write_header)
        self.current_frame_id = -1
        self.current_outer_stage = ''
        self.pending_by_frame = {}
        self.finished_frames = set()

    def set_context(self, frame_id: int, outer_stage: str):
        self.current_frame_id = int(frame_id)
        self.current_outer_stage = str(outer_stage)

    @contextmanager
    def stage(self, name: str):
        stage_definition = self.FULL_STAGES.get(self.current_outer_stage)
        if (
            not self.enabled
            or not torch.cuda.is_available()
            or self.current_frame_id < self.warmup_frames
            or stage_definition is None
            or name != stage_definition[1]
        ):
            yield
            return

        frame_id = self.current_frame_id
        if frame_id in self.finished_frames:
            raise RuntimeError(
                f'frame-total timing received work for finished frame {frame_id}'
            )

        iteration_type = stage_definition[0]
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        try:
            yield
        except BaseException:
            raise
        else:
            end_evt.record()
            frame_events = self.pending_by_frame.setdefault(
                frame_id,
                {'tracking': [], 'bundle_adjustment': []},
            )
            frame_events[iteration_type].append((start_evt, end_evt))

    def flush(self):
        # Tracking flushes before BA for mapping frames. Defer resolving and
        # writing until CoSLAM explicitly marks the whole frame complete.
        pass

    def finish_frame(self, frame_id: int):
        frame_id = int(frame_id)
        if (
            not self.enabled
            or not torch.cuda.is_available()
            or frame_id < self.warmup_frames
        ):
            return
        if frame_id in self.finished_frames:
            raise RuntimeError(
                f'frame-total timing finished frame {frame_id} more than once'
            )

        unfinished_earlier = [
            pending_frame
            for pending_frame in self.pending_by_frame
            if pending_frame < frame_id
        ]
        if unfinished_earlier:
            raise RuntimeError(
                'frame-total timing has unfinished earlier frame(s): '
                + ', '.join(str(value) for value in sorted(unfinished_earlier))
            )

        events = self.pending_by_frame.pop(
            frame_id,
            {'tracking': [], 'bundle_adjustment': []},
        )
        if events['tracking'] or events['bundle_adjustment']:
            torch.cuda.synchronize()

        tracking_ms = sum(
            float(start.elapsed_time(end))
            for start, end in events['tracking']
        )
        bundle_adjustment_ms = sum(
            float(start.elapsed_time(end))
            for start, end in events['bundle_adjustment']
        )
        total_ms = tracking_ms + bundle_adjustment_ms

        write_header = (
            self.write_header
            and (
                not self.output_csv.exists()
                or self.output_csv.stat().st_size == 0
            )
        )
        with open(self.output_csv, 'a', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(self.COLUMNS)
            writer.writerow(
                (
                    frame_id,
                    f'{total_ms:.6f}',
                    f'{tracking_ms:.6f}',
                    f'{bundle_adjustment_ms:.6f}',
                )
            )
        self.finished_frames.add(frame_id)


class IterationBreakdownCudaTimer:
    """Time only full, forward, and backward spans of tracking and BA."""

    COLUMNS = (
        'frame_id',
        'iteration_type',
        'iteration_id',
        'forward_cuda_ms',
        'backward_cuda_ms',
        'full_iteration_cuda_ms',
    )
    STAGE_DEFINITIONS = {
        'Tracking': {
            'iteration_type': 'tracking',
            'forward': 'forward_tr',
            'backward': 'backward_tr',
            'full': 'TR_ITER_PROFILE_TOTAL',
        },
        'Bundle_Adjustment': {
            'iteration_type': 'bundle_adjustment',
            'forward': 'Forward_BA',
            'backward': 'Backward_BA',
            'full': 'BA_ITER_PROFILE_TOTAL',
        },
    }

    def __init__(
        self,
        enabled=False,
        output_csv='./profiling/iteration_breakdown.csv',
        warmup_frames=0,
    ):
        self.enabled = bool(enabled)
        self.output_csv = Path(output_csv)
        if self.enabled:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.warmup_frames = int(warmup_frames)
        self.current_frame_id = -1
        self.current_outer_stage = ''
        self.active_iterations = {}
        self.next_iteration_ids = {}
        self.pending = []

    def set_context(self, frame_id: int, outer_stage: str):
        self.current_frame_id = int(frame_id)
        self.current_outer_stage = str(outer_stage)

    @contextmanager
    def stage(self, name: str):
        definition = self.STAGE_DEFINITIONS.get(self.current_outer_stage)
        if (
            not self.enabled
            or not torch.cuda.is_available()
            or self.current_frame_id < self.warmup_frames
            or definition is None
        ):
            yield
            return

        stage_kind = next(
            (
                kind
                for kind in ('forward', 'backward', 'full')
                if name == definition[kind]
            ),
            None,
        )
        if stage_kind is None:
            yield
            return

        frame_id = self.current_frame_id
        iteration_type = definition['iteration_type']
        sequence_key = (frame_id, iteration_type)

        if stage_kind == 'full':
            if sequence_key in self.active_iterations:
                raise RuntimeError(
                    'iteration timing started a new full iteration before '
                    f'finishing the previous one for frame {frame_id} '
                    f'({iteration_type})'
                )

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            state = {'full': (start_evt, end_evt)}
            self.active_iterations[sequence_key] = state
            start_evt.record()
            try:
                yield
            except BaseException:
                self.active_iterations.pop(sequence_key, None)
                raise
            else:
                end_evt.record()
                self.active_iterations.pop(sequence_key, None)

                if 'forward' not in state or 'backward' not in state:
                    # Tracking may stop after checking the loss but before the
                    # backward pass. It is not a complete optimisation
                    # iteration and must not be logged as one.
                    if (
                        iteration_type == 'tracking'
                        and 'forward' in state
                        and 'backward' not in state
                    ):
                        return
                    raise RuntimeError(
                        'iteration timing did not observe both forward and '
                        f'backward for frame {frame_id} ({iteration_type})'
                    )

                iteration_id = self.next_iteration_ids.get(sequence_key, 0)
                self.next_iteration_ids[sequence_key] = iteration_id + 1
                self.pending.append(
                    (
                        frame_id,
                        iteration_type,
                        iteration_id,
                        state['forward'],
                        state['backward'],
                        state['full'],
                    )
                )
            return

        state = self.active_iterations.get(sequence_key)
        if state is None:
            raise RuntimeError(
                f'iteration timing observed {stage_kind} outside a full '
                f'iteration for frame {frame_id} ({iteration_type})'
            )
        if stage_kind in state:
            raise RuntimeError(
                f'iteration timing observed duplicate {stage_kind} for '
                f'frame {frame_id} ({iteration_type})'
            )

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        try:
            yield
        except BaseException:
            raise
        else:
            end_evt.record()
            state[stage_kind] = (start_evt, end_evt)

    def flush(self):
        if not self.enabled or not self.pending:
            return
        if self.active_iterations:
            raise RuntimeError(
                'cannot flush iteration timing during an active iteration'
            )

        torch.cuda.synchronize()
        rows = []
        for (
            frame_id,
            iteration_type,
            iteration_id,
            forward_events,
            backward_events,
            full_events,
        ) in self.pending:
            forward_ms = float(forward_events[0].elapsed_time(forward_events[1]))
            backward_ms = float(
                backward_events[0].elapsed_time(backward_events[1])
            )
            full_ms = float(full_events[0].elapsed_time(full_events[1]))
            if full_ms < forward_ms or full_ms < backward_ms:
                raise RuntimeError(
                    'full CUDA iteration timing is shorter than a component '
                    f'for frame {frame_id} ({iteration_type}, '
                    f'iteration {iteration_id})'
                )
            rows.append(
                (
                    frame_id,
                    iteration_type,
                    iteration_id,
                    f'{forward_ms:.6f}',
                    f'{backward_ms:.6f}',
                    f'{full_ms:.6f}',
                )
            )

        write_header = (
            not self.output_csv.exists()
            or self.output_csv.stat().st_size == 0
        )
        with open(self.output_csv, 'a', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(self.COLUMNS)
            writer.writerows(rows)
        self.pending.clear()

    def finish_frame(self, frame_id: int):
        pass
