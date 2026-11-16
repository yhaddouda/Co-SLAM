"""Nested CUDA-event partitions, resolved only after a complete selected call."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import csv
import json
from pathlib import Path
import time


class ComponentTimer:
    def __init__(self, output, selected, event_factory=None):
        self.output = Path(output)
        self.selected = {k: set(v) for k, v in selected.items()}
        self.event_factory = event_factory
        self.active = False
        self.enabled = True
        self.counter = 0
        self.model = None
        self.call_count = defaultdict(int)
        self.closure_max_ms = 0.

    def event(self):
        if self.event_factory is None:
            import torch
            value = torch.cuda.Event(enable_timing=True)
        else:
            value = self.event_factory()
        value.record()
        return value

    def transition(self, labels):
        event = self.event()
        self.segments.append((self.last, event, self.labels.copy(), tuple(p['id'] for p in self.stack), time.perf_counter()-self.last_host))
        self.last, self.last_host, self.labels = event, time.perf_counter(), labels.copy()

    def begin(self, stage, frame):
        assert not self.active
        self.active = self.enabled and frame in self.selected.get(stage, set())
        if not self.active:
            return
        self.stage_name, self.frame = stage, frame
        self.labels = dict(call_phase='call_setup', iteration_component='', forward_component='', iteration_id=-1, forward_id=-1)
        self.parents, self.segments, self.queries, self.stack = [], [], [], []
        self.last = self.event()
        self.last_host = time.perf_counter()
        self.new_parent('call', self.last)
        self.forward_count = 0

    def new_parent(self, level, start):
        self.counter += 1
        parent = dict(id=self.counter, parent_id=self.stack[-1]['id'] if self.stack else -1,
                      scope=level, frame_id=self.frame, stage=self.stage_name,
                      iteration_id=self.labels['iteration_id'], forward_id=self.labels['forward_id'], start=start)
        self.stack.append(parent)
        return parent

    @contextmanager
    def scope(self, kind, label=''):
        if not self.active:
            yield
            return
        levels = {p['scope'] for p in self.stack}
        if kind == 'forward_category' and 'forward' not in levels:
            yield
            return
        if kind == 'iteration_component' and 'iteration' not in levels:
            yield
            return
        saved = self.labels.copy()
        labels = saved.copy()
        parent = None
        if kind == 'call_phase':
            labels['call_phase'] = label
        elif kind == 'iteration_component':
            labels['iteration_component'] = label
        elif kind == 'forward_category':
            labels['forward_component'] = label
        elif kind == 'iteration':
            labels.update(call_phase='iterations', iteration_component='iteration_other',
                          iteration_id=int(self.model._r2_iteration))
        elif kind == 'forward_parent':
            labels.update(iteration_component='model_forward', forward_component='other_forward', forward_id=self.forward_count)
            self.forward_count += 1
        else:
            raise ValueError(kind)
        self.transition(labels)
        if kind in ('iteration', 'forward_parent'):
            parent = self.new_parent('iteration' if kind == 'iteration' else 'forward', self.last)
        try:
            yield
        finally:
            self.transition(saved)
            if parent is not None:
                assert self.stack.pop() is parent
                parent['end'] = self.last
                self.parents.append(parent)

    def stage(self, name):
        # The original coarse ranges remain in the source but add no events.
        if name in ('TR_ITER_PROFILE_TOTAL', 'BA_ITER_PROFILE_TOTAL'):
            return self.scope('iteration')
        return nullcontext()

    def finish(self):
        if not self.active:
            return
        assert len(self.stack) == 1 and self.stack[0]['scope'] == 'call'
        self.transition(self.labels)
        parent = self.stack.pop()
        parent['end'] = self.last
        self.parents.append(parent)

    def append_csv(self, name, rows):
        if not rows:
            return
        path = self.output / name
        exists = path.exists()
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def resolve(self):
        if not self.active:
            return None
        # Caller has synchronized the complete call before entering here.
        parts = defaultdict(lambda: defaultdict(float))
        segments = []
        parent_lookup = {p['id']: p for p in self.parents}
        keys = dict(call='call_phase', iteration='iteration_component', forward='forward_component')
        for start, end, labels, ancestors, host_seconds in self.segments:
            elapsed = float(start.elapsed_time(end))
            assert elapsed >= 0
            segments.append(dict(frame_id=self.frame, stage=self.stage_name,
                                 **labels, ancestor_ids=json.dumps(ancestors), cuda_ms=elapsed,
                                 host_submit_ms=host_seconds*1000))
            for parent_id in ancestors:
                parent = parent_lookup[parent_id]
                name = labels[keys[parent['scope']]]
                assert name, (parent, labels)
                parts[parent_id][name] += elapsed
        rows, components = [], []
        for parent in self.parents:
            elapsed = float(parent['start'].elapsed_time(parent['end']))
            total = sum(parts[parent['id']].values())
            error = total - elapsed
            self.closure_max_ms = max(self.closure_max_ms, abs(error))
            # Declared before collection; no rescaling/clipping of components.
            assert abs(error) <= .02 + .005*elapsed, (parent, elapsed, total)
            info = {k: parent[k] for k in ('id', 'parent_id', 'scope', 'frame_id', 'stage', 'iteration_id', 'forward_id')}
            rows.append(dict(**info, cuda_ms=elapsed, partition_sum_ms=total,
                             closure_error_ms=error, executed_components=json.dumps(sorted(parts[parent['id']]))))
            for name, value in parts[parent['id']].items():
                components.append(dict(**info, component=name, cuda_ms=value))
        self.append_csv('parents.csv', rows)
        self.append_csv('components.csv', components)
        self.append_csv('segments.csv', segments)
        self.append_csv('query_calls.csv', self.queries)
        self.call_count[self.stage_name] += 1
        self.active = False
        return rows[-1]['cuda_ms']

    def wrap_modules(self, model):
        self.model = model
        modules = [('embed_fn', 'hash_grid'), ('embed_fn_color', 'hash_grid'), ('decoder', 'decoder')]
        for name, category in modules:
            if not hasattr(model, name):
                continue
            module = getattr(model, name)
            native = module.forward
            def wrapped(*args, _native=native, _name=name, _category=category, **kwargs):
                if self.active and _category == 'hash_grid':
                    self.queries.append(dict(frame_id=self.frame, stage=self.stage_name,
                                             iteration_id=self.labels['iteration_id'], forward_id=self.labels['forward_id'],
                                             module=_name, points=int(args[0].numel()//args[0].shape[-1]),
                                             dtype=str(args[0].dtype), context=self.labels['iteration_component']))
                with self.scope('forward_category', _category):
                    return _native(*args, **kwargs)
            module.forward = wrapped
