"""Per-worker nested events; native threads/streams and handshake retained."""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from component_profile_timer import ComponentTimer


class WorkerProfiler:
    def __init__(self, output, selected, progress=None):
        self.output = Path(output)
        self.output.mkdir(exist_ok=True, parents=True)
        self.selected = selected
        self.local = threading.local()
        self.engines = {}
        self.rows = []
        self.enabled = True
        self.progress = progress
        self.lock = threading.Lock()

    def engine(self):
        return getattr(self.local, 'engine', None)

    def scope(self, kind, label=''):
        engine = self.engine()
        return engine.scope(kind, label) if engine else nullcontext()

    @contextmanager
    def iteration(self, index):
        engine = self.engine()
        if engine:
            engine.model._r2_iteration = int(index)
        with self.scope('iteration'):
            yield

    @contextmanager
    def query(self, points):
        engine = self.engine()
        if engine and engine.active:
            engine.queries.append(dict(frame_id=engine.frame, stage=engine.stage_name,
                                       iteration_id=engine.labels['iteration_id'], forward_id=engine.labels['forward_id'],
                                       module='decoder.embed_fn', points=int(points.numel()//points.shape[-1]),
                                       dtype=str(points.dtype), context=engine.labels['iteration_component']))
        with self.scope('forward_category', 'hash_grid'):
            yield

    @contextmanager
    def wait(self):
        start = time.perf_counter()
        with self.scope('call_phase', 'handshake_wait'):
            try:
                yield
            finally:
                self.local.wait_wall_ms += (time.perf_counter()-start)*1000

    @contextmanager
    def call(self, stage, frame):
        frame = int(frame)
        if self.progress and stage == 'tracking' and frame % 100 == 0:
            self.progress(frame)
        if stage not in self.engines:
            directory = self.output/stage
            directory.mkdir(exist_ok=True)
            engine = ComponentTimer(directory, self.selected)
            engine.model = SimpleNamespace(_r2_iteration=-1)
            with self.lock:
                self.engines[stage] = engine
        engine = self.engines[stage]
        assert self.engine() is None
        self.local.engine = engine
        self.local.wait_wall_ms = 0.
        engine.enabled = self.enabled
        wall = time.perf_counter()
        engine.begin(stage, frame)
        selected = engine.active
        try:
            yield
        finally:
            if selected:
                engine.finish()
                # Only resolve this worker's selected call endpoint. No boundary
                # or device-wide synchronization; native peers are not serialized.
                engine.last.synchronize()
                wall_ms = (time.perf_counter()-wall)*1000
                cuda_ms = engine.resolve()
                row = dict(frame_id=frame, stage=stage, wall_ms=wall_ms, cuda_ms=cuda_ms,
                           handshake_wait_wall_ms=self.local.wait_wall_ms,
                           active_body_wall_ms=wall_ms-self.local.wait_wall_ms,
                           scope='Selected worker body; raw envelope includes explicit handshake interval; event completion included in wall')
                engine.append_csv('calls.csv', [row])
                with self.lock:
                    self.rows.append(row)
            self.local.engine = None

    def summary(self):
        return dict(selected_calls={stage:sum(e.call_count.values()) for stage,e in self.engines.items()},
                    closure_max_ms=max((e.closure_max_ms for e in self.engines.values()), default=0.),
                    worker_timing='Native concurrent workers/streams retained. CUDA stream intervals may overlap/include peer contention or host idle. Never sum workers as scene time/FPS.',
                    call_timing='Selected worker-body host wall with separate actual handshake wait and derived active-body wall; native loader yield, publish/logger and outer handshakes excluded as documented.')
