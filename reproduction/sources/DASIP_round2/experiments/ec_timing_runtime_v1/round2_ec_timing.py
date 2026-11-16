"""Thread-local EC call intervals, explicitly excluding native handshake waits.

Native workers/streams remain in use. Overlapping worker intervals are not
additive device utilization or scene FPS. CSV resolution occurs outside calls.
"""
import csv
from contextlib import contextmanager
import threading
import time
import torch


class Timer:
    def __init__(self):
        self.local=threading.local();self.lock=threading.Lock();self.calls=[];self.iterations=[]

    def start_segment(self):
        event=torch.cuda.Event(enable_timing=True);event.record();event.synchronize()
        self.local.segment=(event,time.perf_counter())

    def end_segment(self):
        start,wall=self.local.segment
        end=torch.cuda.Event(enable_timing=True);end.record();end.synchronize()
        self.local.wall_ms+=(time.perf_counter()-wall)*1000
        self.local.cuda_ms+=start.elapsed_time(end)
        self.local.segments+=1

    @contextmanager
    def call(self,stage,frame):
        self.local.stage=stage;self.local.frame=int(frame);self.local.wall_ms=0.;self.local.cuda_ms=0.;self.local.wait_ms=0.;self.local.segments=0;self.local.pending=[]
        self.start_segment()
        try:yield
        finally:
            self.end_segment()
            row=dict(frame_id=int(frame),stage=stage,wall_ms=self.local.wall_ms,cuda_ms=self.local.cuda_ms,handshake_wait_wall_ms=self.local.wait_ms,segments=self.local.segments)
            iterations=[]
            for index,full_start,full_end,parts in self.local.pending:
                assert set(parts)=={'forward_cuda_ms','backward_cuda_ms'}
                entry=dict(frame_id=int(frame),iteration_type=stage,iteration_id=index)
                for key,(a,b) in parts.items():entry[key]=a.elapsed_time(b)
                entry['full_iteration_cuda_ms']=full_start.elapsed_time(full_end);iterations.append(entry)
            with self.lock:self.calls.append(row);self.iterations.extend(iterations)
            self.local.stage=None

    @contextmanager
    def wait(self):
        self.end_segment();start=time.perf_counter()
        try:yield
        finally:
            self.local.wait_ms+=(time.perf_counter()-start)*1000;self.start_segment()

    @contextmanager
    def iteration(self,index):
        self.local.parts={};start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        try:yield
        finally:
            end.record();self.local.pending.append((int(index),start,end,self.local.parts))

    @contextmanager
    def component(self,key):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        try:yield
        finally:end.record();self.local.parts[key]=(start,end)

    def write(self,output):
        for name,data,columns in [('calls.csv',self.calls,['frame_id','stage','wall_ms','cuda_ms','handshake_wait_wall_ms','segments']),('iterations.csv',self.iterations,['frame_id','iteration_type','iteration_id','forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms'])]:
            with (output/name).open('w',newline='') as file:
                writer=csv.DictWriter(file,columns);writer.writeheader();writer.writerows(sorted(data,key=lambda r:(r['frame_id'],r.get('stage',r.get('iteration_type')),r.get('iteration_id',0))))
        result={}
        for stage in ('tracking','bundle_adjustment'):
            rows=[r for r in self.calls if r['stage']==stage and r['frame_id']>=20]
            allrows=[r for r in self.calls if r['stage']==stage]
            result[stage]=dict(count=len(rows),mean_wall_ms=sum(r['wall_ms'] for r in rows)/len(rows),mean_cuda_ms=sum(r['cuda_ms'] for r in rows)/len(rows),all_call_mean_wall_ms=sum(r['wall_ms'] for r in allrows)/len(allrows),total_excluded_handshake_wait_wall_ms=sum(r['handshake_wait_wall_ms'] for r in allrows))
        return result
