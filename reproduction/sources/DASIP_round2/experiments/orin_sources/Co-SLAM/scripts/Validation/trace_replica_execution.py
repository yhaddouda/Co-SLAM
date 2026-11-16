"""Add complete-call NVTX scopes for separate, profiled diagnostic runs.

These scopes include an additional end fence for reliable GPU attribution.
Trace-run timings/ATE are diagnostic artifacts, never headline comparisons.
"""
import argparse
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import nvtx


def scoped(method, label):
    @functools.wraps(method)
    def call(*inputs, **kwargs):
        with nvtx.annotate(label):
            result = method(*inputs, **kwargs)
            torch.cuda.synchronize()
            return result
    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=('coslam','ec_slam'), required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.system == 'coslam':
        import coslam
        coslam.CoSLAM.tracking_render = scoped(coslam.CoSLAM.tracking_render, 'ExperimentTrackingCall')
        from scripts.Validation.benchmark_replica_tracking import main as benchmark
    else:
        workspace = ROOT / 'experiments/ec_slam_execution'
        os.chdir(workspace)
        sys.path.insert(0, str(workspace))
        from src.Tracker import Tracker
        from src.Mapper import Mapper
        Tracker.optimize_cam_in_batch = scoped(Tracker.optimize_cam_in_batch, 'ExperimentTrackingCall')
        Mapper.optimize_map = scoped(Mapper.optimize_map, 'ExperimentMappingCall')
        from scripts.Validation.benchmark_ec_replica_execution import main as benchmark
    benchmark()


if __name__ == '__main__':
    main()
