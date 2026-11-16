"""Refresh partial full-run reports as fresh benchmark summaries arrive."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    roots = [ROOT/'results/execution_experiments'/system/'full' for system in ('coslam','ec_slam')]
    signatures = {}
    while True:
        complete = (roots[-1]/'SUITE_COMPLETE').exists()
        for root in roots:
            signature = (tuple(sorted(str(p) for p in root.glob('*/*/summary.json'))),
                         (root/'SUITE_COMPLETE').exists())
            if not signature[0] or signature == signatures.get(root):
                continue
            command = [sys.executable,str(ROOT/'scripts/Validation/summarize_execution_experiments.py'),
                       '--root',str(root)]
            result = subprocess.run(command,check=False)
            if result.returncode == 0:
                signatures[root] = signature
        if complete:
            print('REPORT_WATCH_COMPLETE',flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
