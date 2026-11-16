#!/usr/bin/env python3
"""Run the preserved MAXN auditor with relocation-aware artifact lookup."""
import hashlib
import json
from pathlib import Path

import import_orin_maxn_results as original
from round2_paths import local_path


def main():
    native_sha = original.sha
    original.sha = lambda path: native_sha(local_path(original.BASE,path))
    original.main()
    status = original.OUT/'ANALYSIS_STATUS.json'
    data = json.loads(status.read_text())
    for path in (Path(__file__).resolve(), Path(__file__).with_name('round2_paths.py'),
                 original.BASE/'metadata/FILE_LOCATIONS.json'):
        data['input_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    status.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
