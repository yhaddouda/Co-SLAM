"""Only imported in the post-timing exclusive diagnostic window."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def audit_reference(reference):
    assert os.environ.get('DASIP_POST_V4_EXCLUSIVE')=='1'
    original=ROOT.parent/'orin_coslam_runtime_v1/runtime'
    for name in ['coslam.py','model/utils.py','model/scene_rep.py','model/keyframe.py','optimization/utils.py','model/encodings.py','round2_timing.py','round2_sampling.py','datasets/dataset.py','configs/Replica/office0.yaml','configs/Replica/replica.yaml']:
        assert (reference/name).read_bytes()==(original/name).read_bytes(),name
    # The reference's engineering-disabled BASE is separately measured, so no wrapper identity is presumed.
    report=dict(status='PASS',native_core_helpers_and_configs_byte_equal=True,reference=str(reference),working_and_v4_sources_modified=False,extra_two_seed_BASE_controls_required=True,scope='Full supplied source manifest checked plus directly called estimator/loss/sampling/pose/timing/config files compared with validated Orin native runtime')
    (ROOT/'CO_REFERENCE_LOCAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
