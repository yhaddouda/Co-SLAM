#!/usr/bin/env python3
"""Package the concrete night agenda, preserving the runtime-v1 archive."""
import hashlib
import json
from pathlib import Path
import zipfile
from round2_paths import local_path


def main():
    root = Path(__file__).resolve().parents[1]
    entries = {}
    for name in ('RUN_AGENDA.md', 'ORIN_PAPER_NIGHT_PROMPT.md', 'NOTIFICATION_POLICY.md'):
        entries[name] = local_path(root,name).read_bytes()
    entries['README.md'] = b'''# Orin paper-night plan v2

Use this plan together with orin_coslam_runtime_v1.zip. This plan contains no
replacement Co runtime; do not omit runtime v1. It supersedes the earlier
small MAXN addendum and its stop-after-first-four-full-jobs interpretation.

Read ORIN_PAPER_NIGHT_PROMPT.md and execute its ordered preparation/launch
instructions. RUN_AGENDA.md distinguishes actual RTX runs, unknown local Orin
status, planned full blocks, and unprepared remaining paper work.

Co manifests are full-sequence, timing-only. O1 BASE/DP is identical to the
original runtime handoff; reuse matching completed jobs. New manifests add
MC/MC+DP, capacity19 and TUM blocks. Verify local data paths; change only paths
in separately named local manifests if necessary. Never enable quality/mesh
evaluation on Orin. Use MAXN only until the author manually switches to 15 W.

ec_public_reference/ is a hash-recorded pinned public source snapshot to help
prepare O2. It is NOT a prevalidated/instrumented EC execution runtime. The
Orin session must preserve native behavior, add declared controls/timers,
validate that wrapper and then run; do not count source export as EC readiness.
The engineering block likewise needs isolation validation before execution.

Prepare executable serial queues, not just a plan. Verify detached parent and
actual GPU child before telling the author they can close their client. Budget
whole paired blocks using measured pilot durations; report deferrals honestly.
Never switch modes, tune DP, invent public real-data EC configs, or report that
all MAXN paper work is complete while required EC/engineering work is pending.

File hashes are in SHA256SUMS.json. Return ORIN_RUN_AGENDA.md, live status and
ORIN_MAXN_RESULTS.zip; include a source-only validated EC runtime if prepared.
No routine task or document emails. Unattended email is not configured here.
'''
    for name in ('orin_coslam_first_full.json', 'orin_coslam_diagnostic.json',
                 'orin_office0_mc_additional.json', 'orin_office0_capacity19.json',
                 'orin_tum_native_four_conditions.json'):
        entries['manifests/' + name] = (root / 'experiments/manifests' / name).read_bytes()
    entries['watch_round2_queue.py'] = (root / 'experiments/watch_round2_queue.py').read_bytes()
    source = root / 'experiments/ec_public_reference_source_v1'
    native = json.loads((source / 'NATIVE_SOURCE_MANIFEST.json').read_text())
    for name, digest in native['file_hashes'].items():
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        entries['ec_public_reference/' + name] = data
    entries['ec_public_reference/NATIVE_SOURCE_MANIFEST.json'] = (source / 'NATIVE_SOURCE_MANIFEST.json').read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items())}
    entries['SHA256SUMS.json'] = (json.dumps(hashes, indent=2) + '\n').encode()
    archive = root / 'bundles/to_orin/orin_paper_night_v2.zip'
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as saved:
        for name, data in sorted(entries.items()):
            saved.writestr(name, data)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(check.read(name)).hexdigest() == digest
    print(json.dumps(dict(archive=str(archive), entries=len(entries), bytes=archive.stat().st_size,
                          sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
