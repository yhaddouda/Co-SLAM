# Additional Orin sources

These exact Orin source/configuration files supplement the fixed RTX backup
`1e3b30c0de49ead6f603d2fb4fb04651737b0691`. That reference contains the active
Orin Co-SLAM implementation and the principal native EC-SLAM source. The new
files preserve additional Orin orchestration, recovery, timing tests, plotting,
profiling utilities, and native dependency specifications.

`SOURCE_MANIFEST.json` maps original paths in three separate repositories to
either a new `stored_path` or an existing `shared_path` in that fixed commit.
Identical additions are stored once. Existing RTX root files, tests and frozen
sources remain unchanged. In particular, the Orin BA/tracking tests exercise the
Orin implementation; they are separate exact versions, not replacements for the
later RTX tests. Historical controller revisions remain distinguishable.

Restore only the runtime/source subset needed for a study into a fresh directory.
This example previews restoration of the additional plotting scripts:

```bash
python reproduction/orin/restore_sources.py --root Co-SLAM \
  --prefix scripts/plot/ --shared-repo . --destination /path/to/fresh-orin
```

Add `--apply` to write missing files after all selected checksums and existing
files pass inspection. Conflicting files cause an error. The helper does not
launch experiments. For later experiments, use a separate development branch and
fresh output roots; keep the original run/runtime paths frozen.

Run observations, manuscripts, datasets, archives, installed environment dumps,
notification agents, meshes and checkpoints remain local. The source overlay
does not reconstruct those assets. Original machine paths and controller power
policies are preserved inside source: inspect them before any launch. Preserve
MAXN and 15 W run identity; do not substitute timing values between modes.

Co-SLAM and EC-SLAM require their own native environments. Rebuild CUDA
extensions for the target Jetson/CUDA/PyTorch combination and retain encoder
precision and timing boundaries. CPU unit checks passed on the preserved Orin
implementation; GPU equivalence, throughput and paper results were not rerun.
