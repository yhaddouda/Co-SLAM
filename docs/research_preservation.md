# Research preservation

Keep all three stages of the same paper locally: `Neuslam paper/`,
`DASIP paper/` and `DASIP_round2/paper_round2/`. The latest revision has a
12-page paper and a separate rebuttal. Preserve each version's PDF together
with its LaTeX sources, bibliography, styles, assets and supporting results.
Historical PDFs are preserved as artifacts; equivalence to rebuilt source is
not assumed.

The revised paper's `paper_round2_placeholder/RESULTS_PROGRESS.json` lists
6,272 evidence inputs. All matched their recorded hashes on 16 November 2026.
The ledger and the manuscript's `review_material/` contain the table-import
and kernel-selection provenance. Preserve both. Later thesis experiments such
as `results/pruning_joint8/` are also retained locally.

Git stores implementation, tests, configs and reproduction scripts. The
`reproduction/sources/` snapshot makes paper runtime code available without
publishing manuscript text or result data. Restore it using the instructions
in `reproduction/README.md`. Keep new experiment outputs in separate roots.

Create an inventory and verified archives with standard Python:

```bash
python scripts/maintenance/preserve_research.py plan --out _preservation/NEW/inventory.json
python scripts/maintenance/preserve_research.py build --inventory _preservation/NEW/inventory.json --group code --out _preservation/NEW/code.tar.gz
python scripts/maintenance/preserve_research.py build --inventory _preservation/NEW/inventory.json --group evidence --out _preservation/NEW/evidence.tar.gz
python scripts/maintenance/preserve_research.py verify _preservation/NEW/evidence.tar.gz
```

Every archive contains a manifest, and has a sidecar checksum and verification
receipt. Existing archives are never overwritten. `review_large` in the
inventory means the original needs separate retention review; it never means
the file is safe to delete. Meshes, checkpoints and large arrays are excluded
from the compact archive unless explicitly referenced by the result ledger.
Keep their originals. The compact archives are not a complete dataset backup.

The legacy `output/**/depths_virt_cams_1000.npz` arrays are generated depth
renders. Their 637 culled reconstructed meshes remain in place. The matching
saved evaluation camera poses, GT meshes, unseen points and virtual cameras
were found under `../datasets/vv_data/` and preserved separately before cache
cleanup. `clean_render_caches.py` defaults to inspection and requires verified
local evidence and evaluation-asset archives before deletion. It writes a
receipt containing the exact removed paths and retained dependencies.

Raw public RGB-D datasets remain in the dataset location; do not delete those
as part of workspace cleanup. Dataset-specific corrections, fixed evaluation
poses, runtime hashes and config choices matter for reproduction. Do not
replace frozen paper runtimes with the current experimental implementation.
