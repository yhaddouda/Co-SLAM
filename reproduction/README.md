# Paper reproduction sources

`sources/` preserves exact copies of the scripts, runtime snapshots, CUDA
sources, configurations and experiment plans used across the Neuslam, DASIP
and revised DASIP versions of the same paper. `SOURCE_MANIFEST.json` records
their original repository paths and SHA-256 hashes. These copies must stay
frozen; use separate directories for new experiments.

Paper sources/PDFs, recorded measurements, trajectories, evaluation assets,
meshes and checkpoints remain local and are excluded from new Git commits.
The local `_preservation/` directory holds the archive inventories, checksums
and restoration notes. Git history predating this organization already contains
some result CSVs; removing them from the current tree does not rewrite history.

Restore the frozen source files to their original paths from the repository root:

```bash
python scripts/maintenance/restore_paper_sources.py
```

Matching files are retained. A different existing file stops restoration before
any file is copied. Restore into a different checkout for independent experiments:

```bash
python scripts/maintenance/restore_paper_sources.py --destination /path/to/checkout
```

Extract the local evidence archive into that checkout to restore measurements
and provenance at their original paths. Restore evaluation assets separately
into a dataset directory. Original manifests can contain machine-specific dataset
and output paths; set those for the destination before launching experiments.
Co-SLAM and EC-SLAM require their own native environments. Install CUDA
extensions for the destination hardware; compiled libraries are not portable.

Archive verification checks file integrity. It does not rerun experiments or
prove that a fresh environment reproduces every published value.
