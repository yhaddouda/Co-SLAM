# Transfer packaging

From the Co-SLAM root:

```bash
python scripts/Experiments/package_rtx_coslam.py
```

This builds `transfer/rtx4070/coslam_experiments.tar.gz`, its `.sha256` checksum,
and a manifest. An existing archive is preserved; pass `--output <new-path>` to
build a newer snapshot. The script reads the working tree, including untracked
experimental source files. It verifies every archived file against its manifest.

The package contains Co-SLAM runtime source, Replica configs, CUDA-extension
source, benchmark/check/report scripts, semantic tests, experiment documentation
and Orin reference results. It also contains the small optional isolated EC
source snapshot/patch and EC reference observations. EC execution still needs
its own native environment and correctly staged dataset on the destination.

Native compiled libraries, Arm environments, build caches, checkpoints,
datasets and large Nsight trace databases are excluded. The tiny-cuda-nn source
commit is recorded; verify that the RTX native installation uses the matching
source rather than copying Orin bindings. Local datasets are configured separately.

Extract to a staging directory on the RTX rather than over an existing checkout:

```bash
sha256sum -c coslam_experiments.tar.gz.sha256
mkdir -p /tmp/rtx_coslam_transfer
tar -xzf coslam_experiments.tar.gz -C /tmp/rtx_coslam_transfer
python /tmp/rtx_coslam_transfer/coslam_experiments/scripts/Experiments/package_rtx_coslam.py \
  --verify /tmp/rtx_coslam_transfer/coslam_experiments/RTX_TRANSFER_MANIFEST.json
```

Have the RTX Codex compare the staged source with its existing checkout,
preserve independent local edits, then apply the matching experiment files.
The manifest supplies exact relative paths, byte counts and hashes. The package
uses repository-relative paths under `coslam_experiments/`; it does not contain
an authentication directory or an environment archive.

Follow `docs/rtx4070_coslam_handoff.md` inside the extracted package for the
matched full tracking command and the optional earlier BA suite. New RTX results
must use separate roots. A short smoke run prepares imports/extensions before
the measured full suite.
