# Buffered interval-support pruning

## Method and scientific rationale

The method combines **interval-based support selection** with **dense anchor updates**. It starts from the existing mixed-ray, depth-guided lattice and tail-pruning strategies. It changes which RGB/SDF network points are evaluated, without reducing rays, optimizer iterations, the hash representation or the geometry-loss reference measure.

For ray r, let d_r be measured depth, τ the scaled geometry truncation distance, and [l_ri,u_ri] the original stratified jitter interval for sample i. At an intermediate update, retain i when l_ri ≤ d_r + τ + b, where b=0.15 in the tested scene units. Also retain the first two samples and every sample on missing-depth rays. Every scene here has scale factor 1, so the buffer is 15 cm. This is a common buffer across the three scenes, not a scene-specific tuning parameter. A sample beyond the cutoff is omitted only if its entire jitter interval lies behind the buffered supervised surface. Retained samples use their original jitter; no new points are inserted or moved.

Tracking uses the same interval identities throughout a frame’s pose optimization. This avoids iteration-to-iteration changes of the discrete subset, while sample jitter, pose-dependent coordinates, network outputs and gradients are freshly computed. BA retains mixed keyframe/current-frame rays and recomputes support for each newly drawn batch. The first and last optimization updates, and updates whose zero-based index is divisible by 3, use the complete original lattice. For the tested schedule this gives 4 dense and 6 pruned updates in a ten-update call, and 8 dense and 12 pruned updates in TUM’s twenty-update BA call. The first evaluates full point support for its sampled ray batch at the incoming estimate; the last reintroduces all point support before the final update. These are full point lattices for the sampled rays, not evaluations of every image pixel. These anchors do not guarantee convergence or remove all rendering error. The original best-pose selection rule is unchanged: on Replica and Synthetic it can select an intermediate pose scored with the pruned renderer. TUM uses its configured final-iterate pose rule.

### Connection to the original degradation

The earlier bucket design altered three scientific properties together: it preferred shallow tracking rays, separated BA rays into depth-homogeneous updates, and changed geometry supervision by rebuilding shorter sampling intervals and their normalization. This method keeps the original random tracking rays and mixed BA batches, and prunes only within each ray’s existing lattice. Its geometry denominator and balancing weights stay unchanged. Shared pose-storage and missing-depth RGB-weight corrections from the earlier investigation are present in both dense and pruned controls, so they cannot explain this study’s between-condition gains.

### What is preserved and what is approximated

All omitted points satisfy z_ri > d_r + τ for every possible original jitter value. Their free-space and SDF-band masks are therefore zero. Evaluating the original geometry losses on the original lattice, with unchanged denominators and global mask-balancing factors, preserves those losses and gradients for a fixed field and pose. No inverse probability correction is required because no geometry-supervised sample is removed. The original full jitter draws are consumed, preserving the paired random sequence.

The normalized RGB/depth renderer is approximated during intermediate updates. A field may have a late or incorrect crossing beyond the measured depth; removed samples can then have nonzero rendering weight. There is no certified opacity/error bound and no claim that all losses remain identical. The buffer and dense anchors are empirical compensation for this approximation. Training trajectories can differ even though each fixed-field geometry calculation is preserved.

### Why it can be faster

Each pruned update makes one packed network evaluation. There is no second field-dependent recovery evaluation, feature detector or extra neural inference. The full sample coordinates, original geometry masks and rendering tensor are still constructed. Index selection, gather/scatter, CUDA synchronization, losses, backward and optimizer work are all charged to complete stage wall time. The method saves expensive encoding/decoder evaluations rather than assuming that fewer theoretical samples imply lower latency.

A matched full-sample control uses this identical support implementation and anchor schedule with b=1,000,000, retaining every point. It retains the same selection handling and packing path. Comparison to this control checks that the benefit requires actual point omission. Native dense controls additionally measure the cost of introducing the pruning machinery. No claim of an isolated percentage attribution to individual kernels is made: the measured gains are complete-call differences.

## Relationship to prior work

Surface-guided sampling is established: [NICE-SLAM (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_NICE-SLAM_Neural_Implicit_Scalable_Encoding_for_SLAM_CVPR_2022_paper.pdf) combines stratified and depth-near samples. [EC-SLAM](https://arxiv.org/html/2404.13346v2) motivates economical depth-guided sampling and feature/grid ray selection. [Hua and Wang (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Hua_Benchmarking_Implicit_Neural_Representation_and_Geometric_Rendering_in_Real-Time_RGB-D_CVPR_2024_paper.html) study geometric rendering and sampling choices in RGB-D SLAM.

This implementation does not reproduce EC-SLAM’s sampling implementation: it keeps the original Co-SLAM lattice and random ray selection, prunes complete existing jitter intervals behind the buffered supervised surface, fixes interval identities during tracking and interleaves dense anchors. It does not adopt EC-SLAM’s resampling intervals or feature/grid pixel mixture. The scientific proposition is this combination and its measured quality/cost tradeoff; these experiments alone do not establish publication novelty. [VolSDF](https://lioryariv.github.io/volsdf/) uses an opacity-error-bound formulation; no such bound is claimed here.

## Fixed sampling and optimization settings

| Scene | Original uniform + depth-near samples per ray | Depth-near half-window | Geometry truncation | Tracking rays × updates | BA updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| TUM fr1_desk | 64 + 21 | 0.25 m | 0.05 m | 1024 × 10 | 20 |
| Replica office0 | 32 + 11 | 0.10 m | 0.10 m | 1024 × 10 | 10 |
| Synthetic thin_geometry | 32 + 11 | 0.10 m | 0.10 m | 1024 × 10 | 10 |

Each BA update draws the original 2048 global keyframe rays plus max(2048 / number of keyframes, 100) current-frame rays, with integer division as in the original implementation. Both references and pruning use the same draws, original learning rates, CoherentPrime hash size 19 and no importance sampling. The configured tracking patience is 100, exceeding the ten updates; there is no early-stop reduction explaining the savings. Initialization remains dense. Complete inherited scene configs are archived per run.

## Reproducibility and limits

Activate `conda activate coslam`. A fresh complete study, retaining output only after all gates and matched full-sample controls pass:

```bash
python scripts/Validation/run_accepted_pruning_study.py \
  --output results/pruning_reproduction --seeds 0 1 2 \
  --pruning-margin 0.15 --anchor-every 3
```

This command alternates dense/candidate order by seed. The reported development study reused dense controls collected in the same resumed study; candidate runs were sequential, without overlapping GPU benchmarks. Rejected candidate outputs were discarded as requested and are not presented as accepted versions. These are three development scenes used for selecting the method, not held-out evidence across their complete datasets. Three seeds describe observed variation and are not a confidence guarantee. CUDA hash-grid training uses atomic updates, so identical seeds need not reproduce trajectories bit for bit. The acceptance threshold is the ratio of each scene’s arithmetic mean ATEs, not the arithmetic mean of per-seed percentages.

For a mesh run preserving the paper benchmark’s frame/phase seed protocol, use the benchmark and append `--export-mesh`, for example:

```bash
python scripts/Validation/benchmark_ba_accuracy.py \
  --dataset Replica --scene office0 --seed 0 --frames 0 \
  --tracking baseline --variant baseline --pruning support \
  --pruning-margin 0.15 --anchor-every 3 \
  --output output/pruning_support_mesh/Replica_office0_seed0 --export-mesh
```

Change dataset/scene to `Tum/fr1_desk` or `Synthetic/tg`, and run the corresponding dense control with `--pruning dense`. Mesh extraction is excluded from the recorded SLAM timings. Mesh accuracy/completion and ScanNet are untested in this study. Normal opt-in scene presets use the same sampling parameters, but ordinary runs do not automatically acquire the benchmark’s paired frame/phase seed protocol.
