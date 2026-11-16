## Method: buffered ray support plus sampled feature smoothness

This method extends the previously accepted interval-support strategy. Tracking and BA keep the original rays, keyframe mixing, iteration counts, pose updates, representation, learning rates and loss weights. Initialization is dense. No execution-speed variant, preload, pose cache, Morton sort, feature detector or additional neural predictor is enabled.

**Ray samples.** At intermediate updates, retain an existing jitter interval when its lower endpoint is at most measured depth + geometry truncation + 0.15. Keep the first two samples and all samples for missing-depth rays. Retained points keep their original jitter, and the original full-lattice random draws are consumed. Tracking fixes the interval identities within a frame while recomputing coordinates and neural outputs at each pose. BA keeps its original mixed keyframe/current-frame batches. The first and last update and indices divisible by 3 use the complete ray lattice. Both stages use this schedule; these anchors concern ray samples, not feature regularization.

All omitted ray samples have zero original free-space/SDF supervision. The original geometry-loss balancing factors and denominators are retained, preserving fixed-field geometry losses and gradients. Intermediate RGB/depth rendering remains an approximation: late or incorrect field crossings can have nonzero weight. Dense anchors and the support buffer compensate empirically; there is no certified rendering-error bound. The original best-pose selection rule remains in effect.

**Feature samples.** Original Co-SLAM also queries a dense, jittered feature lattice on every BA update, then penalizes squared differences between neighboring feature vectors. Despite the code name `smoothness`/TV, this is a quadratic finite-difference energy, not an L1 TV penalty. Earlier ray pruning did not reduce this work. The new stage estimates this same energy from uniformly sampled directed neighbor edges, stratified equally across the three axes. It does not change the regularizer weight or its spatial measure.

Let the original `smooth_pts` value be s, n=s−1, and E=(n−1)n² edges per axis. With a common endpoint-query fraction ρ=0.125, draw K=max(1, floor(ρn³/6)) edges independently with replacement for each axis. Let f be the encoded feature vector, v the original voxel spacing, o the original random offset, and j the original shared grid jitter. For a sampled integer edge start a and axis unit vector e, use the original points x=(a+j)v+o+bounding_box_min and x+ve. Query both endpoints. The estimator is

    L_hat = E / (K s³) × Σ_axis Σ_k ||f(x_k + v e_axis) − f(x_k)||².

The original dense loss has exactly this s³ denominator, even though its lattice has n³ vertices. The estimator uses 6K feature queries. Conditional on a fixed field, offset and jitter, its loss and parameter gradients are unbiased for the native dense energy and gradients. This is stratified Monte Carlo sampling with replacement (population/count weighting). It is not a unique-edge inclusion-probability estimator. Duplicate edges and endpoints are permitted and counted as actual queries. Unbiased gradients do not imply identical Adam updates, optimizer trajectories, meshes or ATE.

Original CPU Torch offset/jitter draws remain unchanged. A private NumPy generator selects edges using run seed, frame and iteration; it does not consume the ray sampler's random stream. Features, coordinates and gradients are freshly evaluated on every BA update, including ray-anchor updates. Setting ρ=1 restores the native dense regularizer. Initialization and all non-BA regularization remain native dense. Selection, CPU/GPU transfer, query, weighting, backward and optimizer costs are included in complete BA wall time.

| Scene | s | Dense feature queries / BA update | K per axis | Sampled endpoint queries / BA update | Feature-query reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| TUM fr1_desk | 64 | 250,047 | 5,209 | 31,254 | 87.5007% |
| Replica office0 | 32 | 29,791 | 620 | 3,720 | 87.5130% |
| Synthetic thin_geometry | 32 | 29,791 | 620 | 3,720 | 87.5130% |

Feature-only encoding queries cost differently from joint RGB/SDF encoding/decoder queries. Their counts are reported separately; adding them without a cost model would be misleading. Complete measured BA wall time is the performance criterion. The larger native TUM regularization lattice explains why this pruning can matter much more there. Paired ablations measure aggregate changes, not an isolated kernel-time decomposition.

## Scientific context and limits

[Co-SLAM (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Co-SLAM_Joint_Coordinate_and_Sparse_Parametric_Encodings_for_Neural_Real-Time_CVPR_2023_paper.pdf) uses feature-grid smoothness to regularize unobserved space and mitigate hash-grid artifacts. [Ehret, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Ehret_A_Generic_and_Flexible_Regularization_Framework_for_NeRFs_WACV_2024_paper.pdf) discusses finite-difference and sampled regularization for neural fields. Sampling regularization and Monte Carlo expectation compensation are established ideas; this study does not claim to invent them.

[EC-SLAM](https://arxiv.org/html/2404.13346v2) motivates economical geometry-guided sampling. This implementation does not copy its feature/grid pixel-selection mixture or reconstruct its ray-sampling intervals. It keeps the Co-SLAM ray lattice and random rays, prunes buffered whole intervals with dense anchors, and adds an unbiased edge estimator for the original feature-grid energy. This combination is a scientific sampling proposition with a measured quality/cost tradeoff. Experiments on three development scenes do not establish publication novelty, generalization, statistical significance or mesh quality.

## Reproduction and mesh follow-up

Activate `conda activate coslam`. Use a fresh output directory:

```bash
python scripts/Validation/run_ba30_pruning_study.py \
  --output results/pruning_ba30_reproduction --seeds 0 1 2 \
  --tv-fraction 0.125 --pruning-margin 0.15 --anchor-every 3
```

Optional `--prior-root results/pruning_accepted` additionally enforces the ATE and BA gates against the earlier published dense reference. The primary comparison always uses fresh paired controls. The runner retains results only after all scene-mean criteria pass; it then includes the matched no-prune and interval-only controls at seed 0. Rejected trials stay in automatically removed scratch storage. `--plot-python` can select an interpreter with Matplotlib.

For a later mesh comparison, preserve the validated seed protocol and append `--export-mesh`:

```bash
python scripts/Validation/benchmark_ba_accuracy.py \
  --dataset Replica --scene office0 --seed 0 --frames 0 \
  --tracking baseline --variant baseline --pruning support_tv \
  --tv-fraction 0.125 --anchor-every 3 --tracking-anchor-every 3 \
  --pruning-margin 0.15 --output output/pruning_ba30_mesh/Replica_office0_seed0 \
  --export-mesh
```

Compare with `--pruning dense` at the same seed; substitute `Tum/fr1_desk` or `Synthetic/tg` for the other scenes. Mesh export occurs after SLAM and is excluded from stage timing. The opt-in presets under `configs/PruningBA30` use the tested sampling parameters. Ordinary preset runs do not automatically use the benchmark's paired frame/phase seed protocol. ScanNet and meshes have not been evaluated in this study.
