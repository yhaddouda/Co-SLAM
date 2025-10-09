# tests/test_patch_sampling.py
# Unified patch sampler validation for Co-SLAM (headless/SSH-safe).
# - Tracking/Mapping: select_samples_colmajor_patch (COL-MAJOR indices)
# - Keyframe save   : sample_kf_rowmajor_patch (ROW-MAJOR indices)
# - Global BA       : sample_global_rays_like (ROW-MAJOR, dedup)
# Patches, horizontal/vertical "stripes" all use the same function:
#   square: ph=K, pw=K
#   horiz : ph=1, pw=K
#   vert  : ph=K, pw=1

import math, random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless for SSH
import matplotlib.pyplot as plt

# =========================
# CONFIG — edit here
# =========================
SEED = 42

# Toy image size
H, W = 120, 160

# Tracking/Mapping batch size
N_TRACK = 1024

# KF save fraction (~5% like Co-SLAM)
KF_SAVE_FRAC = 0.05

# Tracking/Mapping shapes to visualize
# square 16x16
TRACK_SQ = dict(ph=4, pw=4, sh=16, sw=16, cap_per_patch=None)
# horizontal "segments": thickness 1, length 16 (many short lines)
TRACK_H  = dict(ph=1,  pw=16, sh=4,  sw=16, cap_per_patch=None)
# vertical "segments": thickness 1, length 16
TRACK_V  = dict(ph=16, pw=1,  sh=16, sw=4,  cap_per_patch=None)

# KF saved patch shape (change to ph=1,pw=16 for horizontal, etc.)
KF_PATCH = dict(ph=4, pw=4, sh=16, sw=16, alpha_patch=0.9)

# BA batch size
N_BA = 2048

# =========================
# Index helpers (Co-SLAM conventions)
# =========================
def col_major_flat(h, w, H):
    return w * H + h  # h = idx % H, w = idx // H

def row_major_flat(h, w, W):
    return h * W + w  # idx = h*W + w

def patch_indices_colmajor(H, W, top, left, ph, pw):
    h = np.arange(top, min(top + ph, H))
    w = np.arange(left, min(left + pw, W))
    hh, ww = np.meshgrid(h, w, indexing="ij")
    return (ww * H + hh).reshape(-1)

def patch_indices_rowmajor(H, W, top, left, ph, pw):
    h = np.arange(top, min(top + ph, H))
    w = np.arange(left, min(left + pw, W))
    hh, ww = np.meshgrid(h, w, indexing="ij")
    return (hh * W + ww).reshape(-1)

# =========================
# Stratified grid + jitter (non-overlap by default if sh>=ph and sw>=pw)
# =========================
def grid_jitter_toplefts(H, W, ph, pw, sh=None, sw=None, seed=None):
    """
    Return top-lefts for ph×pw patches on a stride grid (sh,sw) with bounded jitter
    so neighboring cells don't overlap. If sh/sw is None, default to ph/pw (tiling).
    """
    if seed is not None:
        random.seed(seed)
    if sh is None: sh = ph
    if sw is None: sw = pw
    coords = []
    vmax = max(0, H - ph); umax = max(0, W - pw)
    for v0 in range(0, vmax + 1, max(1, sh)):
        for u0 in range(0, umax + 1, max(1, sw)):
            dv_max = max(0, min(sh - ph, vmax - v0))  # jitter bounded so patches don't overlap
            du_max = max(0, min(sw - pw, umax - u0))
            dv = random.randint(0, dv_max) if dv_max > 0 else 0
            du = random.randint(0, du_max) if du_max > 0 else 0
            coords.append((v0 + dv, u0 + du))
    random.shuffle(coords)
    return coords

# =========================
# 1) Tracking/Mapping sampler (COL-MAJOR)
# =========================
def select_samples_colmajor_patch(H, W, samples, *, ph, pw, sh=None, sw=None,
                                  cap_per_patch=None, seed=None):
    """
    Return exactly `samples` unique flat COL-MAJOR indices.
    Unified patch interface:
      - square: ph=K, pw=K
      - horiz : ph=1, pw=K
      - vert  : ph=K, pw=1
    Stride:
      - default (None): sh=ph, sw=pw (tiling)
      - sh/sw < size  : overlap     ; sh/sw > size: gaps
    cap_per_patch: maximum samples to draw from any single patch.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for (top, left) in grid_jitter_toplefts(H, W, ph, pw, sh, sw, seed):
        p = patch_indices_colmajor(H, W, top, left, ph, pw)
        if cap_per_patch is not None and len(p) > cap_per_patch:
            rng.shuffle(p); p = p[:cap_per_patch]
        chunks.append(p)
        if sum(len(c) for c in chunks) >= samples:
            break

    idx = np.unique(np.concatenate(chunks)) if chunks else np.array([], dtype=np.int64)
    rng.shuffle(idx)
    if len(idx) < samples:
        pool = np.setdiff1d(np.arange(H * W), idx)
        extra = rng.choice(pool, size=samples - len(idx), replace=False)
        idx = np.concatenate([idx, extra]); rng.shuffle(idx)
    return idx[:samples].astype(np.int64)

# =========================
# 2) Keyframe saver (ROW-MAJOR)
# =========================
@dataclass
class KFConfig:
    H: int; W: int; num_rays_to_save: int
    ph: int; pw: int
    sh: int | None = None; sw: int | None = None
    alpha_patch: float = 0.7  # fraction from patches (rest random)

@dataclass
class KFSampleResult:
    rays: np.ndarray        # [1, N, 7]
    idxs: np.ndarray        # [N] row-major indices
    patches: List[Tuple[int,int,int,int]]  # (top,left,ph,pw)

def sample_kf_rowmajor_patch(rays_1HW7: np.ndarray, cfg: KFConfig, seed=None) -> KFSampleResult:
    """rays_1HW7 was row-major flattened from [1,H,W,7]."""
    H, W, N = cfg.H, cfg.W, cfg.num_rays_to_save
    rng = np.random.default_rng(seed)
    flat = rays_1HW7.reshape(-1, rays_1HW7.shape[-1])  # [H*W,7]

    Np = int(round(cfg.alpha_patch * N)); Nr = N - Np
    idx_list, patches_used = [], []
    for (top, left) in grid_jitter_toplefts(H, W, cfg.ph, cfg.pw, cfg.sh, cfg.sw, seed):
        p = patch_indices_rowmajor(H, W, top, left, cfg.ph, cfg.pw)
        idx_list.append(p); patches_used.append((top,left,cfg.ph,cfg.pw))
        if sum(len(x) for x in idx_list) >= Np:
            break

    idxp = np.unique(np.concatenate(idx_list)) if idx_list else np.array([], dtype=np.int64)
    rng.shuffle(idxp); idxp = idxp[:Np]
    pool = np.setdiff1d(np.arange(H * W), idxp)
    extr = rng.choice(pool, size=Nr, replace=False)
    idx = np.concatenate([idxp, extr]); rng.shuffle(idx)
    return KFSampleResult(flat[idx][None, ...], idx.astype(np.int64), patches_used)

# =========================
# 3) BA over toy KFs (ROW-MAJOR, dedup)
# =========================
@dataclass
class ToyKF:
    H: int; W: int
    rays: np.ndarray
    idxs: np.ndarray
    patch_buckets: List[np.ndarray]  # each array is a set of saved row-major idx within one saved patch

def build_toy_kf(H, W, Nsave, *, ph, pw, sh=None, sw=None, alpha=0.7, seed=None) -> ToyKF:
    # Build a synthetic 1 x H*W x 7 tensor (row-major flatten)
    rays_full = np.zeros((1, H*W, 7), dtype=np.float32)
    for v in range(H):
        for u in range(W):
            idx = row_major_flat(v, u, W)
            rays_full[0, idx, 0:3] = [u/max(1,W-1), v/max(1,H-1), 1.0]  # dummy direction
            rays_full[0, idx, 3:6] = [u/W, v/H, 0.5]                    # dummy rgb
            rays_full[0, idx, 6]   = math.hypot(u, v)                   # dummy depth

    res = sample_kf_rowmajor_patch(
        rays_full,
        KFConfig(H=H, W=W, num_rays_to_save=Nsave, ph=ph, pw=pw, sh=sh, sw=sw, alpha_patch=alpha),
        seed=seed
    )

    # Build patch buckets from recorded patches; keep only indices that were actually saved
    buckets = []
    for (top,left,ph0,pw0) in res.patches:
        pidx = patch_indices_rowmajor(H, W, top, left, ph0, pw0)
        kept = pidx[np.isin(pidx, res.idxs)]
        if len(kept) > 0:
            buckets.append(kept)

    return ToyKF(H=H, W=W, rays=res.rays.reshape(-1, 7), idxs=res.idxs, patch_buckets=buckets)

def sample_global_rays_like(kfs: List[ToyKF], batch_size=2048, seed=None):
    rng = np.random.default_rng(seed)
    selected = []
    seen = set()
    # shuffle patch order inside each KF
    for kf in kfs:
        rng.shuffle(kf.patch_buckets)

    # round-robin patches, dedup on the fly
    ptr = [0] * len(kfs)
    while len(selected) < batch_size:
        progressed = False
        for ki, kf in enumerate(kfs):
            if len(selected) >= batch_size: break
            if ptr[ki] < len(kf.patch_buckets):
                p = kf.patch_buckets[ptr[ki]]; ptr[ki] += 1
                for idx in p:
                    ii = int(idx)
                    if ii not in seen:
                        selected.append(ii); seen.add(ii)
                        if len(selected) >= batch_size: break
                progressed = True
        if not progressed:
            break

    # top up uniquely from remaining saved rays if needed
    if len(selected) < batch_size:
        remaining = np.unique(np.concatenate([kf.idxs for kf in kfs], axis=0))
        pool = remaining[~np.isin(remaining, np.array(list(seen), dtype=np.int64))]
        need = batch_size - len(selected)
        if need > 0 and len(pool) >= need:
            extra = rng.choice(pool, size=need, replace=False)
            selected.extend(map(int, extra))
            seen.update(map(int, extra))

    out = np.array(selected, dtype=np.int64)
    assert len(np.unique(out)) == len(out), "BA sampler produced duplicates"
    return out

# =========================
# Visualization & metrics
# =========================
def show_pixel_map(H, W, idxs, *, convention: str, title: str, outpath: str):
    img = np.zeros((H, W), dtype=np.float32)
    if convention == "row":
        for i in idxs:
            h = i // W; w = i % W; img[h, w] = 1.0
    elif convention == "col":
        for i in idxs:
            h = i % H; w = i // H; img[h, w] = 1.0
    else:
        raise ValueError("convention must be 'row' or 'col'")
    plt.figure()
    plt.imshow(img, interpolation="nearest")
    plt.title(title); plt.xlabel("w (x)"); plt.ylabel("h (y)")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def tile_coverage(H, W, idxs, th, tw, *, convention: str):
    tileH = math.ceil(H / th); tileW = math.ceil(W / tw)
    counts = np.zeros((tileH, tileW), dtype=int)
    if convention == "row":
        hw = [(i // W, i % W) for i in idxs]
    else:
        hw = [(i % H, i // H) for i in idxs]
    for h, w in hw:
        ti = min(h // th, tileH - 1); tj = min(w // tw, tileW - 1)
        counts[ti, tj] += 1
    return counts, {"min": int(counts.min()), "med": float(np.median(counts)), "max": int(counts.max())}

# =========================
# Main validations
# =========================
def main():
    rng = np.random.default_rng(SEED)

    # ---------- Tracking/Mapping (COL-MAJOR) ----------
    idx_rand = rng.choice(H*W, size=N_TRACK, replace=False).astype(np.int64)

    idx_sq = select_samples_colmajor_patch(H, W, N_TRACK, seed=SEED, **TRACK_SQ)
    idx_h  = select_samples_colmajor_patch(H, W, N_TRACK, seed=SEED, **TRACK_H)
    idx_v  = select_samples_colmajor_patch(H, W, N_TRACK, seed=SEED, **TRACK_V)

    # round-trip checks (col-major)
    for i in idx_sq[:10]:
        h = i % H; w = i // H; assert col_major_flat(h,w,H) == i

    show_pixel_map(H, W, idx_rand, convention="col",
                   title="Tracking RANDOM (col)", outpath="plot_track_random.png")
    show_pixel_map(H, W, idx_sq,   convention="col",
                   title=f"Tracking PATCH {TRACK_SQ['ph']}x{TRACK_SQ['pw']} (col)",
                   outpath="plot_track_patch.png")
    show_pixel_map(H, W, idx_h,    convention="col",
                   title=f"Tracking H-SEG ph={TRACK_H['ph']} pw={TRACK_H['pw']} (col)",
                   outpath="plot_track_hseg.png")
    show_pixel_map(H, W, idx_v,    convention="col",
                   title=f"Tracking V-SEG ph={TRACK_V['ph']} pw={TRACK_V['pw']} (col)",
                   outpath="plot_track_vseg.png")

    cov_r, stats_r = tile_coverage(H, W, idx_rand, 16, 16, convention="col")
    cov_p, stats_p = tile_coverage(H, W, idx_sq,   16, 16, convention="col")
    print("Tile coverage (16x16) — RANDOM (col):", stats_r)
    print("Tile coverage (16x16) — PATCH  (col):", stats_p)

    # ---------- Keyframe saving (ROW-MAJOR) ----------
    total_px = H * W
    Nsave = int(KF_SAVE_FRAC * total_px)
    rays_full = np.random.rand(1, total_px, 7).astype(np.float32)

    # random save
    idxs_rand_kf = rng.choice(H*W, size=Nsave, replace=False).astype(np.int64)
    res_rand = dict(rays=rays_full.reshape(-1,7)[idxs_rand_kf][None,...], idxs=idxs_rand_kf)

    # patch save
    res_patch = sample_kf_rowmajor_patch(
        rays_full,
        KFConfig(H=H, W=W, num_rays_to_save=Nsave, **KF_PATCH),
        seed=SEED
    )

    # round-trip checks (row-major)
    for i in res_patch.idxs[:10]:
        h = i // W; w = i % W; assert row_major_flat(h,w,W) == i

    show_pixel_map(H, W, res_rand["idxs"], convention="row",
                   title="KF RANDOM (row)", outpath="plot_kf_random.png")
    show_pixel_map(H, W, res_patch.idxs,   convention="row",
                   title=f"KF PATCH ph={KF_PATCH['ph']} pw={KF_PATCH['pw']} (row)",
                   outpath="plot_kf_patch.png")

    # ---------- Global BA (ROW-MAJOR, patch-aware) ----------
    # Build 3 toy keyframes with different shapes to visualize mixing
    kf1 = build_toy_kf(H, W, Nsave, ph=4, pw=4, sh=16, sw=16, alpha=0.9, seed=1)
    kf2 = build_toy_kf(H, W, Nsave, ph=1,  pw=4, sh=4,  sw=16, alpha=0.7, seed=2)  # many horizontal segments
    kf3 = build_toy_kf(H, W, Nsave, ph=4, pw=1,  sh=16, sw=4,  alpha=0.7, seed=3)  # many vertical segments

    idx_ba = sample_global_rays_like([kf1,kf2,kf3], batch_size=N_BA, seed=123)
    assert len(np.unique(idx_ba)) == len(idx_ba) == N_BA

    show_pixel_map(H, W, idx_ba, convention="row",
                   title="BA sample (row)", outpath="plot_ba.png")

    print("All tests passed. PNGs written:")
    print("  plot_track_random.png, plot_track_patch.png, plot_track_hseg.png, plot_track_vseg.png")
    print("  plot_kf_random.png, plot_kf_patch.png, plot_ba.png")

if __name__ == "__main__":
    main()

