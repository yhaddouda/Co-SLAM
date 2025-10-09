# tests/test_patch_sampling.py
# Patch/stripe sampling validations with real Co-SLAM shapes & signatures.
# - select_samples(H,W,samples) -> Long[samples]  (COL-MAJOR)
# - KeyframeDB.sample_single_keyframe_rays([1,H*W,7]) -> ([1,40800,7], Long[40800]) (ROW-MAJOR)
# - KeyframeDB.sample_global_rays(bs) -> (Float[bs,7], Long[bs]) using DB.rays[K,40800,7]
#
# Horizontal/vertical stripes are just ph=1 or pw=1 (same sampler).
# Headless PNG saving (Agg backend) + loud progress prints.

import os, sys, math, random, traceback
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG — edit here
# =========================
SEED = 42

# Real Co-SLAM frame size (Replica): 680 x 1200
H, W = 680, 1200
TOTAL_PX = H * W  # 816000

# Tracking/Mapping batch size
TRACK_SAMPLES = 2048

# Keyframe saving (~5%): 40800 for 680x1200
KF_SAVE = int(0.05 * TOTAL_PX)  # 40800

# Number of toy keyframes for BA (keep small for speed)
NUM_KF = 12

# BA batch size
BA_SAMPLES = 2048

# Patch shapes to test (unified; stripes => ph=1 or pw=1)
TRACK_PATCH = dict(ph=4, pw=4, sh=4, sw=4, cap_per_patch=None)   # square segments
TRACK_HSEG  = dict(ph=1,  pw=8, sh=4,  sw=8, cap_per_patch=None)   # many short horizontal segments
TRACK_VSEG  = dict(ph=8, pw=1,  sh=8, sw=4,  cap_per_patch=None)   # many short vertical segments

# KF save patch params (same interface)
KF_PATCH = dict(ph=4, pw=4, sh=4, sw=4, alpha_patch=0.9)  # change to ph=1,pw=16 to test horiz stripes

OUTDIR = os.path.abspath("./patch_out_4")


# =========================
# Helpers: index conventions
# =========================
def col_major_flat(h, w, H):
    return w * H + h  # h = idx % H ; w = idx // H

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

def grid_jitter_toplefts(H, W, ph, pw, sh=None, sw=None, seed=None):
    if seed is not None: random.seed(seed)
    if sh is None: sh = ph
    if sw is None: sw = pw
    coords = []
    vmax = max(0, H - ph); umax = max(0, W - pw)
    for v0 in range(0, vmax + 1, max(1, sh)):
        for u0 in range(0, umax + 1, max(1, sw)):
            # Bounded jitter so neighbors do not overlap if sh>=ph and sw>=pw
            dv_max = max(0, min(sh - ph, vmax - v0))
            du_max = max(0, min(sw - pw, umax - u0))
            dv = random.randint(0, dv_max) if dv_max > 0 else 0
            du = random.randint(0, du_max) if du_max > 0 else 0
            coords.append((v0 + dv, u0 + du))
    random.shuffle(coords)
    return coords


# =========================
# 1) Tracking/Mapping sampler — SIGNATURE MATCH
# =========================
def select_samples(H, W, samples, *, ph, pw, sh=None, sw=None,
                   cap_per_patch=None, seed=None) -> torch.LongTensor:
    """
    Returns exactly `samples` COL-MAJOR flat indices as LongTensor[samples].
    Unified patch interface:
      - square: ph=K, pw=K
      - horiz : ph=1, pw=K
      - vert  : ph=K, pw=1
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for (top, left) in grid_jitter_toplefts(H, W, ph, pw, sh, sw, seed):
        p = patch_indices_colmajor(H, W, top, left, ph, pw)
        if cap_per_patch is not None and len(p) > int(cap_per_patch):
            rng.shuffle(p); p = p[:int(cap_per_patch)]
        chunks.append(p)
        if sum(len(c) for c in chunks) >= samples:
            break
    if not chunks:
        idx = rng.choice(H * W, size=samples, replace=False)
        return torch.from_numpy(idx.astype(np.int64))

    idx = np.unique(np.concatenate(chunks))
    rng.shuffle(idx)
    if len(idx) < samples:
        pool = np.setdiff1d(np.arange(H * W), idx)
        extra = rng.choice(pool, size=samples - len(idx), replace=False)
        idx = np.concatenate([idx, extra]); rng.shuffle(idx)
    return torch.from_numpy(idx[:samples].astype(np.int64))


# =========================
# 2) Keyframe DB + saver — SIGNATURE MATCH
# =========================
@dataclass
class KFSaveCfg:
    ph: int; pw: int
    sh: int | None = None; sw: int | None = None
    alpha_patch: float = 0.7
    seed: int | None = None

class KeyframeDB:
    def __init__(self, H, W, num_kf: int, rays_per_kf: int, device="cpu"):
        self.H, self.W = H, W
        self.K = num_kf
        self.N = rays_per_kf
        self.device = device
        self.rays = torch.zeros((num_kf, rays_per_kf, 7), dtype=torch.float32, device=device)
        self.rays_idxs = torch.full((num_kf, rays_per_kf), -1, dtype=torch.long, device=device)  # row-major
        self.patch_buckets: List[List[torch.LongTensor]] = [ [] for _ in range(num_kf) ]
        self.frame_ids = torch.arange(num_kf, dtype=torch.long, device=device)  # 0..K-1
        self._next_slot = 0

    def sample_single_keyframe_rays(self, rays_1HW7: torch.Tensor, cfg: KFSaveCfg):
        """
        Input:  rays_1HW7 [1, H*W, 7]  (ROW-MAJOR flatten of [1,H,W,7])
        Output: (rays_sel [1,N,7], idxs [N] row-major)
        """
        assert rays_1HW7.ndim == 3 and rays_1HW7.shape[0] == 1
        H, W, N = self.H, self.W, self.N
        rng = np.random.default_rng(cfg.seed)
        Np = int(round(cfg.alpha_patch * N)); Nr = N - Np

        idx_list = []
        for (top, left) in grid_jitter_toplefts(H, W, cfg.ph, cfg.pw, cfg.sh, cfg.sw, cfg.seed):
            p = patch_indices_rowmajor(H, W, top, left, cfg.ph, cfg.pw)
            idx_list.append(p)
            if sum(len(x) for x in idx_list) >= Np:
                break

        if not idx_list:
            idx = rng.choice(H*W, size=N, replace=False).astype(np.int64)
            return rays_1HW7[:, torch.from_numpy(idx)], torch.from_numpy(idx)

        idxp = np.unique(np.concatenate(idx_list))
        rng.shuffle(idxp); idxp = idxp[:Np].astype(np.int64)
        pool = np.setdiff1d(np.arange(H*W), idxp)
        extr = rng.choice(pool, size=Nr, replace=False).astype(np.int64)
        idx = np.concatenate([idxp, extr]); rng.shuffle(idx)
        idx = idx[:N].astype(np.int64)
        return rays_1HW7[:, torch.from_numpy(idx)], torch.from_numpy(idx)

    def add_keyframe(self, rays_1HW7: torch.Tensor, cfg: KFSaveCfg):
        assert 0 <= self._next_slot < self.K, "Too many KFs for the test DB"
        slot = self._next_slot
        self._next_slot += 1

        rays_sel, idxs = self.sample_single_keyframe_rays(rays_1HW7, cfg)
        self.rays[slot] = rays_sel[0]
        self.rays_idxs[slot] = idxs

        # Build patch buckets (local indices inside saved rays) using same grid
        pos_map = {int(v): i for i, v in enumerate(self.rays_idxs[slot].tolist())}
        buckets = []
        for (top, left) in grid_jitter_toplefts(self.H, self.W, cfg.ph, cfg.pw, cfg.sh, cfg.sw, cfg.seed):
            p = patch_indices_rowmajor(self.H, self.W, top, left, cfg.ph, cfg.pw)
            local = [pos_map[v] for v in p if v in pos_map]
            if len(local) > 0:
                buckets.append(torch.tensor(local, dtype=torch.long, device=self.device))
        if not buckets:
            buckets = [torch.arange(self.N, dtype=torch.long, device=self.device)]
        self.patch_buckets[slot] = buckets

    def sample_global_rays(self, bs: int):
        """
        Patch-aware BA sampler (round-robin buckets, dedup).
        Returns:
          rays[bs,7], frame_ids[bs]
        """
        selected_pairs = []  # list of (kf, local_idx)
        seen = set()
        rng = np.random.default_rng(SEED)

        # Shuffle patch order inside each KF
        shuffled = [list(range(len(self.patch_buckets[k]))) for k in range(self._next_slot)]
        for k in range(self._next_slot):
            rng.shuffle(shuffled[k])

        ptr = [0]*self._next_slot
        while len(selected_pairs) < bs:
            progressed = False
            for k in range(self._next_slot):
                if len(selected_pairs) >= bs: break
                if ptr[k] >= len(shuffled[k]): continue
                p = self.patch_buckets[k][shuffled[k][ptr[k]]]
                ptr[k] += 1
                for li in p.tolist():
                    key = (k, int(li))
                    if key not in seen:
                        seen.add(key); selected_pairs.append(key)
                        if len(selected_pairs) >= bs: break
                progressed = True
            if not progressed:
                break

        # Top up uniquely if still short
        if len(selected_pairs) < bs:
            for k in range(self._next_slot):
                for li in range(self.N):
                    key = (k, li)
                    if key not in seen:
                        selected_pairs.append(key); seen.add(key)
                        if len(selected_pairs) >= bs: break
                if len(selected_pairs) >= bs: break

        rays_out = torch.stack([self.rays[k, li] for (k, li) in selected_pairs[:bs]], dim=0)
        ids_out  = torch.tensor([k for (k, _) in selected_pairs[:bs]], dtype=torch.long)
        return rays_out, ids_out


# =========================
# Visualization helpers
# =========================
def save_pixel_map_col(H, W, idxs_col, title, path):
    img = np.zeros((H, W), dtype=np.float32)
    for i in idxs_col:
        h = int(i % H); w = int(i // H)
        img[h, w] = 1.0
    plt.figure(figsize=(10,5))
    plt.imshow(img, interpolation="nearest")
    plt.title(title); plt.xlabel("w (x)"); plt.ylabel("h (y)")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

def save_pixel_map_row(H, W, idxs_row, title, path):
    img = np.zeros((H, W), dtype=np.float32)
    for i in idxs_row:
        h = int(i // W); w = int(i % W)
        img[h, w] = 1.0
    plt.figure(figsize=(10,5))
    plt.imshow(img, interpolation="nearest")
    plt.title(title); plt.xlabel("w (x)"); plt.ylabel("h (y)")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

def save_ba_hist(ids, path, title="BA per-KF counts"):
    counts = np.bincount(ids.numpy(), minlength=int(ids.max().item()+1))
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(counts)), counts)
    plt.title(title); plt.xlabel("KF id"); plt.ylabel("#rays in BA")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# =========================
# Main
# =========================
def main():
    print("[test] starting")
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[test] writing PNGs to: {OUTDIR}")
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    # ---------- Tracking / Mapping (COL-MAJOR) ----------
    print("[test] sampling tracking/mapping…")
    idx_rand  = torch.from_numpy(np.random.choice(H*W, size=TRACK_SAMPLES, replace=False).astype(np.int64))
    idx_patch = select_samples(H, W, TRACK_SAMPLES, seed=SEED, **TRACK_PATCH)
    idx_hseg  = select_samples(H, W, TRACK_SAMPLES, seed=SEED, **TRACK_HSEG)
    idx_vseg  = select_samples(H, W, TRACK_SAMPLES, seed=SEED, **TRACK_VSEG)

    # round-trip checks (col-major)
    for i in idx_patch[:10]:
        h = int(i.item() % H); w = int(i.item() // H)
        assert col_major_flat(h,w,H) == int(i)

    save_pixel_map_col(H,W, idx_rand,  "Tracking RANDOM (col)", os.path.join(OUTDIR,"track_random.png"))
    save_pixel_map_col(H,W, idx_patch, f"Tracking PATCH {TRACK_PATCH['ph']}x{TRACK_PATCH['pw']} (col)", os.path.join(OUTDIR,"track_patch.png"))
    save_pixel_map_col(H,W, idx_hseg,  f"Tracking H-SEG ph={TRACK_HSEG['ph']} pw={TRACK_HSEG['pw']} (col)", os.path.join(OUTDIR,"track_hseg.png"))
    save_pixel_map_col(H,W, idx_vseg,  f"Tracking V-SEG ph={TRACK_VSEG['ph']} pw={TRACK_VSEG['pw']} (col)", os.path.join(OUTDIR,"track_vseg.png"))
    print("[test] tracking/mapping PNGs written")

    # ---------- Build toy full rays ----------
    print("[test] building synthetic full rays [1,H*W,7]…")
    rays_full = torch.rand(1, TOTAL_PX, 7)

    # ---------- Keyframe DB ----------
    print(f"[test] building DB with {NUM_KF} keyframes (each saves {KF_SAVE})…")
    db = KeyframeDB(H, W, NUM_KF, KF_SAVE, device="cpu")
    for k in range(NUM_KF):
        if k % 3 == 0:
            cfg = KFSaveCfg(**KF_PATCH, seed=SEED+k)
        elif k % 3 == 1:
            cfg = KFSaveCfg(ph=1, pw=16, sh=4, sw=16, alpha_patch=KF_PATCH['alpha_patch'], seed=SEED+k)
        else:
            cfg = KFSaveCfg(ph=16, pw=1, sh=16, sw=4, alpha_patch=KF_PATCH['alpha_patch'], seed=SEED+k)
        db.add_keyframe(rays_full, cfg)
    print("[test] DB ready")

    # ---------- KF saving viz (ROW-MAJOR) ----------
    print("[test] visualizing KF save (random vs patch)…")
    idxs_rand_kf = torch.from_numpy(np.random.default_rng(SEED).choice(H*W, size=KF_SAVE, replace=False).astype(np.int64))
    rays_patch, idxs_patch = db.sample_single_keyframe_rays(rays_full, KFSaveCfg(**KF_PATCH, seed=SEED))
    # round-trip checks (row-major)
    for i in idxs_patch[:10]:
        h = int(i.item() // W); w = int(i.item() % W)
        assert row_major_flat(h,w,W) == int(i)

    save_pixel_map_row(H,W, idxs_rand_kf, "KF RANDOM (row)", os.path.join(OUTDIR,"kf_random.png"))
    save_pixel_map_row(H,W, idxs_patch,   f"KF PATCH ph={KF_PATCH['ph']} pw={KF_PATCH['pw']} (row)", os.path.join(OUTDIR,"kf_patch.png"))
    print("[test] KF PNGs written")

    # ---------- Global BA (ROW-MAJOR) ----------
    print("[test] sampling BA batch…")
    rays_ba, ids_ba = db.sample_global_rays(BA_SAMPLES)
    assert rays_ba.shape == (BA_SAMPLES, 7)
    assert ids_ba.shape  == (BA_SAMPLES,)
    save_ba_hist(ids_ba, os.path.join(OUTDIR,"ba_hist.png"))

    # Spatial preview for one KF (first few buckets)
    k_star = int(torch.bincount(ids_ba, minlength=db._next_slot).argmax().item())
    preview = []
    for b in db.patch_buckets[k_star][:8]:
        preview.extend(b.tolist())
    preview = preview[:min(2000, len(preview))]
    chosen_global = db.rays_idxs[k_star, torch.tensor(preview, dtype=torch.long)].numpy()
    save_pixel_map_row(H,W, chosen_global,
                       f"BA sample preview — KF {k_star} first buckets (row)",
                       os.path.join(OUTDIR,"ba_kf_spatial.png"))
    print("[test] BA PNGs written")

    print("\n[test] DONE. Files written to:")
    for fn in ["track_random.png","track_patch.png","track_hseg.png","track_vseg.png",
               "kf_random.png","kf_patch.png","ba_hist.png","ba_kf_spatial.png"]:
        print("  ", os.path.join(OUTDIR, fn))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] Exception while running test_patch_sampling.py")
        traceback.print_exc()
        sys.exit(1)

