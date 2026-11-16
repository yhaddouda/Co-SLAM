import random
import torch
import numpy as np
import random


def patch_indices_colmajor(H, W, top, left, ph, pw):
    h = np.arange(top, min(top + ph, H))
    print("h", h)
    w = np.arange(left, min(left + pw, W))
    print("w", w)
    hh, ww = np.meshgrid(h, w, indexing="ij")
    print("mesh grid", np.meshgrid(h, w, indexing="ij"))
    print("hh", hh)
    print("ww", ww)
    print("before", ww * H + hh)
    return (ww * H + hh).reshape(-1)

H, W = 680, 1200
ph, pw = 4, 4
res = patch_indices_colmajor(H,W, 4,4,ph, pw)
print("res", res)

