import os
import torch
from pathlib import Path
from torch.utils.cpp_extension import load

# Ensure we build for the correct GPU architecture (Orin/Ampere = 8.7)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")

# Locate the CUDA source file relative to this python file
_src_path = Path(__file__).with_name("morton_cuda.cu")

# JIT Compile
_morton = load(
    name="fast_morton_ext",
    sources=[str(_src_path)],
    extra_cuda_cflags=["-O3"],
    verbose=False 
)

def morton2d_keys_cuda(uv01, frame_ids=None, R: int=4096):
    """
    Computes 64-bit keys: (Frame_ID << 32) | Morton2D(uv)
    """
    if frame_ids is None:
        # FIX: Create a zero-filled CUDA tensor of the correct size.
        # This satisfies the C++ signature (must be Tensor) and the device check (must be CUDA).
        # It costs negligible memory/time but guarantees stability.
        frame_ids = torch.zeros(uv01.shape[0], dtype=torch.int64, device=uv01.device)
    else:
        frame_ids = frame_ids.contiguous()
        
    return _morton.morton2d_keys(uv01.contiguous(), frame_ids, int(R))

def morton3d_keys_cuda(positions01, R: int):
    """Legacy 3D sorting support"""
    return _morton.morton3d_keys(positions01.contiguous(), int(R))