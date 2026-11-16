#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

// --- Original 3D Morton Utils (Keep for legacy support if needed) ---
__device__ __forceinline__ uint32_t part1by2(uint32_t v){
    v &= 0x3ffu;                              // 10 bits
    v = (v | (v << 16)) & 0x030000ffu;
    v = (v | (v << 8))  & 0x0300f00fu;
    v = (v | (v << 4))  & 0x030c30c3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}

__global__ void morton_kernel(const float* __restrict__ pos,
                              uint32_t* __restrict__ out,
                              int N, int Rm1){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float fx = fminf(fmaxf(pos[3*i+0], 0.f), 1.f) * Rm1;
    float fy = fminf(fmaxf(pos[3*i+1], 0.f), 1.f) * Rm1;
    float fz = fminf(fmaxf(pos[3*i+2], 0.f), 1.f) * Rm1;

    uint32_t x = (uint32_t)fx;
    uint32_t y = (uint32_t)fy;
    uint32_t z = (uint32_t)fz;

    uint32_t key = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
    out[i] = key;
}

torch::Tensor morton3d_keys(torch::Tensor pos, int R){
    TORCH_CHECK(pos.is_cuda(), "pos must be CUDA");
    TORCH_CHECK(pos.is_contiguous(), "pos must be contiguous");
    const int N = (int)pos.size(0);
    auto keys = torch::empty({N}, pos.options().dtype(torch::kInt32));
    const int block = 256;
    const int grid  = (N + block - 1) / block;
    morton_kernel<<<grid, block>>>(pos.data_ptr<float>(), reinterpret_cast<uint32_t*>(keys.data_ptr<int32_t>()), N, R - 1);
    return keys;
}

// --- NEW: 2D Morton Utils for fast_morton ---

// Interleaves 16 bits of x and y to produce a 32-bit Morton code
// Expands 16 bits: xxxx... -> x0x0x0...
__device__ __forceinline__ uint32_t part1by1(uint32_t v){
    v &= 0x0000ffff;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

__global__ void morton2d_kernel(const float* __restrict__ uv,
                                const int64_t* __restrict__ frame_ids,
                                int64_t* __restrict__ out,
                                int N, int Rm1){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 1. Normalize and Scale Coordinates
    // Expects UV in range [0, 1]. Clamping for safety.
    float fu = fminf(fmaxf(uv[2*i+0], 0.f), 1.f) * Rm1;
    float fv = fminf(fmaxf(uv[2*i+1], 0.f), 1.f) * Rm1;

    uint32_t x = (uint32_t)fu;
    uint32_t y = (uint32_t)fv;

    // 2. Compute 32-bit Spatial Key
    // Interleave bits of X and Y
    uint32_t m_code = part1by1(x) | (part1by1(y) << 1);

    // 3. Handle Frame ID (High 32 bits)
    // If frame_ids is NULL, we assume single-frame batch (ID=0)
    int64_t fid = (frame_ids != nullptr) ? frame_ids[i] : 0;
    
    // 4. Pack into 64-bit Integer
    // Structure: [ Frame_ID (32 bits) | Morton_Code (32 bits) ]
    // This ensures sorting by Frame first, then spatially within the frame.
    // Casting m_code to int64 avoids overflow issues during bitwise OR.
    out[i] = (fid << 32) | (int64_t)m_code;
}

torch::Tensor morton2d_keys(torch::Tensor uv, torch::Tensor frame_ids, int R){
    TORCH_CHECK(uv.is_cuda(), "uv must be CUDA");
    TORCH_CHECK(uv.dim() == 2 && uv.size(1) == 2, "uv shape [N,2]");
    
    const int N = (int)uv.size(0);
    // Output is Int64 to accommodate the packed (Frame | Morton) key
    auto keys = torch::empty({N}, uv.options().dtype(torch::kInt64)); 
    
    const int64_t* fid_ptr = nullptr;
    if (frame_ids.defined()){
        TORCH_CHECK(frame_ids.is_cuda(), "frame_ids must be CUDA");
        fid_ptr = frame_ids.data_ptr<int64_t>();
    }

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    morton2d_kernel<<<grid, block>>>(
        uv.data_ptr<float>(),
        fid_ptr,
        keys.data_ptr<int64_t>(),
        N, R - 1
    );
    return keys;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("morton3d_keys", &morton3d_keys, "Morton keys 3D (CUDA)");
    m.def("morton2d_keys", &morton2d_keys, "Morton keys 2D packed (CUDA)");
}