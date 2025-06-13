# Co-SLAM Installation Guide: RTX 4090/4070 with CUDA 12.4

## Environment Setup

```bash
# Create conda environment
conda create -n coslam python=3.7
conda activate coslam

# Install PyTorch with CUDA 11.3 support
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install Co-SLAM dependencies
pip install -r requirements.txt
```

## tiny-cuda-nn Installation (The Challenging Part)

The main challenge is C++ compiler compatibility and GPU architecture targeting for RTX 4090/4070.

```bash
# Clone tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn
```

### Fix GCC Compatibility
CUDA 11.3's nvcc has issues with modern GCC versions (11+), causing C++ compilation errors.

```bash
# Install compatible GCC version
sudo apt update
sudo apt install gcc-9 g++-9

# Set compiler environment variables
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9
```

### Configure GPU Architecture
RTX 4090/4070 use compute capability 8.9 which needs explicit targeting.

```bash
# Set CUDA architectures (89 for RTX 4090/4070)
export TCNN_CUDA_ARCHITECTURES="75;80;86;89"
```

### Build and Install

```bash
# Configure build with specific compiler and GPU targets
cmake . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTCNN_CUDA_ARCHITECTURES="75;80;86;89" \
    -DCMAKE_C_COMPILER=gcc-9 \
    -DCMAKE_CXX_COMPILER=g++-9 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-9

# Build (takes 5-10 minutes)
cmake --build build --config RelWithDebInfo -j

# Install Python bindings
cd bindings/torch
CC=gcc-9 CXX=g++-9 CUDAHOSTCXX=g++-9 python setup.py install
```

## Verification

```bash
python -c "import tinycudann as tcnn; print('✓ tinycudann installed successfully')"
```

## Complete Co-SLAM Installation

```bash
# Clone Co-SLAM
git clone https://github.com/HengyiWang/Co-SLAM.git
cd Co-SLAM

# Build marching cubes extension
cd external/NumpyMarchingCubes
python setup.py install
```

## Key Issues Solved

- **GCC Compatibility**: Use GCC 9 instead of GCC 11+ to avoid C++ compilation errors
- **GPU Architecture**: Include compute capability 89 for RTX 4090/4070 support
- **Alternative if build fails**: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`