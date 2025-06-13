# Test basic import
try:
    import tinycudann as tcnn
    print("✓ tinycudann imported successfully!")
    print(f"Version: {tcnn.__version__ if hasattr(tcnn, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"✗ Import failed: {e}")

import torch
import tinycudann as tcnn

# Check if CUDA is available to PyTorch (prerequisite)
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

# Check GPU details if available
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")


import torch
import tinycudann as tcnn
import json

# Test creating a simple network configuration
# This mimics what Co-SLAM and other NeRF methods do
config = {
    "otype": "FullyFusedMLP",    # The key component that makes tiny-cuda-nn fast
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,             # Small network for testing
    "n_hidden_layers": 2,
}

try:
    # Create a simple MLP network
    # n_input_dims=3 (like 3D coordinates), n_output_dims=1 (like density)
    network = tcnn.Network(n_input_dims=3, n_output_dims=1, network_config=config)
    print("✓ FullyFusedMLP network created successfully!")
    
    # Test with some dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1024, 3, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        output = network(dummy_input)
    
    print(f"✓ Network inference successful! Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    
except Exception as e:
    print(f"✗ Network creation/inference failed: {e}")


import torch
import tinycudann as tcnn

# Test the famous hash encoding from Instant-NeRF
hash_config = {
    "otype": "HashGrid",           # The revolutionary hash encoding
    "n_levels": 16,                # Multiple resolution levels
    "n_features_per_level": 2,     # Features per level
    "log2_hashmap_size": 19,       # Hash table size (2^19 entries)
    "base_resolution": 16,         # Coarsest level resolution
    "per_level_scale": 2.0,        # How much resolution increases per level
}

try:
    # Create hash encoding for 3D coordinates (typical for NeRF)
    encoding = tcnn.Encoding(n_input_dims=3, encoding_config=hash_config)
    print("✓ Hash encoding created successfully!")
    print(f"Hash encoding output dimensions: {encoding.n_output_dims}")
    
    # Test encoding some 3D points
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create random 3D coordinates in [0,1] range (typical for NeRF)
    coords = torch.rand(1024, 3, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        encoded = encoding(coords)
    
    print(f"✓ Hash encoding successful! Input: {coords.shape} → Output: {encoded.shape}")
    
except Exception as e:
    print(f"✗ Hash encoding failed: {e}")


import torch
import tinycudann as tcnn
import time

def benchmark_network():
    config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU", 
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 3,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = tcnn.Network(3, 3, config).to(device)
    
    # Large batch for performance testing
    batch_size = 65536  # Common size in NeRF applications
    input_data = torch.randn(batch_size, 3, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = network(input_data)
    
    # Timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            output = network(input_data)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    throughput = batch_size / avg_time
    
    print(f"✓ Performance test completed!")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.0f} samples/second")
    
    return avg_time < 0.01  # Should be very fast on modern GPUs

try:
    if torch.cuda.is_available():
        is_fast = benchmark_network()
        if is_fast:
            print("✓ Performance looks good - you're getting GPU acceleration!")
        else:
            print("⚠ Performance seems slower than expected")
    else:
        print("⚠ CUDA not available - performance will be limited")
except Exception as e:
    print(f"✗ Performance test failed: {e}")



# testing the installation of pytorch3d

# Test the basic import and functionality

import torch
import pytorch3d
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch3D version: {pytorch3d.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')

# Test a simple PyTorch3D operation
from pytorch3d.structures import Meshes
print('PyTorch3D basic functionality test passed!')


# Test the marchingcubenumpy lib
try:
    import marching_cubes
    print('✓ marching_cubes module imported successfully!')
    print('  Available attributes:', [attr for attr in dir(marching_cubes) if not attr.startswith('_')])
    
    # Try to access the actual marching cubes function
    if hasattr(marching_cubes, 'marching_cubes'):
        print('  ✓ marching_cubes function found!')
    elif hasattr(marching_cubes, 'marching_cubes_lewiner'):
        print('  ✓ marching_cubes_lewiner function found!')
    else:
        print('  Available functions:', [attr for attr in dir(marching_cubes) if callable(getattr(marching_cubes, attr))])
        
except ImportError as e:
    print('✗ Direct import failed:', str(e))