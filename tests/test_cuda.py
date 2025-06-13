#!/usr/bin/env python3

print("Testing CUDA initialization step by step...")

print("Step 1: Import torch")
import torch
print("✓ PyTorch imported")

print("Step 2: Check CUDA availability")
cuda_available = torch.cuda.is_available()
print(f"✓ CUDA available: {cuda_available}")

if cuda_available:
    print("Step 3: Get device count")
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA devices: {device_count}")
    
    print("Step 4: Create CUDA context")
    device = torch.device("cuda:0")
    print(f"✓ CUDA device created: {device}")
    
    print("Step 5: Allocate test tensor")
    test_tensor = torch.randn(10).to(device)
    print("✓ Test tensor allocated on GPU")
    
    print("Step 6: Test basic GPU operation")
    result = test_tensor * 2
    print("✓ Basic GPU operation successful")

print("Step 7: Test tinycudann import (this often hangs)")
print("About to import tinycudann - this may take several minutes...")
try:
    import tinycudann as tcnn
    print("✓ tinycudann imported successfully")
except Exception as e:
    print(f"✗ tinycudann import failed: {e}")
    import traceback
    traceback.print_exc()

print("CUDA testing complete!")