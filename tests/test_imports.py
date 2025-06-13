#!/usr/bin/env python3

print("Starting import testing...")

print("Testing basic imports...")
try:
    import os
    print("✓ os imported successfully")
    
    import torch
    print("✓ torch imported successfully")
    
    import numpy as np
    print("✓ numpy imported successfully")
    
    import random
    print("✓ random imported successfully")
    
    import torch.nn.functional as F
    print("✓ torch.nn.functional imported successfully")
    
    import argparse
    print("✓ argparse imported successfully")
    
except Exception as e:
    print(f"✗ Basic import failed: {e}")
    exit(1)

print("Testing potentially problematic imports...")
try:
    print("About to import config...")
    import config
    print("✓ config imported successfully")
    
    print("About to import datasets.dataset...")
    from datasets.dataset import get_dataset
    print("✓ datasets.dataset imported successfully")
    
    print("About to import model components...")
    from model.scene_rep import JointEncoding
    print("✓ model.scene_rep imported successfully")
    
    from model.keyframe import KeyFrameDatabase
    print("✓ model.keyframe imported successfully")
    
except Exception as e:
    print(f"✗ Advanced import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("All imports completed successfully!")
print("The hang must be occurring after imports but before main execution")