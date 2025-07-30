#!/usr/bin/env python3
"""Simple CUDA test"""

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU tensor
        x = torch.tensor([1, 2, 3]).cuda()
        print(f"GPU tensor test: {x}")
        
        print("CUDA is working properly!")
    else:
        print("CUDA not available - will train on CPU")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
