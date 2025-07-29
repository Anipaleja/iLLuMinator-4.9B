import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name())
    print("CUDA version:", torch.version.cuda)
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("CUDA not available - using CPU")
