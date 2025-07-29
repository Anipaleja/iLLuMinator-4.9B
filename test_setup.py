import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("CUDA not available")
