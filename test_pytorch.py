try:
    import torch
    print("✅ PyTorch installed:", torch.__version__)
    print("✅ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🖥️ GPU:", torch.cuda.get_device_name(0))
        print("💾 GPU Memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError:
    print("❌ PyTorch not installed")
    print("Run: pip install torch torchvision torchaudio")
