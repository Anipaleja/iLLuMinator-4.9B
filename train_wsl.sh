#!/bin/bash

# iLLuMinator 4.9B WSL Training Launcher
echo "============================================================"
echo "iLLuMinator 4.9B Enhanced Training for WSL"
echo "============================================================"

# Check if we're in WSL
if ! grep -qi microsoft /proc/version; then
    echo "This script is optimized for WSL. Proceeding anyway..."
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Check Python and CUDA
echo "Checking system setup..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name()}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
except ImportError:
    print('PyTorch not installed')
"

# Install requirements if needed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

if ! python3 -c "import transformers" 2>/dev/null; then
    echo "Installing additional ML packages..."
    pip install transformers accelerate datasets wandb matplotlib seaborn psutil GPUtil tqdm numpy scipy scikit-learn tensorboard
fi

# Set optimal environment variables for WSL + CUDA
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3050
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Create necessary directories
mkdir -p checkpoints logs debug_output

echo "============================================================"
echo "Starting 4.9B Parameter Model Training..."
echo "============================================================"

# Run the training
python3 train_4.9B_enhanced.py

echo "============================================================"
echo "Training completed! Starting debug analysis..."
echo "============================================================"

# Run debugging
python3 debug_4.9B_enhanced.py

echo "============================================================"
echo "All processes completed!"
echo "Check results in:"
echo "- checkpoints/ for model checkpoints"
echo "- logs/ for training logs"
echo "- debug_output_*/ for debug analysis"
echo "============================================================"
