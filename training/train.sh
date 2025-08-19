#!/bin/bash

# Enhanced iLLuMinator 4.9B Training Script for RunPod
# This script sets up the environment and starts training

set -e  # Exit on any error

echo "🚀 Enhanced iLLuMinator 4.9B Training Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in the training directory
if [ ! -f "train_illuminator_4_9b.py" ]; then
    print_error "train_illuminator_4_9b.py not found!"
    print_error "Please run this script from the training directory."
    exit 1
fi

# System information
print_status "System Information:"
echo "  - OS: $(uname -s)"
echo "  - CPU: $(nproc) cores"
echo "  - Memory: $(free -h | grep '^Mem:' | awk '{print $2}') total"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | nl -w2 -s'. '
    CUDA_AVAILABLE=true
else
    print_warning "NVIDIA GPU not detected. Training will use CPU (very slow)."
    CUDA_AVAILABLE=false
fi

# Check Python environment
print_status "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Check and install dependencies
print_status "Checking Python dependencies..."

# Function to check if a package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
}

# Required packages
REQUIRED_PACKAGES=("torch" "numpy" "tqdm" "psutil")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! check_package "$package"; then
        MISSING_PACKAGES+=("$package")
    fi
done

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    print_warning "Missing packages detected: ${MISSING_PACKAGES[*]}"
    print_status "Installing missing packages..."
    
    # Install PyTorch with CUDA support if available
    if [ "$CUDA_AVAILABLE" = true ]; then
        print_status "Installing PyTorch with CUDA support..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch CPU version..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other requirements
    pip3 install tqdm psutil wandb transformers
else
    print_status "All required packages are installed."
fi

# Verify PyTorch CUDA
if [ "$CUDA_AVAILABLE" = true ]; then
    TORCH_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [ "$TORCH_CUDA" = "True" ]; then
        print_status "PyTorch CUDA support: ✅ Enabled"
        CUDA_DEVICES=$(python3 -c "import torch; print(torch.cuda.device_count())")
        print_status "CUDA devices available: $CUDA_DEVICES"
    else
        print_warning "PyTorch CUDA support: ❌ Disabled"
    fi
fi

# Setup output directories
print_status "Setting up output directories..."
mkdir -p outputs checkpoints logs cache

# Check disk space
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

print_status "Available disk space: ${AVAILABLE_GB}GB"

if [ $AVAILABLE_GB -lt 50 ]; then
    print_warning "Low disk space detected. Consider cleaning up or using a larger disk."
fi

# Memory requirements
TOTAL_RAM=$(free -g | grep '^Mem:' | awk '{print $2}')
print_status "Total RAM: ${TOTAL_RAM}GB"

if [ $TOTAL_RAM -lt 16 ]; then
    print_warning "Low RAM detected. Consider reducing batch size in training_config.json"
fi

# Backup existing config if present
if [ -f "training_config.json" ]; then
    if [ -f "training_config.json.bak" ]; then
        rm training_config.json.bak
    fi
    cp training_config.json training_config.json.bak
    print_status "Existing config backed up to training_config.json.bak"
fi

# Parse command line arguments
RESUME_CHECKPOINT=""
OUTPUT_DIR="./outputs"
CONFIG_FILE="training_config.json"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resume CHECKPOINT    Resume training from checkpoint"
            echo "  --output-dir DIR       Output directory (default: ./outputs)"
            echo "  --config FILE          Config file (default: training_config.json)"
            echo "  --dry-run              Test setup without starting training"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Training Configuration:"
echo "  - Config file: $CONFIG_FILE"
echo "  - Output directory: $OUTPUT_DIR"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  - Resume from: $RESUME_CHECKPOINT"
fi

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Test configuration
print_status "Validating configuration..."
python3 -c "
import json
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)
    print('✅ Configuration file is valid')
    
    # Print key settings
    model_config = config.get('model_config', {})
    training_config = config.get('training_config', {})
    
    print(f'Model dimension: {model_config.get(\"d_model\", \"N/A\")}')
    print(f'Number of layers: {model_config.get(\"n_layers\", \"N/A\")}')
    print(f'Batch size: {training_config.get(\"batch_size\", \"N/A\")}')
    print(f'Max steps: {training_config.get(\"max_steps\", \"N/A\")}')
    
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Configuration validation failed!"
    exit 1
fi

# Estimate memory requirements
print_status "Estimating memory requirements..."
python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

model_config = config['model_config']
training_config = config['training_config']

# Rough parameter estimation
vocab_size = model_config['vocab_size']
d_model = model_config['d_model']
n_layers = model_config['n_layers']
d_ff = model_config['d_ff']

# Estimate parameters (rough calculation)
embed_params = vocab_size * d_model
attention_params = n_layers * 4 * d_model * d_model
ff_params = n_layers * 3 * d_model * d_ff  # SwiGLU
total_params = embed_params + attention_params + ff_params

# Memory estimates (FP16)
param_memory_gb = (total_params * 2) / (1024**3)
optimizer_memory_gb = param_memory_gb * 2  # AdamW states
activation_memory_gb = training_config['batch_size'] * 0.5  # Rough estimate

total_memory_gb = param_memory_gb + optimizer_memory_gb + activation_memory_gb

print(f'Estimated parameters: {total_params/1e9:.2f}B')
print(f'Estimated memory usage:')
print(f'  - Parameters: {param_memory_gb:.1f}GB')
print(f'  - Optimizer states: {optimizer_memory_gb:.1f}GB')
print(f'  - Activations: {activation_memory_gb:.1f}GB')
print(f'  - Total: {total_memory_gb:.1f}GB')

if total_memory_gb > $TOTAL_RAM:
    print(f'⚠️  Estimated memory usage ({total_memory_gb:.1f}GB) exceeds available RAM ({$TOTAL_RAM}GB)')
    print(f'   Consider reducing batch size or using gradient checkpointing')
"

# Final checks before training
print_status "Pre-flight checks:"

# Check if model file exists
if [ ! -f "../enhanced_illuminator_4_9b.py" ]; then
    print_error "Enhanced model file not found: ../enhanced_illuminator_4_9b.py"
    exit 1
fi
echo "✅ Enhanced model file found"

# Check if tokenizer exists
if [ ! -d "../practical_model" ]; then
    print_warning "Tokenizer directory not found: ../practical_model"
    print_warning "Training may fail if tokenizer is not available"
fi

# Test model creation
print_status "Testing model creation..."
python3 -c "
import sys
sys.path.append('..')
try:
    from enhanced_illuminator_4_9b import iLLuMinator4_9B
    model = iLLuMinator4_9B(vocab_size=1000, d_model=512, n_layers=2, n_heads=8, n_kv_heads=2, d_ff=1024)
    print('✅ Model creation test passed')
except Exception as e:
    print(f'❌ Model creation test failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Model creation test failed!"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    print_status "Dry run completed successfully!"
    print_status "Run without --dry-run to start actual training."
    exit 0
fi

# Start training
print_status "Starting training..."
echo "Press Ctrl+C to stop training gracefully"
echo ""

# Build command
TRAIN_CMD="python3 train_illuminator_4_9b.py --config $CONFIG_FILE --output-dir $OUTPUT_DIR"

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CHECKPOINT"
fi

# Log the command
echo "Executing: $TRAIN_CMD"
echo ""

# Create a training log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Ensure logs directory exists
mkdir -p logs

# Execute training with logging
exec > >(tee -a "$LOG_FILE") 2>&1

print_status "Training started at $(date)"
print_status "Logs are being saved to: $LOG_FILE"

# Set trap for graceful shutdown
trap 'print_warning "Received interrupt signal. Stopping training gracefully..."; kill $TRAIN_PID; wait $TRAIN_PID' INT TERM

# Start training in background to allow signal handling
$TRAIN_CMD &
TRAIN_PID=$!

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    print_status "Training completed successfully at $(date)"
    print_status "Outputs saved to: $OUTPUT_DIR"
    print_status "Logs saved to: $LOG_FILE"
else
    print_error "Training failed with exit code: $TRAIN_EXIT_CODE"
    print_error "Check the logs for details: $LOG_FILE"
fi

exit $TRAIN_EXIT_CODE
