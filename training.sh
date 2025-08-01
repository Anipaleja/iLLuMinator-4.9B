#!/bin/bash
# Apple Silicon 4.9B Training Setup Script
# Prepares system and launches training with monitoring

echo "Apple Silicon 4.9B Parameter Training Setup"
echo "=============================================="

# Check system requirements 
echo "Checking system requirements..."

# Check macOS version
OS_VERSION=$(sw_vers -productVersion)
echo "   macOS Version: $OS_VERSION"

# Check available memory
TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')
echo "   Total RAM: $TOTAL_RAM"

# Check available disk space
DISK_SPACE=$(df -h / | awk 'NR==2 {print $4}')
echo "   Available disk: $DISK_SPACE"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null || echo "Python not found")
echo "   Python: $PYTHON_VERSION"

# Check PyTorch and MPS availability
echo "Testing PyTorch MPS availability..."
python3 -c "
import torch
print(f'   PyTorch version: {torch.__version__}')
if torch.backends.mps.is_available():
    print('MPS Backend: Available')
    # Test MPS with a simple operation
    try:
        test_tensor = torch.randn(100, 100).to('mps')
        result = test_tensor @ test_tensor
        print('MPS Test: Passed')
    except Exception as e:
        print(f'MPS Test: Failed - {e}')
else:
    print('MPS Backend: Not Available')
"

echo ""
echo "Memory optimization..."

# Clear system caches (if possible)
if command -v purge >/dev/null 2>&1; then
    echo "   Clearing system memory caches..."
    sudo purge 2>/dev/null || echo "   Memory purge requires sudo privileges"
else
    echo "   Purge command not available"
fi

# Kill unnecessary processes
echo "   Checking for memory-hungry processes..."
python3 -c "
import psutil
processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
    try:
        if proc.info['memory_percent'] > 5.0:
            processes.append(proc.info)
    except:
        pass

processes.sort(key=lambda x: x['memory_percent'], reverse=True)
print('   Top memory users:')
for i, proc in enumerate(processes[:5]):
    print(f'     {i+1}. {proc[\"name\"]} - {proc[\"memory_percent\"]:.1f}%')
"

echo ""
echo "Training Configuration:"
echo "   Model: iLLuMinator 4.9B parameters"
echo "   Device: Apple Silicon MPS"
echo "   Memory: Optimized for 16GB RAM"
echo "   Sequence Length: 1024 tokens"
echo "   Batch Size: 1 (conservative)"

echo ""
read -p "Ready to start training? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting 4.9B parameter training..."
    
    # Create necessary directories
    mkdir -p checkpoints
    mkdir -p logs
    
    # Start training with monitoring
    echo "Starting system monitor in background..."
    python3 apple_silicon_monitor.py > logs/monitor.log 2>&1 &
    MONITOR_PID=$!
    
    echo "Starting model training..."
    python3 train_4_9b_apple_silicon.py 2>&1 | tee logs/training.log
    
    # Stop monitor
    echo "Stopping system monitor..."
    kill $MONITOR_PID 2>/dev/null
    
    echo "Training session complete!"
    echo "Logs saved in logs/ directory"
    echo "Checkpoints saved in checkpoints/ directory"
else
    echo "Training cancelled"
fi
