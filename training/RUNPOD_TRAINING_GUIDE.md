# 🚀 RunPod Training Guide for Enhanced iLLuMinator 4.9B

## Overview

This guide provides step-by-step instructions for training the Enhanced iLLuMinator 4.9B model on RunPod, a cloud GPU platform optimized for machine learning workloads.

## 📋 Prerequisites

- RunPod account (sign up at [runpod.io](https://www.runpod.io))
- Basic familiarity with Linux command line
- Understanding of machine learning concepts

## 🖥️ RunPod Setup

### 1. Choose a Pod Template

**Recommended Templates:**
- **PyTorch 2.0**: Pre-configured with PyTorch and CUDA
- **Jupyter PyTorch**: Includes Jupyter notebooks for development
- **Custom Template**: Ubuntu 22.04 with CUDA 11.8+

**Recommended GPU:**
- **Minimum**: RTX 4090 (24GB VRAM) - $0.34/hour
- **Recommended**: RTX A6000 (48GB VRAM) - $0.79/hour  
- **Optimal**: A100 40GB/80GB - $1.89/$2.89/hour

### 2. Pod Configuration

```yaml
Recommended Settings:
- Container Disk: 50GB minimum (100GB recommended)
- Volume Disk: 100GB+ for datasets and checkpoints
- GPU: RTX A6000 or A100
- CPU: 8+ cores
- RAM: 64GB+
```

### 3. Create and Start Pod

1. Log into RunPod dashboard
2. Click "Deploy" → "GPU Pods"
3. Select template and GPU
4. Configure storage and network
5. Click "Deploy"

## 📦 Environment Setup

### 1. Connect to Your Pod

```bash
# Via RunPod web terminal or SSH
# Pod will provide connection details
```

### 2. Update System

```bash
apt update && apt upgrade -y
apt install -y git wget curl htop nvtop vim
```

### 3. Verify CUDA

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Project Setup

### 1. Clone Repository

```bash
cd /workspace
git clone https://github.com/Anipaleja/iLLuMinator-4.9B.git
cd iLLuMinator-4.9B
```

### 2. Install Dependencies

```bash
# Navigate to training directory
cd training

# Install Python packages
pip install -r requirements.txt

# Or install manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers wandb tqdm psutil accelerate
```

### 3. Verify Installation

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## ⚙️ Configuration

### 1. Training Configuration

Edit `training_config.json` based on your GPU:

**For RTX 4090 (24GB):**
```json
{
  "training_config": {
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "max_steps": 50000,
    "use_mixed_precision": true
  },
  "model_config": {
    "max_seq_length": 1024
  }
}
```

**For RTX A6000 (48GB):**
```json
{
  "training_config": {
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "max_steps": 100000,
    "use_mixed_precision": true
  },
  "model_config": {
    "max_seq_length": 2048
  }
}
```

**For A100 (80GB):**
```json
{
  "training_config": {
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_steps": 150000,
    "use_mixed_precision": true
  },
  "model_config": {
    "max_seq_length": 4096
  }
}
```

### 2. Weights & Biases Setup (Optional)

```bash
# Install and login to wandb for monitoring
pip install wandb
wandb login
# Enter your API key from wandb.ai
```

## 🚀 Training Execution

### 1. Test Setup (Dry Run)

```bash
./train.sh --dry-run
```

This will:
- Verify all dependencies
- Test model creation
- Estimate memory requirements
- Validate configuration

### 2. Start Training

```bash
# Basic training
./train.sh

# With custom config
./train.sh --config my_config.json

# Resume from checkpoint
./train.sh --resume checkpoints/checkpoint_step_5000.pt

# Custom output directory
./train.sh --output-dir /workspace/my_outputs
```

### 3. Monitor Training

**Option 1: Terminal Monitoring**
```bash
# Watch training logs in real-time
tail -f logs/training_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

**Option 2: Weights & Biases Dashboard**
- Visit [wandb.ai](https://wandb.ai)
- Navigate to your project
- Monitor metrics in real-time

**Option 3: TensorBoard (if enabled)**
```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

## 📊 Memory and Performance Optimization

### Memory Usage Guidelines

| GPU | VRAM | Recommended Batch Size | Max Sequence Length |
|-----|------|----------------------|-------------------|
| RTX 4090 | 24GB | 2-4 | 1024-2048 |
| RTX A6000 | 48GB | 4-8 | 2048-4096 |
| A100 40GB | 40GB | 6-12 | 2048-4096 |
| A100 80GB | 80GB | 8-16 | 4096-8192 |

### Performance Tips

1. **Use Mixed Precision**: Always enable for faster training
2. **Gradient Accumulation**: Simulate larger batch sizes
3. **Model Compilation**: Enable PyTorch 2.0 compilation
4. **Efficient Data Loading**: Use multiple workers
5. **Checkpoint Regularly**: Save every 1000-5000 steps

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
"batch_size": 1,
"gradient_accumulation_steps": 32
```

**2. Import Errors**
```bash
# Ensure Python path is correct
export PYTHONPATH=/workspace/iLLuMinator-4.9B:$PYTHONPATH
```

**3. Slow Training**
```bash
# Enable optimizations
"use_mixed_precision": true,
"compile_model": true
```

**4. Connection Timeouts**
```bash
# Use screen or tmux for persistent sessions
screen -S training
./train.sh
# Detach with Ctrl+A, D
```

### Memory Optimization

If you encounter memory issues:

1. **Reduce batch size**: Start with batch_size=1
2. **Increase gradient accumulation**: Maintain effective batch size
3. **Reduce sequence length**: Use max_seq_length=1024
4. **Enable gradient checkpointing**: Trade compute for memory

## 📈 Training Monitoring

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Learning Rate**: Should follow warmup → decay schedule
3. **GPU Utilization**: Should be >90%
4. **Memory Usage**: Should be stable, not growing
5. **Training Speed**: Steps per second

### Expected Performance

| GPU | Steps/Second | Time to 100k Steps |
|-----|-------------|-------------------|
| RTX 4090 | 0.8-1.2 | ~24-36 hours |
| RTX A6000 | 1.0-1.5 | ~18-28 hours |
| A100 40GB | 1.2-1.8 | ~15-23 hours |
| A100 80GB | 1.5-2.2 | ~12-18 hours |

## 💾 Checkpoints and Outputs

### Automatic Saves

The training script automatically saves:
- **Checkpoints**: Every 5000 steps (`checkpoints/`)
- **Final Model**: At completion (`outputs/`)
- **Training Logs**: Real-time (`logs/`)
- **Configuration**: Used settings (`outputs/training_config.json`)

### Manual Checkpoint

To manually save during training:
```bash
# Send SIGTERM to gracefully stop and save
pkill -TERM -f train_illuminator_4_9b.py
```

## 📁 File Structure

```
training/
├── train.sh                    # Main execution script
├── train_illuminator_4_9b.py   # Training implementation
├── training_config.json        # Configuration file
├── requirements.txt            # Python dependencies
├── outputs/                    # Final models and configs
├── checkpoints/               # Training checkpoints
├── logs/                      # Training logs
└── cache/                     # Tokenized dataset cache
```

## 🎯 Cost Optimization

### Cost-Effective Training

1. **Use Spot Instances**: 50-70% cheaper, but can be interrupted
2. **Monitor Usage**: Stop when not actively training
3. **Efficient Scheduling**: Train during off-peak hours
4. **Checkpoint Frequently**: Resume after interruptions

### Estimated Costs

| GPU | Cost/Hour | 100k Steps | Full Training |
|-----|-----------|------------|---------------|
| RTX 4090 | $0.34 | $8-12 | $15-25 |
| RTX A6000 | $0.79 | $14-22 | $25-40 |
| A100 40GB | $1.89 | $28-43 | $50-80 |
| A100 80GB | $2.89 | $35-52 | $65-105 |

## 🚀 Advanced Features

### Multi-GPU Training

For multiple GPUs:
```bash
# The script automatically detects multiple GPUs
./train.sh  # Will use all available GPUs
```

### Custom Datasets

To use custom datasets, modify the `prepare_comprehensive_dataset()` function in `train_illuminator_4_9b.py`.

### Hyperparameter Tuning

Use Weights & Biases sweeps for automated hyperparameter optimization:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

## 📞 Support

If you encounter issues:

1. **Check Logs**: Always review training logs first
2. **RunPod Community**: Active Discord and forums
3. **GitHub Issues**: Report bugs in the repository
4. **Documentation**: Refer to PyTorch and Transformers docs

## ✅ Quick Start Checklist

- [ ] RunPod account created
- [ ] Pod deployed with sufficient GPU/RAM
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] CUDA verified working
- [ ] Configuration customized for your GPU
- [ ] Dry run completed successfully
- [ ] Monitoring setup (wandb/tensorboard)
- [ ] Training started with `./train.sh`

Happy training! 🎉
