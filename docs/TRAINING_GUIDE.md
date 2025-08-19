# iLLuMinator 4.9B Training Guide

This comprehensive guide covers everything you need to know about training the iLLuMinator 4.9B language model from setup to deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Configuration](#model-configuration)
- [Training Process](#training-process)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Optimization Techniques](#optimization-techniques)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

#### Minimum Configuration
- **GPU**: 8GB VRAM (RTX 3060, RTX 4060)
- **CPU**: 8 cores, 3.0GHz+
- **RAM**: 32GB system memory
- **Storage**: 500GB SSD

#### Recommended Configuration
- **GPU**: 24GB+ VRAM (RTX 4090, A100)
- **CPU**: 16+ cores, 3.5GHz+
- **RAM**: 128GB+ system memory
- **Storage**: 2TB+ NVMe SSD

#### Professional Configuration
- **GPU**: Multiple A100/H100 (40-80GB each)
- **CPU**: 64+ cores, server-grade
- **RAM**: 512GB+ system memory
- **Storage**: 10TB+ enterprise SSD
- **Network**: InfiniBand for multi-node training

### Software Requirements

```bash
# Operating System
Ubuntu 20.04+ or CentOS 8+ (recommended)
Windows 10/11 (supported)
macOS 12+ (Apple Silicon supported)

# Python
Python 3.8+ (3.10+ recommended)

# CUDA (for NVIDIA GPUs)
CUDA 11.8+ or 12.0+
cuDNN 8.6+

# Drivers
NVIDIA Driver 525.0+
```

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/Anipaleja/iLLuMinator-4.9B.git
cd iLLuMinator-4.9B
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n illuminator python=3.10
conda activate illuminator

# Or using venv
python -m venv illuminator_env
source illuminator_env/bin/activate  # Linux/Mac
# illuminator_env\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# With optional optimizations
pip install flash-attn deepspeed bitsandbytes

# For development
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
python error_handler.py
```

This will validate your environment and create necessary directories.

## Data Preparation

### Professional Datasets

The iLLuMinator 4.9B uses high-quality datasets similar to those used by leading AI models:

```python
from data_sources import ProfessionalDatasetLoader

# Initialize dataset loader
loader = ProfessionalDatasetLoader()

# Create comprehensive dataset
dataset = loader.create_comprehensive_dataset(total_size=25000)

# Save for reuse
loader.save_dataset(dataset, "professional_training_data.json")
```

### Dataset Composition

| Dataset | Examples | Percentage | Quality Score |
|---------|----------|------------|---------------|
| OpenOrca | 10,000 | 40% | 0.90 |
| UltraChat | 5,000 | 20% | 0.85 |
| CodeAlpaca | 5,000 | 20% | 0.88 |
| WizardLM | 2,500 | 10% | 0.92 |
| Math | 1,250 | 5% | 0.95 |
| Scientific | 1,250 | 5% | 0.90 |

### Custom Data Preparation

```python
# Format your own data
def format_custom_data(examples):
    formatted = []
    for example in examples:
        text = f"Human: {example['question']}\nAssistant: {example['answer']}"
        formatted.append(text)
    return formatted
```

## Model Configuration

### Using Configuration Manager

```python
from config_manager import create_professional_config

# Auto-detected configuration
config = create_professional_config("4.9B")

# Custom overrides
overrides = {
    "training": {
        "learning_rate": 2e-4,
        "batch_size": 4
    },
    "model": {
        "max_seq_length": 4096
    }
}

config = create_professional_config("4.9B", overrides)
config.save("my_config.json")
```

### Configuration Presets

#### Memory-Constrained Setup (8GB VRAM)
```python
config = create_professional_config("4.9B", {
    "training": {
        "batch_size": 1,
        "gradient_accumulation_steps": 128,
        "gradient_checkpointing": True,
        "mixed_precision": True
    },
    "data": {
        "max_seq_length": 1024
    }
})
```

#### High-Performance Setup (24GB+ VRAM)
```python
config = create_professional_config("4.9B", {
    "training": {
        "batch_size": 4,
        "gradient_accumulation_steps": 16,
        "mixed_precision": True
    },
    "data": {
        "max_seq_length": 2048
    }
})
```

## Training Process

### Single GPU Training

```bash
# Basic training
python train_professional.py

# With custom configuration
python train_professional.py --config my_config.json

# Resume from checkpoint
python train_professional.py --resume checkpoints/best_model.pt
```

### Multi-GPU Training

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 train_professional.py \
    --distributed \
    --config multi_gpu_config.json

# Multiple nodes
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29400 \
    train_professional.py --distributed
```

### Apple Silicon Training

```bash
# Optimized for Apple Silicon
python train_apple_silicon.py
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Initial learning rate |
| `weight_decay` | 0.1 | L2 regularization |
| `warmup_steps` | 2000 | Learning rate warmup |
| `max_steps` | 100000 | Maximum training steps |
| `save_every` | 1000 | Checkpoint frequency |
| `eval_every` | 500 | Evaluation frequency |

## Monitoring and Debugging

### Professional Logging

The system automatically creates comprehensive logs:

```
logs/
├── illuminator.log      # All logs
├── errors.log          # Error-only logs
└── training.log        # Training-specific logs
```

### Real-time Monitoring

```bash
# Monitor training progress
tail -f logs/training.log

# Monitor GPU usage
python mps-monitor.py --watch-memory
```

### Weights & Biases Integration

```python
# Enable W&B logging
config.system.use_wandb = True
```

### Memory Analysis

```bash
# Analyze memory usage
python mfu_analyzer.py --model-path checkpoints/model.pt

# Monitor Apple Silicon memory
python mps-monitor.py
```

## Optimization Techniques

### Memory Optimization

1. **Gradient Checkpointing**
   ```python
   config.training.gradient_checkpointing = True
   ```

2. **Mixed Precision Training**
   ```python
   config.training.mixed_precision = True
   ```

3. **Gradient Accumulation**
   ```python
   config.training.gradient_accumulation_steps = 32
   ```

### Speed Optimization

1. **Flash Attention**
   ```bash
   pip install flash-attn
   python train_professional.py --flash-attention
   ```

2. **Model Compilation** (PyTorch 2.0+)
   ```python
   config.system.compile_model = True
   ```

3. **Efficient Data Loading**
   ```python
   config.system.num_workers = 8
   config.system.pin_memory = True
   ```

### Distributed Training

1. **Data Parallel**
   ```bash
   torchrun --nproc_per_node=4 train_professional.py
   ```

2. **DeepSpeed Integration**
   ```bash
   pip install deepspeed
   python train_professional.py --deepspeed ds_config.json
   ```

## Troubleshooting

### Common Issues

#### Out of Memory Errors

**Symptoms**: CUDA out of memory, allocation failures
**Solutions**:
```python
# Reduce batch size
config.training.batch_size = 1

# Enable gradient checkpointing
config.training.gradient_checkpointing = True

# Reduce sequence length
config.data.max_seq_length = 1024

# Increase gradient accumulation
config.training.gradient_accumulation_steps = 64
```

#### Slow Training Speed

**Symptoms**: Low GPU utilization, slow steps/second
**Solutions**:
```python
# Enable mixed precision
config.training.mixed_precision = True

# Optimize data loading
config.system.num_workers = min(cpu_count(), 8)

# Use flash attention
pip install flash-attn
```

#### Loss Not Decreasing

**Symptoms**: Training loss plateaus or increases
**Solutions**:
```python
# Adjust learning rate
config.training.learning_rate = 1e-4

# Increase warmup steps
config.training.warmup_steps = 5000

# Check data quality
# Ensure diverse, high-quality training data
```

#### Import Errors

**Symptoms**: Module not found, compatibility issues
**Solutions**:
```bash
# Reinstall dependencies
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check Python version
python --version  # Should be 3.8+

# Verify CUDA installation
nvcc --version
```

### Debug Mode

Enable comprehensive debugging:

```bash
python train_professional.py --debug --log-level DEBUG
```

### Emergency Recovery

If training crashes, the system automatically saves emergency checkpoints:

```python
# Resume from emergency checkpoint
python train_professional.py --resume emergency_checkpoint.pt
```

### Performance Profiling

```bash
# Profile training step
python -m torch.profiler train_professional.py --profile

# Memory profiling
python train_professional.py --profile-memory
```

## Advanced Training Strategies

### Curriculum Learning

```python
# Start with shorter sequences, gradually increase
def create_curriculum_config(step):
    seq_length = min(512 + step // 1000 * 256, 2048)
    return {"data": {"max_seq_length": seq_length}}
```

### Learning Rate Scheduling

```python
# Cosine annealing with restarts
config.training.scheduler = "cosine_with_restarts"
config.training.scheduler_t0 = 1000
config.training.scheduler_t_mult = 2
```

### Regularization Techniques

```python
# Dropout and weight decay
config.model.dropout = 0.1
config.training.weight_decay = 0.1

# Label smoothing
config.training.label_smoothing = 0.1
```

This guide provides comprehensive information for training the iLLuMinator 4.9B model. For additional help, consult the API documentation or create an issue on GitHub.