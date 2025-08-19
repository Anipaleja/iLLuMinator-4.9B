# Enhanced iLLuMinator 4.9B Training Setup

## 📁 Training Directory Structure

```
training/
├── train.sh                     # 🚀 Main execution script (run this!)
├── train_illuminator_4_9b.py    # 🧠 Training implementation
├── training_config.json         # ⚙️ Configuration file
├── requirements.txt             # 📦 Python dependencies
├── test_setup.py               # 🧪 Setup verification script
├── RUNPOD_TRAINING_GUIDE.md    # 📚 Comprehensive RunPod guide
└── README.md                   # 📖 This file
```

## 🚀 Quick Start

### 1. **One-Command Training**
```bash
# Navigate to training directory
cd training

# Start training (this handles everything!)
./train.sh
```

### 2. **Test Setup First** (Recommended)
```bash
# Verify everything is working
./train.sh --dry-run

# Or run the verification script
python test_setup.py
```

## 🎯 Key Features

✅ **One-Click Training**: `./train.sh` handles everything  
✅ **Automatic Setup**: Installs dependencies, creates directories  
✅ **Memory Optimization**: Adaptive batch sizes based on GPU  
✅ **Distributed Training**: Automatic multi-GPU support  
✅ **Monitoring**: Weights & Biases integration  
✅ **Checkpointing**: Automatic saves every 5000 steps  
✅ **RunPod Ready**: Optimized for cloud GPU training  
✅ **Error Handling**: Graceful shutdowns and recovery  

## ⚙️ Configuration Options

### Training Script Arguments

```bash
./train.sh --help                # Show all options
./train.sh --dry-run             # Test setup without training
./train.sh --config my_config.json    # Use custom config
./train.sh --resume checkpoint.pt     # Resume from checkpoint
./train.sh --output-dir ./my_outputs  # Custom output directory
```

### Configuration File (`training_config.json`)

Key settings you might want to adjust:

```json
{
  "training_config": {
    "batch_size": 4,              // Reduce if out of memory
    "max_steps": 100000,          // Total training steps
    "learning_rate": 1e-4,        // Learning rate
    "save_interval": 5000         // Checkpoint frequency
  },
  "model_config": {
    "max_seq_length": 2048        // Context length (reduce for memory)
  }
}
```

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU**: RTX 3090 (24GB VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 100GB free space
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Recommended for RunPod
- **GPU**: RTX A6000 (48GB) or A100 (40GB/80GB)
- **RAM**: 64GB+ system RAM
- **Storage**: 200GB+ for datasets and checkpoints

## 📊 Memory Usage Guide

| GPU | VRAM | Recommended Batch Size | Max Sequence Length |
|-----|------|----------------------|-------------------|
| RTX 3090 | 24GB | 2 | 1024 |
| RTX 4090 | 24GB | 2-4 | 1024-2048 |
| RTX A6000 | 48GB | 4-8 | 2048-4096 |
| A100 40GB | 40GB | 6-12 | 2048-4096 |
| A100 80GB | 80GB | 8-16 | 4096-8192 |

## 🚀 RunPod Training

### Step-by-Step RunPod Setup

1. **Create RunPod Account**: [runpod.io](https://www.runpod.io)

2. **Deploy Pod**:
   - Template: PyTorch 2.0
   - GPU: RTX A6000 or A100
   - Storage: 100GB+ container disk
   - Volume: 200GB+ for datasets

3. **Setup Training**:
   ```bash
   # Connect to your pod terminal
   cd /workspace
   
   # Clone repository
   git clone https://github.com/Anipaleja/iLLuMinator-4.9B.git
   cd iLLuMinator-4.9B/training
   
   # Start training
   ./train.sh
   ```

4. **Monitor Training**:
   - Watch terminal output
   - Check Weights & Biases dashboard
   - Monitor GPU usage with `nvidia-smi`

### Cost Estimates

| GPU | Cost/Hour | 100k Steps | Estimated Total |
|-----|-----------|------------|-----------------|
| RTX 4090 | $0.34 | ~30 hours | ~$10 |
| RTX A6000 | $0.79 | ~20 hours | ~$16 |
| A100 40GB | $1.89 | ~15 hours | ~$28 |
| A100 80GB | $2.89 | ~12 hours | ~$35 |

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Edit training_config.json
"batch_size": 1,
"gradient_accumulation_steps": 32
```

**2. Model Import Error**
```bash
# Ensure you're in the training directory
cd training
# Ensure parent directory has the model file
ls ../enhanced_illuminator_4_9b.py
```

**3. Slow Training**
```bash
# Enable optimizations in training_config.json
"use_mixed_precision": true,
"compile_model": true
```

**4. Connection Lost**
```bash
# Use screen for persistent sessions
screen -S training
./train.sh
# Detach with Ctrl+A, D
# Reattach with: screen -r training
```

## 📈 Training Progress

### What to Expect

- **Initial Loss**: ~10-12 (random initialization)
- **After 1k steps**: ~8-9 (model starts learning)
- **After 10k steps**: ~5-7 (good progress)
- **After 50k steps**: ~3-5 (well-trained)
- **After 100k steps**: ~2-4 (production ready)

### Monitoring

The training script provides:
- **Real-time loss tracking**
- **Learning rate scheduling**
- **GPU utilization monitoring**
- **Memory usage tracking**
- **Automatic checkpointing**

## 💾 Outputs

After training completes, you'll find:

```
outputs/
├── illuminator_4_9b_final.pt      # Final trained model
├── training_config.json           # Configuration used
└── model_info.json               # Model metadata

checkpoints/
├── checkpoint_step_5000.pt       # Regular checkpoints
├── checkpoint_step_10000.pt
└── ...

logs/
└── training_YYYYMMDD_HHMMSS.log  # Training logs
```

## 🎯 Next Steps After Training

1. **Model Evaluation**: Test on validation data
2. **Fine-tuning**: Adapt for specific tasks
3. **Deployment**: Set up inference server
4. **Optimization**: Quantization, distillation

## 📞 Support

- **Documentation**: Check `RUNPOD_TRAINING_GUIDE.md` for detailed instructions
- **Issues**: Report bugs on GitHub
- **Community**: Join RunPod Discord for support

## ✅ Quick Checklist

Before training:
- [ ] GPU has sufficient VRAM (24GB+ recommended)
- [ ] Adequate storage space (100GB+)
- [ ] Python 3.8+ installed
- [ ] CUDA drivers installed
- [ ] Repository cloned
- [ ] In training directory

To start:
- [ ] Run `./train.sh --dry-run` first
- [ ] If successful, run `./train.sh`
- [ ] Monitor progress
- [ ] Check outputs after completion

**Happy Training! 🎉**
