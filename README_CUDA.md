# iLLuMinator 4.9B CUDA Setup Guide

## 🔥 RTX 3070 Optimization

This setup is specifically optimized for NVIDIA RTX 3070 with 8GB VRAM.

## 📋 Prerequisites

### CUDA Installation
1. **NVIDIA Driver**: Latest driver (>=525.60)
2. **CUDA Toolkit**: CUDA 12.1 or compatible
3. **cuDNN**: Latest version for CUDA 12.1

### Verify CUDA Installation
```bash
nvidia-smi
nvcc --version
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install CUDA-optimized requirements
pip install -r requirements_cuda.txt

# Verify PyTorch CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"
```

### 2. Test CUDA Model
```bash
# Test the CUDA-optimized model
python illuminator_cuda.py
```

### 3. Train the Model
```bash
# Start CUDA training (optimized for RTX 3070)
python train_cuda.py
```

### 4. Run API Server
```bash
# Start CUDA-optimized API server
python cuda_api_server.py
```

## 🧠 Model Architecture

### CUDA Optimizations
- **Mixed Precision Training**: FP16/FP32 for 2x speed, 50% memory reduction
- **Flash Attention**: Optimized attention computation
- **Gradient Checkpointing**: Trade compute for memory
- **CUDA Kernels**: Optimized low-level operations
- **TensorFloat-32**: Automatic acceleration on RTX 30xx series

### Memory Management
- **Model Size**: ~19GB in FP32, ~10GB in FP16
- **Training Memory**: ~6-7GB with optimizations
- **Inference Memory**: ~5-6GB
- **Batch Size**: 1-2 for training, 1-4 for inference

## ⚡ Performance Targets

### RTX 3070 (8GB VRAM)
- **Training Speed**: ~0.5-1 tokens/second
- **Inference Speed**: ~5-10 tokens/second
- **Memory Usage**: ~6-7GB during training
- **Batch Size**: 1 (with gradient accumulation)

### Optimizations Applied
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Mixed precision
with torch.cuda.amp.autocast():
    output = model(input)

# Gradient checkpointing
model.gradient_checkpointing_enable()
```

## 🔧 Configuration

### Model Configuration
```python
config = {
    'vocab_size': 50260,
    'd_model': 3584,           # Hidden dimension
    'num_layers': 30,          # Transformer layers
    'num_heads': 28,           # Attention heads
    'd_ff': 14336,             # Feed-forward dimension
    'max_seq_length': 2048,    # Context length
    'dropout': 0.1,
    'use_gradient_checkpointing': True
}
```

### Training Configuration
```python
training_config = {
    'batch_size': 1,                    # Per device
    'gradient_accumulation_steps': 8,   # Effective batch size: 8
    'learning_rate': 1e-4,
    'weight_decay': 0.1,
    'mixed_precision': True,
    'max_grad_norm': 1.0
}
```

## 📊 Memory Optimization Strategies

### 1. Gradient Checkpointing
- Saves ~40% memory during training
- Increases training time by ~20%

### 2. Mixed Precision (FP16)
- Reduces memory by ~50%
- Increases speed by ~1.5-2x
- Automatic loss scaling

### 3. Gradient Accumulation
- Simulate larger batch sizes
- Effective batch size = batch_size × accumulation_steps

### 4. Model Parallelism (if needed)
```python
# For models too large for single GPU
model = torch.nn.DataParallel(model)
```

## 🚨 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
batch_size = 1

# Enable gradient checkpointing
use_gradient_checkpointing = True

# Reduce sequence length
max_seq_length = 1024

# Clear cache
torch.cuda.empty_cache()
```

### Slow Training
```bash
# Enable optimizations
torch.backends.cudnn.benchmark = True

# Use mixed precision
use_mixed_precision = True

# Optimize data loading
num_workers = 2
pin_memory = True
```

### Memory Leaks
```bash
# Monitor memory
nvidia-smi -l 1

# Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()
```

## 📈 Monitoring

### GPU Utilization
```bash
# Real-time monitoring
nvidia-smi -l 1

# Python monitoring
import GPUtil
GPUtil.showUtilization()
```

### Memory Usage
```python
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
```

## 🎯 API Endpoints

### Health Check
```bash
curl http://localhost:8002/health
```

### Chat
```bash
curl -X POST "http://localhost:8002/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain CUDA acceleration", "max_tokens": 200}'
```

### Memory Info
```bash
curl http://localhost:8002/model/memory
```

### Clear Cache
```bash
curl -X POST http://localhost:8002/model/clear_cache
```

## 🔬 Advanced Usage

### Custom Generation
```python
from illuminator_cuda import iLLuMinatorCUDA
from tokenizer import iLLuMinatorTokenizer

model = iLLuMinatorCUDA().cuda()
tokenizer = iLLuMinatorTokenizer()

# Custom generation
input_ids = tokenizer.encode("The future of AI is")
input_tensor = torch.tensor([input_ids]).cuda()

with torch.cuda.amp.autocast():
    generated = model.generate(
        input_tensor,
        max_length=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(generated[0].tolist())
print(response)
```

### Batch Inference
```python
# Process multiple prompts efficiently
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
responses = []

for prompt in prompts:
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).cuda()
    
    with torch.no_grad():
        generated = model.generate(input_tensor, max_length=50)
    
    response = tokenizer.decode(generated[0].tolist())
    responses.append(response)
```

## 💡 Tips for RTX 3070

1. **Start Conservative**: Begin with batch_size=1, sequence_length=512
2. **Monitor Memory**: Use nvidia-smi to watch VRAM usage
3. **Gradient Accumulation**: Use steps=8-16 for effective larger batches
4. **Mixed Precision**: Always enable for RTX 30xx series
5. **Regular Cleanup**: Clear cache every 50-100 steps
6. **Temperature Control**: Keep GPU cool for sustained performance

## 🎉 Expected Results

With proper setup on RTX 3070:
- ✅ 4.9B parameter model fits in memory
- ✅ ~5-10 tokens/second inference speed
- ✅ Stable training without OOM errors
- ✅ Professional-quality text generation
- ✅ API response times under 10 seconds

## 📞 Support

If you encounter issues:
1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Monitor memory: `nvidia-smi -l 1`
4. Reduce batch size or sequence length
5. Enable all memory optimizations
