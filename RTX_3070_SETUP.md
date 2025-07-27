# Nvidia CUDA Deployment Guide

## Quick Setup for Your Desktop

### 1. Clone Repository
```bash
git clone https://github.com/Anipaleja/iLLuMinator-4.7B.git
cd iLLuMinator-4.7B
```

### 2. Verify CUDA Setup
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Should show RTX 3070 and CUDA 12.x
```

### 3. Install CUDA Dependencies
```bash
# Install PyTorch with CUDA support
pip install -r requirements_cuda.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}')"
```

### 4. Test CUDA Model
```bash
# Test the 4.9B parameter model
python illuminator_cuda.py

# Should show:
# GPU: NVIDIA GeForce RTX 3070
# VRAM: 8.0GB
# Model moved to GPU
```

### 5. Start Training (Optional)
```bash
# Train the model on your RTX 3070
python train_cuda.py

# Expected performance:
# - Memory usage: ~6-7GB VRAM
# - Speed: ~0.5-1 tokens/second during training
# - Time: Several hours for meaningful training
```

### 6. Run Production API
```bash
# Start CUDA-optimized API server
python cuda_api_server.py

# Server will be available at:
# http://localhost:8002
# Interactive docs: http://localhost:8002/docs
```

### 7. Test API
```bash
# Health check
curl http://localhost:8002/health

# Chat test
curl -X POST "http://localhost:8002/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain transformer architecture", "max_tokens": 200}'

# Monitor GPU memory
curl http://localhost:8002/model/memory
```

## Optimization Tips for RTX 3070

### Memory Management
- **Training batch size**: Keep at 1
- **Gradient accumulation**: Use 8-16 steps
- **Sequence length**: Start with 1024, increase to 2048 if stable
- **Mixed precision**: Always enabled (automatic)

### Performance Monitoring
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Clear GPU cache if needed
curl -X POST http://localhost:8002/model/clear_cache
```

### Expected Performance
- **Inference**: 5-10 tokens/second
- **Memory usage**: 6-7GB during inference
- **Training**: 0.5-1 tokens/second, ~7GB VRAM
- **API response**: 5-15 seconds for complex queries

## Production Deployment

### Run as Service (Ubuntu/Linux)
```bash
# Create systemd service
sudo nano /etc/systemd/system/illuminator.service
```

```ini
[Unit]
Description=iLLuMinator CUDA API Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/iLLuMinator-4.7B
ExecStart=/usr/bin/python3 cuda_api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable illuminator
sudo systemctl start illuminator
sudo systemctl status illuminator
```

### Docker Deployment (Advanced)
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements_cuda.txt .
RUN pip3 install -r requirements_cuda.txt

COPY . /app
WORKDIR /app

EXPOSE 8002
CMD ["python3", "cuda_api_server.py"]
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in train_cuda.py
batch_size = 1

# Increase gradient accumulation
gradient_accumulation_steps = 16

# Reduce sequence length
max_seq_length = 1024
```

### Slow Performance
```bash
# Check GPU utilization
nvidia-smi

# Ensure TF32 is enabled (automatic on RTX 30xx)
# Check model is on GPU
curl http://localhost:8002/model/info
```

### Driver Issues
```bash
# Update NVIDIA driver
sudo apt update
sudo apt install nvidia-driver-525

# Reboot after driver update
sudo reboot
```

## Monitoring Dashboard

Create a simple monitoring script:

```python
import requests
import time

while True:
    try:
        # Health check
        health = requests.get("http://localhost:8002/health").json()
        memory = requests.get("http://localhost:8002/model/memory").json()
        
        print(f"Status: {health['status']}")
        print(f"GPU Memory: {memory['memory_allocated']}")
        print(f"GPU Util: {memory['utilization']}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(30)
```

> The model is now specifically optimized for Nvidia RTX cards and should run smoothly on your device!
