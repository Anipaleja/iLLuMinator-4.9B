# iLLuMinator 4.9B Deployment Guide

Complete guide for deploying iLLuMinator 4.9B in production environments, from local testing to enterprise-scale deployment.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [API Server Setup](#api-server-setup)
- [Load Balancing](#load-balancing)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Performance Optimization](#performance-optimization)
- [Scaling Strategies](#scaling-strategies)

## Deployment Overview

### Deployment Options

| Deployment Type | Use Case | Requirements | Scalability |
|----------------|----------|--------------|-------------|
| **Local** | Development, testing | Single GPU | Low |
| **Single Server** | Small-scale production | Dedicated server | Medium |
| **Docker** | Containerized deployment | Docker, GPU support | Medium |
| **Kubernetes** | Enterprise, auto-scaling | K8s cluster, GPU nodes | High |
| **Cloud** | Managed deployment | AWS/GCP/Azure | Very High |

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   API Gateway   │────│   Model Servers │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │    Monitoring   │    │     Storage     │
                       └─────────────────┘    └─────────────────┘
```

## Local Deployment

### Quick Start

```bash
# 1. Ensure model is trained
python train_professional.py

# 2. Start local API server
python api_server.py --model-path checkpoints/best_model.pt

# 3. Test the deployment
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

### Local Configuration

```python
# local_config.py
DEPLOYMENT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "workers": 1,
    "model_path": "checkpoints/best_model.pt",
    "device": "auto",
    "max_batch_size": 1,
    "timeout": 30
}
```

### Interactive Testing

```bash
# Start interactive client
python interactive_client.py --server-url http://localhost:8000

# Or use the practical model for quick testing
cd practical_model
python simple_test.py
```

## Docker Deployment

### Dockerfile

```dockerfile
# Multi-stage build for optimized image
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Runtime stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /app .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python3", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  illuminator-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/checkpoints/best_model.pt
      - DEVICE=cuda
      - WORKERS=2
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - illuminator-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

### Build and Deploy

```bash
# Build the image
docker build -t illuminator:latest .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs illuminator-api

# Scale the service
docker-compose up -d --scale illuminator-api=3
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

```bash
# Launch EC2 instance (p3.2xlarge or g4dn.xlarge)
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type p3.2xlarge \
    --key-name my-key-pair \
    --security-group-ids sg-0123456789abcdef0 \
    --subnet-id subnet-0123456789abcdef0 \
    --user-data file://setup-script.sh

# Setup script (setup-script.sh)
#!/bin/bash
apt-get update
apt-get install -y docker.io nvidia-docker2
systemctl start docker
systemctl enable docker

# Clone and deploy
git clone https://github.com/Anipaleja/iLLuMinator-4.9B.git
cd iLLuMinator-4.9B
docker-compose up -d
```

#### AWS ECS Deployment

```json
{
  "family": "illuminator-task",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "illuminator-api",
      "image": "your-registry/illuminator:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/checkpoints/best_model.pt"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/illuminator",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ]
    }
  ]
}
```

### Google Cloud Platform

#### GKE Deployment

```yaml
# illuminator-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: illuminator-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: illuminator-api
  template:
    metadata:
      labels:
        app: illuminator-api
    spec:
      containers:
      - name: illuminator
        image: gcr.io/your-project/illuminator:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
        env:
        - name: MODEL_PATH
          value: "/app/checkpoints/best_model.pt"
        volumeMounts:
        - name: model-storage
          mountPath: /app/checkpoints
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      nodeSelector:
        accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: illuminator-service
spec:
  selector:
    app: illuminator-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name illuminator-rg --location eastus

# Deploy container
az container create \
    --resource-group illuminator-rg \
    --name illuminator-api \
    --image your-registry/illuminator:latest \
    --cpu 4 \
    --memory 16 \
    --gpu-count 1 \
    --gpu-sku V100 \
    --ports 8000 \
    --dns-name-label illuminator-api \
    --environment-variables MODEL_PATH=/app/checkpoints/best_model.pt
```

## API Server Setup

### FastAPI Implementation

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from typing import Optional, List
import asyncio
import time

app = FastAPI(title="iLLuMinator 4.9B API", version="1.0.0")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0

class GenerationResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    generation_time: float
    model_version: str

class ModelManager:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model = self.load_model(model_path, device)
        self.tokenizer = self.load_tokenizer()
        self.device = device
        
    def load_model(self, model_path: str, device: str):
        # Model loading implementation
        pass
        
    def load_tokenizer(self):
        # Tokenizer loading implementation
        pass
        
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer.encode(request.prompt)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0])
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=generated_text,
            tokens_generated=len(outputs[0]) - len(inputs),
            generation_time=generation_time,
            model_version="4.9B"
        )

# Global model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    global model_manager
    model_manager = ModelManager("checkpoints/best_model.pt")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        response = await model_manager.generate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_manager is not None}

@app.get("/info")
async def model_info():
    return {
        "model_name": "iLLuMinator 4.9B",
        "version": "1.0.0",
        "parameters": "4.9B",
        "architecture": "Transformer with GQA, RoPE, SwiGLU"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )
```

### Load Balancing with Nginx

```nginx
# nginx.conf
upstream illuminator_backend {
    least_conn;
    server illuminator-api-1:8000 max_fails=3 fail_timeout=30s;
    server illuminator-api-2:8000 max_fails=3 fail_timeout=30s;
    server illuminator-api-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location /generate {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://illuminator_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long generation requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
    
    location /health {
        proxy_pass http://illuminator_backend;
        access_log off;
    }
    
    location / {
        proxy_pass http://illuminator_backend;
    }
}
```

## Monitoring and Observability

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter('illuminator_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('illuminator_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('illuminator_active_connections', 'Active connections')
GPU_MEMORY_USAGE = Gauge('illuminator_gpu_memory_bytes', 'GPU memory usage')
MODEL_INFERENCE_TIME = Histogram('illuminator_inference_duration_seconds', 'Model inference time')

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            REQUEST_COUNT.labels(method=scope["method"], endpoint=scope["path"]).inc()
            ACTIVE_CONNECTIONS.inc()
            
            try:
                await self.app(scope, receive, send)
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.observe(duration)
                ACTIVE_CONNECTIONS.dec()
        else:
            await self.app(scope, receive, send)

# Start metrics server
start_http_server(8001)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "iLLuMinator 4.9B Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(illuminator_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(illuminator_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "illuminator_gpu_memory_bytes / (1024^3)",
            "legendFormat": "GPU Memory (GB)"
          }
        ]
      }
    ]
  }
}
```

## Security Considerations

### API Authentication

```python
# auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import jwt
import os

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    try:
        payload = jwt.decode(
            token.credentials,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Rate Limiting

```python
# rate_limiter.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.state.limiter = limiter
@app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_text(request: Request, data: GenerationRequest):
    # Implementation
    pass
```

### Input Validation

```python
# validation.py
from pydantic import BaseModel, validator
import re

class SecureGenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v) > 1000:
            raise ValueError('Prompt too long')
        
        # Check for potential injection attacks
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*='
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially dangerous content detected')
        
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v > 1000:
            raise ValueError('max_tokens too large')
        return v
```

This deployment guide provides comprehensive coverage of deploying iLLuMinator 4.9B in various environments. Choose the deployment strategy that best fits your use case and scale requirements.