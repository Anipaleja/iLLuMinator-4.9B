#!/usr/bin/env python3
"""
CUDA-Optimized API Server for iLLuMinator 4.9B
High-performance API server with GPU acceleration and streaming support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, AsyncGenerator
import uvicorn
import torch
import asyncio
import time
import json
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from illuminator_cuda import iLLuMinatorCUDA
from tokenizer import iLLuMinatorTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request models
class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stream: Optional[bool] = False

class ModelConfig(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

# Global variables
model: Optional[iLLuMinatorCUDA] = None
tokenizer: Optional[iLLuMinatorTokenizer] = None
device: torch.device = torch.device("cpu")

class CUDAModelManager:
    """Manages CUDA model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.inference_lock = asyncio.Lock()
        
    async def load_model(self, model_path: Optional[str] = None):
        """Load the CUDA-optimized model"""
        try:
            logger.info("🚀 Loading CUDA-optimized iLLuMinator 4.9B...")
            
            # Load tokenizer
            logger.info("📚 Loading tokenizer...")
            self.tokenizer = iLLuMinatorTokenizer()
            
            # Load model
            logger.info("🧠 Loading CUDA model...")
            self.model = iLLuMinatorCUDA(vocab_size=len(self.tokenizer))
            
            # Load trained weights if available
            if model_path and torch.cuda.is_available():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    logger.info(f"✅ Loaded weights from {model_path}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not load weights: {e}")
            
            # Move to GPU and optimize
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.model = self.model.optimize_for_inference()
                
                # Warm up the model
                logger.info("🔥 Warming up GPU model...")
                await self._warmup_model()
                
                memory_info = self.model.get_memory_usage()
                logger.info(f"📊 GPU Memory: {memory_info['allocated']:.2f}GB allocated")
            
            self.model_loaded = True
            logger.info("✅ Model loaded and ready!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.model_loaded = False
            raise
    
    async def _warmup_model(self):
        """Warm up the model with a dummy forward pass"""
        try:
            dummy_input = torch.randint(0, 1000, (1, 10)).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            logger.info("✅ Model warmed up")
        except Exception as e:
            logger.warning(f"⚠️  Warmup failed: {e}")
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False
    ) -> str:
        """Generate response with CUDA acceleration"""
        
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        async with self.inference_lock:
            try:
                # Encode prompt
                input_ids = self.tokenizer.encode(prompt)
                if len(input_ids) > self.model.max_seq_length - max_tokens:
                    input_ids = input_ids[-(self.model.max_seq_length - max_tokens):]
                
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                
                # Generate with CUDA optimizations
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    generated = self.model.generate(
                        input_tensor,
                        max_length=min(len(input_ids) + max_tokens, self.model.max_seq_length),
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.tokenizer.pad_token_id,
                        use_cache=True
                    )
                
                # Decode only the new tokens
                new_tokens = generated[0, len(input_ids):].tolist()
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                return response.strip()
                
            except torch.cuda.OutOfMemoryError:
                # Clear cache and retry with smaller parameters
                torch.cuda.empty_cache()
                logger.warning("⚠️  CUDA OOM, retrying with reduced parameters")
                
                return await self.generate_response(
                    prompt, 
                    max_tokens=min(max_tokens, 50),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stream=False
                )
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response (placeholder for streaming implementation)"""
        
        # For now, return the full response as a single chunk
        # In a full implementation, you'd modify the model to yield tokens
        response = await self.generate_response(
            prompt, max_tokens, temperature, top_p, top_k, stream=False
        )
        
        # Simulate streaming by yielding words
        words = response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield f" {word}"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model_loaded and self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            memory_info = self.model.get_memory_usage() if torch.cuda.is_available() else {}
            
            return {
                "model_type": "iLLuMinator CUDA 4.9B",
                "parameters": f"{total_params:,}",
                "context_length": self.model.max_seq_length,
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "memory_info": memory_info,
                "status": "loaded",
                "optimizations": [
                    "Mixed Precision",
                    "Flash Attention",
                    "CUDA Kernels",
                    "Gradient Checkpointing"
                ]
            }
        else:
            return {
                "model_type": "iLLuMinator CUDA 4.9B",
                "status": "not_loaded",
                "cuda_available": torch.cuda.is_available()
            }

# Initialize model manager
model_manager = CUDAModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    # Startup
    logger.info("🚀 Starting iLLuMinator CUDA API Server...")
    
    try:
        # Try to load trained weights
        model_path = "illuminator_cuda_weights.pth"
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available, model performance will be limited")
        
        await model_manager.load_model(model_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down iLLuMinator CUDA API Server...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize FastAPI with lifespan
app = FastAPI(
    title="iLLuMinator CUDA API",
    description="High-performance AI API with CUDA acceleration for 4.9B parameter model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to iLLuMinator CUDA API",
        "version": "1.0.0",
        "model": "4.9B Parameter Transformer with CUDA",
        "docs": "/docs",
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1e9:.2f}GB"
        }
    
    return {
        "status": "healthy" if model_manager.model_loaded else "model_not_loaded",
        "model_loaded": model_manager.model_loaded,
        "timestamp": datetime.now().isoformat(),
        "device": str(model_manager.device),
        "cuda_available": torch.cuda.is_available(),
        **memory_info
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with the AI"""
    try:
        start_time = time.time()
        
        # Format as conversation
        chat_prompt = f"Human: {request.message}\nAssistant:"
        
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in model_manager.generate_stream(
                    chat_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k
                ):
                    data = {
                        "chunk": chunk,
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Send end marker
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Regular response
            response = await model_manager.generate_response(
                chat_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
            
            end_time = time.time()
            
            return {
                "response": response,
                "generation_time": round(end_time - start_time, 3),
                "model": "iLLuMinator CUDA 4.9B",
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k
                }
            }
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/completion")
async def completion_endpoint(request: CompletionRequest):
    """Text completion"""
    try:
        start_time = time.time()
        
        if request.stream:
            # Streaming completion
            async def generate():
                async for chunk in model_manager.generate_stream(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k
                ):
                    data = {
                        "completion": chunk,
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            response = await model_manager.generate_response(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
            
            end_time = time.time()
            
            return {
                "completion": response,
                "generation_time": round(end_time - start_time, 3),
                "model": "iLLuMinator CUDA 4.9B",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get detailed model information"""
    return model_manager.get_model_info()

@app.get("/model/memory")
async def memory_info():
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "gpu_name": torch.cuda.get_device_name(),
        "memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
        "memory_cached": f"{torch.cuda.memory_reserved() / 1e9:.2f}GB",
        "memory_free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9:.2f}GB",
        "total_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
        "utilization": f"{torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%"
    }

@app.post("/model/clear_cache")
async def clear_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return {"message": "GPU cache cleared", "timestamp": datetime.now().isoformat()}
    else:
        return {"error": "CUDA not available"}

@app.get("/examples")
async def examples():
    """API usage examples"""
    return {
        "chat_example": {
            "url": "/chat",
            "method": "POST",
            "body": {
                "message": "Explain how GPUs accelerate AI training",
                "max_tokens": 200,
                "temperature": 0.8,
                "stream": False
            }
        },
        "completion_example": {
            "url": "/completion", 
            "method": "POST",
            "body": {
                "prompt": "The benefits of CUDA for deep learning are",
                "max_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9
            }
        },
        "streaming_example": {
            "url": "/chat",
            "method": "POST", 
            "body": {
                "message": "What is transformer architecture?",
                "max_tokens": 200,
                "stream": True
            }
        }
    }

def main():
    """Run the CUDA-optimized server"""
    
    print("🚀 iLLuMinator CUDA API Server")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  CUDA not available - performance will be limited")
    
    print("\n📚 API Documentation: http://localhost:8002/docs")
    print("🔍 Health Check: http://localhost:8002/health")
    print("💬 Chat Endpoint: http://localhost:8002/chat")
    print("📝 Completion: http://localhost:8002/completion")
    print("🧠 Model Info: http://localhost:8002/model/info")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,  # Different port for CUDA API
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
