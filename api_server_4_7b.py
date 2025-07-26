#!/usr/bin/env python3
"""
iLLuMinator 4.7B API Server
REST API for the 4.7 billion parameter language model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time
import uvicorn
import torch
import asyncio
from datetime import datetime

from illuminator_4_7b_ai import iLLuMinator4_7BAI

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="The message to send to the AI")
    max_tokens: Optional[int] = Field(150, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.8, description="Sampling temperature")

class ChatResponse(BaseModel):
    response: str
    response_time: float
    tokens_generated: int
    model_info: Dict[str, Any]

class CodeRequest(BaseModel):
    code: str = Field(..., description="Code snippet to complete")
    max_tokens: Optional[int] = Field(100, description="Maximum tokens to generate")

class CodeResponse(BaseModel):
    completion: str
    response_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    system_info: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="iLLuMinator 4.7B API",
    description="REST API for the iLLuMinator 4.7 billion parameter language model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI instance
ai_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI model on startup"""
    global ai_model
    print("🚀 Starting iLLuMinator 4.7B API Server...")
    
    try:
        # Check for trained model
        model_path = "illuminator_4_7b_final.pt"
        ai_model = iLLuMinator4_7BAI(model_path=model_path)
        print("✅ iLLuMinator 4.7B API Server ready!")
    except Exception as e:
        print(f"⚠️  Error initializing model: {e}")
        # Initialize with fallback
        ai_model = iLLuMinator4_7BAI()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "iLLuMinator 4.7B API Server",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=ai_model.model_loaded if ai_model else False,
        timestamp=datetime.now().isoformat(),
        system_info={
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": "3.x",
            "pytorch_version": torch.__version__
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the iLLuMinator 4.7B model"""
    if not ai_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Generate response
        response = ai_model.generate_response(
            request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Estimate token count (rough approximation)
        tokens_generated = len(response.split())
        
        return ChatResponse(
            response=response,
            response_time=response_time,
            tokens_generated=tokens_generated,
            model_info=ai_model.get_model_info()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/complete", response_model=CodeResponse)
async def complete_code(request: CodeRequest):
    """Code completion using the iLLuMinator model"""
    if not ai_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Complete code
        completion = ai_model.complete_code(request.code)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return CodeResponse(
            completion=completion,
            response_time=response_time,
            model_info=ai_model.get_model_info()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code completion failed: {str(e)}")

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get detailed model information"""
    if not ai_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = ai_model.get_model_info()
    
    # Add additional system information
    model_info.update({
        "server_status": "online",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "api_version": "1.0.0"
    })
    
    return model_info

@app.post("/benchmark")
async def benchmark():
    """Benchmark model performance"""
    if not ai_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not ai_model.model_loaded:
        raise HTTPException(status_code=503, detail="4.7B model not loaded, cannot benchmark")
    
    try:
        # Run benchmark
        results = ai_model.inference_engine.benchmark_performance(50)
        return {
            "benchmark_results": results,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

# Example usage endpoints
@app.get("/examples")
async def get_examples():
    """Get example requests for the API"""
    return {
        "chat_example": {
            "endpoint": "/chat",
            "method": "POST",
            "body": {
                "message": "Hello, tell me about artificial intelligence",
                "max_tokens": 150,
                "temperature": 0.8
            }
        },
        "code_example": {
            "endpoint": "/complete",
            "method": "POST", 
            "body": {
                "code": "def fibonacci(n):",
                "max_tokens": 100
            }
        },
        "curl_examples": [
            'curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d \'{"message": "What is machine learning?"}\'',
            'curl -X POST "http://localhost:8000/complete" -H "Content-Type: application/json" -d \'{"code": "class MyClass:"}\''
        ]
    }

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    print(f"🌐 Starting iLLuMinator 4.7B API Server on {host}:{port}")
    
    uvicorn.run(
        "api_server_4_7b:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    start_server(reload=True)
