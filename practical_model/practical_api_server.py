#!/usr/bin/env python3
"""
Fast API Server for Practical iLLuMinator AI
Efficient server using the 120M parameter model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import time
import asyncio
from datetime import datetime

from practical_ai import PracticaliLLuMinatorAI

# Request models
class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8

class CodeRequest(BaseModel):
    code: str

# Initialize FastAPI
app = FastAPI(
    title="iLLuMinator Practical API",
    description="Efficient AI API using 120M parameter model",
    version="1.0.0"
)

# Global AI instance
ai_system: Optional[PracticaliLLuMinatorAI] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI system on startup"""
    global ai_system
    print("Starting Practical iLLuMinator API Server...")
    
    try:
        ai_system = PracticaliLLuMinatorAI()
        
        if ai_system.model_loaded:
            print("AI system loaded successfully!")
        else:
            print("AI system running in fallback mode")
            
    except Exception as e:
        print(f"Failed to initialize AI system: {e}")
        ai_system = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to iLLuMinator Practical API",
        "version": "1.0.0",
        "model": "120M Parameter Transformer",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    return {
        "status": "healthy",
        "model_loaded": ai_system.model_loaded,
        "timestamp": datetime.now().isoformat(),
        "model_info": ai_system.get_model_info()
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with the AI"""
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    try:
        start_time = time.time()
        
        # Generate response
        response = ai_system.chat(
            message=request.message,
            max_tokens=min(request.max_tokens, 150)  # Limit for speed
        )
        
        end_time = time.time()
        
        return {
            "response": response,
            "generation_time": round(end_time - start_time, 3),
            "model": "iLLuMinator Practical 120M",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/completion")
async def completion_endpoint(request: CompletionRequest):
    """Text completion"""
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    try:
        start_time = time.time()
        
        response = ai_system.generate_response(
            prompt=request.prompt,
            max_tokens=min(request.max_tokens, 100),
            temperature=request.temperature
        )
        
        end_time = time.time()
        
        return {
            "completion": response,
            "generation_time": round(end_time - start_time, 3),
            "model": "iLLuMinator Practical 120M",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")

@app.post("/code")
async def code_completion(request: CodeRequest):
    """Code completion endpoint"""
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    try:
        start_time = time.time()
        
        completion = ai_system.complete_code(request.code)
        
        end_time = time.time()
        
        return {
            "code_completion": completion,
            "generation_time": round(end_time - start_time, 3),
            "model": "iLLuMinator Practical 120M",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code completion failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    return ai_system.get_model_info()

@app.get("/benchmark")
async def benchmark():
    """Run performance benchmark"""
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    if not ai_system.model_loaded:
        return {"error": "Model not loaded, cannot benchmark"}
    
    try:
        results = ai_system.benchmark_performance(5)  # Quick benchmark
        return {
            "benchmark_results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.get("/examples")
async def examples():
    """API usage examples"""
    return {
        "chat_example": {
            "url": "/chat",
            "method": "POST",
            "body": {
                "message": "Hello, how are you?",
                "max_tokens": 100,
                "temperature": 0.8
            }
        },
        "completion_example": {
            "url": "/completion",
            "method": "POST",
            "body": {
                "prompt": "The future of AI is",
                "max_tokens": 50,
                "temperature": 0.7
            }
        },
        "code_example": {
            "url": "/code",
            "method": "POST",
            "body": {
                "code": "def fibonacci(n):\n    # Complete this function"
            }
        }
    }

def main():
    """Run the server"""
    print("Starting iLLuMinator Practical API Server")
    print("Interactive docs will be available at: http://localhost:8001/docs")
    print("Health check: http://localhost:8001/health")
    print("Chat endpoint: http://localhost:8001/chat")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port from the big model
        log_level="info"
    )

if __name__ == "__main__":
    main()
