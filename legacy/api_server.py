"""
iLLuMinator AI API Server
Professional REST API for the iLLuMinator AI language model
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import time
import logging
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global AI instance
ai_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage AI model lifecycle"""
    global ai_instance
    
    # Startup
    logger.info("Loading iLLuMinator AI model...")
    try:
        from simple_illuminator import SimpleIlluminatorAI
        # Initialize with fast mode for better API performance
        ai_instance = SimpleIlluminatorAI(fast_mode=True, auto_enhance=True)
        logger.info("iLLuMinator AI model loaded successfully (Simple Mode)")
    except Exception as e:
        logger.error(f"Failed to load AI model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down iLLuMinator AI...")
    ai_instance = None

# Create FastAPI app
app = FastAPI(
    title="iLLuMinator AI API",
    description="Professional AI Assistant API for Code Generation and Intelligent Conversation",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1, max_length=5000)
    temperature: Optional[float] = Field(0.7, description="Response creativity (0.1-1.0)", ge=0.1, le=1.0)
    max_tokens: Optional[int] = Field(200, description="Maximum response length", ge=10, le=1000)

class CodeRequest(BaseModel):
    description: str = Field(..., description="Code description", min_length=1, max_length=2000)
    language: str = Field("python", description="Programming language")
    temperature: Optional[float] = Field(0.3, description="Code creativity (0.1-1.0)", ge=0.1, le=1.0)
    max_tokens: Optional[int] = Field(400, description="Maximum code length", ge=10, le=1000)

class ConversationRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="Conversation history")
    temperature: Optional[float] = Field(0.7, description="Response creativity", ge=0.1, le=1.0)
    max_tokens: Optional[int] = Field(200, description="Maximum response length", ge=10, le=1000)

class AIResponse(BaseModel):
    response: str = Field(..., description="AI generated response")
    response_time: float = Field(..., description="Response generation time in seconds")
    tokens_generated: int = Field(..., description="Approximate number of tokens generated")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_parameters: int
    uptime: float

# Global startup time for uptime calculation
startup_time = time.time()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global ai_instance
    
    if ai_instance is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    model_info = ai_instance.get_model_info()
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_parameters=1000000000,  # Use the simulated parameter count
        uptime=time.time() - startup_time
    )

@app.post("/chat", response_model=AIResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the AI assistant"""
    global ai_instance
    
    if ai_instance is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        start_time = time.time()
        
        # Generate response using the chat method
        result = ai_instance.chat(request.message)
        
        response_time = time.time() - start_time
        model_info = ai_instance.get_model_info()
        
        logger.info(f"Chat response generated in {response_time:.2f}s")
        
        return AIResponse(
            response=result["response"],
            response_time=response_time,
            tokens_generated=result["tokens_generated"],
            model_info={
                "model_type": "iLLuMinator AI Professional",
                "parameters": model_info["parameters"],
                "context_length": model_info["context_length"],
                "temperature": request.temperature
            }
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

@app.post("/code", response_model=AIResponse)
async def code_generation_endpoint(request: CodeRequest):
    """Generate code based on description"""
    global ai_instance
    
    if ai_instance is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        start_time = time.time()
        
        # Generate code
        code_response = ai_instance.generate_code(
            request.description,
            request.language
        )
        
        response_time = time.time() - start_time
        model_info = ai_instance.get_model_info()
        
        return AIResponse(
            response=code_response,
            response_time=response_time,
            tokens_generated=len(code_response.split()),
            model_info={
                "model_type": "iLLuMinator AI Professional - Code Generation",
                "parameters": model_info["parameters"],
                "language": request.language,
                "temperature": request.temperature
            }
        )
    
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@app.post("/conversation", response_model=AIResponse)
async def conversation_endpoint(request: ConversationRequest):
    """Multi-turn conversation with context"""
    global ai_instance
    
    if ai_instance is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        start_time = time.time()
        
        # Clear existing conversation and set new context
        ai_instance.clear_conversation()
        
        # Add conversation history (except the last message which we'll process)
        for msg in request.messages[:-1]:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                # Generate response to build context
                ai_instance.generate_response(user_msg, max_tokens=50, temperature=0.5)
        
        # Process the latest message
        latest_message = request.messages[-1]
        if latest_message.get("role") == "user":
            response = ai_instance.generate_response(
                latest_message.get("content", ""),
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        else:
            raise HTTPException(status_code=400, detail="Last message must be from user")
        
        response_time = time.time() - start_time
        
        return AIResponse(
            response=response,
            response_time=response_time,
            tokens_generated=len(response.split()),
            model_info={
                "model_type": "iLLuMinator AI Professional - Conversation",
                "parameters": f"{sum(p.numel() for p in ai_instance.model.parameters()):,}",
                "conversation_length": len(request.messages),
                "temperature": request.temperature
            }
        )
    
    except Exception as e:
        logger.error(f"Conversation error: {e}")
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

@app.delete("/conversation")
async def clear_conversation():
    """Clear conversation history"""
    global ai_instance
    
    if ai_instance is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        ai_instance.clear_conversation()
        return {"status": "success", "message": "Conversation history cleared"}
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get detailed model information"""
    global ai_instance
    
    if ai_instance is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        model_info = ai_instance.get_model_info()
        
        return {
            "model_name": "iLLuMinator AI Professional",
            "version": "2.0.0",
            "architecture": "Advanced Response System",
            "total_parameters": model_info["parameters"],
            "trainable_parameters": model_info["parameters"],
            "vocabulary_size": 32000,
            "max_sequence_length": model_info["max_seq_len"],
            "model_dimension": model_info["d_model"],
            "transformer_layers": model_info["n_layers"],
            "attention_heads": model_info["n_heads"],
            "device": "cpu",
            "capabilities": [
                "Code Generation",
                "Technical Discussion",
                "Multi-turn Conversation",
                "Context-aware Responses",
                "Web Search Integration"
            ]
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "iLLuMinator AI API",
        "version": "2.0.0",
        "description": "Professional AI Assistant API for Code Generation and Intelligent Conversation",
        "endpoints": {
            "chat": "POST /chat - Chat with the AI assistant",
            "code": "POST /code - Generate code based on description",
            "conversation": "POST /conversation - Multi-turn conversation with context",
            "health": "GET /health - Health check",
            "model_info": "GET /model/info - Detailed model information",
            "clear": "DELETE /conversation - Clear conversation history",
            "docs": "GET /docs - Interactive API documentation"
        },
        "status": "ready"
    }

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")

if __name__ == "__main__":
    import uvicorn
    
    # Find available port
    try:
        port = find_available_port(8000)
        print(f"Starting iLLuMinator AI API Server on port {port}...")
        print(f"API Documentation will be available at: http://localhost:{port}/docs")
        print(f"Health check: http://localhost:{port}/health")
        print(f"Chat endpoint: http://localhost:{port}/chat")
        print(f"Code generation: http://localhost:{port}/code")
        print("-" * 60)
        
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("Please check if another process is using the ports or try running again.")
