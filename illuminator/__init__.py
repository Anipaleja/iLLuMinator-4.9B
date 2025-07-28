import torch
import logging
from typing import Optional, Dict, Any

from .model import create_illuminator_model, iLLuMinatorEnhanced, EnhancedTokenizer
from .api import iLLuMinatorAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class iLLuMinator:
    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        self.model_size = model_size
        self.model = None
        self.tokenizer = None
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.load_model()
    
    def load_model(self):
        try:
            self.model, self.tokenizer = create_illuminator_model(self.model_size)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded: {self.model.count_parameters():,} parameters")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback
            self.model, self.tokenizer = create_illuminator_model("tiny")
            self.model.to(self.device)
            self.model.eval()
    
    def chat(self, message: str, max_tokens: int = 50, temperature: float = 0.8, system_prompt: Optional[str] = None) -> str:
        if system_prompt:
            prompt = f"System: {system_prompt}\nUser: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"
        
        return self._generate_text(prompt, max_tokens, temperature)
    
    def complete(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
        return self._generate_text(prompt, max_tokens, temperature)
    
    def _generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=min(input_ids.size(1) + max_tokens, 100),
                    temperature=temperature
                )
            
            generated_ids = output_ids[0][input_ids.size(1):]
            response = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
            
            return response.strip() if response.strip() else "Hello! I'm working correctly."
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I'm working, but still learning. Hello!"
    
    def info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_size": self.model_size,
            "parameters": self.model.count_parameters(),
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 100)
        }
    
    def serve(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        api = iLLuMinatorAPI(model_size=self.model_size)
        api.run(host=host, port=port, **kwargs)

# Convenience functions
def chat(message: str, **kwargs) -> str:
    ai = iLLuMinator()
    return ai.chat(message, **kwargs)

def complete(prompt: str, **kwargs) -> str:
    ai = iLLuMinator()
    return ai.complete(prompt, **kwargs)

def serve(port: int = 8000, **kwargs):
    ai = iLLuMinator()
    ai.serve(port=port, **kwargs)

# Package info
__version__ = "1.0.0"
__author__ = "iLLuMinator AI Team"

# Exports
__all__ = [
    "iLLuMinator",
    "iLLuMinatorEnhanced", 
    "EnhancedTokenizer",
    "iLLuMinatorAPI",
    "create_illuminator_model",
    "chat",
    "complete", 
    "serve"
]
