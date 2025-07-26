#!/usr/bin/env python3
"""
iLLuMinator Practical AI System
Efficient AI assistant using a 120M parameter model that can actually run
"""

import torch
import time
import os
from typing import Dict, Any, Optional
from illuminator_practical import iLLuMinatorPractical
from tokenizer import iLLuMinatorTokenizer

class PracticaliLLuMinatorAI:
    """Practical AI system with efficient inference"""
    
    def __init__(self, model_path: str = None):
        print("🚀 Initializing Practical iLLuMinator AI...")
        
        # Initialize tokenizer
        print("📚 Loading tokenizer...")
        try:
            self.tokenizer = iLLuMinatorTokenizer()
            print(f"✅ Tokenizer loaded with {len(self.tokenizer)} tokens")
        except Exception as e:
            print(f"❌ Tokenizer failed: {e}")
            return
        
        # Initialize model
        print("🧠 Loading practical model...")
        try:
            self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
            
            # Try to load trained weights first
            if model_path:
                self._load_weights(model_path)
            elif os.path.exists("illuminator_practical_weights.pth"):
                self._load_weights("illuminator_practical_weights.pth")
            else:
                print("⚠️  No trained weights found, using randomly initialized weights")
            
            self.model.eval()
            self.model_loaded = True
            print("✅ Practical model ready!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.model_loaded = False
    
    def _load_weights(self, model_path: str):
        """Load trained weights"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded weights from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        """Generate response efficiently"""
        
        if not self.model_loaded:
            return self._fallback_response(prompt)
        
        try:
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt)
            if len(input_ids) > 512:  # Truncate long prompts
                input_ids = input_ids[-512:]
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_length=min(max_tokens, 50),  # Keep generation short
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.tokenizer.pad_token_id
                )
            
            # Decode only the new tokens
            new_tokens = generated[0, len(input_ids):].tolist()
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if not response:
                response = self._fallback_response(prompt)
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback responses when model fails"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm iLLuMinator, ready to help you with questions and coding tasks."
        
        elif any(word in prompt_lower for word in ['python', 'code', 'programming']):
            if 'function' in prompt_lower:
                return "To create a Python function:\n\n```python\ndef my_function(param):\n    # Your code here\n    return result\n```"
            elif 'class' in prompt_lower:
                return "To create a Python class:\n\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```"
            else:
                return "I can help with Python programming! What specific coding question do you have?"
        
        elif '?' in prompt:
            return f"That's an interesting question about {prompt[:30]}... I'd be happy to help explain that topic."
        
        else:
            return "I understand. How can I assist you further with that?"
    
    def chat(self, message: str, max_tokens: int = 80) -> str:
        """Chat interface with context"""
        # Add conversational context
        chat_prompt = f"Human: {message}\nAssistant:"
        
        response = self.generate_response(chat_prompt, max_tokens=max_tokens, temperature=0.7)
        
        # Clean up chat response
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        return response if response else "I'd be happy to help with that."
    
    def complete_code(self, code_snippet: str) -> str:
        """Code completion"""
        code_prompt = f"# Complete this code:\n{code_snippet}"
        
        response = self.generate_response(code_prompt, max_tokens=60, temperature=0.3)
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model_loaded:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            
            return {
                "model_type": "iLLuMinator Practical 120M",
                "parameters": f"{total_params:,}",
                "context_length": self.model.max_seq_length,
                "temperature": 0.8,
                "status": "loaded",
                "device": "cpu"
            }
        else:
            return {
                "model_type": "Fallback System",
                "parameters": "Pattern-based",
                "status": "fallback"
            }
    
    def benchmark_performance(self, num_iterations: int = 10) -> Dict[str, float]:
        """Benchmark generation speed"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        print(f"🔥 Benchmarking {num_iterations} generations...")
        
        prompt = "The future of AI is"
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.generate_response(prompt, max_tokens=20)
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i % 5 == 0:
                print(f"  Completed {i+1}/{num_iterations}")
        
        avg_time = sum(times) / len(times)
        tokens_per_second = 20 / avg_time  # Rough estimate
        
        results = {
            "avg_generation_time": avg_time,
            "tokens_per_second": tokens_per_second,
            "total_iterations": num_iterations,
            "model_size": "120M parameters"
        }
        
        print(f"⚡ Average generation time: {avg_time:.3f}s")
        print(f"⚡ Estimated speed: {tokens_per_second:.1f} tokens/second")
        
        return results

def main():
    """Test the practical system"""
    print("🧪 Testing Practical iLLuMinator AI")
    print("=" * 50)
    
    # Initialize
    ai = PracticaliLLuMinatorAI()
    
    if not ai.model_loaded:
        print("❌ Model failed to load, exiting...")
        return
    
    # Test queries
    test_queries = [
        "Hello, how are you?",
        "What is Python?",
        "How do I create a function?",
        "Explain machine learning",
        "Write a simple loop",
    ]
    
    for query in test_queries:
        print(f"\n🤔 User: {query}")
        print("🤖 iLLuMinator: ", end="")
        
        start_time = time.time()
        response = ai.chat(query)
        end_time = time.time()
        
        print(response)
        print(f"⏱️  ({end_time - start_time:.3f}s)")
        print("-" * 40)
    
    # Benchmark
    print(f"\n📊 Model Info:")
    info = ai.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Quick benchmark
    print(f"\n⚡ Performance Benchmark:")
    results = ai.benchmark_performance(5)
    
    print(f"\n✅ Practical iLLuMinator AI test completed!")

if __name__ == "__main__":
    main()
