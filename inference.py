"""
iLLuMinator 4.7B Inference Engine
High-performance inference for the 4.7 billion parameter model
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import json
import time
from pathlib import Path

from illuminator_model import iLLuMinator4_7B
from tokenizer import iLLuMinatorTokenizer

class iLLuMinatorInference:
    """Inference engine for iLLuMinator 4.7B model"""
    
    def __init__(self, 
                 model_path: str = None,
                 tokenizer_path: str = "./tokenizer",
                 device: str = None):
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🔧 Initializing iLLuMinator on {self.device}")
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        if Path(tokenizer_path).exists():
            self.tokenizer = iLLuMinatorTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = iLLuMinatorTokenizer()
            
        # Initialize model
        print("🧠 Loading model...")
        self.model = iLLuMinator4_7B(vocab_size=len(self.tokenizer))
        
        # Load trained weights if available
        if model_path and Path(model_path).exists():
            self._load_model_weights(model_path)
        else:
            print("⚠️  No trained weights found, using randomly initialized model")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ iLLuMinator ready for inference!")
        
    def _load_model_weights(self, model_path: str):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"✅ Loaded trained weights from {model_path}")
            
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            print("Using randomly initialized weights")
    
    def generate_text(self, 
                     prompt: str,
                     max_length: int = 100,
                     temperature: float = 0.8,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     repetition_penalty: float = 1.1,
                     do_sample: bool = True) -> str:
        """Generate text from a prompt"""
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        
        if input_ids.size(1) >= self.model.max_seq_length:
            print("⚠️  Prompt too long, truncating...")
            input_ids = input_ids[:, :self.model.max_seq_length-max_length]
        
        original_length = input_ids.size(1)
        
        # Generate
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(input_ids[0].tolist()):
                        next_token_logits[0, token_id] /= repetition_penalty
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[0, indices_to_remove] = -float('Inf')
                    
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for end of text token
                if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break
                
                # Stop if we reach max sequence length
                if input_ids.size(1) >= self.model.max_seq_length:
                    break
        
        # Decode generated text
        generated_ids = input_ids[0, original_length:].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def chat(self, message: str, max_length: int = 200, temperature: float = 0.8) -> str:
        """Chat interface with the model"""
        
        # Format as a chat prompt
        prompt = f"Human: {message}\nAssistant:"
        
        response = self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
        
        # Clean up response
        response = response.strip()
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        return response
    
    def complete_code(self, code_prompt: str, max_length: int = 150) -> str:
        """Code completion functionality"""
        
        # Add code-specific prompt formatting
        prompt = f"# Complete this code:\n{code_prompt}"
        
        response = self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=0.3,  # Lower temperature for code
            top_p=0.95,
            do_sample=True
        )
        
        return response
    
    def benchmark_performance(self, num_tokens: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        
        print(f"🔥 Benchmarking performance for {num_tokens} tokens...")
        
        prompt = "The future of artificial intelligence"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        
        # Warmup
        with torch.no_grad():
            self.model(input_ids)
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if input_ids.size(1) >= self.model.max_seq_length:
                    break
        
        end_time = time.time()
        
        total_time = end_time - start_time
        tokens_per_second = num_tokens / total_time
        
        results = {
            'total_time': total_time,
            'tokens_generated': num_tokens,
            'tokens_per_second': tokens_per_second,
            'device': str(self.device)
        }
        
        print(f"⚡ Performance: {tokens_per_second:.2f} tokens/second")
        
        return results

def interactive_chat():
    """Interactive chat interface"""
    
    print("🤖 iLLuMinator 4.7B Interactive Chat")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    # Initialize model
    inference = iLLuMinatorInference()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            print("iLLuMinator: ", end="", flush=True)
            start_time = time.time()
            
            response = inference.chat(user_input)
            
            end_time = time.time()
            print(response)
            print(f"({end_time - start_time:.2f}s)")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

def demo_inference():
    """Demonstration of model capabilities"""
    
    print("🎯 iLLuMinator 4.7B Demo")
    print("=" * 40)
    
    # Initialize model
    inference = iLLuMinatorInference()
    
    # Demo prompts
    demo_prompts = [
        "The future of artificial intelligence is",
        "Python is a programming language that",
        "Machine learning algorithms can",
        "In the year 2030, technology will",
        "The most important skill for programmers is"
    ]
    
    for prompt in demo_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("🤖 Response: ", end="")
        
        response = inference.generate_text(
            prompt=prompt,
            max_length=50,
            temperature=0.8
        )
        
        print(response)
        print("-" * 60)
    
    # Performance benchmark
    print("\n⚡ Performance Benchmark:")
    results = inference.benchmark_performance(50)
    print(f"Generated {results['tokens_generated']} tokens in {results['total_time']:.2f}s")
    print(f"Speed: {results['tokens_per_second']:.2f} tokens/second")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        interactive_chat()
    else:
        demo_inference()
