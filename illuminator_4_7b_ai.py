#!/usr/bin/env python3
"""
iLLuMinator 4.7B AI System
Professional AI assistant powered by a 4.7 billion parameter transformer model
"""

import json
import re
import random
from typing import List, Dict, Optional
import time
import torch
from inference import iLLuMinatorInference

class iLLuMinator4_7BAI:
    """Professional AI assistant using the 4.7B parameter model"""
    
    def __init__(self, model_path: str = None, fast_mode: bool = False):
        self.fast_mode = fast_mode
        self.conversation_history = []
        
        print("Initializing iLLuMinator 4.7B AI System...")
        
        # Initialize the inference engine
        try:
            self.inference_engine = iLLuMinatorInference(model_path=model_path)
            self.model_loaded = True
            print("4.7B model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load 4.7B model: {e}")
            print("Falling back to rule-based responses...")
            self.inference_engine = None
            self.model_loaded = False
        
        # Conversation patterns for when model isn't available
        self.response_patterns = {
            'greeting': [
                "Hello! I'm iLLuMinator, your AI assistant. How can I help you today?",
                "Hi there! I'm ready to assist you with any questions or tasks.",
                "Greetings! I'm iLLuMinator AI. What would you like to work on?",
            ],
            'programming': [
                "I can help you with programming! What language or concept are you working with?",
                "Programming is one of my specialties. What specific coding challenge can I assist with?",
                "Let's code together! What programming problem would you like to solve?",
            ],
            'general': [
                "That's an interesting question! Let me help you with that.",
                "I'd be happy to assist you with that topic.",
                "Let me provide you with some helpful information about that.",
            ]
        }
        
    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = 0.8) -> str:
        """Generate response using the 4.7B model or fallback patterns"""
        
        # Store conversation
        self.conversation_history.append({"role": "user", "content": prompt})
        
        if self.model_loaded and self.inference_engine:
            try:
                # Use the 4.7B model for generation
                response = self._generate_with_model(prompt, max_tokens, temperature)
            except Exception as e:
                print(f"Model generation failed: {e}")
                response = self._generate_fallback_response(prompt)
        else:
            # Use fallback patterns
            response = self._generate_fallback_response(prompt)
        
        # Store response
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _generate_with_model(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using the 4.7B transformer model"""
        
        # Build context from conversation history
        context = ""
        for turn in self.conversation_history[-6:]:  # Last 3 exchanges
            if turn["role"] == "user":
                context += f"Human: {turn['content']}\n"
            else:
                context += f"Assistant: {turn['content']}\n"
        
        # Add current prompt
        full_prompt = f"{context}Human: {prompt}\nAssistant:"
        
        # Generate response
        response = self.inference_engine.generate_text(
            prompt=full_prompt,
            max_length=max_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        # Clean up response
        response = response.strip()
        
        # Remove any "Human:" that might appear in response
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        # Ensure response isn't empty
        if not response:
            response = "I understand your question. Let me think about that and provide a helpful response."
        
        return response
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response using patterns"""
        prompt_lower = prompt.lower()
        
        # Detect response type
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return random.choice(self.response_patterns['greeting'])
        elif any(word in prompt_lower for word in ['code', 'program', 'python', 'javascript', 'function']):
            return self._handle_programming_query(prompt)
        else:
            return self._handle_general_query(prompt)
    
    def _handle_programming_query(self, prompt: str) -> str:
        """Handle programming-related queries"""
        prompt_lower = prompt.lower()
        
        if 'python' in prompt_lower:
            if 'class' in prompt_lower:
                return """To create a Python class:

```python
class MyClass:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f'Hello, {self.name}!'
        
# Usage:
obj = MyClass('World')
print(obj.greet())
```"""
            elif 'function' in prompt_lower:
                return """In Python, functions are defined using the `def` keyword:

```python
def function_name(parameters):
    # Your code here
    return value

# Example:
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
```"""
            elif 'loop' in prompt_lower:
                return """Python loop examples:

```python
# For loop
for i in range(5):
    print(i)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# Loop through list
items = ['apple', 'banana', 'orange']
for item in items:
    print(item)
```"""
        
        return "I can help you with programming! What specific coding challenge are you working on?"
    
    def _handle_general_query(self, prompt: str) -> str:
        """Handle general queries"""
        return f"That's an interesting question about '{prompt[:50]}...'. I'd be happy to help you explore this topic further. What specific aspect would you like to know more about?"
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the model"""
        if self.model_loaded:
            return {
                "model_type": "iLLuMinator 4.7B Transformer",
                "parameters": "4,700,000,000",
                "context_length": 2048,
                "temperature": 0.8,
                "status": "loaded"
            }
        else:
            return {
                "model_type": "iLLuMinator Fallback System",
                "parameters": "Pattern-based",
                "context_length": 1024,
                "temperature": 0.0,
                "status": "fallback"
            }
    
    def chat(self, message: str) -> str:
        """Chat interface"""
        if self.model_loaded and self.inference_engine:
            return self.inference_engine.chat(message)
        else:
            return self.generate_response(message)
    
    def complete_code(self, code_snippet: str) -> str:
        """Code completion"""
        if self.model_loaded and self.inference_engine:
            return self.inference_engine.complete_code(code_snippet)
        else:
            return f"Code completion: {code_snippet}\n# This would be completed by the 4.7B model"

def main():
    """Test the iLLuMinator system"""
    print("Testing iLLuMinator 4.7B AI System")
    print("=" * 50)
    
    # Initialize system
    ai = iLLuMinator4_7BAI()
    
    # Test queries
    test_queries = [
        "Hello, how are you?",
        "What is Python programming?",
        "How do I create a function in Python?",
        "Explain machine learning",
        "What's the weather like?",
    ]
    
    for query in test_queries:
        print(f"\n User: {query}")
        print("iLLuMinator: ", end="")
        
        start_time = time.time()
        response = ai.generate_response(query)
        end_time = time.time()
        
        print(response)
        print(f"Response time: {end_time - start_time:.3f}s")
        print("-" * 50)
    
    # Model info
    print("\nModel Information:")
    info = ai.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
