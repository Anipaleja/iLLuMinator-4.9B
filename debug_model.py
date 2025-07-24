#!/usr/bin/env python3
"""
Debug script to test the model directly
"""

import sys
import torch
from illuminator_ai import IlluminatorAI

def test_model():
    print("Loading iLLuMinator AI model...")
    
    try:
        # Initialize with fast mode
        ai = IlluminatorAI(fast_mode=True, auto_enhance=False)
        
        print(f"Model loaded successfully!")
        print(f"Device: {ai.device}")
        print(f"Parameters: {sum(p.numel() for p in ai.model.parameters()):,}")
        
        # Test basic generation
        test_prompts = [
            "hi",
            "Hello, how are you?",
            "What is Python?",
            "Tell me a joke"
        ]
        
        for prompt in test_prompts:
            print(f"\n--- Testing prompt: '{prompt}' ---")
            try:
                response = ai.generate_response(prompt, max_tokens=100, temperature=0.7)
                print(f"Response: {response}")
                print(f"Response length: {len(response)}")
            except Exception as e:
                print(f"Error generating response: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
