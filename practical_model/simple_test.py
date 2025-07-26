#!/usr/bin/env python3
"""
Simple Test Script for iLLuMinator Models
Quick testing without complex interfaces
"""

import torch
import os
from illuminator_practical import iLLuMinatorPractical
from tokenizer import iLLuMinatorTokenizer

def test_model(model_path: str):
    """Test a model with simple prompts"""
    
    print(f"📁 Loading model: {model_path}")
    
    # Load tokenizer
    tokenizer = iLLuMinatorTokenizer()
    
    # Load model
    model = iLLuMinatorPractical(vocab_size=len(tokenizer))
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Model file not found - using untrained model")
    
    model.eval()
    
    # Test prompts
    test_prompts = [
        "Human: Hello\nAssistant:",
        "Human: What is Python?\nAssistant:",
        "Human: How do I create a function?\nAssistant:",
        "Human: What is AI?\nAssistant:",
        "Human: Good morning\nAssistant:",
    ]
    
    print("\n🧪 Testing Model Responses:")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        question = prompt.split("Human: ")[1].split("\nAssistant:")[0]
        print(f"\n{i}. Question: {question}")
        
        # Generate response
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            try:
                generated = model.generate(
                    input_tensor,
                    max_length=len(input_ids) + 30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.tokenizer.pad_token_id
                )
                
                response = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                
                # Extract just the assistant's response
                if "Assistant:" in response:
                    assistant_response = response.split("Assistant:")[-1].strip()
                else:
                    assistant_response = response.strip()
                
                print(f"   Response: {assistant_response}")
                
            except Exception as e:
                print(f"   Error: {e}")

def main():
    """Test available models"""
    print("🎯 iLLuMinator Simple Model Test")
    print("=" * 40)
    
    # Available models in order of preference
    models = [
        ("Final Conservative", "illuminator_practical_final.pth"),
        ("Improved Best", "illuminator_practical_improved_best.pth"),
        ("Improved Final", "illuminator_practical_improved.pth"),
        ("Enhanced Best", "illuminator_practical_enhanced_best.pth"),
        ("Enhanced Final", "illuminator_practical_enhanced.pth"),
        ("Basic Trained", "illuminator_practical_weights.pth"),
    ]
    
    for name, path in models:
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"Testing: {name}")
            print(f"{'='*60}")
            test_model(path)
            break
    else:
        print("❌ No trained models found!")
        print("Available training options:")
        print("  python train_improved.py   # Recommended")
        print("  python train_enhanced.py   # Advanced")
        print("  python train_practical.py  # Basic")

if __name__ == "__main__":
    main()
