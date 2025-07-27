#!/usr/bin/env python3
"""
Test script for TinyLlama-inspired iLLuMinator model using correct architecture
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
import os

# Import the actual model class
from illuminator_practical import iLLuMinatorPractical

def test_enhanced_model():
    print("Testing Enhanced iLLuMinator Model")
    print("=" * 50)
    
    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model with correct vocab size
    print("🧠 Creating model...")
    model = iLLuMinatorPractical(vocab_size=len(tokenizer))
    
    # Load trained weights
    print("💾 Loading trained weights...")
    try:
        model_path = 'illuminator_practical_improved_best.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Check vocab size from checkpoint
            if 'vocab_size' in checkpoint:
                trained_vocab_size = checkpoint['vocab_size']
                print(f"📝 Trained vocab size: {trained_vocab_size}")
                print(f"📝 Current vocab size: {len(tokenizer)}")
                
                # Recreate model with correct vocab size
                if trained_vocab_size != len(tokenizer):
                    print(f"🔄 Recreating model with vocab size {trained_vocab_size}")
                    model = iLLuMinatorPractical(vocab_size=trained_vocab_size)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from {model_path}")
                print(f"📊 Model config: {checkpoint.get('model_config', 'N/A')}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Model loaded from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "What is Python?", 
        "Write a simple function.",
        "Explain machine learning in simple terms.",
        "def factorial(n):",
        "How do I learn programming?",
        "What is artificial intelligence?",
        "Write a Python loop."
    ]
    
    print("\n🤖 Testing model responses...")
    print("-" * 50)
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}]")
        print(f"Human: {prompt}")
        
        try:
            # Tokenize input
            inputs = tokenizer(
                prompt, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            
            # Generate response
            with torch.no_grad():
                # Simple generation strategy
                input_ids = inputs['input_ids']
                attention_mask = inputs.get('attention_mask')
                
                generated_ids = input_ids.clone()
                max_new_tokens = 50
                
                for _ in range(max_new_tokens):
                    # Forward pass
                    outputs = model(generated_ids, attention_mask=attention_mask)
                    
                    # Get logits for next token
                    next_token_logits = outputs[:, -1, :]
                    
                    # Apply temperature and sample
                    temperature = 0.8
                    next_token_logits = next_token_logits / temperature
                    
                    # Sample from the distribution
                    probabilities = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                    
                    # Append to generated sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Stop if we hit end of sequence
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    # Update attention mask
                    if attention_mask is not None:
                        attention_mask = torch.cat([
                            attention_mask, 
                            torch.ones((attention_mask.size(0), 1))
                        ], dim=1)
            
            # Decode response
            full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract just the generated part
            if prompt in full_response:
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()
                
            print(f"Assistant: {response}")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params/1e6:.1f}M parameters")

if __name__ == "__main__":
    test_enhanced_model()
