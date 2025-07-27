#!/usr/bin/env python3
"""
Test script for TinyLlama-inspired iLLuMinator model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
import os

# Simple model architecture (matching training script)
class PracticalModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        # Create causal mask for decoder
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Transformer expects (seq_len, batch, hidden_size) but we use batch_first=True
        x = self.transformer(x, x, tgt_mask=causal_mask)
        return self.lm_head(x)
    
    def generate(self, input_ids, attention_mask=None, max_length=100, temperature=0.7, do_sample=True, pad_token_id=50256):
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(generated)
                next_token_logits = outputs[:, -1, :] / temperature
                
                if do_sample:
                    # Sample from the distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we generate the pad token
                if next_token.item() == pad_token_id:
                    break
                    
        return generated

def test_model():
    print("🎯 Testing TinyLlama-inspired iLLuMinator Model")
    print("=" * 50)
    
    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create and load model
    print("🧠 Loading TinyLlama-inspired model...")
    model = PracticalModel()
    
    try:
        model_path = 'illuminator_tinyllama_inspired_best.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
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
        "How do I learn programming?"
    ]
    
    print("\n🤖 Testing model responses...")
    print("-" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/6]")
        print(f"Human: {prompt}")
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_length=min(100, inputs['input_ids'].size(1) + 50),
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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

if __name__ == "__main__":
    test_model()
