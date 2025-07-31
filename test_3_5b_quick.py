#!/usr/bin/env python3
"""
3.5B Apple Silicon Model Test
Quick test to verify the model fits and works
"""

import torch
import torch.backends.mps
import psutil
import os
from legacy.illuminator_model import iLLuMinator4_7B
from practical_model.tokenizer import iLLuMinatorTokenizer

class UltraOptimizedModel(torch.nn.Module):
    """Ultra-optimized 3.5B parameter model"""
    
    def __init__(self, 
                 vocab_size: int = 50260,
                 d_model: int = 2816,
                 n_layers: int = 24,
                 n_heads: int = 22,
                 d_ff: int = 11264,
                 max_seq_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        self.model = iLLuMinator4_7B(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
    
    def forward(self, input_ids):
        return self.model(input_ids)

def main():
    """Test the 3.5B model"""
    print("🧪 Testing 3.5B Apple Silicon Model")
    print("="*40)
    
    # Set conservative memory limits
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.6'
    
    # Initial memory
    memory = psutil.virtual_memory()
    print(f"💾 Initial: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
    
    try:
        # Load tokenizer
        print("\n🔤 Loading tokenizer...")
        tokenizer = iLLuMinatorTokenizer()
        
        # Create model
        print("\n🧠 Creating 3.5B model...")
        model = UltraOptimizedModel(
            vocab_size=len(tokenizer),
            d_model=2816,
            n_layers=24,
            n_heads=22,
            d_ff=11264,
            max_seq_length=512
        )
        
        # Clear caches
        torch.mps.empty_cache()
        
        # Move to MPS
        print("\n🚀 Moving to MPS...")
        device = torch.device('mps')
        model = model.to(device)
        print("✅ Model on MPS")
        
        # Test inference
        print("\n🔍 Testing inference...")
        test_tokens = [1, 2, 3, 4, 5]  # Simple test
        input_ids = torch.tensor([test_tokens], device=device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"✅ Inference works!")
        print(f"   Input: {input_ids.shape}")
        print(f"   Output: {outputs.shape}")
        
        # Memory check
        memory = psutil.virtual_memory()
        print(f"\n💾 Final: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
        
        print(f"\n🎉 3.5B Model Test PASSED!")
        print(f"✅ Ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Run: python train_3_5b_apple_silicon.py")
    exit(0 if success else 1)
