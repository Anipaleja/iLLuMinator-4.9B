#!/usr/bin/env python3
"""
4.2B Memory-Optimized Model Test
Test the memory-optimized version that should fit in 16GB RAM
"""

import torch
import torch.backends.mps
import psutil
import os
from legacy.illuminator_model import iLLuMinator4_7B
from practical_model.tokenizer import iLLuMinatorTokenizer

class MemoryOptimizedModel(torch.nn.Module):
    """Memory-optimized 4.2B parameter model for Apple Silicon"""
    
    def __init__(self, 
                 vocab_size: int = 50260,
                 d_model: int = 3200,      # Reduced from 3584
                 n_layers: int = 28,       # Reduced from 30
                 n_heads: int = 25,        # Reduced from 28
                 d_ff: int = 12800,        # Reduced from 14336
                 max_seq_length: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        # Store config
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Use the existing model class with optimized parameters
        self.model = iLLuMinator4_7B(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Calculate and print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Memory-Optimized iLLuMinator Configuration:")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_params/1e9:.2f}B parameters")
    
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def parameters(self):
        return self.model.parameters()

def test_memory_optimized_model():
    """Test the memory-optimized 4.2B model"""
    print("🧪 Testing Memory-Optimized 4.2B Model")
    print("="*45)
    
    # Check initial memory
    memory = psutil.virtual_memory()
    print(f"💾 Initial Memory: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    # Test MPS
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
    
    print("✅ MPS available")
    
    # Set memory limits
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
    print("🔧 Set MPS memory limit to 70%")
    
    try:
        # Initialize tokenizer
        print("\n🔤 Loading tokenizer...")
        tokenizer = iLLuMinatorTokenizer()
        print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
        
        # Initialize optimized model
        print("\n🧠 Loading 4.2B optimized model...")
        model = MemoryOptimizedModel(
            vocab_size=len(tokenizer),
            d_model=3200,
            n_layers=28,
            n_heads=25,
            d_ff=12800,
            max_seq_length=1024
        )
        
        # Check memory after model creation
        memory = psutil.virtual_memory()
        print(f"💾 Memory after model creation: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
        
        # Try moving to MPS
        print("\n🚀 Moving model to MPS...")
        device = torch.device('mps')
        
        # Clear MPS cache first
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        model = model.to(device)
        print("✅ Model successfully moved to MPS")
        
        # Check memory after moving to MPS
        memory = psutil.virtual_memory()
        print(f"💾 Memory after MPS transfer: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
        
        # Test inference
        print("\n🔍 Testing inference...")
        test_text = "The future of artificial intelligence on Apple Silicon"
        tokens = tokenizer.encode(test_text, max_length=100)
        input_ids = torch.tensor([tokens], device=device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"✅ Inference successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output device: {outputs.device}")
        
        # Final memory check
        memory = psutil.virtual_memory()
        print(f"\n💾 Final Memory: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
        
        # Success!
        print(f"\n🎉 4.2B Model Test PASSED!")
        print(f"   Model fits in memory and runs on MPS")
        print(f"   Ready for training!")
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ Still out of memory: {e}")
            print("💡 Try reducing model size further or use CPU training")
        else:
            print(f"❌ Error: {e}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def suggest_optimizations():
    """Suggest further optimizations if needed"""
    print("\n💡 Memory Optimization Suggestions:")
    print("   1. Close other applications to free RAM")
    print("   2. Reduce sequence length to 256-512 tokens")
    print("   3. Use gradient checkpointing during training")
    print("   4. Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6")
    print("   5. Use mixed precision if available")

def main():
    """Main test function"""
    success = test_memory_optimized_model()
    
    if success:
        print("\n🚀 Ready to start 4.2B training!")
        print("💡 Run: python train_4_2b_optimized.py")
    else:
        print("\n⚠️  Model still too large for available memory")
        suggest_optimizations()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
