#!/usr/bin/env python3
"""
4.9B Model Pre-Training Test
Quick verification that everything works before full training
"""

import torch
import torch.backends.mps
import psutil
from legacy.illuminator_model import iLLuMinator4_7B
from practical_model.tokenizer import iLLuMinatorTokenizer

def test_mps_functionality():
    """Test MPS backend functionality"""
    print("🧪 Testing Apple Silicon MPS...")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
    
    try:
        # Test basic tensor operations
        device = torch.device('mps')
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        
        print(f"✅ MPS tensor operations successful")
        print(f"   Device: {z.device}")
        print(f"   Shape: {z.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        return False

def test_model_loading():
    """Test model loading and basic forward pass"""
    print("\n🧠 Testing 4.9B model loading...")
    
    try:
        # Initialize tokenizer
        tokenizer = iLLuMinatorTokenizer()
        print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
        
        # Initialize model
        model = iLLuMinator4_7B(
            vocab_size=len(tokenizer),
            d_model=3584,
            n_layers=30,
            n_heads=28,
            d_ff=14336,
            max_seq_length=1024
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded: {total_params:,} parameters ({total_params/1e9:.2f}B)")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def test_model_on_mps(model, tokenizer):
    """Test model inference on MPS"""
    print("\n🚀 Testing model on Apple Silicon MPS...")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available for testing")
        return False
    
    try:
        # Move model to MPS
        device = torch.device('mps')
        model = model.to(device)
        print("✅ Model moved to MPS device")
        
        # Create test input
        test_text = "The future of artificial intelligence"
        tokens = tokenizer.encode(test_text, max_length=100)
        input_ids = torch.tensor([tokens], device=device)
        
        print(f"✅ Test input prepared: {input_ids.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"✅ Forward pass successful: {outputs.shape}")
        print(f"   Output device: {outputs.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ MPS inference failed: {e}")
        return False

def check_memory_requirements():
    """Check if system has enough memory"""
    print("\n💾 Checking memory requirements...")
    
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)
    
    print(f"   Total RAM: {total_gb:.1f}GB")
    print(f"   Available RAM: {available_gb:.1f}GB")
    print(f"   Used RAM: {used_gb:.1f}GB ({memory.percent:.1f}%)")
    
    # Estimate memory needs for 4.9B model
    model_memory_gb = 4.9 * 4 / 1024  # ~19GB for FP32, ~10GB for FP16
    print(f"   Estimated model memory: ~{model_memory_gb:.1f}GB")
    
    if available_gb < 8:
        print("⚠️  Warning: Low available memory for 4.9B model")
        return False
    elif available_gb >= 12:
        print("✅ Sufficient memory available")
        return True
    else:
        print("⚠️  Marginal memory - training may be slow")
        return True

def main():
    """Main test function"""
    print("🧪 iLLuMinator 4.9B Pre-Training Test")
    print("="*45)
    
    # Test 1: MPS functionality
    mps_ok = test_mps_functionality()
    
    # Test 2: Model loading
    model, tokenizer = test_model_loading()
    model_ok = model is not None
    
    # Test 3: Model on MPS
    mps_inference_ok = False
    if mps_ok and model_ok:
        mps_inference_ok = test_model_on_mps(model, tokenizer)
    
    # Test 4: Memory check
    memory_ok = check_memory_requirements()
    
    # Summary
    print("\n" + "="*45)
    print("🎯 Test Results Summary:")
    print(f"   MPS Backend: {'✅ Pass' if mps_ok else '❌ Fail'}")
    print(f"   Model Loading: {'✅ Pass' if model_ok else '❌ Fail'}")
    print(f"   MPS Inference: {'✅ Pass' if mps_inference_ok else '❌ Fail'}")
    print(f"   Memory Check: {'✅ Pass' if memory_ok else '⚠️  Warning'}")
    
    all_tests_pass = mps_ok and model_ok and mps_inference_ok and memory_ok
    
    if all_tests_pass:
        print("\n🎉 All tests passed! Ready for 4.9B training!")
        print("💡 Run: ./start_4_9b_training.sh to begin")
    else:
        print("\n⚠️  Some tests failed. Check issues before training.")
        
        if not mps_ok:
            print("   • MPS backend not working - training will be very slow on CPU")
        if not model_ok:
            print("   • Model loading failed - check dependencies")
        if not mps_inference_ok:
            print("   • MPS inference failed - may have memory issues")
        if not memory_ok:
            print("   • Low memory - consider closing other applications")
    
    return all_tests_pass

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
