"""
Simple training verification script
Tests the enhanced model and training setup
"""

import sys
import os

def test_model_creation():
    """Test if enhanced model can be created"""
    print("🧪 Testing Enhanced Model Creation...")
    
    try:
        from enhanced_illuminator_4_9b import iLLuMinator4_9B
        
        # Create a small test model
        model = iLLuMinator4_9B(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            d_ff=512,
            max_seq_length=128
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully!")
        print(f"   Parameters: {param_count:,}")
        print(f"   Model dimension: {model.d_model}")
        print(f"   Layers: {len(model.transformer_blocks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_training_script():
    """Test if training script can be imported"""
    print("\n🧪 Testing Training Script Import...")
    
    try:
        from train_illuminator_4_9b import TrainingConfig, AdvancedTextDataset
        
        # Test config creation
        config = TrainingConfig()
        print(f"✅ Training script imported successfully!")
        print(f"   Default batch size: {config.training_config['batch_size']}")
        print(f"   Default learning rate: {config.training_config['learning_rate']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training script import failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n🧪 Testing Dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('psutil', 'psutil')
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            all_good = False
    
    # Test CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA not available - training will use CPU")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        all_good = False
    
    return all_good

def test_config_file():
    """Test if configuration file is valid"""
    print("\n🧪 Testing Configuration File...")
    
    try:
        import json
        
        if not os.path.exists('training_config.json'):
            print("❌ training_config.json not found")
            return False
        
        with open('training_config.json', 'r') as f:
            config = json.load(f)
        
        required_sections = ['model_config', 'training_config', 'data_config', 'system_config']
        
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section: {section}")
                return False
        
        print("✅ Configuration file is valid")
        print(f"   Model dimension: {config['model_config'].get('d_model', 'N/A')}")
        print(f"   Batch size: {config['training_config'].get('batch_size', 'N/A')}")
        print(f"   Max steps: {config['training_config'].get('max_steps', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced iLLuMinator 4.9B Training Verification")
    print("=" * 60)
    
    tests = [
        test_dependencies,
        test_model_creation,
        test_training_script,
        test_config_file
    ]
    
    results = []
    
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    all_passed = all(results)
    
    if all_passed:
        print("✅ All tests passed! Training setup is ready.")
        print("\nTo start training:")
        print("   ./train.sh")
        print("\nFor a dry run test:")
        print("   ./train.sh --dry-run")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Ensure you're in the training directory")
        print("   - Check that parent directory contains enhanced_illuminator_4_9b.py")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
