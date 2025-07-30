# Training Monitor for iLLuMinator 4.9B
# Monitor training progress and model checkpoints

import os
import json
import time
from pathlib import Path
from datetime import datetime

def check_training_status():
    """Check the current training status"""
    print("iLLuMinator 4.9B Training Monitor")
    print("=" * 50)
    
    # Check checkpoints
    checkpoint_dir = Path("checkpoints_4.9B")
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        print(f"✅ Checkpoints found: {len(checkpoint_files)} files")
        
        for checkpoint in checkpoint_files:
            size_mb = checkpoint.stat().st_size / (1024 * 1024)
            print(f"   📁 {checkpoint.name} ({size_mb:.1f} MB)")
    else:
        print("❌ No checkpoints directory found")
    
    # Check training logs
    log_dir = Path("training_logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        print(f"✅ Training logs found: {len(log_files)} files")
        
        for log_file in log_files:
            size_kb = log_file.stat().st_size / 1024
            print(f"   📄 {log_file.name} ({size_kb:.1f} KB)")
    else:
        print("❌ No training logs directory found")
    
    # Check if training is currently running
    print("\n🔍 Checking for active training processes...")
    
    # Simple check for Python processes
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'train_4_9B_professional.py' in cmdline:
                        python_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print("✅ Training process detected:")
            for proc in python_processes:
                print(f"   🚀 PID {proc['pid']}: {proc['name']}")
        else:
            print("❌ No active training process found")
    except ImportError:
        print("⚠️  psutil not available, cannot check for active processes")
    
    print("\n📊 Summary:")
    print("   - Your 4.9B parameter model is ready for training")
    print("   - Use 'python train_4_9B_professional.py' to start training")
    print("   - Use 'python inference.py' to test trained models")
    print("   - Check 'training_logs/' for detailed progress")

def show_model_info():
    """Show model architecture information"""
    print("\n🏗️  Model Architecture:")
    print("   - Parameters: ~4.8 billion")
    print("   - Layers: 30 transformer blocks")
    print("   - Attention Heads: 28")
    print("   - Embedding Dimension: 3584")
    print("   - Context Length: 1024 tokens")
    print("   - Vocabulary Size: 50,260 tokens")
    print("   - Optimized for: RTX 3050 8GB VRAM")

if __name__ == "__main__":
    check_training_status()
    show_model_info() 