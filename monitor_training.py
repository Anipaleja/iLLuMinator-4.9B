#!/usr/bin/env python3

import os
import time
import json
import glob
from datetime import datetime

def monitor_training():
    print("iLLuMinator RTX 3050 Training Monitor")
    print("=" * 50)
    
    # Check for 4.9B training log files
    log_dirs = ["training_logs"]
    latest_log = None
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            log_files = glob.glob(os.path.join(log_dir, "training_4.9B_*.log"))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                break
    
    if latest_log:
        print(f"4.9B Training Log: {os.path.basename(latest_log)}")
        
        # Show last few lines
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("\nRecent 4.9B training progress:")
                    for line in lines[-15:]:
                        if "Step" in line or "Loss" in line or "GPU" in line:
                            print(f"   {line.strip()}")
        except Exception as e:
            print(f"Error reading log: {e}")
    else:
        print("No 4.9B training logs found yet")
    
    # Check for 4.9B checkpoints
    checkpoint_dirs = ["checkpoints_4.9B", "checkpoints"]
    total_checkpoints = 0
    latest_checkpoint = None
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
            if checkpoints:
                total_checkpoints += len(checkpoints)
                latest_dir_checkpoint = max(checkpoints, key=os.path.getctime)
                if latest_checkpoint is None or os.path.getctime(latest_dir_checkpoint) > os.path.getctime(latest_checkpoint):
                    latest_checkpoint = latest_dir_checkpoint
    
    if latest_checkpoint:
        print(f"\nLatest 4.9B checkpoint: {os.path.basename(latest_checkpoint)}")
        print(f"   Total checkpoints: {total_checkpoints}")
        
        # Try to load checkpoint info
        try:
            import torch
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            if 'global_step' in checkpoint:
                print(f"   Training step: {checkpoint['global_step']:,}")
            if 'loss' in checkpoint:
                print(f"   Loss: {checkpoint['loss']:.4f}")
        except:
            pass
    else:
        print("\nNo 4.9B checkpoints found yet")
    
    # Check GPU usage (if nvidia-smi is available)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            if len(gpu_info) >= 4:
                util, mem_used, mem_total, temp = gpu_info
                print(f"\nGPU Status:")
                print(f"   Utilization: {util}%")
                print(f"   Memory: {mem_used}MB / {mem_total}MB ({int(mem_used)/int(mem_total)*100:.1f}%)")
                print(f"   Temperature: {temp}C")
    except Exception:
        print("\nGPU monitoring not available")
    
    # Training recommendations for 4.9B model
    print(f"\n4.9B Model Training Tips:")
    print(f"   - Training ~4.9 billion parameters on RTX 3050")
    print(f"   - Expected training time: 6-12 hours")
    print(f"   - Watch for GPU memory usage under 7.5GB")
    print(f"   - Loss should decrease gradually from ~10 to ~3")
    print(f"   - Model uses high-quality datasets from LLMDataHub")
    print(f"   - Final model saved as 'final_4.9B_model.pt'")
    print(f"   - Mixed precision training enabled for efficiency")
    
    print(f"\nMonitor updated at: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    monitor_training()
