# Real-time Training Monitor for 4.9B Parameter Model
# Displays live training statistics and system performance

import os
import sys
import time
import json
import psutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TrainingMonitor:
    """Real-time training monitor with system stats"""
    
    def __init__(self):
        self.start_time = time.time()
        self.logs = []
        
        print("🔍 Training Monitor Initialized")
        print("=" * 60)
        self.display_system_info()
        print()
    
    def display_system_info(self):
        """Display system information"""
        print("💻 System Information:")
        print(f"  CPU: {psutil.cpu_count()} cores")
        print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  GPU: Not available or PyTorch not installed")
    
    def monitor_training(self, log_file: str = None):
        """Monitor training progress"""
        if log_file and Path(log_file).exists():
            self.monitor_log_file(log_file)
        else:
            self.monitor_checkpoints()
    
    def monitor_log_file(self, log_file: str):
        """Monitor training from log file"""
        print(f"📊 Monitoring log file: {log_file}")
        print("-" * 60)
        
        with open(log_file, 'r') as f:
            # Read existing logs
            lines = f.readlines()
            for line in lines:
                self.parse_log_line(line.strip())
        
        # Monitor new logs
        f = open(log_file, 'r')
        f.seek(0, 2)  # Go to end of file
        
        while True:
            line = f.readline()
            if line:
                self.parse_log_line(line.strip())
            else:
                time.sleep(1)
    
    def monitor_checkpoints(self):
        """Monitor training by checking checkpoint directories"""
        checkpoint_dirs = [
            "quick_checkpoints",
            "checkpoints_4.9B",
            "training_logs"
        ]
        
        print("📁 Monitoring checkpoint directories...")
        print("-" * 60)
        
        last_checkpoint_count = {}
        
        while True:
            for checkpoint_dir in checkpoint_dirs:
                if Path(checkpoint_dir).exists():
                    checkpoints = list(Path(checkpoint_dir).glob("*.pt"))
                    logs = list(Path(checkpoint_dir).glob("*.log"))
                    
                    current_count = len(checkpoints)
                    if checkpoint_dir not in last_checkpoint_count:
                        last_checkpoint_count[checkpoint_dir] = 0
                    
                    if current_count > last_checkpoint_count[checkpoint_dir]:
                        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                        print(f"💾 New checkpoint: {latest_checkpoint.name}")
                        last_checkpoint_count[checkpoint_dir] = current_count
                        
                        # Try to load and display info
                        self.display_checkpoint_info(latest_checkpoint)
            
            # Display system stats
            self.display_live_stats()
            time.sleep(5)
    
    def parse_log_line(self, line: str):
        """Parse training log line"""
        if "Step" in line and "Loss" in line:
            try:
                # Extract step and loss
                parts = line.split("|")
                step_part = [p for p in parts if "Step" in p][0]
                loss_part = [p for p in parts if "Loss" in p][0]
                
                step = int(step_part.split()[-1])
                loss = float(loss_part.split()[-1])
                
                self.logs.append({
                    'timestamp': datetime.now(),
                    'step': step,
                    'loss': loss
                })
                
                print(f"📈 Step {step:4d} | Loss: {loss:.4f} | Time: {datetime.now().strftime('%H:%M:%S')}")
                
            except:
                pass
    
    def display_checkpoint_info(self, checkpoint_path: Path):
        """Display information about a checkpoint"""
        if not TORCH_AVAILABLE:
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = []
            if 'step' in checkpoint:
                info.append(f"Step: {checkpoint['step']}")
            if 'global_step' in checkpoint:
                info.append(f"Step: {checkpoint['global_step']}")
            if 'loss' in checkpoint:
                info.append(f"Loss: {checkpoint['loss']:.4f}")
            
            if info:
                print(f"  📊 {' | '.join(info)}")
                
        except Exception as e:
            print(f"  ⚠️ Could not load checkpoint: {e}")
    
    def display_live_stats(self):
        """Display live system statistics"""
        current_time = datetime.now().strftime("%H:%M:%S")
        uptime = time.time() - self.start_time
        
        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        stats = [
            f"Time: {current_time}",
            f"Uptime: {uptime/3600:.1f}h",
            f"CPU: {cpu_percent:.1f}%",
            f"RAM: {memory_percent:.1f}%"
        ]
        
        # GPU stats if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_percent = (gpu_memory / gpu_total) * 100
                stats.append(f"GPU: {gpu_percent:.1f}%")
            except:
                pass
        
        print(f"💻 {' | '.join(stats)}")
    
    def generate_report(self):
        """Generate training report"""
        if not self.logs:
            print("📋 No training data to report")
            return
        
        print("\n📊 Training Report")
        print("=" * 50)
        
        first_step = self.logs[0]['step']
        last_step = self.logs[-1]['step']
        first_loss = self.logs[0]['loss']
        last_loss = self.logs[-1]['loss']
        
        print(f"Steps: {first_step} → {last_step} ({last_step - first_step} steps)")
        print(f"Loss: {first_loss:.4f} → {last_loss:.4f} ({last_loss - first_loss:+.4f})")
        
        if last_loss < first_loss:
            improvement = ((first_loss - last_loss) / first_loss) * 100
            print(f"Improvement: {improvement:.1f}% 📈")
        else:
            print("Loss increased - consider adjusting hyperparameters")

def main():
    """Main monitoring function"""
    print("🔍 4.9B Parameter Model Training Monitor")
    print("=" * 60)
    print("Monitoring your RTX 3050 training session...")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    monitor = TrainingMonitor()
    
    try:
        # Check for log files
        log_files = list(Path("training_logs").glob("*.log")) if Path("training_logs").exists() else []
        
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"Found log file: {latest_log}")
            monitor.monitor_training(str(latest_log))
        else:
            print("No log files found, monitoring checkpoints...")
            monitor.monitor_training()
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped")
        monitor.generate_report()
    except Exception as e:
        print(f"❌ Monitor error: {e}")

if __name__ == "__main__":
    main()
