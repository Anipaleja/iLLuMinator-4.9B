#!/usr/bin/env python3
"""
Apple Silicon Training Monitor
Real-time monitoring of training progress and system resources
"""

import psutil
import torch
import time
import os
import json
from datetime import datetime
import threading
import subprocess

class AppleSiliconMonitor:
    """Monitor for Apple Silicon training sessions"""
    
    def __init__(self, log_file: str = "training_monitor.log"):
        self.log_file = log_file
        self.monitoring = False
        self.start_time = None
        
    def get_system_info(self):
        """Get comprehensive system information"""
        # Memory info
        memory = psutil.virtual_memory()
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # Temperature (if available on macOS)
        try:
            temp_output = subprocess.run(
                ['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '-i', '1'],
                capture_output=True, text=True, timeout=5
            )
            # Parse temperature from output (simplified)
            temp_info = "Temperature data requires sudo access"
        except:
            temp_info = "Temperature monitoring not available"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            },
            'cpu': {
                'percent': cpu_percent,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
                'cores': psutil.cpu_count()
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': (disk.used / disk.total) * 100
            },
            'mps_available': torch.backends.mps.is_available(),
            'temperature': temp_info
        }
    
    def log_system_info(self):
        """Log system information to file"""
        info = self.get_system_info()
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(info) + '\n')
    
    def print_system_status(self):
        """Print current system status"""
        info = self.get_system_info()
        
        print("\n" + "="*60)
        print(f"  Apple Silicon System Status - {info['timestamp']}")
        print("="*60)
        
        # Memory
        mem = info['memory']
        print(f"Memory: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['percent']:.1f}%)")
        
        # CPU
        cpu = info['cpu']
        print(f"CPU: {cpu['percent']:.1f}% usage, {cpu['cores']} cores")
        if cpu['frequency_mhz']:
            print(f"    Frequency: {cpu['frequency_mhz']:.0f} MHz")
        
        # Disk
        disk = info['disk']
        print(f"Disk: {disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB ({disk['percent']:.1f}%)")
        print(f"    Free space: {disk['free_gb']:.1f}GB")
        
        # MPS
        mps_status = "Available" if info['mps_available'] else " Not Available"
        print(f"Apple MPS: {mps_status}")
        
        # Training time
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"Training time: {elapsed/3600:.2f} hours")
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        
        def monitor_loop():
            while self.monitoring:
                self.log_system_info()
                self.print_system_status()
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print(f" Monitoring started (interval: {interval}s)")
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        print("Monitoring stopped")
    
    def analyze_training_log(self):
        """Analyze training performance from log"""
        if not os.path.exists(self.log_file):
            print("No log file found")
            return
        
        print("\nTraining Performance Analysis")
        print("="*40)
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print("Empty log file")
                return
            
            # Parse first and last entries
            first_entry = json.loads(lines[0])
            last_entry = json.loads(lines[-1]) if len(lines) > 1 else first_entry
            
            # Memory usage trend
            first_mem = first_entry['memory']['percent']
            last_mem = last_entry['memory']['percent']
            mem_change = last_mem - first_mem
            
            print(f"Memory Usage:")
            print(f"  Start: {first_mem:.1f}%")
            print(f"  Current: {last_mem:.1f}%")
            print(f"  Change: {mem_change:+.1f}%")
            
            # Peak memory usage
            peak_mem = max(json.loads(line)['memory']['percent'] for line in lines)
            print(f"  Peak: {peak_mem:.1f}%")
            
            # Training duration
            start_time = datetime.fromisoformat(first_entry['timestamp'])
            end_time = datetime.fromisoformat(last_entry['timestamp'])
            duration = (end_time - start_time).total_seconds() / 3600
            
            print(f"\nTraining Duration: {duration:.2f} hours")
            print(f"Log entries: {len(lines)}")
            
        except Exception as e:
            print(f"Error analyzing log: {e}")

def main():
    """Main monitoring function"""
    print("Apple Silicon Training Monitor")
    print("="*40)
    
    monitor = AppleSiliconMonitor()
    
    # Check initial system status
    monitor.print_system_status()
    
    # Start monitoring
    try:
        print("\nStarting continuous monitoring...")
        print("Press Ctrl+C to stop and analyze results")
        
        monitor_thread = monitor.start_monitoring(interval=60)  # Every minute
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        monitor.stop_monitoring()
        
        # Analyze results
        monitor.analyze_training_log()
        
        print("\nMonitoring complete!")

if __name__ == "__main__":
    main()
