#!/usr/bin/env python3
"""
Model FLOPs Utilization (MFU) Optimizer for iLLuMinator
Measures and optimizes computational efficiency across Apple Silicon MPS and other devices
"""

import torch
import torch.nn as nn
import torch.backends.mps
import time
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import math
import json
from datetime import datetime


@dataclass
class MFUMetrics:
    """Container for MFU measurement results"""
    theoretical_flops: float
    achieved_flops: float
    mfu_percentage: float
    memory_efficiency: float
    batch_size: int
    sequence_length: int
    model_parameters: int
    device: str
    timestamp: str
    training_time_ms: float
    memory_usage_gb: float
    peak_memory_gb: float


class FLOPsCalculator:
    """Calculate theoretical FLOPs for transformer models"""
    
    @staticmethod
    def transformer_flops(batch_size: int, seq_len: int, vocab_size: int, 
                         d_model: int, n_layers: int, n_heads: int, 
                         d_ff: int, training: bool = True) -> float:
        """
        Calculate FLOPs for transformer model forward pass
        Based on standard transformer FLOP calculations
        """
        # Embedding layer
        embedding_flops = batch_size * seq_len * vocab_size * d_model
        
        # Self-attention layers
        attention_flops = 0
        for _ in range(n_layers):
            # Q, K, V projections
            qkv_flops = 3 * batch_size * seq_len * d_model * d_model
            
            # Attention computation: Q @ K^T
            qk_flops = batch_size * n_heads * seq_len * seq_len * (d_model // n_heads)
            
            # Attention @ V
            av_flops = batch_size * n_heads * seq_len * seq_len * (d_model // n_heads)
            
            # Output projection
            out_proj_flops = batch_size * seq_len * d_model * d_model
            
            # Feed-forward network
            ff_flops = 2 * batch_size * seq_len * d_model * d_ff
            
            attention_flops += qkv_flops + qk_flops + av_flops + out_proj_flops + ff_flops
        
        # Output projection to vocabulary
        output_flops = batch_size * seq_len * d_model * vocab_size
        
        total_forward_flops = embedding_flops + attention_flops + output_flops
        
        # For training, multiply by 3 (forward + backward + optimizer)
        if training:
            total_forward_flops *= 3
        
        return total_forward_flops
    
    @staticmethod
    def calculate_model_flops(model: nn.Module, batch_size: int, seq_len: int, 
                            training: bool = True) -> float:
        """Calculate FLOPs for any model architecture"""
        
        # Try to extract transformer parameters if available
        if hasattr(model, 'model') and hasattr(model.model, 'd_model'):
            # iLLuMinator model structure
            config = model.model
            return FLOPsCalculator.transformer_flops(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=getattr(config, 'vocab_size', 50260),
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                training=training
            )
        elif hasattr(model, 'd_model'):
            # Direct model structure
            return FLOPsCalculator.transformer_flops(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=getattr(model, 'vocab_size', 50260),
                d_model=model.d_model,
                n_layers=getattr(model, 'n_layers', 12),
                n_heads=getattr(model, 'n_heads', 12),
                d_ff=getattr(model, 'd_ff', model.d_model * 4),
                training=training
            )
        else:
            # Fallback: estimate based on parameter count
            param_count = sum(p.numel() for p in model.parameters())
            # Rough estimate: 6 * params * batch_size * seq_len for training
            multiplier = 6 if training else 2
            return multiplier * param_count * batch_size * seq_len


class MFUProfiler:
    """Profile and measure Model FLOPs Utilization"""
    
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        self.flops_calculator = FLOPsCalculator()
        self.measurements: List[MFUMetrics] = []
        
        # Device-specific optimization
        self.is_mps = device == 'mps'
        self.is_cuda = device.startswith('cuda')
        self.is_cpu = device == 'cpu'
        
        print(f"🔬 MFU Profiler initialized for {device}")
        
    def get_memory_info(self) -> Tuple[float, float]:
        """Get current and peak memory usage"""
        if self.is_mps:
            try:
                current_mem = torch.mps.current_allocated_memory() / 1024**3
                # MPS doesn't have direct peak memory, use system memory as approximation
                system_mem = psutil.virtual_memory()
                peak_mem = system_mem.used / 1024**3
                return current_mem, peak_mem
            except:
                # Fallback to system memory
                mem = psutil.virtual_memory()
                return mem.used / 1024**3, mem.used / 1024**3
        elif self.is_cuda:
            current_mem = torch.cuda.memory_allocated() / 1024**3
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            return current_mem, peak_mem
        else:
            # CPU - use system memory
            mem = psutil.virtual_memory()
            return mem.used / 1024**3, mem.used / 1024**3
    
    @contextmanager
    def measure_training_step(self, batch_size: int, seq_len: int):
        """Context manager to measure a single training step"""
        
        # Clear caches and collect garbage
        if self.is_mps:
            torch.mps.empty_cache()
        elif self.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        
        # Reset peak memory tracking
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        
        start_memory, _ = self.get_memory_info()
        start_time = time.perf_counter()
        
        yield
        
        end_time = time.perf_counter()
        current_memory, peak_memory = self.get_memory_info()
        
        # Calculate metrics
        training_time_ms = (end_time - start_time) * 1000
        
        # Calculate theoretical and achieved FLOPs
        theoretical_flops = self.flops_calculator.calculate_model_flops(
            self.model, batch_size, seq_len, training=True
        )
        
        # Achieved FLOPs = theoretical FLOPs / time
        achieved_flops = theoretical_flops / (training_time_ms / 1000) if training_time_ms > 0 else 0
        
        # MFU percentage (achieved / peak device FLOPs)
        peak_device_flops = self.get_peak_device_flops()
        mfu_percentage = (achieved_flops / peak_device_flops * 100) if peak_device_flops > 0 else 0
        
        # Memory efficiency (useful memory / total memory)
        model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3
        memory_efficiency = (model_memory / peak_memory * 100) if peak_memory > 0 else 0
        
        # Create metrics
        metrics = MFUMetrics(
            theoretical_flops=theoretical_flops,
            achieved_flops=achieved_flops,
            mfu_percentage=mfu_percentage,
            memory_efficiency=memory_efficiency,
            batch_size=batch_size,
            sequence_length=seq_len,
            model_parameters=sum(p.numel() for p in self.model.parameters()),
            device=self.device,
            timestamp=datetime.now().isoformat(),
            training_time_ms=training_time_ms,
            memory_usage_gb=current_memory,
            peak_memory_gb=peak_memory
        )
        
        self.measurements.append(metrics)
        return metrics
    
    def get_peak_device_flops(self) -> float:
        """Estimate peak device FLOPs based on device type"""
        if self.is_mps:
            # Apple Silicon estimates (very rough)
            # M1/M2/M3 Pro/Max/Ultra have different capabilities
            # This is a conservative estimate
            return 10e12  # ~10 TFLOPS (very rough estimate)
        elif self.is_cuda:
            # Get GPU info if available
            try:
                gpu_name = torch.cuda.get_device_name(0).lower()
                if 'rtx 4090' in gpu_name:
                    return 165e12  # RTX 4090 ~165 TFLOPS FP16
                elif 'rtx 4080' in gpu_name:
                    return 120e12  # RTX 4080 ~120 TFLOPS FP16
                elif 'rtx 3090' in gpu_name:
                    return 71e12   # RTX 3090 ~71 TFLOPS FP16
                elif 'rtx 3080' in gpu_name:
                    return 58e12   # RTX 3080 ~58 TFLOPS FP16
                elif 'rtx 3070' in gpu_name:
                    return 40e12   # RTX 3070 ~40 TFLOPS FP16
                elif 'rtx 3060' in gpu_name:
                    return 25e12   # RTX 3060 ~25 TFLOPS FP16
                else:
                    return 20e12   # Generic GPU estimate
            except:
                return 20e12
        else:
            # CPU - much lower FLOPs
            cpu_count = psutil.cpu_count()
            return cpu_count * 100e9  # ~100 GFLOPS per core (very rough)
    
    def get_average_mfu(self) -> float:
        """Get average MFU across all measurements"""
        if not self.measurements:
            return 0.0
        return sum(m.mfu_percentage for m in self.measurements) / len(self.measurements)
    
    def get_peak_mfu(self) -> float:
        """Get peak MFU measurement"""
        if not self.measurements:
            return 0.0
        return max(m.mfu_percentage for m in self.measurements)
    
    def print_summary(self):
        """Print MFU measurement summary"""
        if not self.measurements:
            print("❌ No MFU measurements available")
            return
        
        avg_mfu = self.get_average_mfu()
        peak_mfu = self.get_peak_mfu()
        latest = self.measurements[-1]
        
        print(f"\n📊 MFU Performance Summary")
        print(f"=" * 50)
        print(f"🎯 Average MFU: {avg_mfu:.2f}%")
        print(f"🚀 Peak MFU: {peak_mfu:.2f}%")
        print(f"📏 Model Parameters: {latest.model_parameters:,}")
        print(f"🖥️  Device: {latest.device}")
        print(f"💾 Memory Efficiency: {latest.memory_efficiency:.2f}%")
        print(f"⏱️  Last Step Time: {latest.training_time_ms:.2f}ms")
        print(f"🔢 FLOPS: {latest.achieved_flops/1e12:.2f} TFLOPS")
        
        # Performance analysis
        if avg_mfu > 50:
            print("✅ Excellent MFU! Model is well optimized.")
        elif avg_mfu > 30:
            print("🟡 Good MFU. Some optimization opportunities remain.")
        elif avg_mfu > 15:
            print("🟠 Moderate MFU. Consider optimization strategies.")
        else:
            print("🔴 Low MFU. Significant optimization needed.")
    
    def save_measurements(self, filepath: str):
        """Save measurements to JSON file"""
        data = {
            'device': self.device,
            'summary': {
                'average_mfu': self.get_average_mfu(),
                'peak_mfu': self.get_peak_mfu(),
                'total_measurements': len(self.measurements)
            },
            'measurements': [
                {
                    'theoretical_flops': m.theoretical_flops,
                    'achieved_flops': m.achieved_flops,
                    'mfu_percentage': m.mfu_percentage,
                    'memory_efficiency': m.memory_efficiency,
                    'batch_size': m.batch_size,
                    'sequence_length': m.sequence_length,
                    'model_parameters': m.model_parameters,
                    'device': m.device,
                    'timestamp': m.timestamp,
                    'training_time_ms': m.training_time_ms,
                    'memory_usage_gb': m.memory_usage_gb,
                    'peak_memory_gb': m.peak_memory_gb
                }
                for m in self.measurements
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 MFU measurements saved to {filepath}")


class MFUOptimizer:
    """Optimize training for maximum MFU"""
    
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        self.profiler = MFUProfiler(model, device)
        
    def find_optimal_batch_size(self, tokenizer, max_batch_size: int = 16, 
                              seq_len: int = 512) -> int:
        """Find optimal batch size for maximum MFU"""
        print(f"🔍 Finding optimal batch size (max: {max_batch_size})")
        
        best_mfu = 0
        best_batch_size = 1
        
        for batch_size in [1, 2, 4, 8, 16]:
            if batch_size > max_batch_size:
                break
                
            try:
                # Create dummy batch
                dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
                dummy_labels = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
                
                # Simulate training step
                self.model.train()
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                with self.profiler.measure_training_step(batch_size, seq_len) as metrics:
                    optimizer.zero_grad()
                    outputs = self.model(dummy_input)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), dummy_labels.view(-1))
                    loss.backward()
                    optimizer.step()
                
                current_mfu = metrics.mfu_percentage
                print(f"   Batch size {batch_size}: {current_mfu:.2f}% MFU")
                
                if current_mfu > best_mfu:
                    best_mfu = current_mfu
                    best_batch_size = batch_size
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   Batch size {batch_size}: OOM")
                    break
                else:
                    raise e
        
        print(f"✅ Optimal batch size: {best_batch_size} (MFU: {best_mfu:.2f}%)")
        return best_batch_size
    
    def optimize_for_device(self) -> Dict[str, Any]:
        """Apply device-specific optimizations"""
        optimizations = {}
        
        if self.device == 'mps':
            # Apple Silicon MPS optimizations
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.9'
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.8'
            optimizations['mps_memory_fraction'] = 0.9
            print("🍎 Applied Apple Silicon MPS optimizations")
            
        elif self.device.startswith('cuda'):
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations['cudnn_benchmark'] = True
            optimizations['tf32_enabled'] = True
            print("🚀 Applied CUDA optimizations")
            
        elif self.device == 'cpu':
            # CPU optimizations
            torch.set_num_threads(psutil.cpu_count())
            optimizations['cpu_threads'] = psutil.cpu_count()
            print("🧠 Applied CPU optimizations")
        
        return optimizations
    
    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on MFU measurements"""
        suggestions = []
        avg_mfu = self.profiler.get_average_mfu()
        
        if avg_mfu < 15:
            suggestions.extend([
                "Consider reducing model size or sequence length",
                "Increase batch size if memory allows",
                "Enable mixed precision training",
                "Use gradient checkpointing for memory efficiency"
            ])
        elif avg_mfu < 30:
            suggestions.extend([
                "Optimize batch size with grid search",
                "Consider using gradient accumulation",
                "Profile memory usage patterns"
            ])
        elif avg_mfu < 50:
            suggestions.extend([
                "Fine-tune learning rate and optimizer settings",
                "Consider model parallelism for larger models"
            ])
        else:
            suggestions.append("Excellent performance! Model is well optimized.")
        
        return suggestions


def benchmark_mfu(model: nn.Module, tokenizer, device: str, 
                  batch_sizes: List[int] = [1, 2, 4, 8],
                  seq_lengths: List[int] = [256, 512, 1024]) -> MFUProfiler:
    """Comprehensive MFU benchmark"""
    print(f"🏁 Starting MFU benchmark on {device}")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Sequence lengths: {seq_lengths}")
    
    profiler = MFUProfiler(model, device)
    optimizer = MFUOptimizer(model, device)
    
    # Apply device optimizations
    optimizer.optimize_for_device()
    
    # Test different configurations
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            try:
                print(f"\n🧪 Testing batch_size={batch_size}, seq_len={seq_len}")
                
                # Create test data
                dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                dummy_labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                
                # Training step
                model.train()
                test_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                with profiler.measure_training_step(batch_size, seq_len) as metrics:
                    test_optimizer.zero_grad()
                    outputs = model(dummy_input)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), dummy_labels.view(-1))
                    loss.backward()
                    test_optimizer.step()
                
                print(f"   MFU: {metrics.mfu_percentage:.2f}%")
                print(f"   Time: {metrics.training_time_ms:.2f}ms")
                print(f"   Memory: {metrics.memory_usage_gb:.2f}GB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   OOM at batch_size={batch_size}, seq_len={seq_len}")
                    break
                else:
                    print(f"   Error: {e}")
                    continue
    
    # Print summary and suggestions
    profiler.print_summary()
    
    suggestions = optimizer.suggest_improvements()
    if suggestions:
        print(f"\n💡 Optimization Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    
    return profiler


if __name__ == "__main__":
    print("🚀 MFU Optimizer Test")
    
    # Test with a simple model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Create a test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 512
            self.n_layers = 6
            self.n_heads = 8
            self.d_ff = 2048
            self.vocab_size = 10000
            
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    self.d_model, self.n_heads, 
                    dim_feedforward=self.d_ff,
                    batch_first=True
                ),
                num_layers=self.n_layers
            )
            self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            return self.lm_head(x)
    
    model = TestModel().to(device)
    
    # Run benchmark
    profiler = benchmark_mfu(
        model=model,
        tokenizer=None,
        device=device,
        batch_sizes=[1, 2],
        seq_lengths=[256, 512]
    )
    
    # Save results
    profiler.save_measurements("mfu_benchmark_results.json")
