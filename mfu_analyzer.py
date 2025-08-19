#!/usr/bin/env python3
"""
MFU Analysis and Optimization Tool for iLLuMinator Models
Comprehensive analysis of Model FLOPs Utilization and optimization recommendations
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import psutil
from dataclasses import dataclass
from datetime import datetime

from legacy.illuminator_model import iLLuMinator4_7B
from practical_model.tokenizer import iLLuMinatorTokenizer
from mfu_optimizer import MFUProfiler, MFUOptimizer, benchmark_mfu, FLOPsCalculator


@dataclass
class ModelProfile:
    """Complete model profile for MFU analysis"""
    name: str
    parameters: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    max_seq_length: int
    memory_footprint_gb: float
    theoretical_peak_flops: float
    device_compatibility: Dict[str, bool]


class MFUAnalyzer:
    """Comprehensive MFU analysis tool"""
    
    def __init__(self):
        self.device = self.select_best_device()
        self.flops_calculator = FLOPsCalculator()
        self.model_profiles = []
        
        print(f" MFU Analyzer initialized on {self.device}")
    
    def select_best_device(self) -> str:
        """Select the best available device"""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def profile_model(self, model: nn.Module, name: str) -> ModelProfile:
        """Create comprehensive model profile"""
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Extract model configuration
        if hasattr(model, 'model'):
            config = model.model
            d_model = config.d_model
            n_layers = config.n_layers
            n_heads = config.n_heads
            d_ff = config.d_ff
            vocab_size = getattr(config, 'vocab_size', 50260)
            max_seq_length = getattr(config, 'max_seq_length', 1024)
        else:
            # Fallback values
            d_model = getattr(model, 'd_model', 512)
            n_layers = getattr(model, 'n_layers', 12)
            n_heads = getattr(model, 'n_heads', 8)
            d_ff = getattr(model, 'd_ff', d_model * 4)
            vocab_size = getattr(model, 'vocab_size', 50260)
            max_seq_length = getattr(model, 'max_seq_length', 1024)
        
        # Estimate memory footprint
        memory_footprint = self.estimate_memory_footprint(model)
        
        # Calculate theoretical peak FLOPs for standard batch
        theoretical_flops = self.flops_calculator.transformer_flops(
            batch_size=1, seq_len=512, vocab_size=vocab_size,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, training=True
        )
        
        # Test device compatibility
        device_compatibility = self.test_device_compatibility(model)
        
        profile = ModelProfile(
            name=name,
            parameters=total_params,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            memory_footprint_gb=memory_footprint,
            theoretical_peak_flops=theoretical_flops,
            device_compatibility=device_compatibility
        )
        
        self.model_profiles.append(profile)
        return profile
    
    def estimate_memory_footprint(self, model: nn.Module) -> float:
        """Estimate model memory footprint in GB"""
        total_bytes = 0
        
        for param in model.parameters():
            # Parameter memory (FP32)
            param_bytes = param.numel() * 4
            
            # Gradient memory (FP32)
            grad_bytes = param.numel() * 4
            
            # Optimizer state (AdamW: 2x parameters for momentum and variance)
            optimizer_bytes = param.numel() * 8
            
            total_bytes += param_bytes + grad_bytes + optimizer_bytes
        
        # Add activation memory estimate (rough)
        activation_bytes = total_bytes * 0.5
        
        return (total_bytes + activation_bytes) / (1024 ** 3)
    
    def test_device_compatibility(self, model: nn.Module) -> Dict[str, bool]:
        """Test model compatibility with different devices"""
        compatibility = {}
        
        # Test CPU
        try:
            test_model = type(model)(**model.config if hasattr(model, 'config') else {})
            test_input = torch.randint(0, 1000, (1, 64))
            with torch.no_grad():
                _ = test_model(test_input)
            compatibility['cpu'] = True
        except:
            compatibility['cpu'] = False
        
        # Test MPS
        if torch.backends.mps.is_available():
            try:
                test_model = test_model.to('mps')
                test_input = test_input.to('mps')
                with torch.no_grad():
                    _ = test_model(test_input)
                compatibility['mps'] = True
            except:
                compatibility['mps'] = False
        else:
            compatibility['mps'] = False
        
        # Test CUDA
        if torch.cuda.is_available():
            try:
                test_model = test_model.to('cuda')
                test_input = test_input.to('cuda')
                with torch.no_grad():
                    _ = test_model(test_input)
                compatibility['cuda'] = True
            except:
                compatibility['cuda'] = False
        else:
            compatibility['cuda'] = False
        
        return compatibility
    
    def analyze_mfu_potential(self, profile: ModelProfile) -> Dict[str, Any]:
        """Analyze MFU potential for a model profile"""
        
        # Get system specs
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        # Calculate theoretical limits
        memory_limit_gb = memory.total / (1024 ** 3)
        fits_in_memory = profile.memory_footprint_gb < memory_limit_gb * 0.8
        
        # Optimal configurations
        optimal_configs = self.suggest_optimal_configs(profile, memory_limit_gb)
        
        # MFU predictions
        mfu_predictions = self.predict_mfu_performance(profile)
        
        return {
            'profile': profile,
            'fits_in_memory': fits_in_memory,
            'memory_utilization': profile.memory_footprint_gb / memory_limit_gb,
            'optimal_configs': optimal_configs,
            'mfu_predictions': mfu_predictions,
            'bottlenecks': self.identify_bottlenecks(profile),
            'recommendations': self.generate_recommendations(profile)
        }
    
    def suggest_optimal_configs(self, profile: ModelProfile, memory_limit_gb: float) -> List[Dict[str, Any]]:
        """Suggest optimal training configurations"""
        configs = []
        
        # Calculate maximum batch sizes for different sequence lengths
        for seq_len in [256, 512, 1024, 2048]:
            max_batch_size = self.calculate_max_batch_size(profile, seq_len, memory_limit_gb)
            
            if max_batch_size > 0:
                # Estimate MFU for this configuration
                estimated_mfu = self.estimate_mfu_for_config(profile, max_batch_size, seq_len)
                
                configs.append({
                    'sequence_length': seq_len,
                    'max_batch_size': max_batch_size,
                    'estimated_mfu': estimated_mfu,
                    'memory_usage_gb': self.estimate_config_memory(profile, max_batch_size, seq_len)
                })
        
        # Sort by estimated MFU
        configs.sort(key=lambda x: x['estimated_mfu'], reverse=True)
        return configs
    
    def calculate_max_batch_size(self, profile: ModelProfile, seq_len: int, memory_limit_gb: float) -> int:
        """Calculate maximum batch size for given sequence length"""
        
        # Base model memory
        base_memory_gb = profile.memory_footprint_gb
        
        # Activation memory per sample (rough estimate)
        activation_per_sample = (profile.d_model * seq_len * profile.n_layers * 4) / (1024 ** 3)
        
        # Available memory for activations
        available_memory = memory_limit_gb * 0.8 - base_memory_gb
        
        if available_memory <= 0:
            return 0
        
        max_batch_size = int(available_memory / activation_per_sample)
        return max(1, max_batch_size)
    
    def estimate_config_memory(self, profile: ModelProfile, batch_size: int, seq_len: int) -> float:
        """Estimate memory usage for specific configuration"""
        base_memory = profile.memory_footprint_gb
        activation_memory = (profile.d_model * seq_len * batch_size * profile.n_layers * 4) / (1024 ** 3)
        return base_memory + activation_memory
    
    def estimate_mfu_for_config(self, profile: ModelProfile, batch_size: int, seq_len: int) -> float:
        """Estimate MFU for specific configuration"""
        
        # Calculate FLOPs for this configuration
        flops = self.flops_calculator.transformer_flops(
            batch_size=batch_size, seq_len=seq_len,
            vocab_size=profile.vocab_size, d_model=profile.d_model,
            n_layers=profile.n_layers, n_heads=profile.n_heads,
            d_ff=profile.d_ff, training=True
        )
        
        # Rough estimation based on batch size and sequence length efficiency
        # Larger batches and sequences generally give better MFU
        batch_efficiency = min(batch_size / 8, 1.0)  # Optimal around batch size 8
        seq_efficiency = min(seq_len / 1024, 1.0)    # Optimal around 1024 tokens
        
        # Device-specific efficiency
        if self.device == 'mps':
            device_efficiency = 0.6  # Apple Silicon efficiency
        elif self.device == 'cuda':
            device_efficiency = 0.8  # CUDA efficiency
        else:
            device_efficiency = 0.3  # CPU efficiency
        
        # Combine factors
        estimated_mfu = batch_efficiency * seq_efficiency * device_efficiency * 50  # Scale to percentage
        
        return min(estimated_mfu, 95)  # Cap at 95%
    
    def predict_mfu_performance(self, profile: ModelProfile) -> Dict[str, float]:
        """Predict MFU performance for different scenarios"""
        
        scenarios = {
            'conservative': {'batch_size': 1, 'seq_len': 256},
            'balanced': {'batch_size': 4, 'seq_len': 512},
            'aggressive': {'batch_size': 8, 'seq_len': 1024}
        }
        
        predictions = {}
        for scenario_name, config in scenarios.items():
            mfu = self.estimate_mfu_for_config(
                profile, config['batch_size'], config['seq_len']
            )
            predictions[scenario_name] = mfu
        
        return predictions
    
    def identify_bottlenecks(self, profile: ModelProfile) -> List[str]:
        """Identify potential performance bottlenecks"""
        bottlenecks = []
        
        # Memory bottlenecks
        memory = psutil.virtual_memory()
        if profile.memory_footprint_gb > memory.total / (1024 ** 3) * 0.8:
            bottlenecks.append("Model too large for available RAM")
        
        # Architecture bottlenecks
        if profile.n_heads > 32:
            bottlenecks.append("Very high number of attention heads may reduce efficiency")
        
        if profile.d_model % 64 != 0:
            bottlenecks.append("d_model not optimally aligned for hardware (should be multiple of 64)")
        
        if profile.d_ff != profile.d_model * 4:
            bottlenecks.append("Non-standard feed-forward dimension ratio")
        
        # Device compatibility
        if not profile.device_compatibility.get(self.device, False):
            bottlenecks.append(f"Model not compatible with {self.device}")
        
        return bottlenecks
    
    def generate_recommendations(self, profile: ModelProfile) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Memory optimization
        if profile.memory_footprint_gb > 8:
            recommendations.append("Consider gradient checkpointing to reduce memory usage")
            recommendations.append("Use mixed precision training (FP16) to halve memory requirements")
        
        # Architecture optimization
        if profile.parameters > 5e9:  # 5B+ parameters
            recommendations.append("Consider model parallelism for very large models")
        
        # Sequence length optimization
        if profile.max_seq_length > 2048:
            recommendations.append("Consider reducing sequence length for better MFU")
        
        # Batch size optimization
        recommendations.append("Use dynamic batch sizing to maximize GPU utilization")
        recommendations.append("Experiment with gradient accumulation for effective larger batch sizes")
        
        # Device-specific recommendations
        if self.device == 'mps':
            recommendations.append("Use Apple Silicon optimized PyTorch for better MPS performance")
            recommendations.append("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9 for better memory usage")
        elif self.device == 'cuda':
            recommendations.append("Enable torch.backends.cudnn.benchmark = True")
            recommendations.append("Use TF32 precision for better performance on modern GPUs")
        
        return recommendations
    
    def compare_models(self, models: List[Tuple[nn.Module, str]]) -> Dict[str, Any]:
        """Compare multiple models for MFU performance"""
        
        comparisons = []
        
        for model, name in models:
            profile = self.profile_model(model, name)
            analysis = self.analyze_mfu_potential(profile)
            
            comparisons.append({
                'name': name,
                'parameters': profile.parameters,
                'memory_gb': profile.memory_footprint_gb,
                'fits_in_memory': analysis['fits_in_memory'],
                'best_estimated_mfu': max([c['estimated_mfu'] for c in analysis['optimal_configs']]) if analysis['optimal_configs'] else 0,
                'recommendations_count': len(analysis['recommendations'])
            })
        
        # Sort by best estimated MFU
        comparisons.sort(key=lambda x: x['best_estimated_mfu'], reverse=True)
        
        return {
            'comparisons': comparisons,
            'best_model': comparisons[0]['name'] if comparisons else None,
            'summary': self.generate_comparison_summary(comparisons)
        }
    
    def generate_comparison_summary(self, comparisons: List[Dict]) -> str:
        """Generate summary of model comparison"""
        if not comparisons:
            return "No models to compare"
        
        best = comparisons[0]
        summary = f"Best model for MFU: {best['name']}\n"
        summary += f"  - Parameters: {best['parameters']:,}\n"
        summary += f"  - Memory: {best['memory_gb']:.1f}GB\n"
        summary += f"  - Estimated peak MFU: {best['best_estimated_mfu']:.1f}%\n"
        
        return summary
    
    def save_analysis_report(self, analysis: Dict[str, Any], filepath: str):
        """Save comprehensive analysis report"""
        
        # Convert profile to dict for JSON serialization
        if 'profile' in analysis:
            profile = analysis['profile']
            analysis['profile'] = {
                'name': profile.name,
                'parameters': profile.parameters,
                'd_model': profile.d_model,
                'n_layers': profile.n_layers,
                'n_heads': profile.n_heads,
                'd_ff': profile.d_ff,
                'vocab_size': profile.vocab_size,
                'max_seq_length': profile.max_seq_length,
                'memory_footprint_gb': profile.memory_footprint_gb,
                'theoretical_peak_flops': profile.theoretical_peak_flops,
                'device_compatibility': profile.device_compatibility
            }
        
        # Add analysis metadata
        analysis['analysis_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'analyzer_device': self.device,
            'system_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
            'cpu_count': psutil.cpu_count()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f" Analysis report saved: {filepath}")


def create_test_models() -> List[Tuple[nn.Module, str]]:
    """Create test models for MFU analysis"""
    tokenizer = iLLuMinatorTokenizer()
    vocab_size = len(tokenizer)
    
    models = []
    
    # Small model (120M parameters)
    small_model = iLLuMinator4_7B(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_length=1024,
        dropout=0.1
    )
    models.append((small_model, "iLLuMinator-120M"))
    
    # Medium model (1.8B parameters)
    medium_model = iLLuMinator4_7B(
        vocab_size=vocab_size,
        d_model=2048,
        n_layers=18,
        n_heads=16,
        d_ff=8192,
        max_seq_length=1024,
        dropout=0.1
    )
    models.append((medium_model, "iLLuMinator-1.8B"))
    
    # Large model (4.9B parameters)
    large_model = iLLuMinator4_7B(
        vocab_size=vocab_size,
        d_model=3584,
        n_layers=30,
        n_heads=28,
        d_ff=14336,
        max_seq_length=1024,
        dropout=0.1
    )
    models.append((large_model, "iLLuMinator-4.9B"))
    
    return models


def main():
    """Main MFU analysis function"""
    print(" iLLuMinator MFU Analysis Tool")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = MFUAnalyzer()
    
    # Create test models
    print("\n🧠 Creating test models...")
    models = create_test_models()
    
    # Analyze individual models
    print(f"\n Analyzing {len(models)} models...")
    
    analyses = []
    for model, name in models:
        print(f"\n Analyzing {name}...")
        analysis = analyzer.analyze_mfu_potential(analyzer.profile_model(model, name))
        analyses.append((name, analysis))
        
        # Print summary
        profile = analysis['profile']
        print(f"   Parameters: {profile.parameters:,}")
        print(f"   Memory footprint: {profile.memory_footprint_gb:.1f}GB")
        print(f"   Fits in memory: {analysis['fits_in_memory']}")
        print(f"   Best estimated MFU: {max([c['estimated_mfu'] for c in analysis['optimal_configs']]) if analysis['optimal_configs'] else 0:.1f}%")
        
        # Save individual analysis
        analyzer.save_analysis_report(analysis, f"mfu_analysis_{name.lower().replace('-', '_')}.json")
    
    # Compare models
    print(f"\n Model Comparison:")
    comparison = analyzer.compare_models(models)
    print(comparison['summary'])
    
    # Save comparison report
    with open("mfu_model_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate recommendations
    print(f"\n General Recommendations:")
    for i, (name, analysis) in enumerate(analyses, 1):
        print(f"\n{i}. {name}:")
        for rec in analysis['recommendations'][:3]:  # Top 3 recommendations
            print(f"   • {rec}")
    
    print(f"\n MFU analysis complete!")
    print(f" Reports saved in current directory")
    
    return analyzer, analyses


if __name__ == "__main__":
    analyzer, analyses = main()
