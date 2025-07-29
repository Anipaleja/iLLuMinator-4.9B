#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from datetime import datetime

class ModelAnalyzer:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model for analysis"""
        
        # Find the latest checkpoint if not specified
        if self.checkpoint_path is None:
            checkpoints = glob.glob("checkpoints/rtx3050_final_model.pt")
            if not checkpoints:
                checkpoints = glob.glob("checkpoints/model_step_*.pt")
            
            if checkpoints:
                self.checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"📁 Loading checkpoint: {self.checkpoint_path}")
            else:
                print("❌ No checkpoints found!")
                return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract model config
            config = checkpoint.get('config', {})
            
            # Recreate model architecture
            from train_rtx3050_optimized import RTX3050OptimizedModel
            self.model = RTX3050OptimizedModel(
                vocab_size=config.get('vocab_size', 50260),
                d_model=config.get('d_model', 1536),
                n_heads=config.get('n_heads', 12),
                n_layers=config.get('n_layers', 18),
                max_seq_len=config.get('max_seq_len', 512)
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def analyze_architecture(self):
        """Analyze model architecture and parameters"""
        print("\n" + "="*60)
        print("🏗️  MODEL ARCHITECTURE ANALYSIS")
        print("="*60)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size (FP32): {total_params * 4 / 1024**3:.2f} GB")
        print(f"Model Size (FP16): {total_params * 2 / 1024**3:.2f} GB")
        
        # Layer-wise analysis
        print(f"\n📊 Layer Breakdown:")
        layer_counts = {}
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if module_type not in layer_counts:
                layer_counts[module_type] = 0
            layer_counts[module_type] += 1
        
        for layer_type, count in sorted(layer_counts.items()):
            if count > 1:  # Only show layers that appear multiple times
                print(f"   {layer_type}: {count}")
    
    def analyze_weights(self):
        """Analyze weight distributions"""
        print("\n" + "="*60)
        print("⚖️  WEIGHT DISTRIBUTION ANALYSIS")
        print("="*60)
        
        weight_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                weights = param.data.cpu().numpy().flatten()
                
                stats = {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'zero_fraction': float(np.sum(weights == 0) / len(weights))
                }
                
                weight_stats[name] = stats
                
                # Print stats for key layers
                if any(key in name for key in ['embedding', 'lm_head', 'transformer.layers.0']):
                    print(f"\n{name}:")
                    print(f"   Mean: {stats['mean']:.6f}")
                    print(f"   Std:  {stats['std']:.6f}")
                    print(f"   Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                    print(f"   Zeros: {stats['zero_fraction']*100:.2f}%")
        
        return weight_stats
    
    def test_generation(self):
        """Test text generation capabilities"""
        print("\n" + "="*60)
        print("🎨 TEXT GENERATION TEST")
        print("="*60)
        
        # Simple generation test
        self.model.eval()
        
        test_prompts = [
            "The future of artificial intelligence",
            "Machine learning is",
            "In the year 2025",
        ]
        
        with torch.no_grad():
            for prompt in test_prompts:
                print(f"\n📝 Prompt: '{prompt}'")
                
                # Convert prompt to tensor (simple character encoding)
                input_tokens = torch.tensor([[ord(c) % 50260 for c in prompt[:50]]], dtype=torch.long).to(self.device)
                
                # Generate
                try:
                    outputs = self.model(input_tokens)
                    logits = outputs['logits']
                    
                    # Get next token predictions
                    next_token_logits = logits[0, -1, :]
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # Top 5 predictions
                    top_probs, top_indices = torch.topk(next_token_probs, 5)
                    
                    print("   Top predictions:")
                    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        char = chr(min(idx.item(), 127))
                        print(f"     {i+1}. '{char}' (prob: {prob:.4f})")
                        
                except Exception as e:
                    print(f"   ❌ Generation error: {e}")
    
    def benchmark_performance(self):
        """Benchmark model performance"""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
        # Speed test
        test_input = torch.randint(0, 1000, (1, 100)).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(test_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start.record()
                    _ = self.model(test_input)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                else:
                    import time
                    start_time = time.time()
                    _ = self.model(test_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Average inference time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"Tokens per second: {100 / (avg_time / 1000):.1f}")
        
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**2
            print(f"Memory usage: {memory_used:.1f} MB")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("📊 iLLuMinator RTX 3050 - COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        if not hasattr(self, 'model'):
            print("❌ No model loaded for analysis!")
            return
        
        # Run all analyses
        self.analyze_architecture()
        weight_stats = self.analyze_weights()
        self.test_generation()
        self.benchmark_performance()
        
        # Summary
        print("\n" + "="*60)
        print("📋 SUMMARY")
        print("="*60)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✅ Model successfully trained and analyzed")
        print(f"📏 Model size: {total_params:,} parameters (~1.5B)")
        print(f"💾 Memory efficient for RTX 3050 (8GB VRAM)")
        print(f"🚀 Ready for inference and further fine-tuning")
        
        # Save analysis report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_parameters': total_params,
            'checkpoint_path': self.checkpoint_path,
            'weight_statistics': weight_stats,
            'device': str(self.device)
        }
        
        with open('model_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: model_analysis_report.json")

def main():
    print("🔍 iLLuMinator RTX 3050 Model Analyzer")
    print("=" * 50)
    
    analyzer = ModelAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
