import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from datetime import datetime
from tqdm import tqdm
import psutil
import GPUtil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IlluminatorDebugger:
    """Comprehensive debugging and analysis tool for iLLuMinator 4.9B"""
    
    def __init__(self, model_path="illuminator_4.9B_final", config_path="config_4.9B.json"):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_results = {}
        
        # Create debug output directory
        self.debug_dir = f"debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        self.load_model_and_config()
        
    def load_model_and_config(self):
        """Load model and configuration"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load configuration
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config
            self.config = {
                "vocab_size": 50260,
                "d_model": 3584,
                "n_heads": 28,
                "n_layers": 30,
                "d_ff": 14336,
                "max_seq_length": 2048
            }
        
        # Try to load the trained model
        try:
            # Load model checkpoint
            checkpoint_path = os.path.join(self.model_path, 'pytorch_model.bin')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Create model architecture
                self.model = self.create_model_from_config()
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info("Model loaded successfully from checkpoint")
            else:
                logger.warning("No checkpoint found, creating new model for analysis")
                self.model = self.create_model_from_config()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating new model for structural analysis")
            self.model = self.create_model_from_config()
        
        self.model.to(self.device)
        self.model.eval()
        
        # Try to load tokenizer
        try:
            from tokenizer import CustomTokenizer
            self.tokenizer = CustomTokenizer()
        except:
            logger.warning("Using fallback tokenizer for debugging")
            self.tokenizer = self.create_fallback_tokenizer()
    
    def create_model_from_config(self):
        """Create model from configuration"""
        # Try to import your models
        try:
            from legacy.illuminator_cuda import iLLuMinatorCUDA
            return iLLuMinatorCUDA(
                vocab_size=self.config['vocab_size'],
                d_model=self.config['d_model'],
                n_heads=self.config['n_heads'],
                n_layers=self.config['n_layers'],
                d_ff=self.config['d_ff'],
                max_seq_length=self.config['max_seq_length']
            )
        except:
            logger.warning("Using fallback transformer model")
            return self.create_fallback_transformer()
    
    def create_fallback_transformer(self):
        """Create a fallback transformer for debugging"""
        class DebugTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff):
                super().__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Embedding(2048, d_model)
                
                # Transformer layers
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_ff,
                        dropout=0.1,
                        batch_first=True
                    )
                    for _ in range(n_layers)
                ])
                
                self.ln_f = nn.LayerNorm(d_model)
                self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
                
                # Tie weights
                self.lm_head.weight = self.embedding.weight
                
            def forward(self, input_ids, attention_mask=None):
                seq_len = input_ids.size(1)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                
                x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.ln_f(x)
                logits = self.lm_head(x)
                
                return type('Output', (), {'logits': logits})()
        
        return DebugTransformer(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers'],
            d_ff=self.config['d_ff']
        )
    
    def create_fallback_tokenizer(self):
        """Create fallback tokenizer"""
        class FallbackTokenizer:
            def __init__(self):
                self.vocab_size = 50260
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def encode(self, text):
                return [min(ord(c), self.vocab_size-1) for c in text[:100]]
            
            def decode(self, tokens):
                return ''.join([chr(min(max(t, 32), 126)) for t in tokens if t > 0])
        
        return FallbackTokenizer()
    
    def analyze_model_architecture(self):
        """Comprehensive model architecture analysis"""
        logger.info("Analyzing model architecture...")
        
        # Basic model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        architecture_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_gb_fp32': total_params * 4 / 1024**3,
            'model_size_gb_fp16': total_params * 2 / 1024**3,
            'config': self.config
        }
        
        # Layer-wise parameter analysis
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    layer_info.append({
                        'name': name,
                        'type': type(module).__name__,
                        'parameters': param_count,
                        'percentage': param_count / total_params * 100
                    })
        
        # Sort by parameter count
        layer_info.sort(key=lambda x: x['parameters'], reverse=True)
        architecture_info['layer_breakdown'] = layer_info[:20]  # Top 20 layers
        
        # Create visualization
        self.visualize_architecture(layer_info[:15])
        
        self.debug_results['architecture'] = architecture_info
        
        print(f"\n{'='*60}")
        print("MODEL ARCHITECTURE ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size (FP16): {total_params * 2 / 1024**3:.2f} GB")
        print(f"Model Size (FP32): {total_params * 4 / 1024**3:.2f} GB")
        
        print(f"\nTop Parameter-Heavy Layers:")
        for layer in layer_info[:10]:
            print(f"  {layer['name'][:40]:40} {layer['parameters']:>12,} ({layer['percentage']:.1f}%)")
        
        return architecture_info
    
    def visualize_architecture(self, layer_info):
        """Create architecture visualization"""
        plt.figure(figsize=(15, 8))
        
        # Parameter distribution
        names = [info['name'].split('.')[-1][:15] for info in layer_info]
        params = [info['parameters'] for info in layer_info]
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(names)), params)
        plt.xlabel('Layer')
        plt.ylabel('Parameters')
        plt.title('Parameter Distribution by Layer')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, params)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=8)
        
        # Model size comparison
        plt.subplot(2, 2, 2)
        sizes = ['FP32', 'FP16', 'Int8', 'Int4']
        size_values = [
            sum(p.numel() for p in self.model.parameters()) * 4 / 1024**3,
            sum(p.numel() for p in self.model.parameters()) * 2 / 1024**3,
            sum(p.numel() for p in self.model.parameters()) * 1 / 1024**3,
            sum(p.numel() for p in self.model.parameters()) * 0.5 / 1024**3
        ]
        
        colors = ['red', 'orange', 'green', 'blue']
        bars = plt.bar(sizes, size_values, color=colors, alpha=0.7)
        plt.ylabel('Size (GB)')
        plt.title('Model Size by Precision')
        
        # Add value labels
        for bar, val in zip(bars, size_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}GB', ha='center', va='bottom')
        
        # Layer type distribution
        plt.subplot(2, 2, 3)
        layer_types = {}
        for info in layer_info:
            layer_type = info['type']
            if layer_type in layer_types:
                layer_types[layer_type] += info['parameters']
            else:
                layer_types[layer_type] = info['parameters']
        
        plt.pie(layer_types.values(), labels=layer_types.keys(), autopct='%1.1f%%')
        plt.title('Parameter Distribution by Layer Type')
        
        # Memory requirements
        plt.subplot(2, 2, 4)
        scenarios = ['Training\n(FP32)', 'Training\n(Mixed)', 'Inference\n(FP16)', 'Inference\n(Int8)']
        memory_reqs = [
            size_values[0] * 4,  # Training requires ~4x model size
            size_values[1] * 3,  # Mixed precision ~3x
            size_values[1] * 1.2,  # Inference overhead
            size_values[2] * 1.1   # Int8 inference
        ]
        
        bars = plt.bar(scenarios, memory_reqs, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
        plt.ylabel('Memory (GB)')
        plt.title('Estimated Memory Requirements')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, memory_reqs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}GB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, 'architecture_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_weights_and_gradients(self):
        """Analyze weight distributions and potential gradient issues"""
        logger.info("Analyzing weights and gradients...")
        
        weight_stats = []
        gradient_stats = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                # Weight statistics
                weight = param.data.cpu().numpy().flatten()
                weight_stat = {
                    'name': name,
                    'shape': list(param.shape),
                    'mean': float(np.mean(weight)),
                    'std': float(np.std(weight)),
                    'min': float(np.min(weight)),
                    'max': float(np.max(weight)),
                    'zero_fraction': float(np.sum(weight == 0) / len(weight)),
                    'abs_mean': float(np.mean(np.abs(weight))),
                    'parameter_count': param.numel()
                }
                weight_stats.append(weight_stat)
                
                # Gradient statistics (if available)
                if param.grad is not None:
                    grad = param.grad.data.cpu().numpy().flatten()
                    grad_stat = {
                        'name': name,
                        'grad_mean': float(np.mean(grad)),
                        'grad_std': float(np.std(grad)),
                        'grad_norm': float(np.linalg.norm(grad)),
                        'grad_max': float(np.max(np.abs(grad)))
                    }
                    gradient_stats.append(grad_stat)
        
        # Visualize weight distributions
        self.visualize_weight_analysis(weight_stats, gradient_stats)
        
        # Detect potential issues
        issues = self.detect_training_issues(weight_stats, gradient_stats)
        
        analysis_results = {
            'weight_statistics': weight_stats,
            'gradient_statistics': gradient_stats,
            'detected_issues': issues
        }
        
        self.debug_results['weights_and_gradients'] = analysis_results
        
        # Print summary
        print(f"\n{'='*60}")
        print("WEIGHT AND GRADIENT ANALYSIS")
        print(f"{'='*60}")
        
        print(f"Analyzed {len(weight_stats)} parameter tensors")
        if gradient_stats:
            print(f"Found gradients for {len(gradient_stats)} tensors")
        
        # Weight statistics summary
        all_means = [stat['mean'] for stat in weight_stats]
        all_stds = [stat['std'] for stat in weight_stats]
        
        print(f"\nWeight Statistics Summary:")
        print(f"  Mean of means: {np.mean(all_means):.6f}")
        print(f"  Std of means: {np.std(all_means):.6f}")
        print(f"  Mean of stds: {np.mean(all_stds):.6f}")
        
        # Issues
        if issues:
            print(f"\n⚠️  DETECTED ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No critical issues detected")
        
        return analysis_results
    
    def visualize_weight_analysis(self, weight_stats, gradient_stats):
        """Create weight and gradient visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Weight means distribution
        means = [stat['mean'] for stat in weight_stats]
        axes[0, 0].hist(means, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Weight Means Distribution')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Weight standard deviations
        stds = [stat['std'] for stat in weight_stats]
        axes[0, 1].hist(stds, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Weight Standard Deviations')
        axes[0, 1].set_xlabel('Std Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # Zero fraction distribution
        zero_fractions = [stat['zero_fraction'] for stat in weight_stats]
        axes[0, 2].hist(zero_fractions, bins=30, alpha=0.7, color='orange')
        axes[0, 2].set_title('Zero Weight Fractions')
        axes[0, 2].set_xlabel('Zero Fraction')
        axes[0, 2].set_ylabel('Frequency')
        
        # Weight magnitudes by layer
        layer_names = [stat['name'].split('.')[0] for stat in weight_stats]
        abs_means = [stat['abs_mean'] for stat in weight_stats]
        
        axes[1, 0].scatter(range(len(abs_means)), abs_means, alpha=0.6)
        axes[1, 0].set_title('Weight Magnitudes by Layer')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Absolute Mean Weight')
        
        # Gradient analysis (if available)
        if gradient_stats:
            grad_norms = [stat['grad_norm'] for stat in gradient_stats]
            axes[1, 1].hist(grad_norms, bins=30, alpha=0.7, color='red')
            axes[1, 1].set_title('Gradient Norms Distribution')
            axes[1, 1].set_xlabel('Gradient Norm')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_yscale('log')
            
            # Gradient magnitudes by layer
            grad_means = [stat['grad_max'] for stat in gradient_stats]
            axes[1, 2].scatter(range(len(grad_means)), grad_means, alpha=0.6, color='red')
            axes[1, 2].set_title('Max Gradient Magnitudes by Layer')
            axes[1, 2].set_xlabel('Layer Index')
            axes[1, 2].set_ylabel('Max Gradient Magnitude')
            axes[1, 2].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Gradients Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 2].text(0.5, 0.5, 'No Gradients Available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, 'weight_gradient_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def detect_training_issues(self, weight_stats, gradient_stats):
        """Detect potential training issues"""
        issues = []
        
        # Check for dead neurons (very small weights)
        dead_layers = [stat for stat in weight_stats if stat['abs_mean'] < 1e-6]
        if dead_layers:
            issues.append(f"Potential dead neurons in {len(dead_layers)} layers with very small weights")
        
        # Check for weight explosion
        exploded_layers = [stat for stat in weight_stats if stat['abs_mean'] > 10]
        if exploded_layers:
            issues.append(f"Potential weight explosion in {len(exploded_layers)} layers")
        
        # Check gradient issues
        if gradient_stats:
            # Vanishing gradients
            vanishing_grads = [stat for stat in gradient_stats if stat['grad_norm'] < 1e-8]
            if vanishing_grads:
                issues.append(f"Vanishing gradients detected in {len(vanishing_grads)} layers")
            
            # Exploding gradients
            exploding_grads = [stat for stat in gradient_stats if stat['grad_norm'] > 100]
            if exploding_grads:
                issues.append(f"Exploding gradients detected in {len(exploding_grads)} layers")
        
        # Check weight initialization
        mean_of_means = np.mean([stat['mean'] for stat in weight_stats])
        if abs(mean_of_means) > 0.1:
            issues.append(f"Weight initialization may be biased (mean of means: {mean_of_means:.4f})")
        
        # Check for sparse weights
        sparse_layers = [stat for stat in weight_stats if stat['zero_fraction'] > 0.5]
        if sparse_layers:
            issues.append(f"High sparsity detected in {len(sparse_layers)} layers (>50% zeros)")
        
        return issues
    
    def benchmark_performance(self):
        """Comprehensive performance benchmarking"""
        logger.info("Benchmarking model performance...")
        
        # Prepare test inputs
        test_sequences = [
            "The future of artificial intelligence",
            "In a world where technology advances",
            "Machine learning has revolutionized",
            "The quick brown fox jumps over the lazy dog",
            "To be or not to be, that is the question"
        ]
        
        benchmark_results = {
            'device': str(self.device),
            'precision': 'fp16' if next(self.model.parameters()).dtype == torch.float16 else 'fp32',
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'sequences_tested': len(test_sequences)
        }
        
        # Memory benchmarking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()
        
        # Speed benchmarking
        inference_times = []
        token_counts = []
        
        self.model.eval()
        with torch.no_grad():
            for sequence in tqdm(test_sequences, desc="Benchmarking"):
                try:
                    # Tokenize
                    tokens = self.tokenizer.encode(sequence)
                    input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
                    
                    # Warmup
                    for _ in range(3):
                        _ = self.model(input_ids)
                    
                    # Benchmark
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    
                    if torch.cuda.is_available():
                        start_time.record()
                    else:
                        import time
                        start_time = time.time()
                    
                    # Multiple inferences for stable timing
                    for _ in range(10):
                        outputs = self.model(input_ids)
                    
                    if torch.cuda.is_available():
                        end_time.record()
                        torch.cuda.synchronize()
                        elapsed_time = start_time.elapsed_time(end_time) / 1000.0 / 10  # Convert to seconds, average
                    else:
                        end_time = time.time()
                        elapsed_time = (end_time - start_time) / 10
                    
                    inference_times.append(elapsed_time)
                    token_counts.append(len(tokens))
                    
                except Exception as e:
                    logger.warning(f"Error benchmarking sequence '{sequence}': {e}")
                    continue
        
        # Calculate statistics
        if inference_times:
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            avg_tokens = np.mean(token_counts)
            tokens_per_second = avg_tokens / avg_time
            
            benchmark_results.update({
                'avg_inference_time_seconds': avg_time,
                'std_inference_time_seconds': std_time,
                'avg_tokens_per_sequence': avg_tokens,
                'tokens_per_second': tokens_per_second,
                'sequences_per_second': 1.0 / avg_time
            })
        
        # Memory usage
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            benchmark_results.update({
                'memory_used_mb': (memory_after - memory_before) / 1024**2,
                'peak_memory_mb': peak_memory / 1024**2,
                'gpu_utilization': self.get_gpu_utilization()
            })
        
        self.debug_results['performance'] = benchmark_results
        
        # Print results
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        print(f"Device: {benchmark_results['device']}")
        if inference_times:
            print(f"Average inference time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            print(f"Sequences per second: {1.0/avg_time:.2f}")
        
        if torch.cuda.is_available():
            print(f"Memory usage: {benchmark_results.get('memory_used_mb', 0):.1f} MB")
            print(f"Peak memory: {benchmark_results.get('peak_memory_mb', 0):.1f} MB")
        
        return benchmark_results
    
    def get_gpu_utilization(self):
        """Get current GPU utilization"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return None
    
    def test_generation_quality(self):
        """Test text generation quality"""
        logger.info("Testing generation quality...")
        
        test_prompts = [
            "The future of artificial intelligence is",
            "In the year 2050, technology will",
            "The most important aspect of machine learning",
            "Climate change poses significant challenges because",
            "The key to successful communication lies in"
        ]
        
        generation_results = []
        
        self.model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    # Simple greedy generation (you can make this more sophisticated)
                    tokens = self.tokenizer.encode(prompt)
                    input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
                    
                    generated_tokens = tokens.copy()
                    
                    # Generate up to 50 additional tokens
                    for _ in range(50):
                        if len(generated_tokens) >= self.config.get('max_seq_length', 2048):
                            break
                        
                        # Get model prediction
                        current_input = torch.tensor([generated_tokens[-100:]], dtype=torch.long).to(self.device)
                        outputs = self.model(current_input)
                        
                        # Get next token (greedy)
                        next_token = torch.argmax(outputs.logits[0, -1]).item()
                        
                        # Stop if we hit pad/eos token
                        if next_token == 0 or next_token == 1:
                            break
                        
                        generated_tokens.append(next_token)
                    
                    # Decode generated text
                    generated_text = self.tokenizer.decode(generated_tokens)
                    
                    result = {
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'generated_tokens': len(generated_tokens) - len(tokens),
                        'total_length': len(generated_tokens)
                    }
                    
                    generation_results.append(result)
                    
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated_text}")
                    print("-" * 80)
                    
                except Exception as e:
                    logger.warning(f"Error generating for prompt '{prompt}': {e}")
                    continue
        
        self.debug_results['generation_quality'] = generation_results
        return generation_results
    
    def system_analysis(self):
        """Analyze system resources and compatibility"""
        logger.info("Analyzing system resources...")
        
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': f"{torch.__version__}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        # CPU information
        system_info['cpu'] = {
            'count': psutil.cpu_count(),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'usage_percent': psutil.cpu_percent(interval=1)
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        system_info['memory'] = {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'used_gb': memory.used / 1024**3,
            'usage_percent': memory.percent
        }
        
        # GPU information
        if torch.cuda.is_available():
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version()
            }
            
            # GPU utilization
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info['utilization_percent'] = gpus[0].load * 100
                    gpu_info['temperature_c'] = gpus[0].temperature
                    gpu_info['memory_utilization_percent'] = gpus[0].memoryUtil * 100
            except:
                pass
            
            system_info['gpu'] = gpu_info
        
        self.debug_results['system_analysis'] = system_info
        
        # Print system summary
        print(f"\n{'='*60}")
        print("SYSTEM ANALYSIS")
        print(f"{'='*60}")
        print(f"CPU: {system_info['cpu']['count']} cores, {system_info['cpu']['usage_percent']}% usage")
        print(f"Memory: {system_info['memory']['total_gb']:.1f} GB total, {system_info['memory']['usage_percent']:.1f}% used")
        
        if torch.cuda.is_available():
            gpu = system_info['gpu']
            print(f"GPU: {gpu['device_name']}")
            print(f"GPU Memory: {gpu['memory_total_gb']:.1f} GB total, {gpu['memory_allocated_gb']:.1f} GB allocated")
            if 'utilization_percent' in gpu:
                print(f"GPU Utilization: {gpu['utilization_percent']:.1f}%")
        else:
            print("GPU: Not available")
        
        return system_info
    
    def generate_comprehensive_report(self):
        """Generate comprehensive debugging report"""
        logger.info("Generating comprehensive debug report...")
        
        # Run all analyses
        architecture_analysis = self.analyze_model_architecture()
        weight_analysis = self.analyze_weights_and_gradients()
        performance_benchmark = self.benchmark_performance()
        generation_quality = self.test_generation_quality()
        system_analysis = self.system_analysis()
        
        # Compile final report
        final_report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model_path': self.model_path,
                'config_path': self.config_path,
                'debug_version': '1.0'
            },
            'architecture': architecture_analysis,
            'weights_and_gradients': weight_analysis,
            'performance': performance_benchmark,
            'generation_quality': generation_quality,
            'system': system_analysis,
            'recommendations': self.generate_recommendations()
        }
        
        # Save comprehensive report
        report_path = os.path.join(self.debug_dir, 'comprehensive_debug_report.json')
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Generate summary
        self.generate_summary_report(final_report)
        
        logger.info(f"Comprehensive report saved to {report_path}")
        return final_report
    
    def generate_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Memory optimization recommendations
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            model_size_gb = sum(p.numel() for p in self.model.parameters()) * 2 / 1024**3  # FP16
            
            if model_size_gb > gpu_memory_gb * 0.8:
                recommendations.append("Consider using gradient checkpointing to reduce memory usage")
                recommendations.append("Use smaller batch sizes with gradient accumulation")
                recommendations.append("Consider model parallelism for large models")
        
        # Training optimization recommendations
        if 'weights_and_gradients' in self.debug_results:
            issues = self.debug_results['weights_and_gradients']['detected_issues']
            
            if any('vanishing' in issue.lower() for issue in issues):
                recommendations.append("Use residual connections and proper weight initialization")
                recommendations.append("Consider using gradient clipping")
            
            if any('exploding' in issue.lower() for issue in issues):
                recommendations.append("Implement gradient clipping with max_grad_norm=1.0")
                recommendations.append("Reduce learning rate")
        
        # Performance recommendations
        if 'performance' in self.debug_results:
            perf = self.debug_results['performance']
            if perf.get('tokens_per_second', 0) < 5:
                recommendations.append("Consider using mixed precision training for speed")
                recommendations.append("Enable CUDA optimizations (torch.backends.cudnn.benchmark = True)")
                recommendations.append("Use flash attention if available")
        
        return recommendations
    
    def generate_summary_report(self, full_report):
        """Generate human-readable summary report"""
        summary_path = os.path.join(self.debug_dir, 'debug_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("iLLuMinator 4.9B - DEBUG SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Overview
            arch = full_report['architecture']
            f.write("MODEL OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Parameters: {arch['total_parameters']:,}\n")
            f.write(f"Model Size (FP16): {arch['model_size_gb_fp16']:.2f} GB\n")
            f.write(f"Model Size (FP32): {arch['model_size_gb_fp32']:.2f} GB\n\n")
            
            # Performance Summary
            if 'performance' in full_report:
                perf = full_report['performance']
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Device: {perf['device']}\n")
                if 'tokens_per_second' in perf:
                    f.write(f"Speed: {perf['tokens_per_second']:.2f} tokens/second\n")
                if 'memory_used_mb' in perf:
                    f.write(f"Memory Usage: {perf['memory_used_mb']:.1f} MB\n")
                f.write("\n")
            
            # Issues and Recommendations
            issues = full_report['weights_and_gradients']['detected_issues']
            if issues:
                f.write("DETECTED ISSUES\n")
                f.write("-" * 40 + "\n")
                for issue in issues:
                    f.write(f"⚠️  {issue}\n")
                f.write("\n")
            
            recommendations = full_report['recommendations']
            if recommendations:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                for rec in recommendations:
                    f.write(f"💡 {rec}\n")
                f.write("\n")
            
            # System Info
            system = full_report['system']
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"CPU: {system['cpu']['count']} cores\n")
            f.write(f"Memory: {system['memory']['total_gb']:.1f} GB\n")
            if 'gpu' in system:
                f.write(f"GPU: {system['gpu']['device_name']}\n")
                f.write(f"GPU Memory: {system['gpu']['memory_total_gb']:.1f} GB\n")
            
        print(f"\n{'='*60}")
        print("DEBUG SUMMARY")
        print(f"{'='*60}")
        print(f"Report saved to: {self.debug_dir}")
        print(f"Full report: comprehensive_debug_report.json")
        print(f"Summary: debug_summary.txt")
        print(f"Visualizations: *.png files")

def main():
    """Main debugging function"""
    print("iLLuMinator 4.9B Debugger")
    print("=" * 50)
    
    # Initialize debugger
    try:
        debugger = IlluminatorDebugger()
        print("✅ Debugger initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize debugger: {e}")
        return
    
    # Run comprehensive analysis
    try:
        report = debugger.generate_comprehensive_report()
        print("\n✅ Debug analysis completed successfully!")
        
        # Print key findings
        if report['weights_and_gradients']['detected_issues']:
            print("\n⚠️  Issues found - check the full report for details")
        else:
            print("\n✅ No critical issues detected")
        
        print(f"\n📊 Performance: {report['performance'].get('tokens_per_second', 'N/A')} tokens/sec")
        print(f"💾 Model size: {report['architecture']['model_size_gb_fp16']:.2f} GB (FP16)")
        
    except Exception as e:
        print(f"❌ Debug analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
