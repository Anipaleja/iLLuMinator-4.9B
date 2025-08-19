#!/usr/bin/env python3
"""
iLLuMinator MFU-Optimized Training
Training with maximum Model FLOPs Utilization monitoring and optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time
import json
from typing import List, Dict, Any
from tqdm import tqdm
import psutil

from legacy.illuminator_model import iLLuMinator4_7B
from practical_model.tokenizer import iLLuMinatorTokenizer
from mfu_optimizer import MFUProfiler, MFUOptimizer, benchmark_mfu


class MFUOptimizedDataset(Dataset):
    """Dataset optimized for MFU measurements"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }


class MFUOptimizedTrainer:
    """Trainer with integrated MFU optimization"""
    
    def __init__(self, model, tokenizer, device: str):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize MFU components
        self.mfu_profiler = MFUProfiler(model, device)
        self.mfu_optimizer = MFUOptimizer(model, device)
        
        # Apply device optimizations
        self.device_optimizations = self.mfu_optimizer.optimize_for_device()
        
        # Training components
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        
        # MFU tracking
        self.mfu_history = []
        self.best_mfu = 0.0
        
        print(f" MFU-Optimized Trainer initialized on {device}")
        print(f"   Device optimizations: {self.device_optimizations}")
    
    def find_optimal_batch_size(self, sample_texts: List[str], max_batch_size: int = 16, 
                               seq_len: int = 512) -> int:
        """Find optimal batch size for this model and data"""
        print(f"\n Finding optimal batch size for MFU...")
        
        # Create sample dataset
        dataset = MFUOptimizedDataset(sample_texts[:50], self.tokenizer, seq_len)
        
        best_batch_size = self.mfu_optimizer.find_optimal_batch_size(
            self.tokenizer, max_batch_size, seq_len
        )
        
        return best_batch_size
    
    def initialize_optimizer(self, lr: float = 1e-4):
        """Initialize optimizer with MFU-friendly settings"""
        # Use AdamW with optimized settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8
        )
        
        print(f"  Optimizer initialized with lr={lr}")
    
    def train_step_with_mfu(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with MFU measurement"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        batch_size, seq_len = input_ids.shape
        
        # Measure training step with MFU profiler
        with self.mfu_profiler.measure_training_step(batch_size, seq_len) as metrics:
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
        
        # Track MFU
        current_mfu = metrics.mfu_percentage
        self.mfu_history.append(current_mfu)
        
        if current_mfu > self.best_mfu:
            self.best_mfu = current_mfu
        
        return {
            'loss': loss.item(),
            'mfu': current_mfu,
            'flops': metrics.achieved_flops / 1e12,  # TFLOPS
            'memory_gb': metrics.memory_usage_gb,
            'step_time_ms': metrics.training_time_ms
        }
    
    def train_epoch_with_mfu(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch with comprehensive MFU monitoring"""
        print(f"\n Epoch {epoch} - MFU Optimized Training")
        
        total_loss = 0
        total_mfu = 0
        total_flops = 0
        step_count = 0
        
        # Progress bar with MFU metrics
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Training step with MFU measurement
                metrics = self.train_step_with_mfu(batch)
                
                # Update totals
                total_loss += metrics['loss']
                total_mfu += metrics['mfu']
                total_flops += metrics['flops']
                step_count += 1
                
                # Update progress bar
                avg_loss = total_loss / step_count
                avg_mfu = total_mfu / step_count
                avg_flops = total_flops / step_count
                
                progress_bar.set_postfix({
                    'loss': f'{metrics["loss"]:.4f}',
                    'MFU': f'{metrics["mfu"]:.1f}%',
                    'TFLOPS': f'{metrics["flops"]:.2f}',
                    'mem': f'{metrics["memory_gb"]:.1f}GB',
                    'ms': f'{metrics["step_time_ms"]:.1f}'
                })
                
                # Log every 50 steps
                if batch_idx % 50 == 0:
                    self.log_mfu_status(batch_idx, avg_mfu, avg_flops)
                
                # Memory cleanup every 10 steps
                if batch_idx % 10 == 0:
                    self.cleanup_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f" OOM at batch {batch_idx}, cleaning up...")
                    self.cleanup_memory()
                    continue
                else:
                    raise e
        
        # Epoch summary
        avg_loss = total_loss / step_count if step_count > 0 else 0
        avg_mfu = total_mfu / step_count if step_count > 0 else 0
        avg_flops = total_flops / step_count if step_count > 0 else 0
        
        epoch_summary = {
            'avg_loss': avg_loss,
            'avg_mfu': avg_mfu,
            'avg_flops': avg_flops,
            'best_mfu': self.best_mfu,
            'total_steps': step_count
        }
        
        self.print_epoch_summary(epoch, epoch_summary)
        return epoch_summary
    
    def cleanup_memory(self):
        """Clean up memory for optimal MFU"""
        if self.device == 'mps':
            torch.mps.empty_cache()
        elif self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()  # Always try CUDA cleanup
        
    def log_mfu_status(self, step: int, avg_mfu: float, avg_flops: float):
        """Log MFU status"""
        memory = psutil.virtual_memory()
        print(f"   Step {step}: MFU {avg_mfu:.1f}%, {avg_flops:.2f} TFLOPS, "
              f"RAM {memory.percent:.1f}%")
    
    def print_epoch_summary(self, epoch: int, summary: Dict[str, float]):
        """Print detailed epoch summary"""
        print(f"\n Epoch {epoch} Summary:")
        print(f"   Average Loss: {summary['avg_loss']:.4f}")
        print(f"   Average MFU: {summary['avg_mfu']:.2f}%")
        print(f"   Average FLOPS: {summary['avg_flops']:.2f} TFLOPS")
        print(f"   Best MFU: {summary['best_mfu']:.2f}%")
        print(f"   Total Steps: {summary['total_steps']}")
        
        # MFU performance assessment
        if summary['avg_mfu'] > 50:
            print("   🟢 Excellent MFU performance!")
        elif summary['avg_mfu'] > 30:
            print("   🟡 Good MFU performance")
        elif summary['avg_mfu'] > 15:
            print("   🟠 Moderate MFU performance")
        else:
            print("    Low MFU - optimization needed")
    
    def save_mfu_checkpoint(self, epoch: int, metrics: Dict[str, float], path: str):
        """Save checkpoint with MFU metrics"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'mfu_history': self.mfu_history,
            'best_mfu': self.best_mfu,
            'device_optimizations': self.device_optimizations,
            'model_config': {
                'vocab_size': len(self.tokenizer),
                'd_model': getattr(self.model, 'd_model', 'unknown'),
                'device': self.device
            }
        }
        
        torch.save(checkpoint, path)
        print(f" MFU checkpoint saved: {path}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        suggestions = self.mfu_optimizer.suggest_improvements()
        
        return {
            'average_mfu': self.mfu_profiler.get_average_mfu(),
            'peak_mfu': self.mfu_profiler.get_peak_mfu(),
            'device': self.device,
            'device_optimizations': self.device_optimizations,
            'suggestions': suggestions,
            'total_measurements': len(self.mfu_profiler.measurements)
        }


def prepare_sample_data() -> List[str]:
    """Prepare sample training data"""
    return [
        "The development of large language models has revolutionized natural language processing and artificial intelligence research.",
        "Apple Silicon processors with their unified memory architecture provide excellent performance for machine learning workloads.",
        "Model FLOPs Utilization (MFU) measures how efficiently a model uses the available computational resources during training.",
        "Optimizing batch size and sequence length is crucial for achieving maximum throughput in transformer model training.",
        "Memory management and computational efficiency are key factors in scaling language models to billions of parameters.",
        "The Metal Performance Shaders framework enables high-performance GPU computing on Apple devices for AI applications.",
        "Gradient checkpointing and mixed precision training help optimize memory usage while maintaining training stability.",
        "Transformer architectures demonstrate remarkable capabilities in understanding context and generating coherent text.",
        "Advanced optimization techniques like AdamW and learning rate scheduling improve model convergence during training.",
        "The democratization of AI through efficient training methods enables broader access to large language model capabilities."
    ] * 100  # 1000 samples


def main():
    """Main MFU-optimized training function"""
    print(" iLLuMinator MFU-Optimized Training")
    print("=" * 60)
    
    # Device selection
    if torch.backends.mps.is_available():
        device = 'mps'
        print(" Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(" Using CUDA")
    else:
        device = 'cpu'
        print("🧠 Using CPU")
    
    # Initialize components
    print("\n Loading tokenizer...")
    tokenizer = iLLuMinatorTokenizer()
    
    print("\n🧠 Creating model...")
    # Use a smaller model for MFU testing (2.8B parameters)
    model = iLLuMinator4_7B(
        vocab_size=len(tokenizer),
        d_model=2560,      # Reduced for better MFU
        n_layers=20,
        n_heads=20,
        d_ff=10240,
        max_seq_length=512,  # Shorter sequences for better MFU
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    
    # Initialize MFU trainer
    print("\n  Initializing MFU-optimized trainer...")
    trainer = MFUOptimizedTrainer(model, tokenizer, device)
    
    # Prepare data
    print("\n Preparing training data...")
    texts = prepare_sample_data()
    
    # Find optimal batch size
    optimal_batch_size = trainer.find_optimal_batch_size(texts[:50], max_batch_size=8, seq_len=512)
    
    # Create dataset and dataloader
    dataset = MFUOptimizedDataset(texts, tokenizer, max_length=512)
    dataloader = DataLoader(
        dataset,
        batch_size=optimal_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize optimizer
    trainer.initialize_optimizer(lr=1e-4)
    
    # Run initial MFU benchmark
    print(f"\n Running MFU benchmark...")
    benchmark_profiler = benchmark_mfu(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_sizes=[optimal_batch_size],
        seq_lengths=[512]
    )
    
    # Training configuration
    num_epochs = 2
    print(f"\n Starting MFU-Optimized Training!")
    print(f"   Epochs: {num_epochs}")
    print(f"   Optimal batch size: {optimal_batch_size}")
    print(f"   Sequence length: 512")
    print(f"   Training samples: {len(texts):,}")
    print("=" * 60)
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("mfu_logs", exist_ok=True)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        try:
            # Train epoch with MFU monitoring
            epoch_metrics = trainer.train_epoch_with_mfu(dataloader, epoch)
            
            # Save checkpoint
            checkpoint_path = f"checkpoints/illuminator_mfu_epoch_{epoch}.pt"
            trainer.save_mfu_checkpoint(epoch, epoch_metrics, checkpoint_path)
            
            # Save MFU measurements
            mfu_log_path = f"mfu_logs/mfu_measurements_epoch_{epoch}.json"
            trainer.mfu_profiler.save_measurements(mfu_log_path)
            
        except Exception as e:
            print(f" Error in epoch {epoch}: {e}")
            break
    
    # Final summary
    training_time = time.time() - start_time
    
    print(f"\n MFU-Optimized Training Complete!")
    print(f"   Training time: {training_time/3600:.2f} hours")
    print(f"   Device: {device}")
    
    # Print comprehensive MFU summary
    trainer.mfu_profiler.print_summary()
    
    # Generate optimization report
    report = trainer.get_optimization_report()
    
    print(f"\n Optimization Report:")
    print(f"   Average MFU: {report['average_mfu']:.2f}%")
    print(f"   Peak MFU: {report['peak_mfu']:.2f}%")
    print(f"   Suggestions:")
    for i, suggestion in enumerate(report['suggestions'], 1):
        print(f"     {i}. {suggestion}")
    
    # Save final report
    with open("mfu_optimization_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Full report saved: mfu_optimization_report.json")
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = main()
