#!/usr/bin/env python3
"""
iLLuMinator 4.9B Apple Silicon Training Script
Full parameter training with MPS and Neural Engine optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.backends.mps
import math
import json
import os
import time
from typing import Dict, List, Optional
from tqdm import tqdm
import psutil
import gc

from enhanced_illuminator_4_9b import iLLuMinator4_9B
from practical_model.tokenizer import iLLuMinatorTokenizer

class AppleSiliconOptimizer:
    """Apple Silicon specific optimizations"""
    
    @staticmethod
    def is_mps_available():
        """Check if MPS is available"""
        return torch.backends.mps.is_available()
    
    @staticmethod
    def optimize_memory():
        """Optimize memory for Apple Silicon"""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_optimal_batch_size(model_params: int, available_memory_gb: float):
        """Calculate optimal batch size for Apple Silicon with Enhanced 4.9B model"""
        # Conservative estimate for enhanced 4.9B model on various RAM configurations
        if model_params > 4.5e9:  # 4.5B+ parameters (enhanced model)
            if available_memory_gb >= 32:
                return 2  # Higher-end Apple Silicon systems
            elif available_memory_gb >= 16:
                return 1  # Standard configuration
            else:
                return 1  # Ultra conservative for 8GB systems
        return 2
    
    @staticmethod
    def get_memory_info():
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }

class OptimizedTextDataset(Dataset):
    """Memory-optimized dataset for large models"""
    
    def __init__(self, texts: List[str], tokenizer: iLLuMinatorTokenizer, 
                 max_length: int = 1024, use_mps: bool = False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mps = use_mps
        
        # Pre-tokenize for faster training (memory permitting)
        print("Pre-tokenizing dataset for faster training...")
        self.tokenized_data = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text, max_length=max_length)
            
            # Pad or truncate
            if len(tokens) < max_length:
                tokens.extend([self.tokenizer.tokenizer.pad_token_id] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
            
            self.tokenized_data.append(tokens)
        
        print(f"Dataset prepared: {len(self.tokenized_data)} samples")
        
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

class AppleSiliconTrainer:
    """Optimized trainer for Apple Silicon MPS with Enhanced Model"""
    
    def __init__(self, 
                 model: iLLuMinator4_9B,
                 tokenizer: iLLuMinatorTokenizer,
                 use_mixed_precision: bool = True):
        
        # Device selection with Apple Silicon priority
        self.device = self._select_optimal_device()
        print(f"  Training device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        # Apple Silicon specific optimizations
        self.use_mixed_precision = use_mixed_precision and self.device == 'mps'
        
        # Optimized hyperparameters for Apple Silicon
        self.lr = self._get_optimal_lr()
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # Advanced optimizer configuration
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        
        # Memory optimization
        self.memory_optimizer = AppleSiliconOptimizer()
        
        print(f"Trainer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Learning rate: {self.lr}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        
    def _select_optimal_device(self) -> str:
        """Select the best available device"""
        if torch.backends.mps.is_available():
            # Verify MPS works with a test tensor
            try:
                test_tensor = torch.randn(10, 10).to('mps')
                _ = test_tensor @ test_tensor  # Test operation
                return 'mps'
            except Exception as e:
                print(f"MPS test failed: {e}, falling back to CPU")
                return 'cpu'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def _get_optimal_lr(self) -> float:
        """Get optimal learning rate based on device"""
        if self.device == 'mps':
            return 1e-4  # Conservative for Apple Silicon
        elif self.device == 'cuda':
            return 2e-4  # Slightly higher for CUDA
        else:
            return 5e-5  # Very conservative for CPU
    
    def create_lr_scheduler(self, num_training_steps: int, num_warmup_steps: int = 1000):
        """Create cosine learning rate scheduler with warmup"""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Optimized training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Forward pass
        with torch.amp.autocast(device_type='cpu', enabled=False):  # MPS doesn't support autocast yet
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, scheduler=None) -> float:
        """Train one epoch with memory optimization"""
        total_loss = 0
        num_batches = len(dataloader)
        
        # Memory info at start
        mem_info = self.memory_optimizer.get_memory_info()
        print(f"Memory at epoch start: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['percent']:.1f}%)")
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss = self.train_step(batch)
            total_loss += loss
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Memory cleanup every 10 steps
            if batch_idx % 10 == 0:
                self.memory_optimizer.optimize_memory()
            
            # Update progress
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'mem': f'{psutil.virtual_memory().percent:.1f}%'
            })
        
        # Final memory cleanup
        self.memory_optimizer.optimize_memory()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float, save_path: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_length': self.model.max_seq_length
            },
            'training_config': {
                'device': self.device,
                'learning_rate': self.lr,
                'mixed_precision': self.use_mixed_precision
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

def prepare_training_data(size: str = "large") -> List[str]:
    """Prepare comprehensive training data"""
    
    base_texts = [
        "The rapid advancement of artificial intelligence is transforming industries and reshaping how we work, learn, and interact with technology.",
        "Machine learning models with billions of parameters can learn complex patterns from vast datasets, enabling unprecedented capabilities in language understanding.",
        "Transformer architectures have revolutionized natural language processing by introducing the attention mechanism that allows models to focus on relevant information.",
        "Training large language models requires sophisticated optimization techniques, massive computational resources, and carefully curated datasets.",
        "The attention mechanism enables models to weigh the importance of different parts of the input sequence when making predictions.",
        "Deep learning networks utilize multiple layers of neurons to learn hierarchical representations of data, from simple features to complex abstractions.",
        "Natural language processing encompasses various tasks including text generation, translation, summarization, and question answering.",
        "The field of artificial intelligence continues to evolve rapidly with new architectures, training methods, and applications being developed constantly.",
        "Large language models demonstrate emergent abilities that weren't explicitly programmed, such as few-shot learning and complex reasoning.",
        "The development of efficient training algorithms has made it possible to train models with hundreds of billions of parameters.",
        "Apple Silicon processors with their unified memory architecture provide unique advantages for machine learning workloads.",
        "The Metal Performance Shaders framework enables high-performance GPU computing on Apple devices for AI and ML applications.",
        "Neural Engine acceleration on Apple Silicon can significantly speed up inference for optimized machine learning models.",
        "Advanced optimization techniques like gradient checkpointing and mixed precision training help manage memory usage during training.",
        "The democratization of AI through accessible frameworks and pre-trained models is accelerating innovation across various domains."
    ]
    
    # Expand dataset based on size
    if size == "small":
        return base_texts * 50   # 750 samples
    elif size == "medium":
        return base_texts * 100  # 1,500 samples  
    elif size == "large":
        return base_texts * 200  # 3,000 samples
    else:
        return base_texts * 300  # 4,500 samples

def main():
    """Main training function for 4.9B parameter model"""
    print("iLLuMinator 4.9B Apple Silicon Training")
    print("=" * 50)
    
    # System checks
    print("\n System Check:")
    mem_info = AppleSiliconOptimizer.get_memory_info()
    print(f"   Total RAM: {mem_info['total_gb']:.1f}GB")
    print(f"   Available RAM: {mem_info['available_gb']:.1f}GB")
    print(f"   Memory usage: {mem_info['percent']:.1f}%")
    
    if AppleSiliconOptimizer.is_mps_available():
        print("   Apple Silicon MPS: Available")
    else:
        print("   Apple Silicon MPS: Not available")
    
    # Training configuration
    num_epochs = 3
    max_sequence_length = 1024  # Reduced for memory efficiency
    
    print(f"\nTraining Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Sequence Length: {max_sequence_length}")
    print(f"   Target Model Size: 4.9B parameters")
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = iLLuMinatorTokenizer()
    
    # Initialize model (Enhanced 4.9B parameters)
    print("\nInitializing Enhanced 4.9B parameter model...")
    model = iLLuMinator4_9B(
        vocab_size=65536,      # Enhanced vocabulary size
        d_model=4096,          # Enhanced model dimension
        n_layers=32,           # Enhanced layer count
        n_heads=32,            # Enhanced attention heads
        n_kv_heads=8,          # Grouped query attention
        d_ff=14336,            # Enhanced feedforward dimension
        max_seq_length=max_sequence_length,
        dropout=0.0,           # Lower dropout for large models
        tie_embeddings=True    # Tie embeddings for efficiency
    )
    
    # Prepare training data
    print("\nPreparing training data...")
    texts = prepare_training_data("large")
    dataset = OptimizedTextDataset(
        texts, 
        tokenizer, 
        max_length=max_sequence_length,
        use_mps=AppleSiliconOptimizer.is_mps_available()
    )
    
    # Calculate optimal batch size
    model_params = sum(p.numel() for p in model.parameters())
    batch_size = AppleSiliconOptimizer.get_optimal_batch_size(
        model_params, 
        mem_info['available_gb']
    )
    
    print(f"   Batch size: {batch_size} (optimized for Apple Silicon)")
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Single worker for MPS compatibility
        pin_memory=False  # Don't pin memory for MPS
    )
    
    # Initialize trainer
    print("\nInitializing Apple Silicon trainer...")
    trainer = AppleSiliconTrainer(model, tokenizer, use_mixed_precision=True)
    
    # Setup learning rate scheduler
    total_steps = len(dataloader) * num_epochs
    scheduler = trainer.create_lr_scheduler(
        num_training_steps=total_steps,
        num_warmup_steps=min(1000, total_steps // 10)  # 10% warmup
    )
    
    print(f"\nTraining Statistics:")
    print(f"   Model parameters: {model_params:,}")
    print(f"   Training samples: {len(texts):,}")
    print(f"   Batches per epoch: {len(dataloader):,}")
    print(f"   Total training steps: {total_steps:,}")
    print(f"   Learning rate: {trainer.lr}")
    print(f"   Device: {trainer.device}")
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    print(f"\nStarting 4.9B Parameter Training!")
    print("=" * 50)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train epoch
        avg_loss = trainer.train_epoch(dataloader, epoch, scheduler)
        
        # Epoch summary
        current_lr = trainer.optimizer.param_groups[0]['lr']
        elapsed_time = time.time() - start_time
        
        print(f"\nEpoch {epoch} Results:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        print(f"   Elapsed Time: {elapsed_time/3600:.2f} hours")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/illuminator_4_9b_epoch_{epoch}.pt"
        trainer.save_checkpoint(epoch, avg_loss, checkpoint_path)
        
        # Memory status
        mem_info = AppleSiliconOptimizer.get_memory_info()
        print(f"   Memory Usage: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['percent']:.1f}%)")
    
    # Save final model
    print(f"\nSaving final Enhanced 4.9B parameter model...")
    final_model_path = "illuminator_4_9b_enhanced_apple_silicon_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'training_info': {
            'total_epochs': num_epochs,
            'final_loss': avg_loss,
            'total_parameters': model_params,
            'device': trainer.device,
            'training_time_hours': (time.time() - start_time) / 3600
        }
    }, final_model_path)
    
    total_time = time.time() - start_time
    print(f"\nTraining Complete!")
    print(f"   Final model saved: {final_model_path}")
    print(f"   Total parameters: {model_params:,}")
    print(f"   Training time: {total_time/3600:.2f} hours")
    print(f"   Device used: {trainer.device}")
    
    return model, tokenizer

if __name__ == "__main__":
    # Check system compatibility
    if not AppleSiliconOptimizer.is_mps_available():
        print("Warning: MPS not available. Training will use CPU (very slow).")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit(1)
    
    # Start training
    model, tokenizer = main()
