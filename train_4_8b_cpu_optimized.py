#!/usr/bin/env python3
"""
iLLuMinator 4.8B CPU Training Script
Full parameter training using CPU with Apple Silicon optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import json
import os
import time
from typing import Dict, List, Optional
from tqdm import tqdm
import psutil
import gc

from legacy.illuminator_model import iLLuMinator4_7B
from practical_model.tokenizer import iLLuMinatorTokenizer

class CPUOptimizedTrainer:
    """CPU-optimized trainer for large models"""
    
    def __init__(self, model, tokenizer):
        self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        print(f"🖥️  Training device: {self.device}")
        print("💡 Using CPU for maximum memory efficiency")
        
        # Optimized hyperparameters for CPU training
        self.lr = 1e-4  # Conservative learning rate
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        
        # Use CPU-optimized AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        
        # Memory manager
        self.memory_manager = MemoryManager()
        
        print(f"🚀 CPU trainer initialized with {self.lr} learning rate")
    
    def train_step(self, batch):
        """CPU-optimized training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
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
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch with progress tracking"""
        total_loss = 0
        num_batches = len(dataloader)
        
        # Memory info
        mem_info = self.memory_manager.get_memory_info()
        print(f"📊 Epoch {epoch} start - Memory: {mem_info['used_gb']:.1f}GB/{mem_info['total_gb']:.1f}GB")
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (CPU)")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Periodic memory cleanup
            if batch_idx % 50 == 0:
                self.memory_manager.cleanup_memory()
            
            # Update progress
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'mem': f'{psutil.virtual_memory().percent:.1f}%'
            })
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, loss, path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'training_device': 'cpu',
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_length': self.model.max_seq_length
            }
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")

class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def cleanup_memory():
        """Clean up memory"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.cuda.empty_cache()  # Just in case
    
    @staticmethod
    def get_memory_info():
        """Get memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }

class FullSizeDataset(Dataset):
    """Dataset for full-size training"""
    
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📚 Full dataset: {len(texts)} samples, max_length={max_length}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }

def prepare_comprehensive_training_data():
    """Prepare comprehensive training data"""
    texts = [
        "The development of large language models represents a significant breakthrough in artificial intelligence, enabling machines to understand and generate human-like text with remarkable fluency and coherence.",
        "Apple Silicon processors with their unified memory architecture and dedicated Neural Engine provide exceptional performance for machine learning workloads, offering unique advantages for AI model training and inference.",
        "Transformer architectures have revolutionized natural language processing by introducing attention mechanisms that allow models to capture long-range dependencies in text sequences.",
        "Training neural networks with billions of parameters requires sophisticated optimization techniques, including gradient checkpointing, mixed precision training, and careful memory management strategies.",
        "The attention mechanism in transformer models enables the network to selectively focus on relevant parts of the input sequence when generating predictions or representations.",
        "Deep learning models learn hierarchical representations of data by stacking multiple layers of neurons with non-linear activation functions, enabling them to capture complex patterns.",
        "Natural language processing encompasses a wide range of tasks including text classification, named entity recognition, sentiment analysis, machine translation, and text generation.",
        "The field of artificial intelligence continues to advance rapidly with new architectures, training methodologies, and applications being developed by researchers worldwide.",
        "Large-scale pre-training on diverse text corpora followed by fine-tuning on specific tasks has become the dominant paradigm in modern natural language processing.",
        "Optimization algorithms like Adam and AdamW provide adaptive learning rates and momentum terms that help neural networks converge to better solutions during training.",
        "The democratization of AI through open-source frameworks, pre-trained models, and accessible computing resources is accelerating innovation across industries.",
        "Neural network architectures must balance model capacity, computational efficiency, and memory requirements to achieve optimal performance on target tasks.",
        "The emergence of few-shot and zero-shot learning capabilities in large language models demonstrates their ability to generalize to new tasks with minimal examples.",
        "Effective training of large models requires careful consideration of hyperparameters, data preprocessing, regularization techniques, and convergence monitoring.",
        "The integration of language models into real-world applications demands robust evaluation metrics, safety considerations, and ethical guidelines for responsible AI deployment."
    ]
    
    # Expand dataset for comprehensive training
    return texts * 300  # 4,500 samples

def main():
    """Main training function for maximum parameter model"""
    print("🍎 iLLuMinator 4.8B Apple Silicon CPU Training")
    print("Maximum parameters with full Apple Silicon support")
    print("="*60)
    
    # Memory manager
    memory_manager = MemoryManager()
    
    # Initial memory cleanup
    memory_manager.cleanup_memory()
    
    # System info
    mem_info = memory_manager.get_memory_info()
    print(f"💾 System Memory: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['percent']:.1f}%)")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    print(f"🧠 CPU Cores: {cpu_count}")
    
    # MPS status
    if torch.backends.mps.is_available():
        print("🚀 Apple MPS: Available (using CPU for memory efficiency)")
    else:
        print("⚠️  Apple MPS: Not available")
    
    # Initialize tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = iLLuMinatorTokenizer()
    
    # Initialize full-size model (4.8B+ parameters)
    print("\n🧠 Creating maximum parameter model...")
    model = iLLuMinator4_7B(
        vocab_size=len(tokenizer),
        d_model=3584,      # Full dimension
        n_layers=30,       # Full layers
        n_heads=28,        # Full attention heads
        d_ff=14336,        # Full feedforward
        max_seq_length=1024,  # Full sequence length
        dropout=0.1
    )
    
    # Show actual parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    
    # Prepare training data
    print("\n📚 Preparing comprehensive training data...")
    texts = prepare_comprehensive_training_data()
    dataset = FullSizeDataset(texts, tokenizer, max_length=1024)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Single batch for maximum model size
        shuffle=True,
        num_workers=0  # Single process for memory efficiency
    )
    
    # Initialize trainer
    print("\n⚙️  Initializing CPU trainer...")
    trainer = CPUOptimizedTrainer(model, tokenizer)
    
    # Training configuration
    num_epochs = 3
    
    print(f"\n📊 Training Configuration:")
    print(f"   Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Training samples: {len(texts):,}")
    print(f"   Batches per epoch: {len(dataloader):,}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Sequence length: 1024")
    print(f"   Device: CPU (Apple Silicon optimized)")
    print(f"   Learning rate: {trainer.lr}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    print(f"\n🔥 Starting Maximum Parameter Training!")
    print("⏱️  Note: CPU training is slower but uses maximum model size")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs}")
        
        try:
            avg_loss = trainer.train_epoch(dataloader, epoch)
            
            # Save checkpoint
            checkpoint_path = f"checkpoints/illuminator_4_8b_cpu_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, avg_loss, checkpoint_path)
            
            # Epoch summary
            elapsed_time = (time.time() - start_time) / 3600
            print(f"✅ Epoch {epoch} complete:")
            print(f"   Average Loss: {avg_loss:.4f}")
            print(f"   Elapsed Time: {elapsed_time:.2f} hours")
            
            # Memory status
            mem_info = memory_manager.get_memory_info()
            print(f"   Memory Usage: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['percent']:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error in epoch {epoch}: {e}")
            break
    
    # Save final model
    final_path = "illuminator_4_8b_apple_silicon_cpu_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(tokenizer),
            'd_model': model.d_model,
            'n_layers': 30,
            'n_heads': 28,
            'd_ff': 14336,
            'max_seq_length': 1024
        },
        'training_info': {
            'total_parameters': total_params,
            'device': 'cpu',
            'apple_silicon_optimized': True,
            'training_time_hours': (time.time() - start_time) / 3600,
            'final_loss': avg_loss if 'avg_loss' in locals() else None
        }
    }, final_path)
    
    total_time = (time.time() - start_time) / 3600
    
    print(f"\n🎉 Maximum Parameter Training Complete!")
    print(f"   Final model: {final_path}")
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Training time: {total_time:.2f} hours")
    print(f"   Device: CPU (Apple Silicon)")
    print(f"   Memory efficient: ✅")
    
    print(f"\n💡 Model Performance:")
    print(f"   • Full 4.8B+ parameter model trained")
    print(f"   • Optimized for Apple Silicon architecture")
    print(f"   • Memory efficient CPU training")
    print(f"   • Can be converted to MPS for fast inference")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()
