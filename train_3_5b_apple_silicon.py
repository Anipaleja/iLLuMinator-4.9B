#!/usr/bin/env python3
"""
iLLuMinator 3.5B Apple Silicon Training Script
Ultra-optimized for 16GB Apple Silicon with Neural Engine support
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

from legacy.illuminator_model import iLLuMinator4_7B
from practical_model.tokenizer import iLLuMinatorTokenizer

class UltraOptimizedModel(nn.Module):
    """Ultra-optimized 3.5B parameter model for 16GB Apple Silicon"""
    
    def __init__(self, 
                 vocab_size: int = 50260,
                 d_model: int = 2816,      # Optimized dimension (divisible by 32)
                 n_layers: int = 24,       # Reduced layers
                 n_heads: int = 22,        # Optimized heads (divisible by d_model)
                 d_ff: int = 11264,        # 4x d_model
                 max_seq_length: int = 512, # Reduced sequence length
                 dropout: float = 0.1):
        super().__init__()
        
        # Store config
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Use existing model with ultra-optimized parameters
        self.model = iLLuMinator4_7B(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Ultra-Optimized iLLuMinator Configuration:")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_params/1e9:.2f}B parameters")
        print(f"Target: 3.5B parameters for Apple Silicon")
    
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

class AppleSiliconTrainer:
    """Ultra-efficient trainer for Apple Silicon"""
    
    def __init__(self, model, tokenizer):
        # Aggressive memory settings
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.6'  # Use only 60% of available memory
        
        # Device selection
        self.device = self._select_device()
        print(f"🖥️  Training device: {self.device}")
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Ultra-conservative hyperparameters
        self.lr = 5e-5  # Very small learning rate
        self.weight_decay = 0.01
        
        # Lightweight optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Simple loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.tokenizer.pad_token_id
        )
        
        print(f"🚀 Ultra-efficient trainer initialized")
    
    def _select_device(self):
        """Select optimal device"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def move_model_safely(self):
        """Safely move model to device"""
        try:
            # Clear all caches
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            # Move model
            self.model = self.model.to(self.device)
            print(f"✅ Model moved to {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to move model: {e}")
            return False
    
    def train_step(self, batch):
        """Ultra-efficient training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Forward pass
        logits = self.model(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Light gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                loss = self.train_step(batch)
                total_loss += loss
                
                # Memory cleanup every batch
                if batch_idx % 1 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                
                # Update progress
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'mem': f'{psutil.virtual_memory().percent:.1f}%'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"💥 OOM at batch {batch_idx}")
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    continue
                else:
                    raise e
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, loss, path):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, path)
        print(f"💾 Saved: {path}")

class EfficientDataset(Dataset):
    """Ultra-efficient dataset"""
    
    def __init__(self, texts, tokenizer, max_length=256):  # Very short sequences
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📚 Efficient dataset: {len(texts)} samples, max_length={max_length}")
    
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

def main():
    """Main training function"""
    print("🍎 iLLuMinator 3.5B Apple Silicon Training")
    print("Ultra-optimized for 16GB RAM + MPS + Neural Engine")
    print("="*55)
    
    # Memory check
    memory = psutil.virtual_memory()
    print(f"💾 System Memory: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    if memory.percent > 80:
        print("⚠️  High memory usage - close other apps first!")
        input("Press Enter to continue...")
    
    # Initialize components
    print("\n🔤 Loading tokenizer...")
    tokenizer = iLLuMinatorTokenizer()
    
    print("\n🧠 Creating ultra-optimized 3.5B model...")
    model = UltraOptimizedModel(
        vocab_size=len(tokenizer),
        d_model=2816,
        n_layers=24,
        n_heads=22,
        d_ff=11264,
        max_seq_length=512,
        dropout=0.1
    )
    
    # Training data
    print("\n📚 Preparing training data...")
    texts = [
        "Artificial intelligence on Apple Silicon processors demonstrates exceptional performance with optimized neural networks.",
        "Machine learning models benefit significantly from the unified memory architecture of Apple Silicon chips.",
        "The Metal Performance Shaders framework provides high-performance GPU acceleration for AI workloads on Apple devices.",
        "Neural Engine acceleration enables efficient inference for machine learning models on Apple Silicon.",
        "Large language models can be optimized for Apple Silicon through careful memory management and MPS utilization.",
        "Deep learning training on Apple Silicon requires thoughtful optimization of batch sizes and memory usage.",
        "Transformer architectures work exceptionally well on Apple Silicon when properly configured for MPS backend.",
        "The combination of CPU, GPU, and Neural Engine on Apple Silicon creates a powerful AI computing platform.",
        "Gradient checkpointing and mixed precision training help optimize memory usage on Apple Silicon devices.",
        "Apple Silicon's unified memory allows for efficient data sharing between CPU and GPU during model training."
    ]
    
    # Expand dataset
    texts = texts * 200  # 2,000 samples
    
    dataset = EfficientDataset(texts, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Initialize trainer
    print("\n⚙️  Initializing ultra-efficient trainer...")
    trainer = AppleSiliconTrainer(model, tokenizer)
    
    # Move model to device
    if not trainer.move_model_safely():
        print("❌ Cannot fit model in memory")
        return None, None
    
    # Training info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Training Configuration:")
    print(f"   Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Training samples: {len(texts):,}")
    print(f"   Sequence length: 256")
    print(f"   Batch size: 1")
    print(f"   Device: {trainer.device}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training
    print(f"\n🔥 Starting Ultra-Optimized Training!")
    start_time = time.time()
    
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs}")
        
        try:
            avg_loss = trainer.train_epoch(dataloader, epoch)
            
            # Save checkpoint
            checkpoint_path = f"checkpoints/illuminator_3_5b_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, avg_loss, checkpoint_path)
            
            print(f"✅ Epoch {epoch} - Loss: {avg_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Error in epoch {epoch}: {e}")
            break
    
    # Save final model
    final_path = "illuminator_3_5b_apple_silicon_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(tokenizer),
            'd_model': model.d_model,
            'n_layers': model.n_layers,
            'n_heads': model.n_heads,
            'd_ff': model.d_ff,
            'max_seq_length': model.max_seq_length
        },
        'training_info': {
            'total_parameters': total_params,
            'device': str(trainer.device),
            'apple_silicon_optimized': True,
            'mps_backend': torch.backends.mps.is_available(),
            'training_time_hours': (time.time() - start_time) / 3600
        }
    }, final_path)
    
    print(f"\n🎉 Apple Silicon Training Complete!")
    print(f"   Model: {final_path}")
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"   Device: {trainer.device}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()
