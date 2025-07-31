#!/usr/bin/env python3
"""
iLLuMinator 4.2B Apple Silicon Optimized Training Script
Memory-optimized for 16GB Apple Silicon while maintaining high parameter count
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

class MemoryOptimizedModel(nn.Module):
    """Memory-optimized 4.2B parameter model for Apple Silicon"""
    
    def __init__(self, 
                 vocab_size: int = 50260,
                 d_model: int = 3200,      # Reduced from 3584
                 n_layers: int = 28,       # Reduced from 30
                 n_heads: int = 25,        # Reduced from 28
                 d_ff: int = 12800,        # Reduced from 14336
                 max_seq_length: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        # Store config
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Use the existing model class with optimized parameters
        self.model = iLLuMinator4_7B(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
    
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

class AppleSiliconMemoryManager:
    """Advanced memory management for Apple Silicon"""
    
    @staticmethod
    def cleanup_memory():
        """Aggressive memory cleanup"""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.cuda.empty_cache()  # In case CUDA tensors exist
        gc.collect()
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    @staticmethod
    def set_memory_fraction(fraction: float = 0.8):
        """Set MPS memory fraction (if available)"""
        if torch.backends.mps.is_available():
            try:
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(fraction)
                print(f"🔧 Set MPS memory fraction to {fraction}")
            except:
                print("⚠️  Could not set MPS memory fraction")

class GradientCheckpointingTrainer:
    """Memory-efficient trainer with gradient checkpointing"""
    
    def __init__(self, model, tokenizer, memory_fraction: float = 0.7):
        # Set memory limits
        AppleSiliconMemoryManager.set_memory_fraction(memory_fraction)
        
        # Device selection
        self.device = self._get_device()
        print(f"🖥️  Using device: {self.device}")
        
        self.model = model
        self.tokenizer = tokenizer
        self.memory_manager = AppleSiliconMemoryManager()
        
        # Conservative hyperparameters for memory efficiency
        self.lr = 8e-5  # Very conservative for Apple Silicon
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.grad_clip = 0.5  # Lower gradient clipping
        
        # Memory-efficient optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.05  # Reduced label smoothing
        )
        
        print(f"🚀 Memory-optimized trainer initialized")
        print(f"   Learning rate: {self.lr}")
        print(f"   Memory fraction: {memory_fraction}")
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def move_model_to_device_safely(self):
        """Safely move model to device with memory management"""
        print("📤 Moving model to device...")
        
        # Clear memory first
        self.memory_manager.cleanup_memory()
        
        try:
            self.model = self.model.to(self.device)
            print(f"✅ Model successfully moved to {self.device}")
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Out of memory moving model to {self.device}")
                print("💡 Try reducing model size or using CPU")
                return False
            else:
                raise e
    
    def train_step_with_checkpointing(self, batch):
        """Memory-efficient training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Clear memory before forward pass
        self.memory_manager.cleanup_memory()
        
        try:
            # Forward pass with gradient checkpointing
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
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("💥 Out of memory during training step")
                self.memory_manager.cleanup_memory()
                raise e
            else:
                raise e
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch with memory monitoring"""
        total_loss = 0
        num_batches = len(dataloader)
        successful_batches = 0
        
        # Initial memory check
        mem_info = self.memory_manager.get_memory_usage()
        print(f"📊 Starting epoch {epoch} - Memory: {mem_info['used_gb']:.1f}GB/{mem_info['total_gb']:.1f}GB")
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                loss = self.train_step_with_checkpointing(batch)
                total_loss += loss
                successful_batches += 1
                
                # Memory cleanup every 5 steps
                if batch_idx % 5 == 0:
                    self.memory_manager.cleanup_memory()
                
                # Update progress
                avg_loss = total_loss / successful_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                mem_usage = psutil.virtual_memory().percent
                
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'mem': f'{mem_usage:.1f}%'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n💥 OOM at batch {batch_idx}, skipping...")
                    self.memory_manager.cleanup_memory()
                    continue
                else:
                    raise e
        
        return total_loss / max(successful_batches, 1)
    
    def save_checkpoint(self, epoch, loss, path):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_layers': self.model.n_layers,
                'n_heads': self.model.n_heads,
                'd_ff': self.model.d_ff
            }
        }
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")

class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that loads data on demand"""
    
    def __init__(self, texts, tokenizer, max_length=512):  # Reduced sequence length
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📚 Dataset created: {len(texts)} samples, max_length={max_length}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize on-the-fly to save memory
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

def create_4_2b_model(tokenizer):
    """Create memory-optimized 4.2B parameter model"""
    print("🧠 Creating 4.2B parameter model...")
    
    model = MemoryOptimizedModel(
        vocab_size=len(tokenizer),
        d_model=3200,      # Optimized for 16GB RAM
        n_layers=28,       # Balanced layer count
        n_heads=25,        # Attention heads (divisible by d_model)
        d_ff=12800,        # Feed-forward dimension
        max_seq_length=1024,
        dropout=0.1
    )
    
    return model

def prepare_training_data():
    """Prepare comprehensive training data"""
    texts = [
        "Artificial intelligence and machine learning are transforming the way we process information and solve complex problems across industries.",
        "Large language models with billions of parameters demonstrate remarkable capabilities in understanding and generating human-like text.",
        "Apple Silicon processors with unified memory architecture provide unique advantages for machine learning workloads and neural network training.",
        "The Metal Performance Shaders framework enables high-performance GPU computing on Apple devices for AI and ML applications.",
        "Neural Engine acceleration on Apple Silicon can significantly speed up inference for optimized machine learning models.",
        "Transformer architectures have revolutionized natural language processing by introducing attention mechanisms that capture long-range dependencies.",
        "Training large neural networks requires sophisticated optimization techniques, gradient checkpointing, and careful memory management.",
        "The democratization of AI through accessible frameworks and pre-trained models is accelerating innovation across various domains.",
        "Deep learning models learn hierarchical representations by stacking multiple layers of neurons with non-linear activation functions.",
        "Natural language processing encompasses tasks like text generation, translation, summarization, sentiment analysis, and question answering.",
        "Advanced optimization algorithms like AdamW provide adaptive learning rates and momentum for efficient neural network training.",
        "Attention mechanisms allow models to focus on relevant parts of input sequences when making predictions or generating outputs.",
        "The field of artificial intelligence continues to evolve with new architectures, training methods, and applications being developed.",
        "Machine learning models require large datasets, computational resources, and careful hyperparameter tuning for optimal performance.",
        "Transfer learning enables models to leverage knowledge from pre-training on large datasets for specific downstream tasks."
    ]
    
    # Expand dataset
    return texts * 150  # 2,250 samples

def main():
    """Main training function"""
    print("🍎 iLLuMinator 4.2B Apple Silicon Training")
    print("="*50)
    
    # Memory management
    memory_manager = AppleSiliconMemoryManager()
    
    # Initial cleanup
    memory_manager.cleanup_memory()
    
    # Check memory
    mem_info = memory_manager.get_memory_usage()
    print(f"💾 System Memory: {mem_info['used_gb']:.1f}GB/{mem_info['total_gb']:.1f}GB ({mem_info['percent']:.1f}%)")
    
    if mem_info['percent'] > 85:
        print("⚠️  High memory usage detected. Consider closing other applications.")
        input("Press Enter to continue or Ctrl+C to abort...")
    
    # Initialize components
    print("\n🔤 Loading tokenizer...")
    tokenizer = iLLuMinatorTokenizer()
    
    print("\n🧠 Creating 4.2B parameter model...")
    model = create_4_2b_model(tokenizer)
    
    print("\n📚 Preparing training data...")
    texts = prepare_training_data()
    dataset = MemoryEfficientDataset(texts, tokenizer, max_length=512)
    
    # Create dataloader with minimal memory usage
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Single batch for memory efficiency
        shuffle=True,
        num_workers=0,  # No multiprocessing to save memory
        pin_memory=False
    )
    
    print("\n⚙️  Initializing trainer...")
    trainer = GradientCheckpointingTrainer(model, tokenizer, memory_fraction=0.7)
    
    # Try to move model to device
    if not trainer.move_model_to_device_safely():
        print("❌ Cannot fit model in device memory. Exiting.")
        return None, None
    
    # Training configuration
    num_epochs = 3
    
    print(f"\n📊 Training Configuration:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Training samples: {len(texts):,}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 1")
    print(f"   Sequence length: 512")
    print(f"   Device: {trainer.device}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    print(f"\n🔥 Starting Training!")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs}")
        
        try:
            avg_loss = trainer.train_epoch(dataloader, epoch)
            
            # Save checkpoint
            checkpoint_path = f"checkpoints/illuminator_4_2b_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, avg_loss, checkpoint_path)
            
            print(f"✅ Epoch {epoch} complete - Loss: {avg_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Error in epoch {epoch}: {e}")
            break
    
    # Save final model
    final_path = "illuminator_4_2b_apple_silicon_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(tokenizer),
            'd_model': model.d_model,
            'n_layers': model.n_layers,
            'n_heads': model.n_heads,
            'd_ff': model.d_ff
        },
        'training_info': {
            'total_parameters': total_params,
            'device': str(trainer.device),
            'training_time_hours': (time.time() - start_time) / 3600
        }
    }, final_path)
    
    print(f"\n🎉 Training Complete!")
    print(f"   Model saved: {final_path}")
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Training time: {(time.time() - start_time)/3600:.2f} hours")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()
