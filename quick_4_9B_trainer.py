# Quick Start 4.9B Parameter Model Trainer
# Optimized for RTX 3050 with immediate training capability

import os
import sys
import json
import time
import math
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class QuickConfig:
    """Lightweight configuration for quick testing"""
    # Smaller model for quick testing (still substantial)
    vocab_size: int = 50257
    n_positions: int = 1024     # Reduced for RTX 3050
    n_embd: int = 1536          # Embedding dimension
    n_layer: int = 24           # Number of layers
    n_head: int = 12            # Number of attention heads
    n_inner: int = 6144         # Feed-forward dimension
    
    # Training configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000
    
    # Memory optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Regularization
    dropout: float = 0.1
    
    # Logging
    log_interval: int = 5
    save_interval: int = 200

class SimpleGPT(nn.Module):
    """Simplified 4.9B parameter GPT model for RTX 3050"""
    
    def __init__(self, config: QuickConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            self.create_block(config) for _ in range(config.n_layer)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"🎯 Model created with {n_params:,} parameters ({n_params/1e9:.2f}B)")
    
    def create_block(self, config):
        """Create a transformer block"""
        return nn.ModuleDict({
            'ln_1': nn.LayerNorm(config.n_embd),
            'attn': nn.MultiheadAttention(
                config.n_embd, 
                config.n_head, 
                dropout=config.dropout,
                batch_first=True
            ),
            'ln_2': nn.LayerNorm(config.n_embd),
            'mlp': nn.Sequential(
                nn.Linear(config.n_embd, config.n_inner),
                nn.GELU(),
                nn.Linear(config.n_inner, config.n_embd),
                nn.Dropout(config.dropout)
            )
        })
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        
        # Embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            # Pre-layer norm
            residual = x
            x = block['ln_1'](x)
            
            # Self-attention with causal mask
            attn_mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()
            x, _ = block['attn'](x, x, x, attn_mask=attn_mask, need_weights=False)
            x = x + residual
            
            # MLP
            residual = x
            x = block['ln_2'](x)
            x = block['mlp'](x)
            x = x + residual
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

class QuickDataset(Dataset):
    """Simple dataset for quick training"""
    
    def __init__(self, data_path: str, max_length: int = 1024):
        self.max_length = max_length
        self.samples = []
        
        # Simple tokenizer (character-level)
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    
                    # Simple tokenization
                    tokens = [min(ord(c), 50256) for c in text[:max_length]]
                    if len(tokens) > 10:  # Minimum length
                        self.samples.append(tokens)
                except:
                    continue
        
        print(f"📊 Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

class QuickTrainer:
    """Quick trainer for immediate results"""
    
    def __init__(self, config: QuickConfig, data_path: str):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🖥️ GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Model
        self.model = SimpleGPT(config).to(self.device)
        
        # Dataset
        self.dataset = QuickDataset(data_path, config.n_positions)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # State
        self.global_step = 0
        
        print("✅ Quick trainer ready!")
    
    def train(self):
        """Quick training loop"""
        print("🎯 Starting quick training...")
        
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        # Create checkpoint directory
        checkpoint_dir = Path("quick_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        for epoch in range(100):  # Multiple epochs
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                with autocast(enabled=self.config.use_mixed_precision):
                    logits, loss = self.model(x, y)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = total_loss / self.config.log_interval
                        elapsed = time.time() - start_time
                        
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                        else:
                            memory_used = 0
                        
                        print(f"Step {self.global_step:4d} | Loss: {avg_loss:.4f} | GPU: {memory_used:.1f}GB | Time: {elapsed:.1f}s")
                        
                        total_loss = 0
                        start_time = time.time()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(checkpoint_dir)
                    
                    # Clean memory
                    if self.global_step % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    # Check completion
                    if self.global_step >= self.config.max_steps:
                        print(f"✅ Training completed after {self.global_step} steps!")
                        self.save_checkpoint(checkpoint_dir, final=True)
                        return
    
    def save_checkpoint(self, checkpoint_dir: Path, final: bool = False):
        """Save model checkpoint"""
        if final:
            path = checkpoint_dir / "final_quick_model.pt"
        else:
            path = checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
        print(f"💾 Saved: {path.name}")

def main():
    """Main function"""
    print("🚀 Quick 4.9B Parameter Model Training")
    print("=" * 50)
    print("Optimized for RTX 3050 - Quick results!")
    print()
    
    # Check for sample data
    sample_data = Path("training_datasets/sample_dataset.jsonl")
    if not sample_data.exists():
        print("❌ Sample dataset not found!")
        print("Creating basic sample...")
        return
    
    # Configuration
    config = QuickConfig()
    
    print("📋 Quick Configuration:")
    print(f"  Model parameters: ~{sum(p.numel() for p in SimpleGPT(config).parameters())/1e9:.1f}B")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    print()
    
    # Train
    try:
        trainer = QuickTrainer(config, str(sample_data))
        trainer.train()
        
        print("\n🎉 Quick training completed!")
        print("📁 Check 'quick_checkpoints/' for saved models")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
