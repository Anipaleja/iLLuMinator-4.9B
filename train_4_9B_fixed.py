#!/usr/bin/env python3
"""Fixed 4.9B training without legacy dependencies"""

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
import numpy as np

warnings.filterwarnings("ignore")

@dataclass
class Config4_9B:
    """Configuration for 4.9B parameter model optimized for RTX 3050"""
    # Model architecture - targeting exactly 4.9B parameters
    vocab_size: int = 50260
    n_positions: int = 1024
    n_embd: int = 3584
    n_layer: int = 30
    n_head: int = 28
    n_inner: int = 14336
    
    # Training configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 64
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    
    # Optimization settings
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 25000
    
    # Memory optimizations
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Regularization
    dropout: float = 0.1
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1000

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT4_9B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying
        
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params:,} parameters ({n_params/1e9:.2f}B)")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_positions
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

class SimpleDataset(Dataset):
    def __init__(self, seq_length=1024, num_samples=1000):
        self.seq_length = seq_length
        self.data = []
        
        # Create synthetic training data
        vocab_size = 50260
        for i in range(num_samples):
            seq = torch.randint(10, vocab_size-10, (seq_length,))
            self.data.append(seq)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

def main():
    print("Fixed 4.9B Parameter Model Training")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    config = Config4_9B()
    
    print("Configuration:")
    print(f"  Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding")
    print(f"  Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
    print(f"  Learning rate: {config.learning_rate}")
    print()
    
    try:
        # Initialize model
        model = GPT4_9B(config).to(device)
        
        # Create dataset
        dataset = SimpleDataset(config.n_positions)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Mixed precision
        scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
        
        print("Starting training...")
        
        model.train()
        global_step = 0
        total_loss = 0.0
        
        checkpoint_dir = Path("checkpoints_4.9B")
        checkpoint_dir.mkdir(exist_ok=True)
        
        for epoch in range(100):
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                
                if scaler:
                    with autocast():
                        logits, loss = model(x, y)
                        loss = loss / config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps
                    loss.backward()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % config.log_interval == 0:
                        avg_loss = total_loss / config.log_interval
                        
                        if torch.cuda.is_available():
                            mem_used = torch.cuda.memory_allocated() / 1024**3
                            print(f"Step {global_step:5d} | Loss: {avg_loss:.4f} | GPU: {mem_used:.1f}GB")
                        else:
                            print(f"Step {global_step:5d} | Loss: {avg_loss:.4f}")
                        
                        total_loss = 0.0
                    
                    if global_step % config.save_interval == 0:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'config': config
                        }, checkpoint_dir / f"checkpoint_step_{global_step}.pt")
                        print(f"Checkpoint saved at step {global_step}")
                    
                    if global_step % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if global_step >= config.max_steps:
                        print(f"Training completed after {global_step} steps!")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'global_step': global_step
                        }, checkpoint_dir / "final_4.9B_model.pt")
                        return
                        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
