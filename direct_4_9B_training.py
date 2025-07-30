#!/usr/bin/env python3
"""Direct 4.9B training script - no dependencies on external datasets"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

class SimpleConfig:
    """Simple configuration for 4.9B model targeting RTX 3050"""
    vocab_size = 50257
    n_positions = 1024  # Reduced for memory
    n_embd = 3200
    n_layer = 36
    n_head = 25
    n_inner = 12800
    
    batch_size = 1
    gradient_accumulation_steps = 16
    learning_rate = 1e-4
    max_steps = 10000
    log_interval = 10
    save_interval = 500

class SimpleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Simple transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_inner,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model has {n_params:,} parameters ({n_params/1e9:.2f}B)")
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        
        # Token + position embeddings
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        # Create causal mask
        mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, x, tgt_mask=mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

class SimpleDataset(Dataset):
    def __init__(self, seq_length=1024, num_samples=1000):
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # Create simple synthetic data
        self.data = []
        for i in range(num_samples):
            # Random sequence
            seq = torch.randint(10, 50000, (seq_length,))
            self.data.append(seq)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target

def main():
    print("Starting Simple 4.9B Training")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Initialize
    config = SimpleConfig()
    model = SimpleGPT(config).to(device)
    
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    
    print("Starting training...")
    
    try:
        for epoch in range(100):
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
                
                # Gradient step
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % config.log_interval == 0:
                        avg_loss = total_loss / config.log_interval
                        
                        if torch.cuda.is_available():
                            mem_used = torch.cuda.memory_allocated() / 1024**3
                            print(f"Step {global_step:5d} | Loss: {avg_loss:.4f} | GPU: {mem_used:.1f}GB")
                        else:
                            print(f"Step {global_step:5d} | Loss: {avg_loss:.4f}")
                        
                        total_loss = 0
                    
                    # Save checkpoint
                    if global_step % config.save_interval == 0:
                        checkpoint_dir = Path("checkpoints_4.9B")
                        checkpoint_dir.mkdir(exist_ok=True)
                        
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'config': config
                        }, checkpoint_dir / f"checkpoint_step_{global_step}.pt")
                        
                        print(f"Checkpoint saved at step {global_step}")
                    
                    # Memory cleanup
                    if global_step % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Stop condition
                    if global_step >= config.max_steps:
                        print(f"Training completed after {global_step} steps!")
                        
                        # Save final model
                        checkpoint_dir = Path("checkpoints_4.9B")
                        checkpoint_dir.mkdir(exist_ok=True)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'global_step': global_step
                        }, checkpoint_dir / "final_4.9B_model.pt")
                        
                        return
                        
    except KeyboardInterrupt:
        print(f"Training interrupted at step {global_step}")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
