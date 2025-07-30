#!/usr/bin/env python3
"""Test parameter count for 4.9B model"""

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config4_9B:
    """Configuration for 4.9B parameter model optimized for RTX 3050"""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 2944
    n_layer: int = 32
    n_head: int = 23
    n_inner: int = 11776

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config4_9B):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

class FeedForward(nn.Module):
    def __init__(self, config: Config4_9B):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)
        self.dropout = nn.Dropout(0.1)

class TransformerBlock(nn.Module):
    def __init__(self, config: Config4_9B):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

class GPT4_9B(nn.Module):
    def __init__(self, config: Config4_9B):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(0.1)
        
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model has {n_params:,} parameters ({n_params/1e9:.2f}B)")

def main():
    print("Testing 4.9B Parameter Model Configuration")
    print("=" * 50)
    
    config = Config4_9B()
    print(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding")
    print(f"Vocab size: {config.vocab_size:,}")
    print(f"Context length: {config.n_positions}")
    print(f"FFN inner: {config.n_inner}")
    print()
    
    model = GPT4_9B(config)
    
    # Test memory usage
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("Testing GPU memory usage...")
        
        try:
            model = model.cuda()
            print(f"Model loaded to GPU successfully!")
            
            # Test forward pass
            test_input = torch.randint(0, config.vocab_size, (1, 512)).cuda()
            with torch.no_grad():
                logits, _ = model(test_input)
            
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory used: {memory_used:.2f} GB")
            print("Model fits in GPU memory!")
            
        except torch.OutOfMemoryError:
            print("Model too large for GPU memory")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("CUDA not available")

if __name__ == "__main__":
    main()
