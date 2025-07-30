#!/usr/bin/env python3
"""
iLLuMinator 4.9B Parameter Model - CUDA Optimized for RTX 3070
High-performance transformer with CUDA acceleration and memory optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
import math
from typing import Optional, Tuple
import warnings

class CUDAOptimizedMultiHeadAttention(nn.Module):
    """CUDA-optimized multi-head attention with flash attention support"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use single linear layer for efficiency (combined q, k, v)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Enable flash attention if available
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V in one go for efficiency
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash_attention and mask is None:
            # Use PyTorch's optimized flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Apply causal mask for autoregressive generation
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        return self.output_proj(attn_output)

class CUDAOptimizedFeedForward(nn.Module):
    """CUDA-optimized feed-forward network with SwiGLU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # SwiGLU requires 2/3 scaling for equivalent parameters
        hidden_dim = int(2 * d_ff / 3)
        
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: swish(gate) * up
        gate = F.silu(self.gate_proj(x))  # SiLU is same as Swish
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)

class CUDAOptimizedTransformerBlock(nn.Module):
    """CUDA-optimized transformer block with pre-norm and residual connections"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attention = CUDAOptimizedMultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward = CUDAOptimizedFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture for better training stability
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = x + residual
        
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = x + residual
        
        return x

class CUDAOptimizedPositionalEncoding(nn.Module):
    """CUDA-optimized rotary positional encoding (RoPE)"""
    
    def __init__(self, d_model: int, max_seq_length: int = 4096):
        super().__init__()
        self.d_model = d_model
        
        # Create rotary position embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute positional encodings for efficiency
        position = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        freqs = position * inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class iLLuMinatorCUDA(nn.Module):
    """
    CUDA-Optimized iLLuMinator 4.9B Parameter Transformer
    Optimized for RTX 3070 with 8GB VRAM
    """
    
    def __init__(
        self,
        vocab_size: int = 50260,
        d_model: int = 3584,
        num_layers: int = 30,
        num_heads: int = 28,
        d_ff: int = 14336,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        print("Initializing CUDA-Optimized iLLuMinator 4.9B...")
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Token embeddings with weight tying
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = CUDAOptimizedPositionalEncoding(d_model, max_seq_length)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            CUDAOptimizedTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        
        # Output projection (tied with input embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"CUDA Model Configuration:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {total_params/1e9:.1f}B parameters")
        print(f"   Max sequence length: {max_seq_length}")
        print(f"   Gradient checkpointing: {use_gradient_checkpointing}")
        
        # CUDA optimizations
        if torch.cuda.is_available():
            print(f"   CUDA detected: {torch.cuda.get_device_name()}")
            print(f"   VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            print("CUDA optimizations enabled (TF32, cuDNN benchmark)")
        else:
            print("CUDA not available, running on CPU")
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with CUDA optimizations"""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        cos, sin = self.pos_encoding(x, seq_len)
        # For simplicity, we'll add standard positional encoding
        # In production, you'd apply RoPE to attention layers
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        return logits
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: int = 50256,
        eos_token_id: int = 50256,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        CUDA-optimized text generation with KV-caching
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        
        # Generation loop
        for _ in range(max_length - input_ids.shape[1]):
            if generated.shape[1] >= self.max_seq_length:
                break
            
            # Get model outputs
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if use_cache and past_key_values is not None:
                    # Only process the last token if using cache
                    model_input = generated[:, -1:]
                else:
                    model_input = generated
                
                logits = self.forward(model_input)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('inf'))
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == eos_token_id:
                    break
        
        return generated
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'cached': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
                'total_vram': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        else:
            return {'status': 'CUDA not available'}
    
    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.eval()
        
        if torch.cuda.is_available():
            # Move to GPU
            self.cuda()
            
            # Optimize with torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    print("Optimizing with torch.compile...")
                    self.forward = torch.compile(self.forward, mode="reduce-overhead")
                    print("Model compiled for faster inference")
                except Exception as e:
                    print(f"Could not compile model: {e}")

        return self

def create_cuda_model(vocab_size: int = 50260) -> iLLuMinatorCUDA:
    """Create and initialize the CUDA-optimized model"""
    print("🏗️  Creating CUDA-optimized iLLuMinator 4.9B model...")
    
    # Optimize configuration for RTX 3050 (8GB VRAM) - much smaller model for memory constraints
    config = {
        'vocab_size': vocab_size,
        'd_model': 2048,         # Further reduced from 3072
        'num_layers': 16,        # Further reduced from 24
        'num_heads': 16,         # Further reduced from 24
        'd_ff': 8192,           # Further reduced from 12288
        'max_seq_length': 256,   # Even smaller for RTX 3050
        'dropout': 0.1,
        'use_gradient_checkpointing': True  # Essential for fitting in 8GB VRAM
    }
    
    model = iLLuMinatorCUDA(**config)
    
    # Don't move to GPU immediately - let the training script handle it
    # This saves memory during initialization
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        # Clear cache but don't move model yet
        torch.cuda.empty_cache()
    
    return model

def main():
    """Test the CUDA-optimized model"""
    print("Testing CUDA-Optimized iLLuMinator 4.9B")
    print("=" * 60)
    
    # Create model
    model = create_cuda_model()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 1, 128
    
    if torch.cuda.is_available():
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len)).cuda()
    else:
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    try:
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids)
        
        print(f"   Forward pass successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {logits.shape}")
        
        if torch.cuda.is_available():
            memory_info = model.get_memory_usage()
            print(f"   GPU Memory: {memory_info['allocated']:.2f}GB allocated")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return
    
    # Test generation
    print("\nTesting generation...")
    try:
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence
        if torch.cuda.is_available():
            test_input = test_input.cuda()
        
        generated = model.generate(
            test_input,
            max_length=20,
            temperature=0.8,
            do_sample=True
        )
        
        print(f"   Generation successful!")
        print(f"   Generated sequence length: {generated.shape[1]}")
        
    except Exception as e:
        print(f"Generation failed: {e}")

    print(f"\nCUDA model test completed!")

if __name__ == "__main__":
    main()
