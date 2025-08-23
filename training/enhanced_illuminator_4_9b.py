"""
Enhanced iLLuMinator 4.9B Parameter Language Model
Advanced transformer architecture with 4.9 billion parameters
Enhanced with modern techniques: RMSNorm, SwiGLU, RoPE, Grouped Query Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import json
import torch.utils.checkpoint as checkpoint

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more stable than LayerNorm)"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - More effective than standard positional encoding"""
    
    def __init__(self, d_model: int, max_seq_length: int = 4096):
        super().__init__()
        self.d_model = d_model
        
        # Create rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cache_max_length = max_seq_length
        self._cache_cos = None
        self._cache_sin = None
        
    def _update_cache(self, seq_length: int, device: torch.device):
        """Update cached cos/sin values"""
        if (self._cache_cos is None or 
            seq_length > self._cache_max_length or 
            self._cache_cos.device != device):
            
            self._cache_max_length = max(seq_length, self._cache_max_length)
            position = torch.arange(seq_length, device=device).unsqueeze(1)
            freqs = position * self.inv_freq.unsqueeze(0)
            
            self._cache_cos = torch.cos(freqs)
            self._cache_sin = torch.sin(freqs)
    
    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding to input tensor"""
        seq_length = x.size(-2)
        self._update_cache(seq_length, x.device)
        
        # Split into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        cos = self._cache_cos[:seq_length].unsqueeze(0).unsqueeze(0)
        sin = self._cache_sin[:seq_length].unsqueeze(0).unsqueeze(0)
        
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return rotated

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention - More efficient than standard multi-head attention"""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Repetition factor
        self.d_k = d_model // n_heads
        
        # Query, Key, Value projections
        self.w_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Apply RoPE to Q and K
        Q = self.rope.apply_rope(Q)
        K = self.rope.apply_rope(K)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        K = K.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)
        V = V.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_k)
        
        # Repeat K and V for grouped query attention
        if self.n_rep > 1:
            K = K.repeat_interleave(self.n_rep, dim=1)
            V = V.repeat_interleave(self.n_rep, dim=1)
        
        # Scaled dot-product attention with Flash Attention optimization
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Make sure we do masking in float32
        attention_scores = attention_scores.float()
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, torch.finfo(torch.float32).min)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output

class SwiGLU(nn.Module):
    """SwiGLU activation function - more effective than GELU for large models"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU requires 3 linear layers instead of 2
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x @ W1) * (x @ W3) @ W2
        gate = F.silu(self.w1(x))  # SiLU (Swish) activation
        up = self.w3(x)
        return self.w2(self.dropout(gate * up))

class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with modern improvements"""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.feed_forward = SwiGLU(d_model, d_ff, dropout)
        
        # Use RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Pre-norm architecture (more stable training)
        self.pre_norm = True
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-norm: normalize before each sublayer
            x = x + self.attention(self.norm1(x), mask)
            x = x + self.feed_forward(self.norm2(x))
        else:
            # Post-norm: normalize after each sublayer
            x = self.norm1(x + self.attention(x, mask))
            x = self.norm2(x + self.feed_forward(x))
        
        return x

class iLLuMinator4_9B(nn.Module):
    """Enhanced iLLuMinator 4.9B Parameter Language Model"""
    
    def __init__(self, 
                 vocab_size: int = 65536,      # Larger vocabulary for better tokenization
                 d_model: int = 3328,          # Optimized model dimension for 4.9B target
                 n_layers: int = 32,           # Number of transformer layers
                 n_heads: int = 32,            # Number of attention heads (divisible by n_kv_heads)
                 n_kv_heads: int = 8,          # Number of key-value heads (for GQA)
                 d_ff: int = 9984,             # Feed-forward dimension (3x d_model for SwiGLU)
                 max_seq_length: int = 4096,   # Longer context length
                 dropout: float = 0.0,         # Lower dropout for large models
                 gradient_cp: bool = False,    # Reduce VRAM 
                 tie_embeddings: bool = True): # Tie input/output embeddings
        super().__init__()
        
        # Store configuration
        self.gradient_cp = gradient_cp
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.n_layers = n_layers
        self.tie_embeddings = tie_embeddings
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = RMSNorm(d_model)
        
        # Output projection
        if tie_embeddings:
            # Tie input and output embeddings to reduce parameters
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate and display parameter count
        self._calculate_parameters()
        
    def _init_weights(self, module):
        """Initialize weights using modern best practices"""
        if isinstance(module, nn.Linear):
            # Use scaled initialization for better training stability
            std = 0.02
            if hasattr(module, 'scale_init'):
                std *= module.scale_init
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def _calculate_parameters(self):
        """Calculate and print the total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate memory usage (approximate)
        param_memory_gb = (total_params * 4) / (1024**3)  # 4 bytes per float32
        training_memory_gb = param_memory_gb * 4  # Rough estimate for gradients + activations
        
        print(f"Enhanced iLLuMinator 4.9B Model Configuration:")
        print(f"├── Total parameters: {total_params:,}")
        print(f"├── Trainable parameters: {trainable_params:,}")
        print(f"├── Model size: {total_params / 1e9:.2f}B parameters")
        print(f"├── Estimated memory (params): {param_memory_gb:.2f}GB")
        print(f"├── Estimated memory (training): {training_memory_gb:.2f}GB")
        print(f"└── Architecture: Enhanced Transformer with GQA, RoPE, SwiGLU, RMSNorm")
        
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Create causal mask for autoregressive modeling
        causal_mask = self.create_causal_mask(seq_length, device)
        
        if attention_mask is not None:
            # Combine attention mask with causal mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * attention_mask
        
        # Token embeddings (scaled for better gradient flow)
        x = self.token_embedding(input_ids).float() * math.sqrt(self.d_model)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            if self.gradient_cp and self.training:
                def custom_forward(*inputs):
                    return block(*inputs)
                x = torch.utils.checkpoint.checkpoint(custom_forward, x, causal_mask)
            else:
                x = block(x, causal_mask)

        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Output projection to vocabulary
        if self.tie_embeddings:
            # Use transposed token embedding weights
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_p: float = 0.9, top_k: int = 50,
                 do_sample: bool = True, pad_token_id: int = 0) -> torch.Tensor:
        """Enhanced text generation with multiple sampling strategies"""
        self.eval()
        original_length = input_ids.size(1)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate input if it exceeds max sequence length
                if input_ids.size(1) > self.max_seq_length:
                    input_ids = input_ids[:, -self.max_seq_length:]
                
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get logits for next token prediction
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Apply top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits.scatter_(1, indices_to_remove, -float('inf'))
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if we hit pad token or exceed reasonable length
                if next_token.item() == pad_token_id:
                    break
        
        return input_ids

    def get_config(self) -> dict:
        """Get model configuration for serialization"""
        return {
            "model_name": "iLLuMinator-4.9B-Enhanced",
            "architecture": "enhanced-transformer",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.transformer_blocks[0].attention.n_heads,
            "n_kv_heads": self.transformer_blocks[0].attention.n_kv_heads,
            "d_ff": self.transformer_blocks[0].feed_forward.w1.out_features,
            "max_seq_length": self.max_seq_length,
            "tie_embeddings": self.tie_embeddings,
            "activation": "swiglu",
            "norm_type": "rmsnorm",
            "position_embedding": "rope",
            "attention_type": "grouped_query_attention"
        }

# Simple tokenizer for training
class SimpleTokenizer:
    """Simple tokenizer for training purposes"""
    
    def __init__(self, vocab_size: int = 65536):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        
    def encode(self, text: str, max_length: int = 2048) -> list:
        """Simple character-based encoding"""
        # Convert text to character codes
        tokens = [ord(c) % self.vocab_size for c in text]
        
        # Add BOS token
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, tokens: list) -> str:
        """Simple character-based decoding"""
        # Filter out special tokens
        filtered_tokens = [t for t in tokens if t not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        
        # Convert back to characters
        chars = [chr(t) for t in filtered_tokens if 0 <= t <= 1114111]  # Valid Unicode range
        
        return ''.join(chars)

if __name__ == "__main__":
    # Test model creation
    print("🚀 Testing Enhanced iLLuMinator 4.9B Model")
    print("=" * 60)
    
    # Create model
    model = iLLuMinator4_9B(
        vocab_size=1000,  # Smaller for testing
        d_model=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        d_ff=1536,
        max_seq_length=128
    )
    
    # Test forward pass
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    print(f"\n🧪 Testing forward pass...")
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Output shape: {logits.shape}")
        print(f"✅ Forward pass successful!")
    
    print(f"\n✨ Enhanced iLLuMinator 4.9B model test completed!")
