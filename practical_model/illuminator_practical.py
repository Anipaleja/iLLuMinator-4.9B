"""
iLLuMinator Practical Model - Efficient 120M Parameter Architecture
A smaller but functional transformer model for real-world usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import json

class EfficientMultiHeadAttention(nn.Module):
    """Efficient multi-head attention for smaller models"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_qkv = nn.Linear(d_model, d_model * 3)  # Combined QKV projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Combined QKV projection
        qkv = self.w_qkv(x).view(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output

class EfficientFeedForward(nn.Module):
    """Efficient feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EfficientTransformerBlock(nn.Module):
    """Efficient transformer block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = EfficientMultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = EfficientFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (more stable)
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class SimplePositionalEncoding(nn.Module):
    """Simple learnable positional encoding"""
    
    def __init__(self, d_model: int, max_seq_length: int = 2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length, :]

class iLLuMinatorPractical(nn.Module):
    """Practical iLLuMinator model - 120M parameters"""
    
    def __init__(self, 
                 vocab_size: int = 50257,
                 d_model: int = 768,        # Smaller model dimension
                 n_layers: int = 12,        # Fewer layers
                 n_heads: int = 12,         # Fewer attention heads
                 d_ff: int = 3072,          # Smaller feed-forward
                 max_seq_length: int = 1024, # Shorter context
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = SimplePositionalEncoding(d_model, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (reduce parameters)
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate parameters
        self._calculate_parameters()
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def _calculate_parameters(self):
        """Calculate total parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"iLLuMinator Practical Model Configuration:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params / 1e6:.1f}M parameters")
        
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal mask"""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_length, device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * attention_mask
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.position_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                 temperature: float = 1.0, top_p: float = 0.9, 
                 do_sample: bool = True, pad_token_id: int = 50256) -> torch.Tensor:
        """Fast text generation"""
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_length):
                # Truncate if too long
                if input_ids.size(1) >= self.max_seq_length:
                    input_ids = input_ids[:, -self.max_seq_length+1:]
                
                # Forward pass
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = -float('Inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Stop at pad token
                if next_token.item() == pad_token_id:
                    break
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids

def create_practical_model():
    """Create the practical model"""
    config = {
        "model_name": "iLLuMinator-Practical-120M",
        "vocab_size": 50257,
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "d_ff": 3072,
        "max_seq_length": 1024,
        "dropout": 0.1
    }
    
    with open('illuminator_practical_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return iLLuMinatorPractical()

if __name__ == "__main__":
    print("Creating iLLuMinator Practical Model...")
    
    model = create_practical_model()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_length = 50
    
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length))
    
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        
        # Test generation
        print("\nTesting generation...")
        prompt_ids = torch.randint(0, model.vocab_size, (1, 10))
        generated = model.generate(prompt_ids, max_length=20, temperature=0.8)
        print(f"Generated sequence length: {generated.shape[1]}")
    
    print("\n✅ iLLuMinator Practical model created successfully!")
    print("This model is efficient enough for CPU inference and training.")
