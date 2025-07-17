import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class iLLuMinator(nn.Module):
    def __init__(self, vocab_size, block_size=512, n_embd=256, n_head=8, n_layer=6, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Custom transformer decoder layers with causal masking
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, context_mask=None):
        """
        Forward pass with optional context masking for RAG
        
        Args:
            idx: input token indices (B, T)
            context_mask: optional mask to separate context from query (B, T)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    def generate_with_context(self, context_tokens, query_tokens, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Generate text given context (retrieved documents) and query
        
        Args:
            context_tokens: retrieved document tokens (B, context_len)
            query_tokens: query tokens (B, query_len) 
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
        """
        self.eval()
        
        # Combine context and query
        if context_tokens.size(1) + query_tokens.size(1) > self.block_size:
            # Truncate context if too long
            available_context = self.block_size - query_tokens.size(1) - max_new_tokens
            if available_context > 0:
                context_tokens = context_tokens[:, -available_context:]
            else:
                context_tokens = context_tokens[:, :0]  # Empty context
        
        # Concatenate context and query
        input_tokens = torch.cat([context_tokens, query_tokens], dim=1)
        
        # Generate tokens
        for _ in range(max_new_tokens):
            if input_tokens.size(1) >= self.block_size:
                # Sliding window: keep recent tokens
                input_tokens = input_tokens[:, -self.block_size:]
            
            with torch.no_grad():
                logits = self(input_tokens)
                logits = logits[:, -1, :] / temperature  # Focus on last token
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
        
        return input_tokens


class TransformerDecoderBlock(nn.Module):
    """Custom transformer decoder block with causal masking"""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention with proper masking for autoregressive generation"""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x