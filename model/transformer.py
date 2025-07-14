import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTMini(nn.Module):
    def __init__(self, vocab_size, block_size=64, n_embd=128, n_head=4, n_layer=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits