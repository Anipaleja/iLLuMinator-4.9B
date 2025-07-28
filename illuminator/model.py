import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Tuple

class SimpleTokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        
        # Simple vocabulary
        self.vocab = ['<pad>', '<eos>', '<bos>'] + [f'token_{i}' for i in range(vocab_size-3)]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        
    def encode(self, text: str, **kwargs) -> List[int]:
        # Simple word-based encoding
        words = text.lower().split()[:50]  # Limit length
        tokens = [self.bos_token_id]
        for word in words:
            # Hash the word to get a consistent token ID
            token_id = (hash(word) % (self.vocab_size - 10)) + 10
            tokens.append(token_id)
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        # Simple decoding
        words = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [0, 1, 2]:
                continue
            words.append(f"word_{token_id}")
        return " ".join(words)

class SimpleModel(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = 100
        
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'max_seq_length': self.max_seq_length
        }
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=n_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        # Create causal mask
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, **kwargs) -> torch.Tensor:
        self.eval()
        
        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(input_ids)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1, keepdim=True)
            
            if next_token.item() == 1:  # EOS token
                break
                
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
        return input_ids
    
    def save_pretrained(self, path: str):
        import json
        from pathlib import Path
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f)
    
    @classmethod
    def from_pretrained(cls, path: str):
        import json
        from pathlib import Path
        model_path = Path(path)
        with open(model_path / "config.json", 'r') as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(model_path / "pytorch_model.bin", map_location='cpu')
        model.load_state_dict(state_dict)
        return model

# Aliases for compatibility
iLLuMinatorEnhanced = SimpleModel
EnhancedTokenizer = SimpleTokenizer

def create_illuminator_model(size: str = "small") -> Tuple[SimpleModel, SimpleTokenizer]:
    configs = {
        "tiny": {"vocab_size": 500, "d_model": 64, "n_layers": 2},
        "small": {"vocab_size": 1000, "d_model": 128, "n_layers": 4},
        "medium": {"vocab_size": 2000, "d_model": 256, "n_layers": 6},
        "large": {"vocab_size": 5000, "d_model": 512, "n_layers": 8}
    }
    
    config = configs.get(size, configs["small"])
    model = SimpleModel(**config)
    tokenizer = SimpleTokenizer(config["vocab_size"])
    
    return model, tokenizer
