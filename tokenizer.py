"""
iLLuMinator Tokenizer
Custom tokenizer for the iLLuMinator 4.7B model
"""

from transformers import GPT2Tokenizer, GPT2TokenizerFast
import json
import os
from typing import List, Dict, Any

class iLLuMinatorTokenizer:
    """Custom tokenizer wrapper for iLLuMinator model"""
    
    def __init__(self, tokenizer_path: str = None):
        """Initialize tokenizer, using GPT-2 tokenizer as base"""
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        else:
            # Use GPT-2 tokenizer as base
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            
        # Add special tokens for iLLuMinator
        special_tokens = {
            'pad_token': '<|pad|>',
            'eos_token': '<|endoftext|>',
            'bos_token': '<|startoftext|>',
            'unk_token': '<|unknown|>',
        }
        
        # Add special tokens to tokenizer
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Store vocabulary info
        self.vocab_size = len(self.tokenizer)
        self.max_length = 2048
        
    def encode(self, text: str, max_length: int = None) -> List[int]:
        """Encode text to token IDs"""
        if max_length is None:
            max_length = self.max_length
            
        return self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(self, texts: List[str], max_length: int = None, 
                    padding: bool = True) -> Dict[str, List[List[int]]]:
        """Batch encode multiple texts"""
        if max_length is None:
            max_length = self.max_length
            
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=padding,
            return_tensors='pt'
        )
    
    def save(self, save_path: str):
        """Save tokenizer to directory"""
        os.makedirs(save_path, exist_ok=True)
        self.tokenizer.save_pretrained(save_path)
        
        # Save additional config
        config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'model_name': 'iLLuMinator-4.7B'
        }
        
        with open(os.path.join(save_path, 'tokenizer_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, tokenizer_path: str):
        """Load tokenizer from directory"""
        return cls(tokenizer_path)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        return self.tokenizer.get_vocab()
    
    def __len__(self):
        return self.vocab_size

def create_tokenizer():
    """Create and save the iLLuMinator tokenizer"""
    print("Creating iLLuMinator tokenizer...")
    
    tokenizer = iLLuMinatorTokenizer()
    
    print(f"Tokenizer created with vocabulary size: {len(tokenizer)}")
    
    # Test the tokenizer
    test_text = "Hello, I am iLLuMinator, a 4.7 billion parameter language model!"
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Test text: {test_text}")
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    
    # Save tokenizer
    tokenizer.save('./tokenizer')
    print("Tokenizer saved to ./tokenizer/")
    
    return tokenizer

if __name__ == "__main__":
    tokenizer = create_tokenizer()
