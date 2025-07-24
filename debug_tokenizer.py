#!/usr/bin/env python3
"""
Debug tokenizer to see what's happening
"""

import sys
import re
from typing import Optional, List

class ProfessionalTokenizer:
    """Advanced tokenizer with code-aware capabilities"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
            '<CODE>': 4, '<CHAT>': 5, '<SYSTEM>': 6, '<USER>': 7,
            '<ASSISTANT>': 8, '<FUNCTION>': 9, '<CLASS>': 10, '<IMPORT>': 11
        }
        self.vocab = self._build_vocabulary()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
    def _build_vocabulary(self) -> List[str]:
        """Build comprehensive vocabulary for code and chat"""
        vocab = list(self.special_tokens.keys())
        
        # Programming keywords and symbols
        programming_tokens = [
            # Python keywords
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'as', 'in', 'not', 'and', 'or', 'is', 'None',
            'True', 'False', 'lambda', 'yield', 'break', 'continue', 'pass', 'global', 'nonlocal',
            
            # Common words and phrases
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
            'I', 'you', 'he', 'she', 'it', 'we', 'they', 'hello', 'hi', 'how', 'are',
            'what', 'which', 'who', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        ]
        
        vocab.extend(programming_tokens)
        
        # Add alphabet
        for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            vocab.append(char)
            
        # Pad vocabulary to desired size
        while len(vocab) < self.vocab_size:
            vocab.append(f'<UNK_{len(vocab)}>')
            
        return vocab[:self.vocab_size]
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs with smart tokenization"""
        if not text:
            return [self.special_tokens['<PAD>']]
            
        tokens = []
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # Character-level fallback
                for char in word:
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.special_tokens['<UNK>'])
        
        if max_length:
            tokens = tokens[:max_length]
            
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if not token.startswith('<') or not token.endswith('>'):
                    tokens.append(token)
                    
        return ' '.join(tokens)

def test_tokenizer():
    print("Testing tokenizer...")
    tokenizer = ProfessionalTokenizer()
    
    test_texts = [
        "hi",
        "Hello, how are you?",
        "What is Python?",
        "I am an AI assistant ready to help you."
    ]
    
    for text in test_texts:
        print(f"\n--- Testing: '{text}' ---")
        encoded = tokenizer.encode(text)
        print(f"Encoded: {encoded}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
        print(f"Decoded length: {len(decoded)}")

if __name__ == "__main__":
    test_tokenizer()
