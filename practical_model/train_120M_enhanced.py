#!/usr/bin/env python3
"""
Enhanced Training Script for 120M iLLuMinator Model
Uses premium datasets and optimized training techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from illuminator_practical import iLLuMinatorPractical
from enhanced_dataset_loader import DatasetProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedTokenizer:
    """Lightweight tokenizer for 120M model"""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
            '<USER>': 4, '<ASSISTANT>': 5
        }
        self.vocab = self._build_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def _build_vocab(self) -> List[str]:
        """Build vocabulary optimized for 120M model"""
        vocab = list(self.special_tokens.keys())
        
        # Essential programming and conversation tokens
        essential_tokens = [
            # Core programming
            'def', 'class', 'import', 'return', 'if', 'else', 'for', 'while',
            'function', 'var', 'let', 'const', 'print', 'input', 'output',
            # Common symbols
            '(', ')', '[', ']', '{', '}', '=', '+', '-', '*', '/', '%',
            '.', ',', ':', ';', '!', '?', '"', "'", '\n', ' ',
            # Numbers
            *[str(i) for i in range(100)],
            # Letters
            *'abcdefghijklmnopqrstuvwxyz',
            *'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            # Common words
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'about', 'how', 'what',
            'when', 'where', 'why', 'who', 'which', 'this', 'that',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'get', 'make', 'go', 'see', 'know', 'take', 'give',
            'find', 'think', 'say', 'work', 'try', 'use', 'need', 'want'
        ]
        
        vocab.extend(essential_tokens)
        
        # Pad to desired size
        while len(vocab) < self.vocab_size:
            vocab.append(f'<UNK_{len(vocab)}>')
            
        return vocab[:self.vocab_size]
    
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """Encode text to token IDs"""
        if not text:
            return [self.special_tokens['<PAD>']]
        
        tokens = [self.special_tokens['<BOS>']]
        
        # Simple word-based tokenization
        words = text.lower().replace('\n', ' <NEWLINE> ').split()
        
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # Character-level fallback for unknown words
                for char in word:
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.special_tokens['<UNK>'])
        
        tokens.append(self.special_tokens['<EOS>'])
        
        # Pad or truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.special_tokens['<EOS>']]
        else:
            tokens.extend([self.special_tokens['<PAD>']] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                    tokens.append(token)
        
        return ' '.join(tokens).replace(' <NEWLINE> ', '\n')

class Enhanced120MDataset(Dataset):
    """Dataset class for 120M model training"""
    
    def __init__(self, texts: List[str], tokenizer: OptimizedTokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, self.max_length)
        
        # Create input and target sequences
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }

class Enhanced120MTrainer:
    """Optimized trainer for 120M model"""
    
    def __init__(self, model: iLLuMinatorPractical, tokenizer: OptimizedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Training on device: {self.device}")
        
        # Optimized optimizer settings
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=5e-4,  # Higher learning rate for smaller model
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens['<PAD>'])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,
            eta_min=1e-6
        )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def train(self, dataset: Enhanced120MDataset, epochs: int = 10, batch_size: int = 4):
        """Main training loop"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Batch size: {batch_size}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = self.train_epoch(dataloader, epoch + 1)
            
            logger.info(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {epoch_loss:.4f}")
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_model(f"best_120M_model_epoch_{epoch+1}.pth")
                logger.info(f"New best model saved with loss: {epoch_loss:.4f}")
            
            # Test generation every few epochs
            if (epoch + 1) % 3 == 0:
                self.test_generation()
        
        logger.info("Training completed!")
    
    def save_model(self, filename: str):
        """Save the trained model"""
        save_path = Path("models") / filename
        save_path.parent.mkdir(exist_ok=True)
        
        # Create config dict for saving
        model_config = {
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_layers': len(self.model.transformer_blocks),
            'max_seq_length': self.model.max_seq_length,
            'dropout': 0.1
        }
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer_vocab': self.tokenizer.vocab,
            'config': model_config
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def test_generation(self):
        """Test model generation capability"""
        self.model.eval()
        
        test_prompts = [
            "Human: What is Python?\nAssistant:",
            "Human: Write a function to add two numbers\nAssistant:",
            "Human: Explain machine learning\nAssistant:"
        ]
        
        with torch.no_grad():
            for prompt in test_prompts:
                input_ids = torch.tensor([self.tokenizer.encode(prompt, max_length=128)]).to(self.device)
                
                # Simple generation test - just get next token prediction
                try:
                    outputs = self.model(input_ids)
                    next_token_logits = outputs[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits).item()
                    next_token = self.tokenizer.decode([next_token_id])
                    
                    logger.info(f"Test Prompt: {prompt[:30]}...")
                    logger.info(f"Next predicted token: '{next_token}'\n")
                except Exception as e:
                    logger.warning(f"Generation test failed: {e}")
        
        self.model.train()

def main():
    """Main training function for 120M model"""
    print("Enhanced Training for 120M iLLuMinator Model")
    print("=" * 50)
    
    # Create dataset processor
    processor = DatasetProcessor()
    
    # Load optimized dataset for 120M model
    logger.info("Loading optimized dataset for 120M model...")
    texts_120M, dataset_path = processor.create_dataset_for_120M_model()
    
    # Initialize tokenizer
    tokenizer = OptimizedTokenizer(vocab_size=8000)
    logger.info(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = Enhanced120MDataset(texts_120M, tokenizer, max_length=256)
    logger.info(f"Dataset created with {len(dataset)} examples")
    
    # Initialize model with optimized config
    config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': 512,      # Smaller for 120M
        'n_heads': 8,        # Fewer heads
        'n_layers': 12,      # Fewer layers
        'd_ff': 2048,        # Smaller FFN
        'max_seq_length': 256   # Shorter sequences
    }
    
    model = iLLuMinatorPractical(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize trainer
    trainer = Enhanced120MTrainer(model, tokenizer)
    
    # Start training
    try:
        trainer.train(
            dataset=dataset,
            epochs=5,  # Reduced for testing
            batch_size=8 if torch.cuda.is_available() else 4
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    print("\n120M Model Training Complete!")

if __name__ == "__main__":
    main()
