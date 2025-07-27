"""
iLLuMinator 4.7B Training Script
Training pipeline for the 4.7 billion parameter language model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import math
import json
from typing import Dict, List
import os
from tqdm import tqdm
import wandb

from illuminator_model import iLLuMinator4_7B
from tokenizer import iLLuMinatorTokenizer

class TextDataset(Dataset):
    """Dataset for text training"""
    
    def __init__(self, texts: List[str], tokenizer: iLLuMinatorTokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),  # Input
            'labels': torch.tensor(tokens[1:], dtype=torch.long)       # Target (shifted by 1)
        }

class iLLuMinatorTrainer:
    """Training class for iLLuMinator model"""
    
    def __init__(self, 
                 model: iLLuMinator4_7B,
                 tokenizer: iLLuMinatorTokenizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Training parameters
        self.lr = 6e-4  # Learning rate for 4.7B model
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tokenizer.pad_token_id)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})
            
            # Log to wandb if available
            if 'wandb' in globals() and wandb.run is not None:
                wandb.log({
                    'train_loss': loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': epoch * num_batches + batch_idx
                })
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float, save_path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_length': self.model.max_seq_length
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
        
        return epoch, loss

def prepare_sample_data() -> List[str]:
    """Prepare sample training data"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test fonts and keyboards.",
        "Artificial intelligence is revolutionizing the way we work, learn, and interact with technology.",
        "Python is a versatile programming language that's great for beginners and experts alike.",
        "Machine learning models require large amounts of data to train effectively and produce good results.",
        "Natural language processing enables computers to understand and generate human language.",
        "Deep learning networks with billions of parameters can learn complex patterns from data.",
        "Transformer architectures have become the foundation for most modern language models.",
        "Training large language models requires significant computational resources and time.",
        "The attention mechanism allows models to focus on relevant parts of the input sequence.",
        "Fine-tuning pre-trained models can achieve good performance on specific tasks with less data.",
    ]
    
    # Repeat the sample data to have more training examples
    return sample_texts * 100  # 1000 samples

def train_illuminator():
    """Main training function"""
    print("Starting iLLuMinator 4.9B Training")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = iLLuMinatorTokenizer()
    
    # Initialize model
    print("Initializing model...")
    model = iLLuMinator4_7B(vocab_size=len(tokenizer))
    
    # Prepare data
    print("Preparing training data...")
    texts = prepare_sample_data()
    dataset = TextDataset(texts, tokenizer)
    
    # Create data loader
    batch_size = 2 if torch.cuda.is_available() else 1  # Small batch size for 4.7B model
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize trainer
    trainer = iLLuMinatorTrainer(model, tokenizer)
    
    print(f"Training on device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(texts)}")
    print(f"Batch size: {batch_size}")
    
    # Training loop
    num_epochs = 3  # Small number for demo
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📚 Epoch {epoch}/{num_epochs}")
        
        avg_loss = trainer.train_epoch(dataloader, epoch)
        
        print(f"Average loss for epoch {epoch}: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/illuminator_epoch_{epoch}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        trainer.save_checkpoint(epoch, avg_loss, checkpoint_path)
    
    # Save final model
    print("\nSaving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(tokenizer),
            'd_model': model.d_model,
            'max_seq_length': model.max_seq_length
        }
    }, "illuminator_4_7b_final.pt")
    
    print("Training completed!")
    return model, tokenizer

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, training on CPU (will be slow)")
    
    model, tokenizer = train_illuminator()
