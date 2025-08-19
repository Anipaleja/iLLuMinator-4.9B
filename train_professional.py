#!/usr/bin/env python3
"""
Professional Training Script for iLLuMinator 4.9B
Incorporates high-quality datasets used by leading AI models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import math
import time
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

# Optional wandb import for professional experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import our enhanced data sources
from data_sources import ProfessionalDatasetLoader, format_for_training

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalTrainingDataset(Dataset):
    """
    Professional training dataset with quality filtering and proper formatting
    """
    
    def __init__(self, tokenizer, max_length: int = 2048, dataset_size: int = 25000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_professional_data(dataset_size)
        
    def _load_professional_data(self, dataset_size: int) -> List[str]:
        """Load and format professional training data"""
        logger.info("Loading professional training data...")
        
        # Initialize professional dataset loader
        loader = ProfessionalDatasetLoader()
        
        # Check for cached dataset
        cached_dataset = loader.load_saved_dataset("professional_training_data.json")
        
        if cached_dataset and len(cached_dataset) >= dataset_size:
            logger.info(f"Using cached dataset with {len(cached_dataset)} examples")
            examples = cached_dataset[:dataset_size]
        else:
            logger.info("Creating new professional dataset...")
            examples = loader.create_comprehensive_dataset(total_size=dataset_size)
            loader.save_dataset(examples, "professional_training_data.json")
        
        # Format for training
        formatted_texts = format_for_training(examples, format_type="alpaca")
        
        logger.info(f"Loaded {len(formatted_texts)} professionally formatted examples")
        return formatted_texts
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize with proper truncation and padding
        encoding = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding[0]
        
        # Create labels for causal LM (shift by one position)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token in loss calculation
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }

class ProfessionalTrainer:
    """
    Professional training class with advanced optimization and monitoring
    """
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer with advanced settings
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if self.config.get('mixed_precision', True) else None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # Professional logging
        self._setup_logging()
        
    def _setup_optimizer(self):
        """Setup AdamW optimizer with professional settings"""
        # Use AdamW with weight decay (standard for large models)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 3e-4),
            betas=(0.9, 0.95),  # Standard for large LMs
            eps=1e-8,
            weight_decay=self.config.get('weight_decay', 0.1)
        )
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        # Cosine annealing with warmup (standard for large models)
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('scheduler_t0', 1000),
            T_mult=2,
            eta_min=self.config.get('min_lr', 1e-6)
        )
        return scheduler
    
    def _setup_logging(self):
        """Setup professional experiment tracking"""
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="illuminator-4.9b",
                    config=self.config,
                    name=f"training-{int(time.time())}"
                )
                logger.info("Weights & Biases logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
        elif self.config.get('use_wandb', False) and not WANDB_AVAILABLE:
            logger.warning("wandb requested but not available. Install with: pip install wandb")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch with professional practices"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = nn.CrossEntropyLoss()(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels.view(-1)
                    )
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1)
                )
            
            # Backward pass with gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            # Update tracking
            total_loss += loss.item()
            self.current_step += 1
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False) and WANDB_AVAILABLE and batch_idx % 100 == 0:
                try:
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'step': self.current_step
                    })
                except:
                    pass
            
            # Save checkpoint periodically
            if self.current_step % self.config.get('save_every', 1000) == 0:
                self.save_checkpoint(f"checkpoint_step_{self.current_step}.pt")
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.current_step = checkpoint['current_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")

def create_simple_tokenizer():
    """Create a simple tokenizer for testing"""
    class SimpleTokenizer:
        def __init__(self):
            # Create a basic vocabulary
            self.vocab = {
                '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3
            }
            
            # Add common characters and words
            chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-\n'
            for char in chars:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
            
            # Add common words
            common_words = ['the', 'and', 'or', 'to', 'of', 'in', 'a', 'is', 'that', 'it']
            for word in common_words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
            
            self.pad_token_id = self.vocab['<pad>']
            self.unk_token_id = self.vocab['<unk>']
            self.vocab_size = len(self.vocab)
        
        def encode(self, text, max_length=512, truncation=True, padding='max_length', return_tensors='pt'):
            # Simple character-level tokenization
            tokens = []
            for char in text:
                tokens.append(self.vocab.get(char, self.unk_token_id))
            
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            if padding == 'max_length':
                while len(tokens) < max_length:
                    tokens.append(self.pad_token_id)
            
            if return_tensors == 'pt':
                return torch.tensor(tokens).unsqueeze(0)
            return tokens
    
    return SimpleTokenizer()

def main():
    """Main training function"""
    logger.info("Starting Professional iLLuMinator 4.9B Training")
    logger.info("=" * 60)
    
    # Training configuration
    config = {
        'model_name': 'iLLuMinator-4.9B',
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'batch_size': 1,  # Adjust based on GPU memory
        'gradient_accumulation_steps': 32,
        'num_epochs': 5,
        'max_seq_length': 2048,
        'dataset_size': 25000,
        'mixed_precision': True,
        'save_every': 1000,
        'use_wandb': False,  # Set to True if you have wandb setup
        'scheduler_t0': 1000,
        'min_lr': 1e-6
    }
    
    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create tokenizer (in production, use a proper tokenizer)
    tokenizer = create_simple_tokenizer()
    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    logger.info("Creating professional training dataset...")
    try:
        dataset = ProfessionalTrainingDataset(
            tokenizer=tokenizer,
            max_length=config['max_seq_length'],
            dataset_size=config['dataset_size']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Dataset created with {len(dataset)} examples")
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        logger.info("Falling back to simple synthetic dataset...")
        
        # Fallback to simple dataset
        from practical_model.train_comprehensive import ComprehensiveDataset
        dataset = ComprehensiveDataset(tokenizer, max_length=config['max_seq_length'])
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Load model (Note: This is a placeholder - replace with actual model)
    logger.info("Loading model...")
    try:
        # Try to import the actual model
        from illuminator_model import iLLuMinator4_9B
        model = iLLuMinator4_9B(vocab_size=tokenizer.vocab_size)
    except ImportError:
        logger.warning("iLLuMinator4_9B not found, using practical model")
        from practical_model.illuminator_practical import iLLuMinatorPractical
        model = iLLuMinatorPractical()
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model loaded with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = ProfessionalTrainer(model, config)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train epoch
        avg_loss = trainer.train_epoch(dataloader, epoch + 1)
        trainer.current_epoch = epoch + 1
        
        # Log epoch results
        logger.info(f"Epoch {epoch + 1} completed - Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < trainer.best_loss:
            trainer.best_loss = avg_loss
            trainer.save_checkpoint("best_model.pt")
            logger.info(f"New best model saved with loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        trainer.save_checkpoint(f"epoch_{epoch + 1}.pt")
    
    # Training completed
    total_time = time.time() - start_time
    logger.info("Training completed!")
    logger.info(f"Total training time: {total_time / 3600:.2f} hours")
    logger.info(f"Best loss achieved: {trainer.best_loss:.4f}")
    
    # Save final model
    trainer.save_checkpoint("final_model.pt")
    logger.info("Final model saved")

if __name__ == "__main__":
    main()