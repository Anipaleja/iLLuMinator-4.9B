#!/usr/bin/env python3
"""
Enhanced Training Script for 4.7B iLLuMinator Model
Uses premium datasets and advanced training techniques
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
import math

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from illuminator_ai import ProfessionalIlluminatorModel, ProfessionalTokenizer
from practical_model.enhanced_dataset_loader import DatasetProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Enhanced4_7BDataset(Dataset):
    """Dataset class for 4.7B model training"""
    
    def __init__(self, texts: List[str], tokenizer: ProfessionalTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, self.max_length)
        
        # Ensure we have enough tokens
        if len(tokens) < 2:
            tokens = [self.tokenizer.special_tokens['<BOS>'], self.tokenizer.special_tokens['<EOS>']]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.special_tokens['<PAD>']] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input and target sequences
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }

class Enhanced4_7BTrainer:
    """Advanced trainer for 4.7B model with optimization techniques"""
    
    def __init__(self, model: ProfessionalIlluminatorModel, tokenizer: ProfessionalTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")
        
        # Advanced optimizer settings (proven techniques)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=4e-4,  # Proven learning rate
            betas=(0.9, 0.95),  # Proven beta values
            weight_decay=0.1,
            eps=1e-8
        )
        
        # Loss function with label smoothing
        self.criterion = self._create_label_smoothing_loss(
            num_classes=tokenizer.vocab_size,
            smoothing=0.1,
            ignore_index=tokenizer.special_tokens['<PAD>']
        )
        
        # Cosine learning rate scheduler with warmup
        self.warmup_steps = 2000
        self.total_steps = 10000
        self.scheduler = self._create_cosine_schedule_with_warmup()
        
        # Training metrics
        self.step = 0
        self.best_loss = float('inf')
    
    def _create_label_smoothing_loss(self, num_classes: int, smoothing: float, ignore_index: int):
        """Create loss function with label smoothing"""
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, num_classes, smoothing, ignore_index):
                super().__init__()
                self.num_classes = num_classes
                self.smoothing = smoothing
                self.ignore_index = ignore_index
                self.confidence = 1.0 - smoothing
            
            def forward(self, pred, target):
                pred = pred.log_softmax(dim=-1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(pred)
                    true_dist.fill_(self.smoothing / (self.num_classes - 1))
                    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
                    true_dist[:, self.ignore_index] = 0
                    mask = torch.nonzero(target.data == self.ignore_index)
                    if mask.dim() > 0:
                        true_dist.index_fill_(0, mask.squeeze(), 0.0)
                return torch.mean(torch.sum(-true_dist * pred, dim=1))
        
        return LabelSmoothingLoss(num_classes, smoothing, ignore_index)
    
    def _create_cosine_schedule_with_warmup(self):
        """Create cosine learning rate schedule with warmup"""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch with advanced techniques"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = self.model(input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("GPU out of memory, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            
            total_loss += loss.item()
            
            # Progress logging
            if batch_idx % 20 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 50:  # Limit validation batches
                    break
        
        return total_loss / num_batches
    
    def train(self, train_dataset: Enhanced4_7BDataset, epochs: int = 5, batch_size: int = 2):
        """Main training loop with early stopping"""
        # Split dataset for validation
        val_size = min(len(train_dataset) // 10, 500)  # 10% or max 500 examples
        train_size = len(train_dataset) - val_size
        
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")
        logger.info(f"Batch size: {batch_size}")
        
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch + 1)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(f"best_4_7B_model_epoch_{epoch+1}.pth")
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
            
            # Test generation
            if (epoch + 1) % 2 == 0:
                self.test_generation()
        
        logger.info("Training completed!")
    
    def save_model(self, filename: str):
        """Save the trained model"""
        save_path = Path("models") / filename
        save_path.parent.mkdir(exist_ok=True)
        
        # Handle DataParallel wrapper
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'best_loss': self.best_loss
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def test_generation(self):
        """Test model generation capability"""
        self.model.eval()
        
        test_prompts = [
            "Human: What is artificial intelligence?\nAssistant:",
            "Human: Write a Python function to sort a list\nAssistant:",
            "Human: Explain the theory of relativity\nAssistant:"
        ]
        
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    # Simple generation test
                    input_ids = self.tokenizer.encode(prompt, max_length=128)
                    input_tensor = torch.tensor([input_ids]).to(self.device)
                    
                    # Basic forward pass to test
                    outputs = self.model(input_tensor)
                    
                    # Get top prediction
                    predicted_id = torch.argmax(outputs[0, -1, :]).item()
                    predicted_token = self.tokenizer.decode([predicted_id])
                    
                    logger.info(f"Test prompt: {prompt[:50]}...")
                    logger.info(f"Next predicted token: {predicted_token}")
                    
                except Exception as e:
                    logger.warning(f"Generation test failed: {e}")
        
        self.model.train()

def main():
    """Main training function for 4.7B model"""
    print("Enhanced Training for 4.7B iLLuMinator Model")
    print("=" * 50)
    
    # Create dataset processor
    processor = DatasetProcessor()
    
    # Load comprehensive dataset for 4.7B model
    logger.info("Loading comprehensive dataset for 4.7B model...")
    texts_4_7B, dataset_path = processor.create_dataset_for_4_7B_model()
    
    # Initialize tokenizer
    tokenizer = ProfessionalTokenizer(vocab_size=32000)
    logger.info(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = Enhanced4_7BDataset(texts_4_7B, tokenizer, max_length=512)
    logger.info(f"Dataset created with {len(dataset)} examples")
    
    # Initialize model
    model = ProfessionalIlluminatorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=1536,  # 4.7B parameters
        n_layers=32,
        n_heads=24,
        max_seq_len=512
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize trainer
    trainer = Enhanced4_7BTrainer(model, tokenizer)
    
    # Determine batch size based on available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory > 20:  # High-end GPU
            batch_size = 4
        elif gpu_memory > 10:  # Mid-range GPU
            batch_size = 2
        else:  # Limited GPU
            batch_size = 1
    else:
        batch_size = 1  # CPU training
    
    logger.info(f"Using batch size: {batch_size}")
    
    # Start training
    try:
        trainer.train(
            train_dataset=dataset,
            epochs=8,
            batch_size=batch_size
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    print("\n4.7B Model Training Complete!")

if __name__ == "__main__":
    main()
