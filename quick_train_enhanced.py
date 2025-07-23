"""
Enhanced Training Script for iLLuMinator AI
Uses the fetched training data to improve model performance
"""

import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import random
from pathlib import Path
import time

class ChatDataset(Dataset):
    """Dataset for training the iLLuMinator model on conversational data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load training data
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.data = dataset['data']
        print(f"Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create input-output pair
        input_text = f"Human: {item['input']}\nAssistant:"
        target_text = item['output']
        
        # Tokenize
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length//2)
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length//2)
        
        # Combine for training
        full_sequence = input_ids + target_ids + [self.tokenizer.special_tokens['<EOS>']]
        
        # Pad to max_length
        if len(full_sequence) < self.max_length:
            full_sequence.extend([self.tokenizer.special_tokens['<PAD>']] * (self.max_length - len(full_sequence)))
        else:
            full_sequence = full_sequence[:self.max_length]
        
        return {
            'input_ids': torch.tensor(full_sequence[:-1], dtype=torch.long),
            'labels': torch.tensor(full_sequence[1:], dtype=torch.long)
        }

class ModelTrainer:
    """Enhanced trainer for the iLLuMinator model"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def train(self, data_path: str, epochs: int = 1, learning_rate: float = 1e-5, batch_size: int = 2):
        """Train the model on the enhanced dataset"""
        
        # Create dataset and dataloader
        dataset = ChatDataset(data_path, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Setup optimizer with low learning rate for fine-tuning
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Training loop
        self.model.train()
        total_loss = 0
        total_steps = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Total batches per epoch: {len(dataloader)}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                try:
                    logits = self.model(input_ids)
                    
                    # Calculate loss
                    loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<PAD>'])
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    epoch_steps += 1
                    total_steps += 1
                    
                    # Progress update
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        avg_total_loss = total_loss / max(total_steps, 1)
        print(f"Training completed! Average loss: {avg_total_loss:.4f}")
        
        # Save the improved model
        self.save_model("improved_illuminator_model.pth")
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer_vocab_size': self.tokenizer.vocab_size,
                'model_config': {
                    'vocab_size': self.model.vocab_size,
                    'd_model': self.model.d_model,
                    'n_layers': self.model.n_layers,
                    'n_heads': self.model.n_heads,
                    'max_seq_len': self.model.max_seq_len
                }
            }, save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

def quick_training_session():
    """Run a quick training session to improve the model"""
    
    print("=== iLLuMinator AI Enhanced Training ===")
    print("This will improve the model using high-quality training data")
    
    try:
        # Import the model
        from illuminator_ai import IlluminatorAI
        
        # Initialize AI with fast mode
        print("Loading iLLuMinator AI...")
        ai = IlluminatorAI(fast_mode=True)
        
        # Check if training data exists
        data_path = "enhanced_training_data.json"
        if not Path(data_path).exists():
            print(f"Training data not found at {data_path}")
            print("Generating training data first...")
            
            # Generate training data
            from enhanced_training_data_fetcher import TrainingDataFetcher
            fetcher = TrainingDataFetcher()
            fetcher.compile_training_data(data_path)
        
        # Create trainer
        trainer = ModelTrainer(ai.model, ai.tokenizer, ai.device)
        
        # Run training with minimal epochs for speed
        print("Starting enhanced training...")
        trainer.train(
            data_path=data_path,
            epochs=1,  # Quick training
            learning_rate=5e-6,  # Very low learning rate for stability
            batch_size=1  # Small batch size for memory efficiency
        )
        
        print("Training completed! The model has been improved with new data.")
        print("Restart the API server to use the enhanced model.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure PyTorch is installed and the model is properly initialized.")

if __name__ == "__main__":
    quick_training_session()
