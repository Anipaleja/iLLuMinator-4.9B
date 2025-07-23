"""
Enhanced Training Pipeline for iLLuMinator AI with Web Data Integration
Automatically fetches web data and trains the model for improved intelligence
"""

import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
import random
from pathlib import Path
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChatDataset(Dataset):
    """Enhanced dataset that combines multiple data sources"""
    
    def __init__(self, data_paths: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data from multiple sources
        for data_path in data_paths:
            if Path(data_path).exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    if 'data' in dataset:
                        self.data.extend(dataset['data'])
                        logger.info(f"Loaded {len(dataset['data'])} samples from {data_path}")
                    else:
                        self.data.extend(dataset)
                        logger.info(f"Loaded {len(dataset)} samples from {data_path}")
        
        # Shuffle data for better training
        random.shuffle(self.data)
        logger.info(f"Total training samples: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats
        if isinstance(item, dict):
            input_text = item.get('input', '')
            output_text = item.get('output', '')
        else:
            # Fallback for simple text data
            input_text = str(item)
            output_text = ""
        
        # Create training sequence
        full_text = f"Human: {input_text}\nAssistant: {output_text}<EOS>"
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.special_tokens['<PAD>']] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }

class AdvancedModelTrainer:
    """Advanced trainer with web data integration and optimization"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        # Training metrics
        self.training_history = []
        
    def prepare_training_data(self) -> List[str]:
        """Prepare training data by fetching from web and combining sources"""
        data_files = []
        
        # Check if web data exists, if not generate it
        web_data_path = "web_training_data.json"
        if not Path(web_data_path).exists():
            logger.info("Web training data not found. Generating...")
            try:
                from web_data_fetcher import WebDataFetcher
                fetcher = WebDataFetcher()
                fetcher.compile_web_data(web_data_path)
            except Exception as e:
                logger.warning(f"Could not generate web data: {e}")
        
        if Path(web_data_path).exists():
            data_files.append(web_data_path)
        
        # Check for other training data files
        other_files = [
            "enhanced_training_data.json",
            "training_data.json",
            "conversation_data.json"
        ]
        
        for file_path in other_files:
            if Path(file_path).exists():
                data_files.append(file_path)
        
        if not data_files:
            # Create basic training data if nothing exists
            logger.info("No training data found. Creating basic dataset...")
            self._create_basic_training_data()
            data_files.append("basic_training_data.json")
        
        return data_files
    
    def _create_basic_training_data(self):
        """Create basic training data if no other sources available"""
        basic_data = [
            {
                "input": "Hello",
                "output": "Hello! I'm iLLuMinator AI, here to help you with programming, technical questions, and general conversation. How can I assist you today?",
                "category": "greeting"
            },
            {
                "input": "What is Python?",
                "output": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, machine learning, automation, and more.",
                "category": "programming"
            },
            {
                "input": "Write a hello world program",
                "output": "```python\nprint('Hello, World!')\n```\n\nThis is the classic 'Hello, World!' program in Python. It simply prints the text to the console.",
                "category": "code_generation"
            },
            {
                "input": "Explain functions in programming",
                "output": "Functions are reusable blocks of code that perform specific tasks. They help organize code, reduce repetition, and make programs more modular. Functions typically take inputs (parameters), process them, and return outputs.",
                "category": "programming_concepts"
            },
            {
                "input": "Thank you",
                "output": "You're welcome! I'm here whenever you need help with programming, technical questions, or anything else. Feel free to ask anytime!",
                "category": "courtesy"
            }
        ]
        
        dataset = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "total_samples": len(basic_data),
                "description": "Basic training data for iLLuMinator AI"
            },
            "data": basic_data
        }
        
        with open("basic_training_data.json", 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    def train_with_web_data(
        self, 
        epochs: int = 2, 
        learning_rate: float = 5e-6, 
        batch_size: int = 1,
        save_path: str = "enhanced_illuminator_model.pth"
    ):
        """Train the model using web data and other sources"""
        
        logger.info("Starting enhanced training with web data integration...")
        
        # Prepare training data
        data_files = self.prepare_training_data()
        logger.info(f"Using data files: {data_files}")
        
        # Create dataset
        try:
            dataset = EnhancedChatDataset(data_files, self.tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0,
                drop_last=True
            )
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return
        
        # Setup optimizer with gradient accumulation
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(dataloader)
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        total_steps = 0
        gradient_accumulation_steps = 4  # Accumulate gradients
        
        logger.info(f"Training for {epochs} epochs with {len(dataloader)} batches per epoch")
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_steps = 0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    logits = self.model(input_ids)
                    
                    # Calculate loss
                    loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<PAD>'])
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights every gradient_accumulation_steps
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    total_loss += loss.item() * gradient_accumulation_steps
                    epoch_steps += 1
                    total_steps += 1
                    
                    # Progress logging
                    if batch_idx % 20 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                            f"Loss: {loss.item() * gradient_accumulation_steps:.4f}, LR: {current_lr:.2e}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Epoch summary
            epoch_time = time.time() - start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            
            logger.info(
                f"Epoch {epoch+1} completed in {epoch_time:.2f}s. "
                f"Average loss: {avg_epoch_loss:.4f}"
            )
            
            # Save training history
            self.training_history.append({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "time": epoch_time,
                "timestamp": datetime.now().isoformat()
            })
        
        # Final training summary
        avg_total_loss = total_loss / max(total_steps, 1)
        logger.info(f"Training completed! Average total loss: {avg_total_loss:.4f}")
        
        # Save the enhanced model
        self.save_enhanced_model(save_path)
        
        return self.training_history
    
    def save_enhanced_model(self, save_path: str):
        """Save the enhanced model with training metadata"""
        try:
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'tokenizer_vocab_size': self.tokenizer.vocab_size,
                'model_config': {
                    'vocab_size': self.model.vocab_size,
                    'd_model': self.model.d_model,
                    'n_layers': self.model.n_layers,
                    'n_heads': self.model.n_heads,
                    'max_seq_len': self.model.max_seq_len
                },
                'training_history': self.training_history,
                'training_metadata': {
                    'enhanced_with_web_data': True,
                    'training_date': datetime.now().isoformat(),
                    'version': '3.0'
                }
            }
            
            torch.save(save_data, save_path)
            logger.info(f"Enhanced model saved to {save_path}")
            
            # Save training history separately
            history_path = save_path.replace('.pth', '_training_history.json')
            with open(history_path, 'w') as f:
                json.dump({
                    'training_history': self.training_history,
                    'metadata': save_data['training_metadata']
                }, f, indent=2)
            
            logger.info(f"Training history saved to {history_path}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced model: {e}")

def run_enhanced_training():
    """Run enhanced training with web data integration"""
    
    logger.info("=== iLLuMinator AI Enhanced Training with Web Data ===")
    
    try:
        # First, generate web training data
        logger.info("Step 1: Generating comprehensive web training data...")
        from web_data_fetcher import WebDataFetcher
        fetcher = WebDataFetcher()
        fetcher.compile_web_data("web_training_data.json")
        
        # Initialize AI model
        logger.info("Step 2: Loading iLLuMinator AI model...")
        from illuminator_ai import IlluminatorAI
        ai = IlluminatorAI(fast_mode=True)  # Use fast mode for training efficiency
        
        # Create trainer
        trainer = AdvancedModelTrainer(ai.model, ai.tokenizer, ai.device)
        
        # Run enhanced training
        logger.info("Step 3: Starting enhanced training with web data...")
        training_history = trainer.train_with_web_data(
            epochs=3,  # More epochs with web data
            learning_rate=2e-6,  # Conservative learning rate
            batch_size=1,  # Small batch for memory efficiency
            save_path="web_enhanced_illuminator_model.pth"
        )
        
        logger.info("Enhanced training completed successfully!")
        logger.info("The model has been significantly improved with web-sourced data.")
        logger.info("Restart the API server to use the enhanced model.")
        
        return training_history
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        return None

if __name__ == "__main__":
    run_enhanced_training()
