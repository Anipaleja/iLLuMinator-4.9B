#!/usr/bin/env python3
"""
Comprehensive Training Script for iLLuMinator
Using enhanced datasets and proven optimization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import math
from typing import List, Dict
import os
from tqdm import tqdm
from pathlib import Path

from illuminator_practical import iLLuMinatorPractical
from dataset_integration import EnhancedDatasetProcessor

class ComprehensiveDataset(Dataset):
    """Comprehensive dataset using multiple high-quality sources"""
    
    def __init__(self, tokenizer, max_length: int = 512, use_external_data: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_training_data(use_external_data)
        
    def _load_training_data(self, use_external_data: bool) -> List[str]:
        """Load training data from multiple sources"""
        
        if use_external_data:
            # Try to load from saved datasets first
            dataset_path = Path("datasets/comprehensive_training_data.json")
            if dataset_path.exists():
                print(f"Loading existing dataset from {dataset_path}")
                with open(dataset_path, 'r') as f:
                    return json.load(f)
            else:
                print("Creating new comprehensive dataset...")
                processor = EnhancedDatasetProcessor()
                texts = processor.create_comprehensive_dataset()
                return texts
        
        # Fallback to built-in high-quality examples
        return self._create_builtin_dataset()
    
    def _create_builtin_dataset(self) -> List[str]:
        """Create built-in high-quality training dataset"""
        
        # High-quality educational content
        educational_content = [
            "Human: What is machine learning?\nAssistant: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Common applications include image recognition, natural language processing, and recommendation systems.",
            
            "Human: Explain neural networks in simple terms.\nAssistant: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that adjusts as the network learns. Information flows from input layer through hidden layers to output layer, with each layer transforming the data to extract increasingly complex features.",
            
            "Human: What is the difference between supervised and unsupervised learning?\nAssistant: Supervised learning uses labeled training data to learn a mapping from inputs to outputs, like email spam detection with labeled spam/not-spam examples. Unsupervised learning finds patterns in data without labels, like clustering customers by behavior or reducing data dimensions for visualization.",
            
            "Human: How does natural language processing work?\nAssistant: Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. It combines computational linguistics with machine learning and deep learning. Key tasks include tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation.",
        ]
        
        # Programming examples with explanations
        programming_content = [
            "Human: Write a Python function to reverse a string.\nAssistant: Here's a Python function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n\n# Alternative method using built-in function\ndef reverse_string_alt(s):\n    return ''.join(reversed(s))\n\n# Example usage\ntext = 'hello world'\nreversed_text = reverse_string(text)\nprint(reversed_text)  # Output: 'dlrow olleh'\n```\n\nThe first method uses Python's slice notation [::-1] which is the most efficient way.",
            
            "Human: Explain Python list comprehensions.\nAssistant: List comprehensions provide a concise way to create lists in Python. The basic syntax is: [expression for item in iterable if condition]\n\nExamples:\n```python\n# Basic list comprehension\nsquares = [x**2 for x in range(10)]\n# Result: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n\n# With condition\neven_squares = [x**2 for x in range(10) if x % 2 == 0]\n# Result: [0, 4, 16, 36, 64]\n\n# Processing strings\nwords = ['hello', 'world', 'python']\ncapitalized = [word.upper() for word in words]\n# Result: ['HELLO', 'WORLD', 'PYTHON']\n```\n\nList comprehensions are more readable and often faster than equivalent for loops.",
        ]
        
        # Science and mathematics content
        science_content = [
            "Human: Explain the scientific method.\nAssistant: The scientific method is a systematic approach to understanding the natural world through observation and experimentation:\n\n1. **Observation**: Notice something interesting or unexplained\n2. **Question**: Formulate a specific, testable question\n3. **Hypothesis**: Propose a potential explanation or prediction\n4. **Experiment**: Design and conduct controlled tests\n5. **Analysis**: Examine results and draw conclusions\n6. **Communication**: Share findings with the scientific community\n\nKey principles include reproducibility, peer review, and the willingness to revise theories based on new evidence.",
            
            "Human: What is quantum computing?\nAssistant: Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in both states simultaneously.\n\nKey concepts:\n- **Superposition**: Qubits can be in multiple states at once\n- **Entanglement**: Qubits can be correlated in ways that classical physics can't explain\n- **Interference**: Quantum states can amplify correct answers and cancel incorrect ones\n\nPotential applications include cryptography, drug discovery, financial modeling, and optimization problems.",
        ]
        
        # Combine all content
        all_content = []
        all_content.extend(educational_content)
        all_content.extend(programming_content)
        all_content.extend(science_content)
        
        print(f"Created built-in dataset with {len(all_content)} high-quality examples")
        
        return all_content
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize with proper handling
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Pad if necessary
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.pad_token_id)
        
        # Create input/target pairs for language modeling
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {'input_ids': input_ids, 'labels': target_ids}

class ComprehensiveTrainer:
    """Comprehensive trainer with advanced optimization techniques"""
    
    def __init__(self, use_external_data: bool = True):
        from transformers import AutoTokenizer
        
        print("Initializing Comprehensive Training")
        print("Loading tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Creating iLLuMinator model...")
        self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Advanced hyperparameters based on research best practices
        self.lr = 4e-4          # Proven effective learning rate
        self.weight_decay = 0.1  # L2 regularization
        self.beta1 = 0.9         # Adam momentum parameter
        self.beta2 = 0.95        # Adam second moment parameter
        self.grad_clip = 1.0     # Gradient clipping threshold
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=1e-8
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1  # Prevents overconfident predictions
        )
        
        print("Comprehensive trainer initialized")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.lr}")
    
    def get_cosine_schedule_with_warmup(self, num_training_steps: int, num_warmup_steps: int = 2000):
        """Cosine learning rate schedule with linear warmup"""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine annealing after warmup
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)
    
    def train(self, epochs: int = 15, batch_size: int = 2, use_external_data: bool = True):
        """Comprehensive training with advanced techniques"""
        print(f"Starting comprehensive training for {epochs} epochs...")
        
        # Create dataset
        dataset = ComprehensiveDataset(self.tokenizer, use_external_data=use_external_data)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Setup learning rate scheduler
        total_steps = len(dataloader) * epochs
        scheduler = self.get_cosine_schedule_with_warmup(
            num_training_steps=total_steps,
            num_warmup_steps=min(2000, total_steps // 10)  # 10% of training for warmup
        )
        
        print(f"Dataset size: {len(dataset)} examples")
        print(f"Batch size: {batch_size}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {min(2000, total_steps // 10)}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Update weights
                self.optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'LR': f'{current_lr:.2e}'
                })
            
            avg_loss = total_loss / len(dataloader)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model("illuminator_comprehensive_best.pth", avg_loss, epoch)
                print(f"Best model saved! Loss: {best_loss:.4f}")
            
            # Test generation periodically
            if (epoch + 1) % 5 == 0:
                self.test_generation("What is artificial intelligence?")
        
        # Save final model
        self.save_model("illuminator_comprehensive_final.pth", avg_loss, epochs)
        print("\nComprehensive training completed!")
        print(f"Best model: illuminator_comprehensive_best.pth (Loss: {best_loss:.4f})")
        print(f"Final model: illuminator_comprehensive_final.pth")
    
    def test_generation(self, prompt: str, max_tokens: int = 30):
        """Test model generation capability"""
        self.model.eval()
        
        print(f"\n  Generation Test:")
        print(f"    Human: {prompt}")
        
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
            input_ids = inputs['input_ids'].to(self.device)
            
            # Generate response
            generated_ids = input_ids
            for _ in range(max_tokens):
                outputs = self.model(generated_ids)
                next_token_logits = outputs[0, -1, :]
                
                # Apply temperature for controlled randomness
                temperature = 0.7
                next_token_logits = next_token_logits / temperature
                
                # Sample next token
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop at end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Decode response
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if prompt in response:
                response = response[len(prompt):].strip()
            
            print(f"    Assistant: {response}")
    
    def save_model(self, save_path: str, loss: float, epoch: int):
        """Save model checkpoint with comprehensive information"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'vocab_size': len(self.tokenizer),
            'model_config': {
                'vocab_size': len(self.tokenizer),
                'architecture': 'Comprehensive Enhanced',
                'training_method': 'Advanced optimization with multiple datasets'
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model checkpoint saved to {save_path}")

def main():
    """Main training function"""
    print("iLLuMinator Comprehensive Training")
    print("Advanced optimization with multiple high-quality datasets")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize trainer
    trainer = ComprehensiveTrainer(use_external_data=True)
    
    try:
        # Start comprehensive training
        trainer.train(epochs=15, batch_size=2, use_external_data=True)
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("Model files:")
        print("  - illuminator_comprehensive_best.pth (best performing model)")
        print("  - illuminator_comprehensive_final.pth (final model)")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")

if __name__ == "__main__":
    main()
