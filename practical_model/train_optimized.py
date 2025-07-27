#!/usr/bin/env python3
"""
Enhanced Training for iLLuMinator
Incorporating proven techniques and optimizations for better model performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from typing import List, Dict
import os
from tqdm import tqdm

from illuminator_practical import iLLuMinatorPractical

class EnhancedDataset(Dataset):
    """Dataset with optimized data preprocessing"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._create_enhanced_training_data()
        
    def _create_enhanced_training_data(self) -> List[str]:
        """Create data following optimized approach - mix of natural language and code"""
        
        # Natural language examples (70% of data for balanced training)
        natural_language = [
            # Educational content
            "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
            "Machine learning is a subset of AI that enables computers to automatically learn and improve from experience without being explicitly programmed.",
            "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
            "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
            "Computer vision enables machines to interpret and make decisions based on visual input from the world around them.",
            
            # Conversational patterns
            "Hello! I'm an AI assistant designed to help with various tasks. How can I assist you today?",
            "I'm here to help you learn about programming, artificial intelligence, and answer any questions you might have.",
            "Programming is the process of creating a set of instructions that tell a computer how to perform a task.",
            "Python is a high-level programming language known for its simplicity and readability, making it great for beginners.",
            "Functions in programming are reusable blocks of code that perform specific tasks when called.",
            
            # Technical explanations
            "Variables in programming are containers that store data values, which can be numbers, text, or other types of information.",
            "Loops allow you to execute a block of code repeatedly, which is useful for processing large amounts of data efficiently.",
            "Conditional statements like if-else allow programs to make decisions and execute different code paths based on certain conditions.",
            "Data structures like lists, dictionaries, and arrays help organize and store data in ways that make it easy to access and manipulate.",
        ]
        
        # Code examples (30% of data for coding capability)
        code_examples = [
            "def greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('World'))",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def multiply(self, a, b):\n        return a * b",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            "import numpy as np\n\ndef linear_regression(X, y):\n    return np.linalg.inv(X.T @ X) @ X.T @ y",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        ]
        
        # Conversation examples 
        conversations = [
            "Human: How do I learn Python?\nAssistant: Start with the basics like variables, loops, and functions. Practice with small projects and gradually work on more complex problems.",
            "Human: What is machine learning?\nAssistant: Machine learning is a method of data analysis that automates analytical model building using algorithms that learn from data.",
            "Human: Can you explain recursion?\nAssistant: Recursion is when a function calls itself to solve a smaller version of the same problem until it reaches a base case.",
            "Human: What's the difference between a list and a tuple in Python?\nAssistant: Lists are mutable and use square brackets, while tuples are immutable and use parentheses.",
        ]
        
        # Combine all data
        training_data = []
        training_data.extend(natural_language)
        training_data.extend(code_examples)
        training_data.extend(conversations)
        
        print(f"Created {len(training_data)} enhanced training examples")
        print(f"Natural Language: {len(natural_language)} examples")
        print(f"Code: {len(code_examples)} examples") 
        print(f"Conversations: {len(conversations)} examples")
        
        return training_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize with proper padding
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.tokenizer.pad_token_id)
        
        # Create input/target pairs
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {'input_ids': input_ids, 'labels': target_ids}

class EnhancedTrainer:
    """Enhanced trainer with proven optimization techniques"""
    
    def __init__(self):
        from transformers import AutoTokenizer
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Creating enhanced model...")
        self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimized hyperparameters
        self.lr = 4e-4  # Proven learning rate
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95  # Conservative beta2
        
        # Initialize optimizer with proven configuration
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
            label_smoothing=0.1
        )
        
        print(f"Enhanced trainer ready")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.lr}")
    
    def get_cosine_schedule_with_warmup(self, num_training_steps: int, num_warmup_steps: int = 2000):
        """Cosine learning rate schedule with warmup"""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine annealing
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)
    
    def train(self, epochs: int = 10, batch_size: int = 2):
        """Enhanced training loop"""
        print(f"Starting enhanced training for {epochs} epochs...")
        
        dataset = EnhancedDataset(self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        total_steps = len(dataloader) * epochs
        scheduler = self.get_cosine_schedule_with_warmup(total_steps)
        
        print(f"Dataset size: {len(dataset)} examples")
        print(f"Total training steps: {total_steps}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{current_lr:.2e}'
                })
            
            avg_loss = total_loss / len(dataloader)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model("illuminator_enhanced_best.pth", avg_loss)
                print(f"  Best model saved! Loss: {best_loss:.4f}")
            
            # Test generation every few epochs
            if (epoch + 1) % 3 == 0:
                self.test_generation("Hello")
        
        # Save final model
        self.save_model("illuminator_enhanced.pth", best_loss)
        print("Enhanced training completed!")
    
    def test_generation(self, prompt: str):
        """Test model generation"""
        self.model.eval()
        
        print(f"\n  Generation Test:")
        print(f"    Human: {prompt}")
        
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True)
            input_ids = inputs['input_ids'].to(self.device)
            
            # Simple generation
            generated_ids = input_ids
            for _ in range(20):  # Generate 20 tokens
                outputs = self.model(generated_ids)
                next_token_logits = outputs[0, -1, :]
                next_token = torch.multinomial(torch.softmax(next_token_logits / 0.8, dim=-1), 1)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            print(f"Assistant: {response}")
    
    def save_model(self, save_path: str, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': len(self.tokenizer),
            'model_config': {
                'vocab_size': len(self.tokenizer),
                'architecture': 'Enhanced'  
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

def main():
    """Run enhanced training"""
    print("iLLuMinator Enhanced Training")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    
    trainer = EnhancedTrainer()
    
    try:
        # Train with enhanced techniques
        trainer.train(epochs=10, batch_size=2)
        
        print(f"\nEnhanced training completed!")
        print(f"Best model: illuminator_enhanced_best.pth")
        print(f"Final model: illuminator_enhanced.pth")
        print(f"Test with: python simple_test.py")
        
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
