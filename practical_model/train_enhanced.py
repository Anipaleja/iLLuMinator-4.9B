#!/usr/bin/env python3
"""
Enhanced Training for Practical iLLuMinator
Features anti-overfitting techniques, validation, and better data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from typing import List, Dict, Tuple
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from illuminator_practical import iLLuMinatorPractical
from tokenizer import iLLuMinatorTokenizer

class EnhancedConversationDataset(Dataset):
    """Enhanced conversation dataset with anti-overfitting measures"""
    
    def __init__(self, tokenizer, max_length: int = 256, augment_data: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_data = augment_data
        self.data = self._create_comprehensive_dataset()
        
    def _create_comprehensive_dataset(self) -> List[str]:
        """Create comprehensive training data with variations"""
        
        # Base conversational patterns
        base_conversations = [
            # Greetings and basic interaction
            ("Hello", "Hello! I'm iLLuMinator, your AI assistant. How can I help you today?"),
            ("Hi there", "Hi! I'm here to assist you with any questions or tasks you have."),
            ("Good morning", "Good morning! What would you like to explore or learn about today?"),
            ("How are you?", "I'm doing great and ready to help! What's on your mind?"),
            ("What's up?", "Hello! I'm here and ready to assist. What can I help you with?"),
            
            # AI and technology
            ("What is AI?", "Artificial Intelligence (AI) is the field of creating computer systems that can perform tasks requiring human-like intelligence, such as learning, reasoning, and problem-solving."),
            ("Explain machine learning", "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each specific task."),
            ("What is deep learning?", "Deep learning uses neural networks with multiple layers to automatically learn complex patterns from data. It's particularly powerful for tasks like image recognition and natural language processing."),
            ("How do neural networks work?", "Neural networks consist of interconnected nodes (neurons) that process information through weighted connections. They learn by adjusting these weights based on training data."),
            ("What are transformers?", "Transformers are a neural network architecture that revolutionized AI by using attention mechanisms to process sequences efficiently, enabling better understanding of context and relationships."),
            
            # Programming basics
            ("What is Python?", "Python is a high-level programming language known for its simplicity and readability. It's widely used for web development, data science, AI, and automation."),
            ("How do I create a function?", "In Python, create a function using 'def':\n\ndef my_function(parameter):\n    # Your code here\n    return result\n\n# Call it like this:\nresult = my_function('hello')"),
            ("What are variables?", "Variables store data values in Python. Simply assign a value: name = 'Alice', age = 25, is_student = True. Python automatically determines the data type."),
            ("Explain Python lists", "Lists store multiple items in order: my_list = [1, 2, 3, 'text']. Access items with brackets: my_list[0] gets the first item. Lists are mutable and can be modified."),
            ("What are dictionaries?", "Dictionaries store key-value pairs: person = {'name': 'Alice', 'age': 25}. Access values with keys: person['name'] returns 'Alice'."),
            ("How do loops work?", "Python has for loops for iterating: 'for item in list:', and while loops for conditions: 'while condition:'. Both execute code repeatedly."),
            ("What are classes?", "Classes are blueprints for objects:\n\nclass Person:\n    def __init__(self, name):\n        self.name = name\n    \n    def greet(self):\n        return f'Hello, I am {self.name}'"),
            
            # Data science and ML
            ("What is data science?", "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting data to solve problems."),
            ("Explain supervised learning", "Supervised learning uses labeled training data where the correct answers are known. The algorithm learns from input-output pairs to make predictions on new, unseen data."),
            ("What is unsupervised learning?", "Unsupervised learning finds hidden patterns in data without labeled examples. It discovers structures like clusters, associations, or anomalies in datasets."),
            ("How does training work?", "Model training involves feeding data to an algorithm, which adjusts its parameters to minimize prediction errors. This process repeats until the model performs well."),
            ("What is overfitting?", "Overfitting occurs when a model learns training data too well, including noise and irrelevant patterns. This causes poor performance on new data."),
            
            # Problem solving
            ("How do I debug code?", "Debug by: 1) Reading error messages carefully, 2) Using print statements to trace values, 3) Checking logic step by step, 4) Using a debugger, 5) Testing with simple inputs."),
            ("Best practices for coding?", "Write clean, readable code with meaningful names. Comment your code, use functions to organize logic, test thoroughly, and follow consistent formatting."),
            ("How to learn programming?", "Start with basics, practice regularly with small projects, read others' code, join communities, build real projects, and don't be afraid to make mistakes."),
            ("What makes good code?", "Good code is readable, maintainable, efficient, and well-tested. It follows conventions, has clear structure, and is documented appropriately."),
        ]
        
        # Create training examples with different formats
        training_texts = []
        
        # Multiple conversation formats for robustness
        formats = [
            ("Human: {}\nAssistant: {}", "Human", "Assistant"),
            ("User: {}\nAI: {}", "User", "AI"),
            ("Q: {}\nA: {}", "Q", "A"),
            ("Question: {}\nAnswer: {}", "Question", "Answer"),
            ("{}\n\n{}", "", ""),  # Direct format
        ]
        
        # Generate conversations with format variations
        for question, answer in base_conversations:
            for format_template, q_prefix, a_prefix in formats:
                if format_template == "{}\n\n{}":
                    training_texts.append(format_template.format(question, answer))
                else:
                    training_texts.append(format_template.format(question, answer))
        
        # Add educational standalone content
        educational_content = [
            "Python syntax is designed to be readable and intuitive. Indentation defines code blocks, making the structure visually clear.",
            "Version control with Git helps track changes, collaborate with others, and manage different versions of your code effectively.",
            "APIs (Application Programming Interfaces) allow different software applications to communicate and share data with each other.",
            "Data structures like lists, dictionaries, and sets organize information efficiently for different use cases and operations.",
            "Algorithm complexity describes how runtime or memory usage grows with input size, helping choose efficient solutions.",
            "Testing ensures code works correctly. Unit tests verify individual functions, while integration tests check system components together.",
            "Documentation explains how code works, making it easier for others (and future you) to understand and maintain.",
            "Regular expressions (regex) are patterns for matching and manipulating text, useful for data validation and processing.",
        ]
        
        training_texts.extend(educational_content)
        
        # Add code examples with explanations
        code_examples = [
            "# List comprehension - concise way to create lists\nnumbers = [1, 2, 3, 4, 5]\nsquares = [x**2 for x in numbers]\nprint(squares)  # [1, 4, 9, 16, 25]",
            
            "# Dictionary usage example\nstudent = {'name': 'Alice', 'grade': 85, 'subjects': ['Math', 'Science']}\nprint(f\"{student['name']} scored {student['grade']} points\")",
            
            "# Function with default parameters\ndef greet(name, greeting='Hello'):\n    return f'{greeting}, {name}!'\n\nprint(greet('Alice'))  # Hello, Alice!\nprint(greet('Bob', 'Hi'))  # Hi, Bob!",
            
            "# Error handling example\ntry:\n    result = 10 / int(input('Enter number: '))\n    print(f'Result: {result}')\nexcept ValueError:\n    print('Please enter a valid number')\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')",
            
            "# Working with files\nwith open('data.txt', 'r') as file:\n    content = file.read()\n    lines = content.splitlines()\nprint(f'File has {len(lines)} lines')",
        ]
        
        training_texts.extend(code_examples)
        
        # Add data augmentation if enabled
        if self.augment_data:
            augmented = self._augment_data(training_texts)
            training_texts.extend(augmented)
        
        return training_texts
    
    def _augment_data(self, texts: List[str]) -> List[str]:
        """Augment data with variations to prevent overfitting"""
        augmented = []
        
        for text in texts[:len(texts)//2]:  # Augment subset to avoid too much repetition
            # Add slight variations
            if "Human:" in text:
                # Replace with synonymous phrases
                variations = [
                    text.replace("Human:", "Person:"),
                    text.replace("Assistant:", "AI Assistant:"),
                    text.replace("Hello", "Greetings") if "Hello" in text and random.random() > 0.7 else text,
                ]
                augmented.extend([v for v in variations if v != text])
        
        return augmented[:len(texts)//3]  # Limit augmentation to prevent overfitting
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Fixed length for all sequences
        max_len = self.max_length
        
        # Truncate if too long
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # Pad if too short
        while len(tokens) < max_len:
            tokens.append(self.tokenizer.tokenizer.pad_token_id)
        
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

class EnhancedTrainer:
    """Enhanced trainer with validation and anti-overfitting techniques"""
    
    def __init__(self, model_save_path: str = "illuminator_practical_enhanced.pth"):
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = iLLuMinatorTokenizer()
        
        # Initialize model with dropout for regularization
        print("Creating enhanced practical model...")
        self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        self.model.to(self.device)
        
        # Enhanced optimizer with proper regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=5e-4,  # Slightly higher learning rate
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Loss function with label smoothing to prevent overfitting
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"Enhanced trainer initialized")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
    
    def train(self, epochs: int = 15, batch_size: int = 4, validation_split: float = 0.2):
        """Train with validation and early stopping"""
        
        print(f"Starting enhanced training for {epochs} epochs...")
        
        # Create dataset
        full_dataset = EnhancedConversationDataset(self.tokenizer, augment_data=True)
        
        # Split into train/validation
        total_size = len(full_dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"Dataset split: {train_size} train, {val_size} validation samples")
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(is_best=True)
                print(f"  Best validation loss so far!")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            # Test generation every few epochs
            if epoch % 3 == 0:
                self._test_generation()
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Final save and plot results
        self.save_model()
        self._plot_training_history()
        print(f"Enhanced training completed!")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for input_ids, target_ids in pbar:
            input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids)
            
            # Calculate loss
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _test_generation(self):
        """Test generation during training"""
        self.model.eval()
        
        test_prompts = [
            "Human: Hello\nAssistant:",
            "User: What is Python?\nAI:",
            "Q: How do loops work?\nA:"
        ]
        
        print(f"\n  Generation Test:")
        
        for prompt in test_prompts[:1]:  # Test one prompt to save time
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_length=min(len(input_ids) + 30, 150),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            print(f"    {response}")
        
        self.model.train()
    
    def save_model(self, is_best: bool = False):
        """Save trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'vocab_size': len(self.tokenizer),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
        }
        
        save_path = self.model_save_path
        if is_best:
            save_path = save_path.replace('.pth', '_best.pth')
        
        torch.save(checkpoint, save_path)
        if is_best:
            print(f"Best model saved to {save_path}")
    
    def _plot_training_history(self):
        """Plot training curves"""
        try:
            plt.figure(figsize=(12, 4))
            
            # Loss curves
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
            plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Learning rate
            plt.subplot(1, 2, 2)
            plt.plot(self.learning_rates, label='Learning Rate', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Training curves saved to training_history.png")
            
        except Exception as e:
            print(f"Could not create plots: {e}")

def main():
    """Run enhanced training"""
    print("iLLuMinator Enhanced Training")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = EnhancedTrainer()
    
    try:
        trainer.train(epochs=10, batch_size=1, validation_split=0.2)
        
        print(f"\nEnhanced training completed successfully!")
        print(f"Best model: illuminator_practical_enhanced_best.pth")
        print(f"Final model: illuminator_practical_enhanced.pth")
        print(f"Run 'python enhanced_test.py' to test the trained model!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
