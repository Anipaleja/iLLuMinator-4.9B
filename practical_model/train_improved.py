#!/usr/bin/env python3
"""
Improved Training for Practical iLLuMinator
Better anti-overfitting techniques with more data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Dict
import random

from illuminator_practical import iLLuMinatorPractical
from tokenizer import iLLuMinatorTokenizer

class ImprovedConversationDataset(Dataset):
    """Improved conversation dataset with more data and better structure"""
    
    def __init__(self, tokenizer, max_length: int = 200):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._create_training_data()
        
    def _create_training_data(self) -> List[str]:
        """Create comprehensive training data"""
        
        conversations = [
            # Basic greetings - various formats
            ("Hello", "Hello! I'm iLLuMinator, your AI assistant. How can I help you today?"),
            ("Hi", "Hi there! I'm here to assist you with any questions you have."),
            ("Hey", "Hey! What can I help you with?"),
            ("Good morning", "Good morning! What would you like to explore today?"),
            ("How are you?", "I'm doing well and ready to help! What's on your mind?"),
            ("What's up?", "Hello! I'm here and ready to assist. What can I do for you?"),
            
            # Programming questions - comprehensive coverage
            ("What is Python?", "Python is a high-level programming language known for its simplicity and readability. It's used for web development, data science, AI, and automation."),
            ("How do I create a function?", "To create a function in Python, use the 'def' keyword:\n\ndef my_function(parameter):\n    # Your code here\n    return result"),
            ("What are variables?", "Variables store data values. In Python: name = 'Alice', age = 25, is_student = True. Python determines the type automatically."),
            ("Explain lists", "Lists store multiple items: my_list = [1, 2, 3, 'text']. Access with brackets: my_list[0]. Lists are ordered and changeable."),
            ("What are dictionaries?", "Dictionaries store key-value pairs: person = {'name': 'Alice', 'age': 25}. Access values with keys: person['name']."),
            ("How do loops work?", "Python has for loops: 'for item in list:' and while loops: 'while condition:'. Both repeat code blocks."),
            ("What are classes?", "Classes are blueprints for objects:\n\nclass Person:\n    def __init__(self, name):\n        self.name = name"),
            ("How to handle errors?", "Use try-except blocks:\n\ntry:\n    risky_code()\nexcept Exception as e:\n    print(f'Error: {e}')"),
            ("What is a string?", "Strings are text data: my_string = 'Hello, world!'. Use quotes to create them."),
            ("How to print output?", "Use the print() function: print('Hello, world!') or print(variable_name)."),
            
            # AI and technology
            ("What is AI?", "Artificial Intelligence (AI) creates computer systems that can perform tasks requiring human-like intelligence, such as learning and problem-solving."),
            ("Explain machine learning", "Machine learning lets computers learn patterns from data to make predictions without being explicitly programmed for each task."),
            ("What is deep learning?", "Deep learning uses neural networks with multiple layers to learn complex patterns from data, especially good for images and text."),
            ("How do neural networks work?", "Neural networks have connected nodes that process information through weighted connections, learning by adjusting these weights."),
            
            # Data science
            ("What is data science?", "Data science combines statistics, programming, and domain knowledge to extract insights from data for decision-making."),
            ("What is supervised learning?", "Supervised learning uses labeled training data where correct answers are known to train models that predict on new data."),
            ("What is unsupervised learning?", "Unsupervised learning finds hidden patterns in data without labeled examples, like discovering clusters or groups."),
            
            # Problem solving
            ("How to debug code?", "Debug by reading error messages, using print statements, checking logic step by step, and testing with simple inputs."),
            ("Best coding practices?", "Write clean, readable code with meaningful names. Comment your code, test thoroughly, and follow consistent formatting."),
            ("How to learn programming?", "Start with basics, practice regularly, build small projects, read others' code, and don't be afraid to make mistakes."),
            
            # Simple factual questions
            ("What is coding?", "Coding is writing instructions in a programming language to create software that solves problems or performs tasks."),
            ("What is an algorithm?", "An algorithm is a step-by-step procedure for solving a problem or completing a task."),
            ("What is debugging?", "Debugging is the process of finding and fixing errors or bugs in computer programs."),
            ("What is a database?", "A database is an organized collection of structured information stored electronically in a computer system."),
            ("What is an API?", "An API (Application Programming Interface) allows different software applications to communicate and share data."),
        ]
        
        # Create training examples with multiple formats
        training_texts = []
        
        # Standard Human/Assistant format
        for question, answer in conversations:
            training_texts.append(f"Human: {question}\nAssistant: {answer}")
        
        # Add some variations for robustness (but not too many to avoid overfitting)
        for question, answer in conversations[:len(conversations)//3]:
            training_texts.append(f"User: {question}\nAI: {answer}")
            training_texts.append(f"Q: {question}\nA: {answer}")
        
        # Add some educational content
        educational_content = [
            "Python uses indentation to define code blocks, making it visually clear and readable.",
            "Variables in Python don't need type declarations - Python figures out the type automatically.",
            "Functions help organize code into reusable blocks that can be called multiple times.",
            "Lists and dictionaries are fundamental data structures in Python for storing collections of data.",
            "Error handling with try-except prevents programs from crashing when something goes wrong.",
            "Comments in code explain what the code does, making it easier to understand later.",
            "Good programming involves writing clear, maintainable code that others can understand.",
            "Testing code ensures it works correctly before deploying it for others to use.",
        ]
        
        training_texts.extend(educational_content)
        
        print(f"Created {len(training_texts)} training examples")
        return training_texts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Ensure consistent length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Pad to max length
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.tokenizer.pad_token_id)
        
        # Create input/target pairs
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

class ImprovedTrainer:
    """Improved trainer with better anti-overfitting techniques"""
    
    def __init__(self, model_save_path: str = "illuminator_practical_improved.pth"):
        self.model_save_path = model_save_path
        
        # Initialize tokenizer
        print("📚 Loading tokenizer...")
        self.tokenizer = iLLuMinatorTokenizer()
        
        # Initialize model
        print("🧠 Creating improved practical model...")
        self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        
        # Better optimizer settings for stability
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=3e-4,  # Lower learning rate
            weight_decay=0.01,  # Regularization
            betas=(0.9, 0.95)  # Better momentum
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.8
        )
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.1  # Prevents overconfidence
        )
        
        print(f"✅ Improved trainer initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train(self, epochs: int = 15, batch_size: int = 2):
        """Train with improved techniques"""
        
        print(f"🚀 Starting improved training for {epochs} epochs...")
        
        # Create dataset
        dataset = ImprovedConversationDataset(self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"📊 Dataset size: {len(dataset)} examples")
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            print(f"\n📖 Epoch {epoch + 1}/{epochs}")
            
            for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
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
                
                # Gradient clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            avg_loss = total_loss / num_batches
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_model(is_best=True)
                print("  ⭐ Best loss so far - model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  🛑 Early stopping at epoch {epoch + 1}")
                    break
            
            # Test generation every few epochs
            if epoch % 2 == 0:
                self._test_generation()
        
        # Final save
        self.save_model()
        print("✅ Improved training completed!")
    
    def _test_generation(self):
        """Test generation during training"""
        self.model.eval()
        
        test_prompts = [
            "Human: Hello\nAssistant:",
            "Human: What is Python?\nAssistant:",
            "Human: How do loops work?\nAssistant:"
        ]
        
        print("  🤖 Generation Test:")
        
        for prompt in test_prompts[:1]:  # Test one to save time
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_length=min(len(input_ids) + 25, 100),
                    temperature=0.8,
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
            'vocab_size': len(self.tokenizer),
        }
        
        save_path = self.model_save_path
        if is_best:
            save_path = save_path.replace('.pth', '_best.pth')
        
        torch.save(checkpoint, save_path)
        if is_best:
            print(f"💎 Best model saved to {save_path}")

def main():
    """Run improved training"""
    print("🎯 iLLuMinator Improved Training")
    print("=" * 50)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    trainer = ImprovedTrainer()
    
    try:
        trainer.train(epochs=12, batch_size=2)
        
        print(f"\n🎉 Improved training completed successfully!")
        print(f"📁 Best model: illuminator_practical_improved_best.pth")
        print(f"📁 Final model: illuminator_practical_improved.pth")
        print(f"🚀 Test with: python enhanced_test.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
