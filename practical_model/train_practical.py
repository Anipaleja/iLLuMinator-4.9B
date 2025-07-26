#!/usr/bin/env python3
"""
Quick Training for Practical iLLuMinator
Simple training on basic conversational data
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

class ConversationDataset(Dataset):
    """Simple conversation dataset"""
    
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create basic training data
        self.conversations = [
            ("Hello", "Hello! I'm iLLuMinator, nice to meet you!"),
            ("Hi there", "Hi! How can I help you today?"),
            ("How are you?", "I'm doing well, thank you for asking!"),
            ("What is Python?", "Python is a powerful programming language known for its simplicity and readability."),
            ("What is programming?", "Programming is the process of creating instructions for computers to solve problems."),
            ("How do I create a function?", "To create a function in Python: def function_name(parameters): return result"),
            ("What is machine learning?", "Machine learning is a branch of AI that enables computers to learn from data."),
            ("Write a loop", "Here's a simple loop: for i in range(10): print(i)"),
            ("Hello world", "Hello world is the classic first program: print('Hello, world!')"),
            ("What is AI?", "AI (Artificial Intelligence) is technology that enables machines to simulate human intelligence."),
            ("Explain variables", "Variables store data values. In Python: name = 'value' creates a variable."),
            ("What are lists?", "Lists store multiple items: my_list = [1, 2, 3, 'hello']"),
            ("How to print?", "Use the print() function: print('your message here')"),
            ("What is coding?", "Coding is writing instructions in a programming language to create software."),
            ("Define a class", "Classes are blueprints: class MyClass: def __init__(self): pass"),
            ("Import modules", "Import modules with: import module_name or from module import function"),
            ("Handle errors", "Use try-except blocks: try: code except Exception as e: handle_error"),
            ("What are dictionaries?", "Dictionaries store key-value pairs: my_dict = {'key': 'value'}"),
            ("Explain if statements", "If statements make decisions: if condition: do_something"),
            ("What is a string?", "Strings are text data: my_string = 'Hello, world!'"),
        ]
        
        # Extend with more variations
        self.data = []
        for question, answer in self.conversations:
            # Create training examples
            text = f"Human: {question}\nAssistant: {answer}"
            self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Pad if too short
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.tokenizer.pad_token_id)
        
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

class PracticalTrainer:
    """Simple trainer for the practical model"""
    
    def __init__(self, model_save_path: str = "illuminator_practical_weights.pth"):
        self.model_save_path = model_save_path
        
        # Initialize tokenizer
        print("📚 Loading tokenizer...")
        self.tokenizer = iLLuMinatorTokenizer()
        
        # Initialize model
        print("🧠 Creating practical model...")
        self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        
        # Setup training
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tokenizer.pad_token_id)
        
        print(f"✅ Trainer initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train(self, epochs: int = 5, batch_size: int = 4):
        """Train the model on conversation data"""
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Create dataset
        dataset = ConversationDataset(self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 2 == 0:
                    print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"  Average Loss: {avg_loss:.4f}")
            
            # Test generation
            if epoch % 2 == 0:
                self._test_generation()
        
        # Save model
        self.save_model()
        print(f"✅ Training completed!")
    
    def _test_generation(self):
        """Test generation during training"""
        self.model.eval()
        
        test_prompt = "Human: Hello\nAssistant:"
        input_ids = self.tokenizer.encode(test_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_length=min(len(input_ids) + 20, 100),
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print(f"  🤖 Test: {response}")
        
        self.model.train()
    
    def save_model(self):
        """Save trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': len(self.tokenizer),
        }
        
        torch.save(checkpoint, self.model_save_path)
        print(f"💾 Model saved to {self.model_save_path}")

def main():
    """Run training"""
    print("🎯 iLLuMinator Practical Training")
    print("=" * 50)
    
    trainer = PracticalTrainer()
    
    print(f"\n📊 Dataset size: {len(ConversationDataset(trainer.tokenizer))} examples")
    
    try:
        trainer.train(epochs=10, batch_size=2)  # Small batch size for memory
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model weights saved to: illuminator_practical_weights.pth")
        print(f"🚀 Run 'python practical_ai.py' to test the trained model!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
