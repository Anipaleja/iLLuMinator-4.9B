#!/usr/bin/env python3
"""
Final Training for Practical iLLuMinator
Ultra-conservative approach to prevent overfitting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

from illuminator_practical import iLLuMinatorPractical
from tokenizer import iLLuMinatorTokenizer

class ConservativeDataset(Dataset):
    """Conservative dataset with minimal overfitting risk"""
    
    def __init__(self, tokenizer, max_length: int = 150):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._create_conservative_data()
        
    def _create_conservative_data(self) -> list:
        """Create conservative training data with variety"""
        
        # Core conversation patterns - keep it simple and diverse
        conversations = [
            # Basic greetings
            ("Hello", "Hello! I'm iLLuMinator, an AI assistant. How can I help you?"),
            ("Hi", "Hi there! What can I assist you with today?"),
            ("Hey", "Hey! I'm here to help. What do you need?"),
            ("Good morning", "Good morning! How can I help you today?"),
            ("How are you?", "I'm doing well, thank you! How can I assist you?"),
            
            # Python basics
            ("What is Python?", "Python is a programming language known for being easy to read and write. It's used for many things like websites and data analysis."),
            ("How do I print something?", "Use the print() function. For example: print('Hello World') will display Hello World."),
            ("What is a variable?", "A variable stores a value. Like: name = 'Alice' stores the text Alice in a variable called name."),
            ("What is a function?", "A function is reusable code. You create one with def function_name(): and put your code inside."),
            ("What is a list?", "A list stores multiple items. Like: fruits = ['apple', 'banana', 'orange']. You can access items with fruits[0]."),
            
            # Simple AI concepts
            ("What is AI?", "AI stands for Artificial Intelligence. It means making computers smart enough to solve problems and learn from data."),
            ("What is machine learning?", "Machine learning is when computers learn patterns from examples, like recognizing pictures or understanding text."),
            ("What is programming?", "Programming is writing instructions for computers. You tell the computer what to do step by step."),
            
            # Helpful responses
            ("Can you help me?", "Of course! I'm here to help. What would you like to know about?"),
            ("Thank you", "You're welcome! Is there anything else I can help you with?"),
            ("I don't understand", "No problem! I'm here to explain things. What would you like me to clarify?"),
            
            # Problem solving
            ("How do I learn programming?", "Start with basic concepts, practice with small programs, and don't be afraid to make mistakes. Practice is key!"),
            ("I'm stuck on my code", "That's normal when learning! Try reading your code line by line, or break the problem into smaller pieces."),
        ]
        
        # Create training examples
        training_examples = []
        
        # Use only Human/Assistant format for consistency
        for question, answer in conversations:
            training_examples.append(f"Human: {question}\nAssistant: {answer}")
        
        # Add a few educational snippets (very few to avoid overfitting)
        educational = [
            "Python uses indentation to organize code. This makes it easy to read.",
            "Variables in Python can store text, numbers, or lists of items.",
            "Functions help you organize code so you can reuse it multiple times.",
        ]
        
        training_examples.extend(educational)
        
        print(f"Created {len(training_examples)} conservative training examples")
        return training_examples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Fixed length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.tokenizer.pad_token_id)
        
        # Create input/target pairs
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

class ConservativeTrainer:
    """Very conservative trainer to minimize overfitting"""
    
    def __init__(self, model_save_path: str = "illuminator_practical_final.pth"):
        self.model_save_path = model_save_path
        
        print("📚 Loading tokenizer...")
        self.tokenizer = iLLuMinatorTokenizer()
        
        print("🧠 Creating conservative model...")
        self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        
        # Very conservative optimizer settings
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-4,  # Very low learning rate
            weight_decay=0.05,  # Strong regularization
            betas=(0.9, 0.95)
        )
        
        # Loss with strong label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.2  # Higher smoothing
        )
        
        print(f"✅ Conservative trainer ready with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train(self, epochs: int = 8, batch_size: int = 1):
        """Conservative training with minimal epochs"""
        
        print(f"🚀 Starting conservative training for {epochs} epochs...")
        
        # Create dataset
        dataset = ConservativeDataset(self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"📊 Dataset size: {len(dataset)} examples")
        
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
                
                # Strong gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"  Average Loss: {avg_loss:.4f}")
            
            # Test every 2 epochs
            if epoch % 2 == 0:
                self._test_generation()
                
            # Save model after each epoch (conservative approach)
            if epoch >= 3:  # Only save after some training
                self.save_model()
        
        print("✅ Conservative training completed!")
    
    def _test_generation(self):
        """Quick test during training"""
        self.model.eval()
        
        prompt = "Human: Hello\nAssistant:"
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_length=len(input_ids) + 15,  # Short generation
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print(f"  🤖 Test: {response}")
        
        self.model.train()
    
    def save_model(self):
        """Save model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': len(self.tokenizer),
        }
        
        torch.save(checkpoint, self.model_save_path)
        print(f"💾 Model saved to {self.model_save_path}")

def main():
    """Run conservative training"""
    print("🎯 iLLuMinator Conservative Training")
    print("=" * 50)
    
    # Set seeds
    torch.manual_seed(42)
    random.seed(42)
    
    trainer = ConservativeTrainer()
    
    try:
        # Very few epochs to prevent overfitting
        trainer.train(epochs=6, batch_size=1)
        
        print(f"\n🎉 Conservative training completed!")
        print(f"📁 Model saved: illuminator_practical_final.pth")
        print(f"🧪 Test with: python simple_test.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
