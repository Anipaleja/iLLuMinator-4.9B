#!/usr/bin/env python3
"""
TinyLlama-Inspired Training for iLLuMinator
Incorporating TinyLlama's proven techniques and optimizations
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
from tokenizer import iLLuMinatorTokenizer

class TinyLlamaInspiredDataset(Dataset):
    """Dataset inspired by TinyLlama's data preprocessing"""
    
    def __init__(self, tokenizer, max_length: int = 512):  # Shorter than TinyLlama's 2048 for memory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._create_tinyllama_style_data()
        
    def _create_tinyllama_style_data(self) -> List[str]:
        """Create data following TinyLlama's approach - mix of natural language and code"""
        
        # Natural language examples (70% as per TinyLlama)
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
            "Algorithms are step-by-step procedures for solving problems or performing computations in programming.",
        ]
        
        # Code examples (30% as per TinyLlama)
        code_examples = [
            # Python basics
            "def greet(name):\n    return f'Hello, {name}! Welcome to programming.'\n\n# Call the function\nresult = greet('Alice')\nprint(result)",
            
            "# Working with lists\nfruits = ['apple', 'banana', 'orange']\nfor fruit in fruits:\n    print(f'I like {fruit}')",
            
            "# Simple calculator\ndef add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n\nresult = add(5, 3)\nprint(f'5 + 3 = {result}')",
            
            "# Working with dictionaries\nstudent = {\n    'name': 'John',\n    'age': 20,\n    'grades': [85, 92, 78]\n}\n\nprint(f\"{student['name']} is {student['age']} years old\")",
            
            "# Simple class example\nclass Dog:\n    def __init__(self, name):\n        self.name = name\n    \n    def bark(self):\n        return f'{self.name} says woof!'\n\nmy_dog = Dog('Buddy')\nprint(my_dog.bark())",
            
            "# File handling\nwith open('data.txt', 'r') as file:\n    content = file.read()\n    lines = content.splitlines()\nprint(f'File has {len(lines)} lines')",
            
            "# Error handling\ntry:\n    number = int(input('Enter a number: '))\n    result = 10 / number\n    print(f'Result: {result}')\nexcept ValueError:\n    print('Please enter a valid number')\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')",
        ]
        
        # Combine in 7:3 ratio (natural language : code)
        training_data = []
        
        # Add natural language examples
        training_data.extend(natural_language)
        
        # Add code examples  
        training_data.extend(code_examples)
        
        # Add conversation pairs
        conversations = [
            "Human: What is Python?\nAssistant: Python is a high-level, interpreted programming language known for its clean syntax and readability. It's widely used in web development, data science, artificial intelligence, and automation.",
            
            "Human: How do I create a function?\nAssistant: In Python, you create a function using the 'def' keyword:\n\ndef my_function(parameters):\n    # Your code here\n    return result\n\nFor example:\ndef greet(name):\n    return f'Hello, {name}!'",
            
            "Human: What is machine learning?\nAssistant: Machine learning is a branch of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each specific task.",
            
            "Human: Explain variables in Python\nAssistant: Variables in Python are names that store values. You create them by assignment:\n\nname = 'Alice'  # String variable\nage = 25        # Integer variable\nheight = 5.6    # Float variable\n\nPython automatically determines the data type.",
        ]
        
        training_data.extend(conversations)
        
        print(f"Created {len(training_data)} TinyLlama-inspired training examples")
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
        
        return input_ids, target_ids

class TinyLlamaInspiredTrainer:
    """Trainer incorporating TinyLlama's training techniques"""
    
    def __init__(self, model_save_path: str = "illuminator_tinyllama_inspired.pth"):
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("📚 Loading tokenizer...")
        self.tokenizer = iLLuMinatorTokenizer()
        
        print("🧠 Creating TinyLlama-inspired model...")
        self.model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        self.model.to(self.device)
        
        # TinyLlama-inspired hyperparameters
        self.lr = 4e-4  # TinyLlama's learning rate
        self.weight_decay = 0.1  # Strong regularization like TinyLlama
        self.beta1 = 0.9
        self.beta2 = 0.95  # TinyLlama's beta2
        
        # Initialize optimizer (TinyLlama style)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=1e-8
        )
        
        # Loss function with label smoothing (like TinyLlama)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        
        print(f"✅ TinyLlama-inspired trainer ready")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔧 Device: {self.device}")
        print(f"⚡ Learning rate: {self.lr}")
    
    def get_cosine_schedule_with_warmup(self, num_training_steps: int, num_warmup_steps: int = 2000):
        """TinyLlama's cosine learning rate schedule with warmup"""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine annealing
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)
    
    def train(self, epochs: int = 8, batch_size: int = 2):
        """Train with TinyLlama-inspired techniques"""
        
        print(f"🚀 Starting TinyLlama-inspired training for {epochs} epochs...")
        
        # Create dataset
        dataset = TinyLlamaInspiredDataset(self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup learning rate schedule
        total_steps = len(dataloader) * epochs
        scheduler = self.get_cosine_schedule_with_warmup(total_steps, num_warmup_steps=2000)
        
        print(f"📊 Dataset size: {len(dataset)} examples")
        print(f"📈 Total training steps: {total_steps}")
        
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            print(f"\n📖 Epoch {epoch + 1}/{epochs}")
            
            pbar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, (input_ids, target_ids) in enumerate(pbar):
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
                
                # TinyLlama-style gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                scheduler.step()  # Update learning rate
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{current_lr:.2e}'
                })
            
            avg_loss = total_loss / num_batches
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"  📈 Average Loss: {avg_loss:.4f}")
            print(f"  ⚡ Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(f"{self.model_save_path.replace('.pth', '_best.pth')}")
                print(f"  ⭐ Best model saved! Loss: {best_loss:.4f}")
            
            # Test generation every 2 epochs
            if epoch % 2 == 0:
                self._test_generation()
        
        # Save final model
        self.save_model()
        print("✅ TinyLlama-inspired training completed!")
    
    def _test_generation(self):
        """Test generation during training"""
        self.model.eval()
        
        test_prompts = [
            "Human: Hello\nAssistant:",
            "Human: What is Python?\nAssistant:",
        ]
        
        print(f"\n  🤖 Generation Test:")
        
        for prompt in test_prompts[:1]:  # Test one prompt
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_length=len(input_ids) + 25,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            print(f"    {response}")
        
        self.model.train()
    
    def save_model(self, save_path: str = None):
        """Save model checkpoint"""
        if save_path is None:
            save_path = self.model_save_path
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': len(self.tokenizer),
            'model_config': {
                'vocab_size': len(self.tokenizer),
                'architecture': 'TinyLlama-inspired'
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"💾 Model saved to {save_path}")

def main():
    """Run TinyLlama-inspired training"""
    print("🎯 iLLuMinator TinyLlama-Inspired Training")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    
    trainer = TinyLlamaInspiredTrainer()
    
    try:
        # Train with TinyLlama techniques
        trainer.train(epochs=10, batch_size=2)
        
        print(f"\n🎉 TinyLlama-inspired training completed!")
        print(f"📁 Best model: illuminator_tinyllama_inspired_best.pth")
        print(f"📁 Final model: illuminator_tinyllama_inspired.pth")
        print(f"🧪 Test with: python simple_test.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
