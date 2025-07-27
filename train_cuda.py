#!/usr/bin/env python3
"""
CUDA-Optimized Training for iLLuMinator 4.9B
High-performance training with mixed precision, gradient checkpointing, and RTX 3070 optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
import os
import time
from typing import Dict, Any, Optional
import math
from datetime import datetime

from illuminator_cuda import iLLuMinatorCUDA
from tokenizer import iLLuMinatorTokenizer

class CUDATextDataset(Dataset):
    """CUDA-optimized dataset for text training"""
    
    def __init__(self, tokenizer, texts: list, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"🔨 Processing {len(texts)} training examples...")
        
        # Tokenize all texts
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            
            # Split long texts into chunks
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= 32:  # Minimum viable length
                    self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Pad to max length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, labels

class CUDATrainer:
    """CUDA-optimized trainer for iLLuMinator 4.9B"""
    
    def __init__(
        self,
        model_save_path: str = "illuminator_cuda_weights.pth",
        checkpoint_dir: str = "checkpoints",
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 4
    ):
        self.model_save_path = model_save_path
        self.checkpoint_dir = checkpoint_dir
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("Initializing CUDA Trainer...")
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = iLLuMinatorTokenizer()
        
        # Initialize model
        print("Creating CUDA-optimized model...")
        self.model = iLLuMinatorCUDA(vocab_size=len(self.tokenizer))
        
        # Move to GPU and optimize
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            print(f"Model on GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU")
        
        # Initialize optimizer with proper settings for large models
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,  # Lower learning rate for large model
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            print("Mixed precision training enabled")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tokenizer.pad_token_id)
        
        print(f"Trainer Configuration:")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   Device: {self.device}")
    
    def create_training_data(self) -> list:
        """Create comprehensive training data"""
        training_texts = [
            # Conversational data
            "Human: Hello, how are you?\nAssistant: Hello! I'm iLLuMinator, an AI assistant. I'm doing well and ready to help you with any questions or tasks.",
            "Human: What is artificial intelligence?\nAssistant: Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving.",
            "Human: How do I write a Python function?\nAssistant: To write a Python function, use the 'def' keyword followed by the function name and parameters. Here's an example:\n\ndef greet(name):\n    return f'Hello, {name}!'\n\nresult = greet('World')\nprint(result)  # Output: Hello, World!",
            
            # Programming content
            "Python is a versatile programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            "Deep learning uses neural networks with multiple layers to process data and make predictions. It's particularly effective for tasks like image recognition and natural language processing.",
            
            # Educational content
            "The transformer architecture revolutionized natural language processing by introducing the attention mechanism, allowing models to focus on relevant parts of the input sequence.",
            "CUDA (Compute Unified Device Architecture) is a parallel computing platform that allows developers to use GPUs for general-purpose computing, significantly accelerating computation-intensive tasks.",
            "Neural networks are inspired by biological neurons and consist of interconnected nodes that process information through weighted connections and activation functions.",
            
            # Technical explanations
            "Large Language Models (LLMs) are trained on vast amounts of text data to understand and generate human-like text. They use transformer architectures with billions of parameters.",
            "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning by iteratively adjusting model parameters in the direction of steepest descent.",
            "Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to each parameter and updating them accordingly.",
            
            # Q&A format
            "Q: What is the difference between supervised and unsupervised learning?\nA: Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data without predefined correct answers.",
            "Q: How does attention work in transformers?\nA: Attention allows the model to focus on different parts of the input sequence when processing each token, enabling better understanding of context and relationships.",
            "Q: What are the benefits of using GPUs for training?\nA: GPUs provide massive parallel processing capabilities, significantly faster matrix operations, and optimized memory bandwidth for deep learning computations.",
            
            # Code examples
            "# Example of a simple neural network in PyTorch\nimport torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 1)\n    \n    def forward(self, x):\n        return self.linear(x)",
            
            # More conversational data
            "Human: Can you explain what makes a good AI assistant?\nAssistant: A good AI assistant should be helpful, accurate, honest about limitations, respectful, and able to understand context. It should provide clear explanations and be willing to say when it doesn't know something.",
            "Human: What are the ethical considerations in AI development?\nAssistant: AI ethics involves ensuring fairness, transparency, privacy protection, avoiding bias, maintaining human oversight, and considering the societal impact of AI systems.",
            
            # Technical details
            "The attention mechanism computes a weighted sum of values based on the similarity between queries and keys, allowing the model to focus on relevant information dynamically.",
            "Mixed precision training uses both 16-bit and 32-bit floating point representations to speed up training while maintaining numerical stability.",
            "Gradient checkpointing trades computation for memory by recomputing intermediate activations during backpropagation instead of storing them."
        ]
        
        # Extend with variations and additional content
        extended_texts = []
        for text in training_texts:
            extended_texts.append(text)
            
            # Add variations for robustness
            if "Human:" in text and "Assistant:" in text:
                # Create variations with different prompts
                varied = text.replace("Human:", "User:").replace("Assistant:", "AI:")
                extended_texts.append(varied)
        
        return extended_texts
    
    def train(
        self,
        epochs: int = 3,
        batch_size: int = 1,  # Small batch size for large model
        save_every: int = 1000,
        eval_every: int = 500
    ):
        """Train the model with CUDA optimizations"""
        
        print(f"Starting CUDA training for {epochs} epochs...")
        print(f"Training Configuration:")
        print(f"   Batch size: {batch_size}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"   Effective batch size: {batch_size * self.gradient_accumulation_steps}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        
        # Create dataset
        training_texts = self.create_training_data()
        dataset = CUDATextDataset(self.tokenizer, training_texts, max_length=512)  # Shorter for memory
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (input_ids, labels) in enumerate(dataloader):
                # Move to device
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model(input_ids)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        loss = loss / self.gradient_accumulation_steps
                else:
                    logits = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_mixed_precision:
                        # Gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1
                
                # Logging
                if batch_idx % 10 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"  Step {global_step}: Loss = {loss.item() * self.gradient_accumulation_steps:.4f}, LR = {current_lr:.2e}")
                    
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1e9
                        memory_cached = torch.cuda.memory_reserved() / 1e9
                        print(f"    GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
                
                # Evaluation
                if global_step % eval_every == 0 and global_step > 0:
                    self._evaluate()
                
                # Checkpointing
                if global_step % save_every == 0 and global_step > 0:
                    self._save_checkpoint(global_step, total_loss / num_batches)
                
                # Clear cache periodically
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch + 1} completed: Average Loss = {avg_epoch_loss:.4f}")
            
            # Save after each epoch
            self._save_checkpoint(global_step, avg_epoch_loss, is_epoch_end=True)
        
        # Final save
        self.save_model()
        print(f"Training completed! Model saved to {self.model_save_path}")
        
        # Memory summary
        if torch.cuda.is_available():
            print(f"\nFinal GPU Memory Stats:")
            print(f"   Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
            print(f"   Peak cached: {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")
    
    def _evaluate(self):
        """Quick evaluation during training"""
        self.model.eval()
        
        test_prompt = "Human: What is machine learning?\nAssistant:"
        input_ids = self.tokenizer.encode(test_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            if self.use_mixed_precision:
                with autocast():
                    generated = self.model.generate(
                        input_tensor,
                        max_length=min(len(input_ids) + 30, 200),
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.tokenizer.pad_token_id
                    )
            else:
                generated = self.model.generate(
                    input_tensor,
                    max_length=min(len(input_ids) + 30, 200),
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.tokenizer.pad_token_id
                )
        
        response = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print(f"  Eval Sample: {response}")
        
        self.model.train()
    
    def _save_checkpoint(self, step: int, loss: float, is_epoch_end: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'vocab_size': len(self.tokenizer),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if is_epoch_end:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{step}.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pth")
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_model(self):
        """Save final trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': len(self.tokenizer),
            'config': {
                'vocab_size': len(self.tokenizer),
                'd_model': self.model.d_model,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'max_seq_length': self.model.max_seq_length
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, self.model_save_path)
        print(f"Final model saved to {self.model_save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Checkpoint loaded from step {checkpoint.get('step', 'unknown')}")
        return checkpoint.get('step', 0)

def main():
    """Run CUDA-optimized training"""
    print("iLLuMinator CUDA Training")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("   CUDA not available! This training script requires a CUDA-capable GPU.")
        print("   Please run on a system with NVIDIA GPU and CUDA installed.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        trainer = CUDATrainer(
            use_mixed_precision=True,
            gradient_accumulation_steps=8  # Increase for RTX 3070
        )
        
        print(f"\nStarting training optimized for RTX 3070...")
        trainer.train(
            epochs=5,
            batch_size=1,  # Start with batch size 1 for safety
            save_every=500,
            eval_every=250
        )
        
        print(f"\nCUDA training completed successfully!")
        print(f"Model weights: illuminator_cuda_weights.pth")
        print(f"Checkpoints: checkpoints/")

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error: {e}")
        print("Try reducing batch size or sequence length")
        print("Enable gradient checkpointing")
        print("Use gradient accumulation for effective larger batches")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
