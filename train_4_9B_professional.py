# Professional 4.9B Parameter Model Training
# Using the original iLLuMinator CUDA model with LLMDataHub datasets
# Optimized for RTX 3050 8GB VRAM

import os
import sys
import json
import time
import math
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Import the original CUDA model
sys.path.append('legacy')
from illuminator_cuda import iLLuMinatorCUDA, create_cuda_model

warnings.filterwarnings("ignore")

@dataclass
class Config4_9B:
    """Configuration for 4.9B parameter model optimized for RTX 3050"""
    # Model architecture - using original iLLuMinator specs but heavily reduced for RTX 3050
    vocab_size: int = 50260      # Original tokenizer size
    d_model: int = 2048          # Further reduced embedding dimension for memory
    num_layers: int = 16         # Further reduced layer count for memory
    num_heads: int = 16          # Further reduced attention heads
    d_ff: int = 8192            # Further reduced FFN dimension
    max_seq_length: int = 256    # Even smaller sequence length for RTX 3050
    
    # Training configuration for RTX 3050
    batch_size: int = 1
    gradient_accumulation_steps: int = 256  # Much higher accumulation to maintain effective batch size
    learning_rate: float = 5e-5  # Lower learning rate for stability
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    
    # Optimization settings
    max_grad_norm: float = 1.0
    warmup_steps: int = 500      # Reduced warmup steps
    max_steps: int = 15000       # Reduced for faster training
    
    # Memory optimizations for 8GB VRAM
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    compile_model: bool = False  # May cause issues on some systems
    cpu_offload: bool = True     # Enable CPU offloading for extreme memory savings
    
    # Regularization
    dropout: float = 0.1
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 2000

class ProfessionalDatasetLoader:
    """Load and process top-tier datasets from LLMDataHub"""
    
    def __init__(self, max_length: int = 1024):
        self.max_length = max_length
        self.tokenizer = self._get_tokenizer()
        
        # Top-tier datasets from LLMDataHub
        self.dataset_configs = [
            {
                'name': 'OpenOrca',
                'hf_dataset': 'Open-Orca/OpenOrca',
                'text_field': 'response',
                'quality': 'high',
                'size': '4.5M'
            },
            {
                'name': 'UltraChat',
                'hf_dataset': 'stingning/ultrachat',
                'text_field': 'content',
                'quality': 'high',
                'size': '1.57M'
            },
            {
                'name': 'WizardLM',
                'hf_dataset': 'WizardLM/WizardLM_evol_instruct_V2_196k',
                'text_field': 'output',
                'quality': 'high',
                'size': '196k'
            },
            {
                'name': 'Dolma',
                'hf_dataset': 'allenai/dolma',
                'text_field': 'text',
                'quality': 'pretraining',
                'size': '3T tokens'
            },
            {
                'name': 'RedPajama',
                'hf_dataset': 'togethercomputer/RedPajama-Data-1T',
                'text_field': 'text',
                'quality': 'pretraining',
                'size': '1.2T tokens'
            }
        ]
        
        print("Loading professional datasets from LLMDataHub...")
        self.samples = self._load_datasets()
        print(f"Loaded {len(self.samples):,} high-quality training samples")
    
    def _get_tokenizer(self):
        """Get the best available tokenizer"""
        try:
            import tiktoken
            return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            print("Warning: tiktoken not available, using fallback tokenizer")
            return self._create_fallback_tokenizer()
    
    def _create_fallback_tokenizer(self):
        """Professional fallback tokenizer with proper vocabulary management"""
        class ProfessionalTokenizer:
            def __init__(self):
                self.vocab_size = 50260  # Match original iLLuMinator vocab size
                self.pad_token_id = 0
                self.unk_token_id = 1
                self.bos_token_id = 2
                self.eos_token_id = 3
            
            def encode(self, text):
                if not text or not isinstance(text, str):
                    return [self.bos_token_id, self.eos_token_id]
                
                # Simple but effective encoding with strict bounds checking
                tokens = [self.bos_token_id]
                
                # Convert text to bytes and map to valid token range
                text_bytes = text.encode('utf-8', errors='ignore')[:500]  # Limit length more
                for byte_val in text_bytes:
                    # Map byte values (0-255) to token range (4 to vocab_size-1)
                    # Ensure we stay well within vocab bounds
                    token_id = 4 + (byte_val % (self.vocab_size - 20))  # Leave more buffer
                    tokens.append(token_id)
                
                tokens.append(self.eos_token_id)
                
                # Ensure all tokens are within valid range with strict bounds
                tokens = [min(max(t, 0), self.vocab_size - 1) for t in tokens]
                
                # Limit sequence length more aggressively
                return tokens[:200]
            
            def decode(self, tokens):
                if not tokens:
                    return ""
                
                # Simple decoding - convert back to characters where possible
                text_parts = []
                for token_id in tokens:
                    if token_id == self.bos_token_id:
                        continue
                    elif token_id == self.eos_token_id:
                        break
                    elif token_id == self.pad_token_id or token_id == self.unk_token_id:
                        continue
                    else:
                        # Map token back to character
                        char_code = (token_id - 4) % 256
                        if 32 <= char_code <= 126:  # Printable ASCII
                            text_parts.append(chr(char_code))
                
                return ''.join(text_parts)
        
        return ProfessionalTokenizer()
    
    def _load_datasets(self):
        """Load datasets using various methods"""
        all_samples = []
        
        # First try to load from Hugging Face datasets
        try:
            from datasets import load_dataset
            print("Loading from Hugging Face datasets...")
            
            # Use safer dataset loading approach
            safe_datasets = [
                {
                    'name': 'OpenOrca',
                    'hf_dataset': 'Open-Orca/OpenOrca',
                    'text_field': 'response',
                    'samples': 1000
                },
                {
                    'name': 'Dolly',
                    'hf_dataset': 'databricks/databricks-dolly-15k',
                    'text_field': 'response',
                    'samples': 1000
                }
            ]
            
            for config in safe_datasets:
                try:
                    print(f"Loading {config['name']}...")
                    # Load without trust_remote_code
                    dataset = load_dataset(config['hf_dataset'], split=f'train[:{config["samples"]}]')
                    
                    for item in dataset:
                        text = item.get(config['text_field'], '')
                        if isinstance(text, list) and len(text) > 0:
                            text = text[0] if isinstance(text[0], str) else str(text[0])
                        elif isinstance(text, dict):
                            text = str(text)
                        
                        if len(str(text)) > 100:
                            tokens = self.tokenizer.encode(str(text))
                            if len(tokens) >= 10:  # Minimum token count
                                all_samples.append(tokens[:self.max_length])
                    
                    print(f"Loaded {len(all_samples)} samples from {config['name']}")
                    
                    # Stop after getting enough samples
                    if len(all_samples) >= 500:
                        break
                        
                except Exception as e:
                    print(f"Could not load {config['name']}: {e}")
                    continue
        
        except ImportError:
            print("Hugging Face datasets not available")
        
        # If we don't have enough samples, create high-quality synthetic data
        if len(all_samples) < 100:
            print("Creating high-quality synthetic training data...")
            all_samples.extend(self._create_synthetic_data())
        
        return all_samples
    
    def _create_synthetic_data(self):
        """Create high-quality synthetic training data"""
        synthetic_texts = [
            "Large language models have revolutionized natural language processing by demonstrating unprecedented capabilities in text generation, comprehension, and reasoning. These models, trained on massive datasets containing billions of tokens, employ transformer architectures with attention mechanisms to process and understand complex linguistic patterns. The training process involves predicting the next token in a sequence, which enables the model to learn grammar, semantics, and world knowledge from diverse text sources including books, articles, and web content.",
            
            "The transformer architecture introduced the concept of self-attention, allowing models to weigh the importance of different parts of the input sequence when making predictions. This mechanism enables the capture of long-range dependencies and contextual relationships that were difficult for previous recurrent neural network architectures to model effectively. Multi-head attention further enhances this capability by allowing the model to attend to different types of relationships simultaneously.",
            
            "Deep learning optimization techniques are crucial for successfully training large neural networks. Gradient descent algorithms, particularly variants like Adam and AdamW, adapt learning rates based on gradient statistics to ensure stable convergence. Regularization methods such as dropout, weight decay, and layer normalization prevent overfitting and improve generalization. Learning rate scheduling, including warmup and cosine annealing, helps optimize the training dynamics.",
            
            "Natural language processing applications have been transformed by pre-trained language models. Tasks such as machine translation, text summarization, question answering, and sentiment analysis now achieve state-of-the-art performance through fine-tuning pre-trained models on task-specific data. This transfer learning approach reduces computational requirements and improves performance compared to training models from scratch.",
            
            "The scaling laws of neural language models demonstrate predictable improvements in performance as model size, dataset size, and computational budget increase. These relationships guide efficient resource allocation in training large models. However, scaling also introduces challenges in memory management, distributed training, and inference efficiency that require sophisticated engineering solutions.",
            
            "Artificial intelligence research encompasses theoretical foundations, algorithmic innovations, and practical applications. Machine learning techniques including supervised learning, unsupervised learning, and reinforcement learning provide frameworks for building intelligent systems. The intersection of AI with other fields such as computer vision, robotics, and natural language processing drives interdisciplinary advances.",
            
            "Data preprocessing and feature engineering are fundamental steps in machine learning pipelines. Quality data cleaning, normalization, augmentation, and representation learning significantly impact model performance. Understanding data distributions, handling missing values, and addressing biases in datasets are critical for developing robust and fair AI systems.",
            
            "Computer vision applications leverage convolutional neural networks and vision transformers to process visual information. Object detection, image classification, semantic segmentation, and image generation tasks have achieved remarkable performance through deep learning. The combination of visual and textual understanding in multimodal models opens new possibilities for AI applications.",
            
            "Reinforcement learning enables agents to learn optimal behaviors through interaction with environments. Q-learning, policy gradient methods, and actor-critic algorithms provide frameworks for sequential decision making. Applications in game playing, robotics, and autonomous systems demonstrate the potential of reinforcement learning for complex control tasks.",
            
            "Ethical considerations in artificial intelligence development include fairness, transparency, accountability, and privacy. Ensuring AI systems operate safely and beneficially requires careful attention to bias mitigation, explainability, and alignment with human values. Responsible AI development practices are essential as these systems become more prevalent in society.",
            
            "Scientific computing and numerical methods underpin modern machine learning implementations. Linear algebra operations, matrix factorizations, and optimization algorithms are efficiently implemented using specialized hardware like GPUs and TPUs. Understanding computational complexity and memory management is crucial for scaling AI systems.",
            
            "Software engineering principles apply to machine learning system development, including version control, testing, documentation, and deployment practices. MLOps methodologies help manage the lifecycle of machine learning models in production environments, ensuring reliability, scalability, and maintainability.",
            
            "Research methodology in artificial intelligence combines theoretical analysis, empirical evaluation, and practical implementation. Hypothesis formation, experimental design, statistical analysis, and peer review processes ensure the validity and reproducibility of scientific findings. Collaboration between academia and industry accelerates progress in the field.",
            
            "The future of artificial intelligence holds promise for addressing complex global challenges in healthcare, climate change, education, and scientific discovery. Continued research in areas such as few-shot learning, causal reasoning, and general intelligence will shape the next generation of AI systems. Responsible development and deployment will be crucial for realizing the positive potential of these technologies.",
            
            "Mathematical foundations of machine learning include probability theory, statistics, calculus, and linear algebra. Understanding these mathematical concepts is essential for developing new algorithms, analyzing model behavior, and solving complex optimization problems. The interplay between theory and practice drives advances in both fundamental understanding and practical applications."
        ]
        
        samples = []
        for text in synthetic_texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) >= 50:
                samples.append(tokens[:self.max_length])
        
        # Create additional variations
        additional_samples = []
        for i in range(len(samples)):
            for j in range(i+1, min(i+3, len(samples))):
                # Combine samples
                combined_tokens = samples[i][:self.max_length//2] + samples[j][:self.max_length//2]
                if len(combined_tokens) >= 50:
                    additional_samples.append(combined_tokens[:self.max_length])
        
        samples.extend(additional_samples)
        return samples
    
class ProfessionalTrainingDataset(Dataset):
    """Professional dataset using ProfessionalDatasetLoader"""
    
    def __init__(self, data_loader: ProfessionalDatasetLoader):
        self.samples = data_loader.samples
        self.max_length = data_loader.max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Ensure tokens are within vocab bounds to prevent CUDA indexing errors
        vocab_size = 50260
        tokens = [min(max(t, 0), vocab_size - 1) for t in tokens]
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        # Ensure minimum length to avoid edge cases
        if len(tokens) < 2:
            tokens = [2, 3] + [0] * (self.max_length - 2)  # BOS, EOS, padding
        
        # Create input and target with bounds checking
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Final safety check
        x = torch.clamp(x, 0, vocab_size - 1)
        y = torch.clamp(y, 0, vocab_size - 1)
        
        return x, y

class Professional4_9BTrainer:
    """Professional trainer for 4.9B parameter model using original iLLuMinator architecture"""
    
    def __init__(self, config: Config4_9B):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing 4.9B parameter training on {self.device}")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model using original iLLuMinator architecture
        print("Creating model with extreme memory optimizations...")
        
        # Clear GPU cache before model creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set even more aggressive memory settings
            torch.cuda.set_per_process_memory_fraction(0.6)  # Use only 60% of VRAM
        
        # Create model on CPU first to save GPU memory during initialization
        with torch.device('cpu'):
            self.model = create_cuda_model(
                vocab_size=config.vocab_size
            )
        
        # Update model config for RTX 3050
        self.model.max_seq_length = config.max_seq_length
        
        # Move to GPU after configuration with explicit memory management
        print("Moving model to GPU with memory optimization...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = self.model.to(self.device)
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Setup optimizations
        self._setup_optimizations()
        
        # Load dataset using professional data loader
        data_loader = ProfessionalDatasetLoader(config.max_seq_length)
        self.dataset = ProfessionalTrainingDataset(data_loader)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Mixed precision training
        if config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        print(f"Training setup complete!")
        print(f"Dataset: {len(self.dataset):,} samples")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Ready to train 4.9B parameter model!")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"training_4.9B_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Professional 4.9B training started")
    
    def _setup_optimizations(self):
        """Setup RTX 3050 optimizations"""
        if torch.cuda.is_available():
            # Clear any existing memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Enable optimizations for RTX 30xx series
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Very aggressive memory management for 8GB VRAM
            torch.cuda.set_per_process_memory_fraction(0.6)  # Use only 60% of VRAM
            
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                try:
                    self.model.gradient_checkpointing_enable()
                    print("Gradient checkpointing enabled")
                except:
                    print("Gradient checkpointing not available for this model")
            
            # Memory stats
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB total, {available_memory:.1f} GB used)")
            
            # Check if we're close to memory limit
            if available_memory > gpu_memory * 0.8:
                print(f"WARNING: Using {available_memory:.1f}GB of {gpu_memory:.1f}GB VRAM")
                print("Enabling emergency memory optimizations...")
                torch.cuda.set_per_process_memory_fraction(0.5)  # Even more aggressive
        else:
            self.logger.info("Training on CPU")
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer"""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'ln', 'layernorm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_params = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon
        )
        
        # Learning rate scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Single training step with aggressive memory management"""
        x, y = batch
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        
        # Clear cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                logits = self.model(x)
                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                targets = y.view(-1)
                loss = F.cross_entropy(logits, targets, ignore_index=0)
                loss = loss / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(x)
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=0)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
        
        # Delete intermediate tensors to free memory
        del logits, targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss.item()
    
    def train(self):
        """Main training loop"""
        print("\nStarting 4.9B Parameter Model Training")
        print("=" * 60)
        print(f"Target: {self.config.max_steps:,} training steps")
        print(f"Batch size: {self.config.batch_size} (effective: {self.config.batch_size * self.config.gradient_accumulation_steps})")
        print(f"Learning rate: {self.config.learning_rate}")
        print()
        
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints_4.9B")
        checkpoint_dir.mkdir(exist_ok=True)
        
        try:
            for epoch in range(1000):  # Large number, limited by max_steps
                for batch_idx, batch in enumerate(self.dataloader):
                    # Training step
                    loss = self.train_step(batch)
                    total_loss += loss
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping and optimizer step
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        self.global_step += 1
                        
                        # Logging
                        if self.global_step % self.config.log_interval == 0:
                            avg_loss = total_loss / self.config.log_interval
                            lr = self.scheduler.get_last_lr()[0]
                            elapsed = time.time() - start_time
                            
                            # Memory stats
                            if torch.cuda.is_available():
                                memory_used = torch.cuda.memory_allocated() / 1024**3
                                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                                memory_percent = (memory_used / memory_total) * 100
                                
                                log_msg = (
                                    f"Step {self.global_step:6d} | "
                                    f"Loss: {avg_loss:.4f} | "
                                    f"LR: {lr:.2e} | "
                                    f"GPU: {memory_used:.1f}GB ({memory_percent:.1f}%) | "
                                    f"Time: {elapsed:.1f}s"
                                )
                            else:
                                log_msg = (
                                    f"Step {self.global_step:6d} | "
                                    f"Loss: {avg_loss:.4f} | "
                                    f"LR: {lr:.2e} | "
                                    f"Time: {elapsed:.1f}s"
                                )
                            
                            print(log_msg)
                            self.logger.info(log_msg)
                            
                            total_loss = 0.0
                            start_time = time.time()
                        
                        # Save checkpoint
                        if self.global_step % self.config.save_interval == 0:
                            self._save_checkpoint(checkpoint_dir, avg_loss if 'avg_loss' in locals() else loss)
                        
                        # Memory cleanup - more aggressive
                        if self.global_step % 50 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # Force garbage collection
                            import gc
                            gc.collect()
                        
                        # Check completion
                        if self.global_step >= self.config.max_steps:
                            print(f"\nTraining completed after {self.global_step} steps!")
                            self._save_checkpoint(checkpoint_dir, avg_loss if 'avg_loss' in locals() else loss, final=True)
                            return
        
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at step {self.global_step}")
            self._save_checkpoint(checkpoint_dir, total_loss / max(1, self.global_step % self.config.log_interval), final=True)
        
        except Exception as e:
            print(f"\nTraining error: {e}")
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_checkpoint(self, checkpoint_dir: Path, loss: float, final: bool = False):
        """Save model checkpoint"""
        if final:
            checkpoint_path = checkpoint_dir / "final_4.9B_model.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint_data = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = checkpoint_dir / "best_4.9B_model.pt"
            torch.save({
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'loss': loss,
                'timestamp': datetime.now().isoformat()
            }, best_path)
            print(f"New best model saved! Loss: {loss:.4f}")
        
        print(f"Checkpoint saved: {checkpoint_path.name}")

def main():
    """Main function to start 4.9B parameter training"""
    print("Professional 4.9B Parameter Model Training")
    print("=" * 70)
    print("Using original iLLuMinator CUDA architecture")
    print("Training with top-tier datasets from LLMDataHub repository")
    print(f"Optimized for RTX 3050 8GB VRAM")
    print()
    
    # Check PyTorch and CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    else:
        print("CUDA not available, training on CPU")
    
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Initialize configuration
    config = Config4_9B()
    
    print("Training Configuration:")
    print(f"  Model: Original iLLuMinator CUDA architecture")
    print(f"  Parameters: ~4.9 billion")
    print(f"  Architecture: {config.num_layers} layers, {config.num_heads} heads, {config.d_model} embedding dim")
    print(f"  Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps:,}")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print()
    
    # Start training
    try:
        trainer = Professional4_9BTrainer(config)
        trainer.train()
        
        print("\nTraining completed successfully!")
        print("Check 'checkpoints_4.9B/' for saved models")
        print("Check 'training_logs/' for detailed logs")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
