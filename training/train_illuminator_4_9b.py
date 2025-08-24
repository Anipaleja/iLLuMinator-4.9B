#!/usr/bin/env python3
"""
Enhanced iLLuMinator 4.9B Training Script for RunPod/Cloud Training
Optimized for CUDA GPUs with distributed training support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import json
import time
import math
import torch.nn.functional as F
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import psutil
import gc
import wandb
from pathlib import Path
import logging

# Local imports
try:
    from enhanced_illuminator_4_9b import iLLuMinator4_9B, SimpleTokenizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure enhanced_illuminator_4_9b.py is in the training directory.")
    sys.exit(1)

class TrainingConfig:
    """Training configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.model_config = {
            "vocab_size": 65536,
            "d_model": 3328,
            "n_layers": 32,
            "n_heads": 32,
            "n_kv_heads": 8,
            "d_ff": 9984,
            "max_seq_length": 2048,  # Reduced for memory efficiency
            "dropout": 0.0,
            "tie_embeddings": True
        }
        
        self.training_config = {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "max_grad_norm": 1.0,
            "warmup_steps": 2000,
            "max_steps": 100000,
            "eval_interval": 1000,
            "save_interval": 5000,
            "log_interval": 100,
            "use_mixed_precision": True,
            "compile_model": True
        }
        
        self.data_config = {
            "dataset_size": "large",
            "max_length": 2048,
            "num_workers": 4,
            "pin_memory": True
        }
        
        self.system_config = {
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
            "use_wandb": True,
            "project_name": "illuminator-4.9b",
            "run_name": None,
            "resume_from_checkpoint": None
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for section in ['model_config', 'training_config', 'data_config', 'system_config']:
            if section in config:
                getattr(self, section).update(config[section])
    
    def save_config(self, save_path: str):
        """Save current configuration to JSON file"""
        config = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'data_config': self.data_config,
            'system_config': self.system_config
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)

class AdvancedTextDataset(Dataset):
    """Advanced dataset with better text processing"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, 
                 max_length: int = 2048, cache_dir: str = "./cache"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check for cached tokenized data
        cache_file = os.path.join(cache_dir, f"tokenized_data_{len(texts)}_{max_length}.pt")
        
        if os.path.exists(cache_file):
            print(f"Loading cached tokenized data from {cache_file}")
            self.tokenized_data = torch.load(cache_file)
        else:
            print("Tokenizing dataset (this may take a while)...")
            self.tokenized_data = self._tokenize_texts()
            torch.save(self.tokenized_data, cache_file)
            print(f"Cached tokenized data saved to {cache_file}")
    
    def _tokenize_texts(self) -> List[torch.Tensor]:
        """Tokenize all texts with progress bar"""
        tokenized_data = []
        
        for text in tqdm(self.texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text, max_length=self.max_length)
            
            # Pad or truncate
            if len(tokens) < self.max_length:
                tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
            else:
                tokens = tokens[:self.max_length]
            
            tokenized_data.append(torch.tensor(tokens, dtype=torch.long))
        
        return tokenized_data
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx]
        
        # Create input and target sequences
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Apply random token masking
        mask_prob = 0.15
        mask_token_id = self.tokenizer.pad_token_id  # 
        random_mask = torch.rand(input_ids.shape) < mask_prob
        input_ids[random_mask] = mask_token_id
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }

class EnhancedTrainer:
    """Enhanced trainer with modern features"""
    
    def __init__(self, config: TrainingConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.checkpoint_path = "/workspace/best_model.pt"
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup logging
        if self.is_main_process:
            self._setup_logging()
        
        # Initialize model
        self.model = self._setup_model()
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(config.model_config["vocab_size"])
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision
        if config.training_config["use_mixed_precision"]:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create directories
        if self.is_main_process:
            os.makedirs(config.system_config["output_dir"], exist_ok=True)
            os.makedirs(config.system_config["checkpoint_dir"], exist_ok=True)
            os.makedirs(config.system_config["log_dir"], exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.rank}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
            print("⚠️  CUDA not available, using CPU (training will be very slow)")
        
        return device
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        log_dir = self.config.system_config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
    
        log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
        # Reset existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
        # Add handlers
        logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    
        print(f"[INFO] Logs are being saved to: {log_file}")
    
        if self.config.system_config["use_wandb"]:
            wandb.init(
                project=self.config.system_config["project_name"],
                name=self.config.system_config["run_name"],
                config={
                    **self.config.model_config,
                    **self.config.training_config,
                    **self.config.data_config
                },
                dir=log_dir
            )
    
    def _setup_model(self) -> nn.Module:
        """Setup and initialize model"""
        model = iLLuMinator4_9B(**self.config.model_config)
        model = model.to(self.device)

        # Compile model for better performance (PyTorch 2.0+)
        if self.config.training_config["compile_model"] and hasattr(torch, 'compile'):
            model = torch.compile(model)
                
        # Setup distributed training
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank])
        
        return model
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with parameter groups"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embedding']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_params = [
            {'params': decay_params, 'weight_decay': self.config.training_config["weight_decay"]},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        try:
            return optim.AdamW(
                optimizer_params,
                lr=self.config.training_config["learning_rate"],
                betas=(self.config.training_config["beta1"], self.config.training_config["beta2"]),
                eps=self.config.training_config["eps"],
                fused=True
            )
        # Fallback in case fused if pytorch does not support fused
        except TypeError:
            return optim.AdamW(
            optimizer_params,
            lr=self.config.training_config["learning_rate"],
            betas=(self.config.training_config["beta1"], self.config.training_config["beta2"]),
            eps=self.config.training_config["eps"]
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        def lr_lambda(step):
            if step < self.config.training_config["warmup_steps"]:
                return step / self.config.training_config["warmup_steps"]
            else:
                progress = (step - self.config.training_config["warmup_steps"]) / (
                    self.config.training_config["max_steps"] - self.config.training_config["warmup_steps"]
                )
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
        
        # Forward pass with mixed precision
        if self.scaler:
            with torch.amp.autocast('cuda'):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1), 
                    ignore_index=self.tokenizer.pad_token_id,
                    label_smoothing=0.1
                )
                loss = loss / self.config.training_config["gradient_accumulation_steps"]
        else:
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=self.tokenizer.pad_token_id,
                label_smoothing=0.1
            )
            loss = loss / self.config.training_config["gradient_accumulation_steps"]
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Adding small gradient noise to improve generalization
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * 0.01)
        
        return loss.item() * self.config.training_config["gradient_accumulation_steps"]
    
    def train(self, train_dataloader: DataLoader):
        """Main training loop"""
        if self.is_main_process:
            print("🚀 Starting Enhanced iLLuMinator 4.9B Training")
            print(f"   Device: {self.device}")
            print(f"   World Size: {self.world_size}")
            print(f"   Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Max Steps: {self.config.training_config['max_steps']:,}")
        
        self.model.train()
        accumulated_loss = 0.0
        
        while self.global_step < self.config.training_config["max_steps"]:
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(self.epoch)
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.training_config["gradient_accumulation_steps"] == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.training_config["max_grad_norm"]
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.training_config["max_grad_norm"]
                        )
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.training_config["log_interval"] == 0:
                        avg_loss = accumulated_loss / self.config.training_config["log_interval"]
                        
                        if self.is_main_process:
                            lr = self.scheduler.get_last_lr()[0]
                            
                            print(f"Step {self.global_step:6d} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                            
                            if self.config.system_config["use_wandb"]:
                                wandb.log({
                                    "train/loss": avg_loss,
                                    "train/learning_rate": lr,
                                    "train/step": self.global_step
                                })
                        
                        accumulated_loss = 0.0

                        # Stopping early to prevent overfitting if needed
                        if avg_loss < self.best_loss:
                            self.best_loss = avg_loss
                            self.patience_counter = 0

                        # Save best model
                        torch.save({
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "step": self.global_step,
                            "loss": avg_loss,
                        }, "best_model.pt")
                    
                        if self.is_main_process:
                            print(f"💾 Saved new best model with loss {avg_loss:.4f}")
                        else:
                            self.patience_counter += 1
                            # Stop after 10 log intervals without improvement
                            if self.patience_counter >= 10: 
                                if self.is_main_process:
                                    print("⏹️ Early stopping triggered")
                                    print("🔄 Restoring best model from workspace/best_model.pt")

                                # Restore best model before stopping
                                checkpoint = torch.load("/workspace/best_model.pt", map_location="cuda")
                                self.model.load_state_dict(checkpoint["model_state_dict"])
                                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
                                return
                    
                    # Save checkpoint
                    if self.global_step % self.config.training_config["save_interval"] == 0:
                        if self.is_main_process:
                            self.save_checkpoint(avg_loss if 'avg_loss' in locals() else loss)
                    
                    # Check if training is complete
                    if self.global_step >= self.config.training_config["max_steps"]:
                        break
            
            self.epoch += 1
        
        if self.is_main_process:
            print("✅ Training completed!")
            self.save_final_model()
    
    def save_checkpoint(self, loss: float):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config.system_config["checkpoint_dir"],
            f"checkpoint_step_{self.global_step}.pt"
        )
        
        # Get model state dict (handle DDP)
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'loss': loss,
            'config': self.config.__dict__
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Keep only last 3 checkpoints
        self._cleanup_checkpoints()
    
    def save_final_model(self):
        """Save final trained model"""
        final_model_path = os.path.join(
            self.config.system_config["output_dir"],
            "illuminator_4_9b_final.pt"
        )
        
        # Get model state dict (handle DDP)
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
            model_config = self.model.module.get_config()
        else:
            model_state_dict = self.model.state_dict()
            model_config = self.model.get_config()
        
        final_model = {
            'model_state_dict': model_state_dict,
            'model_config': model_config,
            'training_config': self.config.training_config,
            'global_step': self.global_step,
            'epoch': self.epoch
        }
        
        torch.save(final_model, final_model_path)
        print(f"🎉 Final model saved: {final_model_path}")
    
    def _cleanup_checkpoints(self):
        """Keep only the most recent checkpoints"""
        checkpoint_dir = self.config.system_config["checkpoint_dir"]
        checkpoints = sorted([
            f for f in os.listdir(checkpoint_dir) 
            if f.startswith("checkpoint_step_") and f.endswith(".pt")
        ])
        
        # Remove older checkpoints, keep last 3
        for checkpoint in checkpoints[:-3]:
            os.remove(os.path.join(checkpoint_dir, checkpoint))

def prepare_comprehensive_dataset(size: str = "large") -> List[str]:
    """Prepare comprehensive training dataset"""
    
    base_texts = [
        # AI and Machine Learning
        "Artificial intelligence has revolutionized numerous industries by enabling machines to perform tasks that traditionally required human intelligence, including natural language processing, computer vision, and decision making.",
        "Deep learning models with transformer architectures have achieved remarkable success in language understanding tasks, utilizing self-attention mechanisms to capture long-range dependencies in sequential data.",
        "The training of large language models requires massive computational resources and carefully curated datasets, often involving billions of parameters and terabytes of text data from diverse sources.",
        "Neural networks learn hierarchical representations of data through multiple layers of interconnected neurons, each layer extracting increasingly complex features from the input.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence when making predictions, significantly improving performance on sequence-to-sequence tasks.",
        
        # Technology and Science
        "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to process information in ways that classical computers cannot achieve.",
        "Blockchain technology provides a decentralized and immutable ledger system that enables secure transactions without the need for traditional intermediaries.",
        "Cloud computing has transformed how organizations deploy and scale their applications, offering on-demand access to computing resources and services.",
        "Renewable energy sources such as solar and wind power are becoming increasingly cost-effective alternatives to traditional fossil fuels for electricity generation.",
        "Biotechnology advances are enabling new treatments for diseases through gene therapy, personalized medicine, and precision drug development techniques.",
        
        # Literature and Philosophy
        "The exploration of human consciousness and the nature of reality has been a central theme in philosophy for millennia, inspiring countless debates and theories.",
        "Literature serves as a mirror to society, reflecting cultural values, social issues, and the human condition through storytelling and artistic expression.",
        "The concept of free will versus determinism continues to challenge philosophers and scientists as they seek to understand human behavior and choice.",
        "Educational systems worldwide are adapting to incorporate digital technologies and new pedagogical approaches to better serve diverse learning needs.",
        "The relationship between language and thought has fascinated linguists and cognitive scientists, leading to insights about how humans process and understand information.",
        
        # History and Culture
        "Historical civilizations have developed unique systems of governance, art, and knowledge that continue to influence modern society and cultural practices.",
        "The industrial revolution marked a turning point in human history, fundamentally changing how goods are produced and how people live and work.",
        "Cultural exchange through trade, migration, and communication has led to the sharing of ideas, technologies, and artistic traditions across different societies.",
        "Archaeological discoveries continue to reveal new insights about ancient civilizations and their contributions to human knowledge and development.",
        "The preservation of cultural heritage is essential for maintaining the diversity of human experience and passing knowledge to future generations.",
        
        # Science and Discovery
        "Scientific research follows rigorous methodologies to ensure that discoveries are reproducible and contribute to our understanding of the natural world.",
        "Space exploration has expanded our knowledge of the universe and our place within it, leading to technological innovations that benefit life on Earth.",
        "Medical research continues to advance our understanding of human biology and disease, leading to new treatments and improved quality of life.",
        "Environmental science plays a crucial role in understanding and addressing climate change and its impacts on ecosystems and human societies.",
        "The development of new materials with unique properties is enabling innovations in electronics, energy storage, and construction technologies."
    ]
    
    # Expand dataset based on size
    multipliers = {
        "small": 20,     # 500 samples
        "medium": 50,    # 1,250 samples
        "large": 100,    # 2,500 samples
        "xlarge": 200    # 5,000 samples
    }
    
    multiplier = multipliers.get(size, 100)
    return base_texts * multiplier

def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_worker(rank: int, world_size: int, config: TrainingConfig):
    """Worker function for distributed training"""
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    try:
        # Initialize trainer
        trainer = EnhancedTrainer(config, rank, world_size)
        
        # Prepare dataset
        if rank == 0:
            print("📚 Preparing training dataset...")
        
        texts = prepare_comprehensive_dataset(config.data_config["dataset_size"])
        dataset = AdvancedTextDataset(
            texts, 
            trainer.tokenizer, 
            max_length=config.data_config["max_length"]
        )
        
        # Setup data loader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.training_config["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=config.data_config["num_workers"],
            pin_memory=config.data_config["pin_memory"],
            drop_last=True
        )
        
        # Start training
        trainer.train(dataloader)
        
    finally:
        if world_size > 1:
            cleanup_distributed()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Enhanced iLLuMinator 4.9B Training")
    parser.add_argument("--config", type=str, default="training_config.json", help="Training configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainingConfig(args.config if os.path.exists(args.config) else None)
    
    # Override with command line arguments
    if args.output_dir:
        config.system_config["output_dir"] = args.output_dir
        config.system_config["log_dir"] = os.path.join(args.output_dir, "logs")
        config.system_config["checkpoint_dir"] = os.path.join(args.output_dir, "checkpoints")
    if args.resume:
        config.system_config["resume_from_checkpoint"] = args.resume
    
    # Save configuration
    config.save_config(os.path.join(config.system_config["output_dir"], "training_config.json"))
    
    # Check for distributed training
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    print(f"🚀 Starting Enhanced iLLuMinator 4.9B Training")
    print(f"   Available GPUs: {world_size}")
    print(f"   Output Directory: {config.system_config['output_dir']}")
    
    if world_size > 1:
        print(f"   Using distributed training with {world_size} GPUs")
        mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        print(f"   Using single GPU/CPU training")
        train_worker(0, 1, config)

if __name__ == "__main__":
    main()
