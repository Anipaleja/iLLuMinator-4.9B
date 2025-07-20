"""
Training Script for Giant 4-5B Parameter Transformer
High-performance training with memory optimization and advanced techniques
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torch.cuda.amp as amp
import os
import json
import wandb
import time
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

# Import our giant model and components
from giant_transformer import GiantTransformer, GiantModelConfig
from advanced_tokenizer import AdvancedTokenizer
from knowledge_base import ComprehensiveKnowledgeBase

class TextDataset(Dataset):
    """High-performance text dataset for giant model training"""
    
    def __init__(self, 
                 data_paths: List[str], 
                 tokenizer: AdvancedTokenizer,
                 max_length: int = 4096,
                 chunk_size: int = 10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.data_paths = data_paths
        
        # Load and preprocess data
        self.examples = self._load_and_process_data()
        print(f"📚 Loaded {len(self.examples)} training examples")
    
    def _load_and_process_data(self) -> List[List[int]]:
        """Load and tokenize all data efficiently"""
        examples = []
        
        # Add knowledge base content as training data
        kb = ComprehensiveKnowledgeBase()
        for category, entries in kb.knowledge_base.items():
            for entry in entries:
                # Create training text from knowledge entries
                text = f"Category: {category}\nTopic: {entry.topic}\nContent: {entry.content}\nDetails: {' '.join(entry.details)}"
                tokens = self.tokenizer.encode(text)
                
                # Split into chunks if too long
                for i in range(0, len(tokens), self.max_length):
                    chunk = tokens[i:i + self.max_length]
                    if len(chunk) > 50:  # Minimum chunk size
                        examples.append(chunk)
        
        # Load additional text files if provided
        for data_path in self.data_paths:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Tokenize and chunk
                tokens = self.tokenizer.encode(text)
                for i in range(0, len(tokens), self.max_length):
                    chunk = tokens[i:i + self.max_length]
                    if len(chunk) > 50:
                        examples.append(chunk)
        
        # Add synthetic high-quality training data
        synthetic_examples = self._generate_synthetic_data()
        examples.extend(synthetic_examples)
        
        return examples
    
    def _generate_synthetic_data(self) -> List[List[int]]:
        """Generate high-quality synthetic training data"""
        synthetic_texts = [
            # Technical explanations
            "Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn. Machine learning is a subset of AI that focuses on algorithms that improve automatically through experience.",
            
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data. These networks can process vast amounts of information and identify intricate relationships.",
            
            "Natural language processing enables computers to understand, interpret, and generate human language. This involves tasks like text analysis, language translation, and conversational AI.",
            
            "Computer vision allows machines to interpret and understand visual information from the world. This includes image recognition, object detection, and scene understanding.",
            
            # Scientific content
            "Quantum mechanics describes the physical properties of matter and energy at the atomic and subatomic scale. It reveals the wave-particle duality of matter and the probabilistic nature of quantum systems.",
            
            "The theory of relativity, developed by Einstein, fundamentally changed our understanding of space, time, and gravity. It consists of special relativity and general relativity.",
            
            "DNA contains the genetic instructions for the development and function of all living organisms. It consists of nucleotides arranged in a double helix structure.",
            
            "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly burning fossil fuels, are the primary drivers of recent climate change.",
            
            # Technology explanations
            "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, software, and analytics. It offers flexibility and scalability.",
            
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records linked and secured using cryptography. It enables secure, transparent transactions.",
            
            "Cybersecurity protects digital systems, networks, and data from digital attacks. It involves implementing security measures and protocols to prevent unauthorized access.",
            
            "The Internet of Things connects everyday objects to the internet, enabling them to send and receive data. This creates opportunities for automation and smart systems.",
            
            # Historical content
            "The Renaissance was a period of cultural, artistic, political, and economic rebirth following the Middle Ages. It marked the transition from medieval to modern Europe.",
            
            "The Industrial Revolution transformed economies from agriculture-based to manufacturing-based. It introduced mechanization, factory systems, and mass production.",
            
            "World War II was a global conflict that lasted from 1939 to 1945. It involved most of the world's nations and was the deadliest conflict in human history.",
            
            "The Space Race was a 20th-century competition between the Soviet Union and United States to achieve superior spaceflight capability. It led to significant technological advances.",
        ]
        
        synthetic_examples = []
        for text in synthetic_texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 10:
                synthetic_examples.append(tokens)
        
        return synthetic_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with pad token
            pad_token = self.tokenizer.special_tokens['<|pad|>']
            tokens = tokens + [pad_token] * (self.max_length - len(tokens))
        
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # For causal language modeling, input and target are the same shifted by 1
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': (input_ids != self.tokenizer.special_tokens['<|pad|>']).float()
        }

class GiantModelTrainer:
    """High-performance trainer for giant transformer models"""
    
    def __init__(self, 
                 config: GiantModelConfig,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 output_dir: str = "./models/giant_model/checkpoints",
                 use_wandb: bool = True):
        
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = GiantTransformer(config)
        print(f"🚀 Giant model initialized with {self.count_parameters():,} parameters")
        
        # Setup device and distributed training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ddp = False
        
        if torch.cuda.device_count() > 1:
            print(f"🔥 Using {torch.cuda.device_count()} GPUs for training")
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.get_optimal_batch_size(),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.get_optimal_batch_size(),
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision training
        self.scaler = amp.GradScaler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Wandb logging
        if use_wandb:
            wandb.init(
                project="giant-transformer",
                config=config.__dict__,
                name=f"giant-{config.n_layer}L-{config.n_embd}D"
            )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        if hasattr(self.model, 'module'):
            return sum(p.numel() for p in self.model.module.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        if torch.cuda.is_available():
            # Conservative batch size for giant models
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 80:  # A100 80GB
                return 4
            elif gpu_memory_gb >= 40:  # A100 40GB
                return 2
            else:  # Smaller GPUs
                return 1
        else:
            return 1  # CPU fallback
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup AdamW optimizer with weight decay"""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'ln' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config.max_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass with mixed precision
            with amp.autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=self.model.config.pad_token_id if hasattr(self.model, 'config') else -100
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % 10 == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                
                print(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {lr:.2e}")
                
                if wandb.run:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/learning_rate': lr,
                        'train/epoch': self.epoch,
                        'train/global_step': self.global_step
                    })
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return {'loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                        ignore_index=self.model.config.pad_token_id if hasattr(self.model, 'config') else -100
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return {'loss': avg_loss, 'perplexity': perplexity}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint-epoch-{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model checkpoint at epoch {self.epoch}")
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def train(self, max_epochs: int = None):
        """Main training loop"""
        max_epochs = max_epochs or self.config.max_epochs
        
        print(f"🏋️ Starting training for {max_epochs} epochs")
        print(f"📊 Model parameters: {self.count_parameters():,}")
        print(f"🔢 Training batches per epoch: {len(self.train_loader)}")
        print(f"📏 Batch size: {self.train_loader.batch_size}")
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Logging
            print(f"\n📈 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"   Val Loss: {val_metrics['loss']:.4f}")
                print(f"   Val Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"   Time: {epoch_time:.2f}s")
            
            if wandb.run:
                log_dict = {
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_time': epoch_time
                }
                if val_metrics:
                    log_dict.update({
                        'val/loss': val_metrics['loss'],
                        'val/perplexity': val_metrics['perplexity']
                    })
                wandb.log(log_dict)
            
            # Save checkpoint
            is_best = val_metrics and val_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['loss']
            
            if epoch % 5 == 0 or is_best:  # Save every 5 epochs or if best
                self.save_checkpoint(is_best=is_best)
            
            print(f"{'='*60}\n")

def main():
    """Main training function"""
    # Configuration
    config = GiantModelConfig(
        vocab_size=50257,
        n_positions=4096,
        n_embd=2560,
        n_layer=32,
        n_head=32,
        learning_rate=1e-4,
        max_epochs=50,
        warmup_steps=1000,
        use_flash_attention=True
    )
    
    # Initialize tokenizer
    print("🔤 Initializing advanced tokenizer...")
    tokenizer = AdvancedTokenizer(vocab_size=config.vocab_size)
    
    # Create datasets
    print("📚 Creating datasets...")
    train_dataset = TextDataset(
        data_paths=[],  # Add your data files here
        tokenizer=tokenizer,
        max_length=config.n_positions
    )
    
    # Initialize trainer
    print("🚀 Initializing trainer...")
    trainer = GiantModelTrainer(
        config=config,
        train_dataset=train_dataset,
        output_dir="./models/giant_model/checkpoints"
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
