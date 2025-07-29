# Professional 4.9B Parameter Model Trainer for RTX 3050
# Optimized for enterprise-grade datasets and maximum performance

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
import torch.distributed as dist

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class ModelConfig:
    """Configuration for the 4.9B parameter model"""
    # Model architecture (scaled for RTX 3050)
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 2560      # Embedding dimension
    n_layer: int = 32       # Number of layers
    n_head: int = 20        # Number of attention heads
    n_inner: int = 10240    # Feed-forward dimension (4 * n_embd)
    
    # Training configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    
    # Optimization
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Memory optimization for RTX 3050
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False  # May not be available
    compile_model: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 2000

class GPTAttention(nn.Module):
    """Multi-head self-attention with optimizations for RTX 3050"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class GPTMLP(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPTBlock(nn.Module):
    """Transformer block with pre-layer normalization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPTMLP(config)
    
    def forward(self, x):
        # Pre-layer norm
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT4_9B(nn.Module):
    """4.9 Billion Parameter GPT Model optimized for RTX 3050"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language model head (tied with input embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"🎯 Model initialized with {n_params:,} parameters ({n_params/1e9:.2f}B)")
    
    def _init_weights(self, module):
        """Initialize weights using GPT-3 initialization scheme"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_positions, f"Sequence length {t} exceeds maximum {self.config.n_positions}"
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.h:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

class EnterpriseDataset(Dataset):
    """High-quality dataset loader for enterprise training"""
    
    def __init__(self, data_path: str, tokenizer_path: str = None, max_length: int = 2048):
        self.data_path = Path(data_path)
        self.max_length = max_length
        
        # Load tokenizer (using GPT-3 tiktoken if available)
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = self.tokenizer.n_vocab
            print(f"✅ Using tiktoken GPT-3 tokenizer (vocab: {self.vocab_size})")
        except ImportError:
            # Fallback to simple tokenizer
            self.tokenizer = self._create_simple_tokenizer()
            self.vocab_size = 50257
            print("⚠️ Using fallback tokenizer")
        
        # Load and preprocess data
        self.samples = self._load_and_preprocess()
        print(f"📊 Loaded {len(self.samples):,} training samples")
    
    def _create_simple_tokenizer(self):
        """Simple character-level tokenizer fallback"""
        class SimpleTokenizer:
            def encode(self, text):
                return [min(ord(c), 50256) for c in text]
            
            def decode(self, tokens):
                return ''.join([chr(min(t, 127)) for t in tokens if t > 0])
        
        return SimpleTokenizer()
    
    def _load_and_preprocess(self):
        """Load and preprocess training data"""
        samples = []
        
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        
                        if len(text) > 200:  # Filter short texts
                            # Tokenize and chunk
                            tokens = self.tokenizer.encode(text)
                            
                            # Create overlapping chunks
                            for i in range(0, len(tokens), self.max_length // 2):
                                chunk = tokens[i:i + self.max_length]
                                if len(chunk) >= 50:  # Minimum chunk size
                                    samples.append(chunk)
                        
                        if (line_num + 1) % 10000 == 0:
                            print(f"  Processed {line_num + 1:,} lines...")
                    
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input and target tensors
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

class Professional4_9BTrainer:
    """Professional trainer for 4.9B parameter model"""
    
    def __init__(self, config: ModelConfig, data_path: str):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model and move to device
        self.model = GPT4_9B(config).to(self.device)
        
        # Enable optimizations for RTX 3050
        self.setup_optimizations()
        
        # Load dataset
        self.dataset = EnterpriseDataset(data_path, max_length=config.n_positions)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid issues in Windows
            pin_memory=True,
            persistent_workers=False
        )
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Mixed precision training
        if config.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        self.logger.info(f"🚀 Professional trainer initialized for {self.device}")
        self.logger.info(f"📊 Dataset: {len(self.dataset):,} samples")
        self.logger.info(f"🔥 Ready for enterprise-grade training!")
    
    def setup_logging(self):
        """Setup professional logging"""
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"training_4.9B_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizations(self):
        """Setup RTX 3050 specific optimizations"""
        if torch.cuda.is_available():
            # Enable TensorFloat-32 for RTX 30xx series
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Memory management for 8GB VRAM
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            # Compile model for better performance (PyTorch 2.0+)
            if self.config.compile_model and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    self.logger.info("✅ Model compiled for optimal performance")
                except Exception as e:
                    self.logger.warning(f"⚠️ Model compilation failed: {e}")
            
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"🖥️ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    def setup_optimizer(self):
        """Setup AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
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
        
        # Learning rate scheduler (cosine with warmup)
        self.scheduler = self.get_lr_scheduler()
    
    def get_lr_scheduler(self):
        """Get learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                # Cosine annealing
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Single training step with mixed precision"""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_mixed_precision):
            logits, loss = self.model(x, y)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item()
    
    def train(self):
        """Main training loop"""
        self.logger.info("🎯 Starting professional 4.9B parameter training...")
        
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints_4.9B")
        checkpoint_dir.mkdir(exist_ok=True)
        
        for epoch in range(1000):  # Large number, will be limited by max_steps
            for batch_idx, batch in enumerate(self.dataloader):
                # Training step
                loss = self.train_step(batch)
                total_loss += loss
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = total_loss / self.config.log_interval
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        
                        # GPU memory usage
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            memory_percent = memory_used / (torch.cuda.get_device_properties(0).total_memory / 1024**3) * 100
                        else:
                            memory_used = 0
                            memory_percent = 0
                        
                        self.logger.info(
                            f"Step {self.global_step:6d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"GPU: {memory_used:.1f}GB ({memory_percent:.1f}%) | "
                            f"Time: {elapsed:.1f}s"
                        )
                        
                        total_loss = 0.0
                        start_time = time.time()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(checkpoint_dir, avg_loss if 'avg_loss' in locals() else loss)
                    
                    # Memory cleanup for RTX 3050
                    if self.global_step % 100 == 0:
                        torch.cuda.empty_cache()
                    
                    # Check if training is complete
                    if self.global_step >= self.config.max_steps:
                        self.logger.info(f"✅ Training completed after {self.global_step} steps!")
                        self.save_checkpoint(checkpoint_dir, avg_loss if 'avg_loss' in locals() else loss, final=True)
                        return
    
    def save_checkpoint(self, checkpoint_dir: Path, loss: float, final: bool = False):
        """Save model checkpoint"""
        if final:
            checkpoint_path = checkpoint_dir / "final_model_4.9B.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        # Save checkpoint
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'config': self.config,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        # Update best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = checkpoint_dir / "best_model_4.9B.pt"
            torch.save({
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'loss': loss,
                'timestamp': datetime.now().isoformat()
            }, best_path)
            self.logger.info(f"💎 New best model saved! Loss: {loss:.4f}")
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path.name}")

def main():
    """Main training function"""
    print("🚀 Professional 4.9B Parameter Model Training")
    print("=" * 60)
    print("Optimized for RTX 3050 with enterprise-grade datasets")
    print()
    
    # Check for training data
    data_file = Path("training_datasets/enterprise_mixed_dataset.jsonl")
    if not data_file.exists():
        print("❌ Training data not found!")
        print("Please run: python enhanced_training_data_fetcher.py")
        return
    
    # Initialize configuration
    config = ModelConfig()
    
    # Display configuration
    print("📋 Training Configuration:")
    print(f"  Model parameters: ~4.9B")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps:,}")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print()
    
    # Initialize trainer
    try:
        trainer = Professional4_9BTrainer(config, str(data_file))
        
        # Start training
        trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print("📁 Check 'checkpoints_4.9B/' for saved models")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
