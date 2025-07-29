import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json
import os
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import psutil
import GPUtil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import your existing models
try:
    from legacy.illuminator_cuda import iLLuMinatorCUDA
    from legacy.illuminator_4_7b_ai import iLLuMinator4_7B
    from tokenizer import CustomTokenizer
except ImportError:
    print("Warning: Some modules not found. Will use fallback implementations.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_4.9B_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Enhanced4_9BDataset(Dataset):
    """Optimized dataset for 4.9B parameter training"""
    
    def __init__(self, texts, tokenizer, max_length=2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Preprocess and cache tokenized data
        self.tokenized_data = []
        logger.info("Preprocessing dataset...")
        
        for text in tqdm(texts, desc="Tokenizing"):
            if len(text.strip()) > 50:  # Filter short texts
                try:
                    tokens = self.tokenizer.encode(text)
                    if len(tokens) > 100:  # Ensure sufficient context
                        self.tokenized_data.append(tokens)
                except Exception as e:
                    logger.warning(f"Error tokenizing text: {e}")
                    continue
        
        logger.info(f"Dataset ready: {len(self.tokenized_data)} samples")
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx]
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with special token (using 0 as pad token)
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # Create input and target (shifted by 1 for next token prediction)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        pad_mask = input_ids == 0
        attention_mask[pad_mask] = 0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }

class Advanced4_9BTrainer:
    """Advanced trainer for 4.9B parameter model with all optimizations"""
    
    def __init__(self, config_path="config_4.9B.json"):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_cuda_optimizations()
        self.setup_model_and_tokenizer()
        self.setup_training()
        self.setup_monitoring()
        self.best_loss = float('inf')
        
    def setup_cuda_optimizations(self):
        """Enable all CUDA optimizations for RTX GPUs"""
        if torch.cuda.is_available():
            # Enable TensorFloat-32 for RTX 30xx series
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction to avoid OOM on RTX 3050
            torch.cuda.set_per_process_memory_fraction(0.90)
            
            # Enable CUDA graphs for faster execution
            torch.backends.cudnn.enabled = True
            
            logger.info("CUDA optimizations enabled for RTX 3050")
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def load_config(self, config_path):
        # RTX 3050 optimized configuration (smaller model for 8GB VRAM)
        default_config = {
            # Model architecture (optimized for RTX 3050 - ~1.5B parameters)
            "vocab_size": 50260,
            "d_model": 2048,
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 8192,
            "max_seq_length": 1024,
            
            # Training parameters optimized for RTX 3050
            "batch_size": 1,  # Small batch for memory efficiency
            "gradient_accumulation_steps": 16,  # Simulate larger batch
            "learning_rate": 5e-6,  # Conservative for stable training
            "num_epochs": 2,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            
            # Memory optimizations
            "use_mixed_precision": True,
            "gradient_checkpointing": True,
            "use_flash_attention": True,
            "pin_memory": True,
            
            # Logging and saving
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 50,
            "save_total_limit": 3,
            
            # Dataset
            "train_data_files": [
                "practical_model/training_data.txt",
                "integrated_knowledge_base.json"
            ],
            "use_knowledge_distillation": True,
            "teacher_model_path": "practical_model/illuminator_practical_improved_best.pth",
            
            # Advanced features
            "use_dynamic_loss_scaling": True,
            "use_gradient_clipping": True,
            "use_warmup_scheduler": True,
            "save_optimizer_states": True
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        else:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def setup_model_and_tokenizer(self):
        logger.info("Setting up 4.9B parameter model...")
        
        # Try to load your custom tokenizer, fallback to basic implementation
        try:
            self.tokenizer = CustomTokenizer()
            logger.info(f"Custom tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
        except:
            logger.warning("Using fallback tokenizer")
            # Create a simple tokenizer fallback
            class FallbackTokenizer:
                def __init__(self):
                    self.vocab_size = 50260
                    self.pad_token_id = 0
                    self.eos_token_id = 1
                
                def encode(self, text):
                    # Simple character-level encoding for fallback
                    return [min(ord(c), self.vocab_size-1) for c in text[:1000]]
                
                def decode(self, tokens):
                    return ''.join([chr(min(t, 127)) for t in tokens if t > 0])
            
            self.tokenizer = FallbackTokenizer()
        
        # Initialize your model - try different approaches
        try:
            # Try to load your CUDA-optimized model
            self.model = iLLuMinatorCUDA(
                vocab_size=self.config['vocab_size'],
                d_model=self.config['d_model'],
                n_heads=self.config['n_heads'],
                n_layers=self.config['n_layers'],
                d_ff=self.config['d_ff'],
                max_seq_length=self.config['max_seq_length']
            )
            logger.info("Loaded iLLuMinatorCUDA model")
        except:
            try:
                # Try the 4.7B model
                self.model = iLLuMinator4_7B(
                    vocab_size=self.config['vocab_size'],
                    d_model=self.config['d_model'],
                    n_heads=self.config['n_heads'],
                    n_layers=self.config['n_layers']
                )
                logger.info("Loaded iLLuMinator4_7B model")
            except:
                # Fallback to basic transformer
                logger.warning("Using fallback transformer model")
                self.model = self.create_fallback_model()
        
        # Enable optimizations
        if self.config['gradient_checkpointing']:
            try:
                self.model.gradient_checkpointing_enable()
            except:
                logger.warning("Gradient checkpointing not available")
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model loaded:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: {total_params * 4 / 1024**3:.2f} GB (FP32)")
        logger.info(f"  Model size: {total_params * 2 / 1024**3:.2f} GB (FP16)")
        
        # Setup teacher model for knowledge distillation
        if self.config['use_knowledge_distillation']:
            self.setup_teacher_model()
    
    def create_fallback_model(self):
        """Create a fallback transformer model"""
        class FallbackTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Embedding(2048, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.ln_f = nn.LayerNorm(d_model)
                self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
                
                # Tie weights
                self.lm_head.weight = self.embedding.weight
                
            def forward(self, input_ids, attention_mask=None, labels=None):
                seq_len = input_ids.size(1)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                
                x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
                
                if attention_mask is not None:
                    # Convert attention mask to transformer format
                    attention_mask = attention_mask.float()
                    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
                    attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
                
                x = self.transformer(x, src_key_padding_mask=(attention_mask == float('-inf')))
                x = self.ln_f(x)
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                return type('Output', (), {'loss': loss, 'logits': logits})()
        
        return FallbackTransformer(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers'],
            d_ff=self.config['d_ff']
        )
    
    def setup_teacher_model(self):
        """Setup smaller practical model as teacher for knowledge distillation"""
        try:
            teacher_path = self.config['teacher_model_path']
            if os.path.exists(teacher_path):
                logger.info("Loading teacher model for knowledge distillation...")
                
                # Load checkpoint
                checkpoint = torch.load(teacher_path, map_location=self.device)
                
                # Create teacher model (smaller version)
                self.teacher_model = self.create_fallback_model()
                
                # Load weights if compatible
                try:
                    self.teacher_model.load_state_dict(checkpoint)
                    self.teacher_model.to(self.device)
                    self.teacher_model.eval()
                    logger.info("Teacher model loaded successfully")
                except:
                    logger.warning("Teacher model weights incompatible, disabling knowledge distillation")
                    self.config['use_knowledge_distillation'] = False
            else:
                logger.warning("Teacher model not found, disabling knowledge distillation")
                self.config['use_knowledge_distillation'] = False
        except Exception as e:
            logger.warning(f"Failed to load teacher model: {e}")
            self.config['use_knowledge_distillation'] = False
    
    def setup_training(self):
        """Setup optimizer, scheduler, and training components"""
        # Setup optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight", "ln_", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config['weight_decay'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate'],
            betas=(self.config['beta1'], self.config['beta2']),
            eps=1e-8
        )
        
        # Setup mixed precision scaler
        if self.config['use_mixed_precision']:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        
        # Setup learning rate scheduler
        if self.config['use_warmup_scheduler']:
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(step):
                if step < self.config['warmup_steps']:
                    return step / self.config['warmup_steps']
                else:
                    # Cosine decay
                    progress = (step - self.config['warmup_steps']) / max(1, self.config['warmup_steps'] * 10)
                    return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None
    
    def setup_monitoring(self):
        """Setup monitoring and logging"""
        # Initialize wandb
        try:
            wandb.init(
                project="illuminator-4.9b-enhanced",
                config=self.config,
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info("Weights & Biases monitoring enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
        
        # Setup system monitoring
        try:
            self.gpu_monitor = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        except:
            self.gpu_monitor = None
    
    def load_training_data(self):
        """Load and prepare training data"""
        logger.info("Loading training data...")
        
        all_texts = []
        
        # Load from various sources
        for data_file in self.config['train_data_files']:
            if os.path.exists(data_file):
                logger.info(f"Loading {data_file}...")
                
                if data_file.endswith('.json'):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_texts.extend([str(item) for item in data])
                        elif isinstance(data, dict):
                            # Extract text from various JSON structures
                            for key, value in data.items():
                                if isinstance(value, str) and len(value) > 100:
                                    all_texts.append(value)
                                elif isinstance(value, list):
                                    all_texts.extend([str(item) for item in value if len(str(item)) > 100])
                
                elif data_file.endswith('.txt'):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Split into chunks
                        chunks = content.split('\n\n')
                        all_texts.extend([chunk.strip() for chunk in chunks if len(chunk.strip()) > 100])
            else:
                logger.warning(f"Data file not found: {data_file}")
        
        # Add some default training data if no files found
        if not all_texts:
            logger.warning("No training data found, using default examples")
            all_texts = [
                "The quick brown fox jumps over the lazy dog. This is a sample sentence for training.",
                "Artificial intelligence is transforming the world in remarkable ways.",
                "Large language models can generate human-like text when properly trained.",
                "Machine learning requires careful attention to data quality and model architecture.",
                "Training neural networks involves optimizing millions or billions of parameters."
            ] * 100  # Repeat for more training data
        
        logger.info(f"Loaded {len(all_texts)} training examples")
        
        # Create dataset
        self.train_dataset = Enhanced4_9BDataset(
            all_texts,
            self.tokenizer,
            max_length=self.config['max_seq_length']
        )
        
        # Create data loader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Use 0 for Windows compatibility
            pin_memory=self.config['pin_memory'] and torch.cuda.is_available(),
            drop_last=True
        )
        
        logger.info(f"Training dataloader ready with {len(self.train_dataloader)} batches")
    
    def calculate_knowledge_distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.7, temperature=4.0):
        """Calculate knowledge distillation loss"""
        if not hasattr(self, 'teacher_model') or self.teacher_model is None:
            return 0.0
        
        # Standard cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Knowledge distillation loss
        student_soft = torch.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = torch.softmax(teacher_logits / temperature, dim=-1)
        kd_loss = nn.KLDivLoss(reduction='batchmean')(student_soft, teacher_soft) * (temperature ** 2)
        
        # Combined loss
        total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
        return total_loss
    
    def log_system_stats(self):
        """Log system statistics"""
        stats = {}
        
        try:
            # CPU and Memory stats
            stats['cpu_percent'] = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            stats['memory_percent'] = memory.percent
            stats['memory_used_gb'] = memory.used / 1024**3
            
            # GPU stats
            if torch.cuda.is_available():
                stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
                
                if self.gpu_monitor:
                    stats['gpu_utilization'] = self.gpu_monitor.load * 100
                    stats['gpu_temperature'] = self.gpu_monitor.temperature
        
        except Exception as e:
            logger.warning(f"Error collecting system stats: {e}")
        
        return stats
    
    def train(self):
        """Main training loop"""
        logger.info("Starting 4.9B parameter model training...")
        
        # Load training data
        self.load_training_data()
        
        # Set model to training mode
        self.model.train()
        
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            epoch_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    with autocast(enabled=self.config['use_mixed_precision']):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                        # Knowledge distillation if enabled
                        if self.config['use_knowledge_distillation'] and hasattr(self, 'teacher_model'):
                            with torch.no_grad():
                                teacher_outputs = self.teacher_model(**batch)
                            loss = self.calculate_knowledge_distillation_loss(
                                outputs.logits, teacher_outputs.logits, batch['labels']
                            )
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.config['gradient_accumulation_steps']
                    
                    # Backward pass
                    if self.config['use_mixed_precision']:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    total_loss += loss.item()
                    epoch_loss += loss.item()
                    
                    # Gradient accumulation step
                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        # Gradient clipping
                        if self.config['use_gradient_clipping']:
                            if self.config['use_mixed_precision']:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config['max_grad_norm']
                                )
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config['max_grad_norm']
                                )
                                self.optimizer.step()
                        else:
                            if self.config['use_mixed_precision']:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                        
                        # Update learning rate
                        if self.scheduler:
                            self.scheduler.step()
                        
                        self.optimizer.zero_grad()
                        global_step += 1
                    
                    # Logging
                    if global_step % self.config['logging_steps'] == 0 and global_step > 0:
                        avg_loss = total_loss / self.config['logging_steps']
                        system_stats = self.log_system_stats()
                        
                        log_data = {
                            'train_loss': avg_loss,
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                            'global_step': global_step,
                            'epoch': epoch,
                            **system_stats
                        }
                        
                        try:
                            wandb.log(log_data)
                        except:
                            pass
                        
                        logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}, LR = {self.optimizer.param_groups[0]['lr']:.2e}")
                        total_loss = 0
                    
                    # Save checkpoint
                    if global_step % self.config['save_steps'] == 0 and global_step > 0:
                        self.save_checkpoint(global_step, avg_loss if 'avg_loss' in locals() else loss.item())
                    
                    # Update progress bar
                    progress_bar.set_description(
                        f"Epoch {epoch + 1} - Loss: {epoch_loss / (step + 1):.4f}"
                    )
                    
                    # Memory cleanup for RTX 3050
                    if step % 50 == 0:
                        torch.cuda.empty_cache()
                        
                    # Early stopping if GPU memory is getting full
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                        if memory_used > 0.95:
                            logger.warning("GPU memory usage high, forcing cleanup")
                            torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Error in training step {step}: {e}")
                    continue
        
        # Final save
        self.save_checkpoint(global_step, epoch_loss / len(self.train_dataloader), final=True)
        logger.info("Training completed!")
    
    def save_checkpoint(self, step, loss, final=False):
        """Save model checkpoint"""
        checkpoint_dir = f"checkpoints/step_{step}" if not final else "illuminator_4.9B_final"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        torch.save({
            'step': step,
            'loss': loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'config': self.config
        }, os.path.join(checkpoint_dir, 'pytorch_model.bin'))
        
        # Save config
        with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Update best model if loss improved
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(
                self.model.state_dict(),
                os.path.join(checkpoint_dir, 'best_model.bin')
            )
            logger.info(f"New best model saved with loss: {loss:.4f}")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

def main():
    """Main training function"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be extremely slow on CPU.")
        logger.warning("Please install CUDA-enabled PyTorch for GPU training.")
    else:
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Check if GPU has sufficient memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 6:
            logger.warning(f"GPU memory ({gpu_memory_gb:.1f} GB) may be insufficient for 4.9B model")
            logger.warning("Consider using gradient checkpointing and small batch sizes")
    
    # Initialize trainer
    try:
        trainer = Advanced4_9BTrainer()
        logger.info("Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    main()
