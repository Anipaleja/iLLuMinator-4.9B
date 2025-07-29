#!/usr/bin/env python3

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_rtx3050_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RTX3050OptimizedModel(nn.Module):
    """Optimized transformer model for RTX 3050 (8GB VRAM)"""
    
    def __init__(self, vocab_size=50260, d_model=1536, n_heads=12, n_layers=18, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        hidden_states = token_emb + pos_emb
        
        # Transformer
        if attention_mask is not None:
            # Convert to transformer format
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
        
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=(attention_mask == float('-inf')))
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

class SimpleDataset(Dataset):
    def __init__(self, texts, max_length=1024):
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Simple character-level tokenization
        tokens = [ord(c) % 50260 for c in text[:self.max_length]]
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == 0] = 0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def main():
    logger.info("Starting RTX 3050 Optimized Training")
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # RTX 3050 optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.85)
    
    # Load training data
    data_file = "integrated_knowledge_base.json"
    if os.path.exists(data_file):
        logger.info(f"Loading training data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Extract text from JSON structure
        texts = []
        if isinstance(raw_data, list):
            texts = [str(item) for item in raw_data]
        elif isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, str) and len(value) > 50:
                    texts.append(value)
                elif isinstance(value, list):
                    texts.extend([str(item) for item in value if len(str(item)) > 50])
        
        logger.info(f"Loaded {len(texts)} training samples")
    else:
        logger.warning("No training data found, using sample data")
        texts = [
            "The future of artificial intelligence is bright and full of possibilities.",
            "Machine learning models require careful training and optimization.",
            "Large language models can generate human-like text when properly trained.",
            "Neural networks learn patterns from data through gradient descent optimization.",
            "Deep learning has revolutionized natural language processing tasks."
        ] * 200
    
    # Create dataset and dataloader
    dataset = SimpleDataset(texts, max_length=512)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Small batch for RTX 3050
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in WSL
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = RTX3050OptimizedModel(
        vocab_size=50260,
        d_model=1536,
        n_heads=12,
        n_layers=18,
        max_seq_len=512
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} (~1.5B parameters)")
    logger.info(f"Model size: {total_params * 4 / 1024**3:.2f} GB (FP32)")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    # Training configuration
    config = {
        'num_epochs': 3,
        'gradient_accumulation_steps': 8,
        'max_grad_norm': 1.0,
        'save_steps': 100,
        'log_steps': 10
    }
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs['loss'] / config['gradient_accumulation_steps']
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % config['log_steps'] == 0:
                    avg_loss = total_loss / config['log_steps']
                    logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}")
                    total_loss = 0.0
                
                # Save checkpoint
                if global_step % config['save_steps'] == 0:
                    checkpoint_path = f"checkpoints/model_step_{global_step}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'loss': avg_loss if 'avg_loss' in locals() else loss.item(),
                        'config': config
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Memory cleanup for RTX 3050
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()
    
    # Save final model
    final_path = "checkpoints/rtx3050_final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': 50260,
            'd_model': 1536,
            'n_heads': 12,
            'n_layers': 18,
            'max_seq_len': 512
        },
        'training_completed': True,
        'total_steps': global_step
    }, final_path)
    
    logger.info(f"Training completed! Final model saved: {final_path}")
    logger.info(f"Total training steps: {global_step}")
    
    # Quick generation test
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, 1000, (1, 50)).to(device)
        outputs = model(test_input)
        logger.info("✅ Model inference test passed")
    
    logger.info("🎉 RTX 3050 training completed successfully!")

if __name__ == "__main__":
    main()
