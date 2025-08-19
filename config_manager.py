"""
Professional Configuration Management for iLLuMinator 4.9B
Centralized configuration with validation and documentation
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

class ModelSize(Enum):
    """Supported model sizes"""
    TINY = "120M"
    SMALL = "1.3B"
    MEDIUM = "2.8B"
    LARGE = "4.9B"

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 65536
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    max_seq_length: int = 4096
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_ff > self.d_model, "Feed-forward dimension should be larger than model dimension"

@dataclass
class TrainingConfig:
    """Training hyperparameters configuration"""
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 100
    num_epochs: int = 5
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        return self.batch_size * self.gradient_accumulation_steps

@dataclass
class DataConfig:
    """Data processing configuration"""
    max_seq_length: int = 2048
    dataset_size: int = 25000
    train_split: float = 0.9
    val_split: float = 0.1
    use_external_data: bool = True
    quality_threshold: float = 0.8
    data_format: str = "alpaca"  # alpaca, chat, simple
    cache_dir: str = "datasets"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.train_split + self.val_split <= 1.0, "Split ratios must sum to <= 1.0"
        assert 0 < self.quality_threshold <= 1.0, "Quality threshold must be between 0 and 1"

@dataclass
class SystemConfig:
    """System and hardware configuration"""
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4
    pin_memory: bool = True
    compile_model: bool = False  # PyTorch 2.0+ compilation
    use_wandb: bool = False
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)

@dataclass
class ProfessionalConfig:
    """Complete professional configuration"""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        """Initialize defaults if not provided"""
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.system is None:
            self.system = SystemConfig()
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProfessionalConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def get_model_size_estimate(self) -> str:
        """Estimate model size based on parameters"""
        # Rough estimation: parameters ≈ 12 * n_layers * d_model^2 / 1e9
        params = 12 * self.model.n_layers * (self.model.d_model ** 2) / 1e9
        
        if params < 0.5:
            return ModelSize.TINY.value
        elif params < 2.0:
            return ModelSize.SMALL.value
        elif params < 4.0:
            return ModelSize.MEDIUM.value
        else:
            return ModelSize.LARGE.value

def get_preset_config(model_size: ModelSize) -> ProfessionalConfig:
    """Get preset configuration for different model sizes"""
    
    configs = {
        ModelSize.TINY: ProfessionalConfig(
            model=ModelConfig(
                vocab_size=32000,
                d_model=768,
                n_layers=12,
                n_heads=12,
                n_kv_heads=4,
                d_ff=3072,
                max_seq_length=2048
            ),
            training=TrainingConfig(
                batch_size=8,
                max_steps=50000,
                learning_rate=5e-4
            ),
            data=DataConfig(
                max_seq_length=1024,
                dataset_size=10000
            )
        ),
        
        ModelSize.SMALL: ProfessionalConfig(
            model=ModelConfig(
                vocab_size=50000,
                d_model=1536,
                n_layers=18,
                n_heads=16,
                n_kv_heads=4,
                d_ff=6144,
                max_seq_length=2048
            ),
            training=TrainingConfig(
                batch_size=4,
                max_steps=75000,
                learning_rate=4e-4
            ),
            data=DataConfig(
                max_seq_length=1536,
                dataset_size=15000
            )
        ),
        
        ModelSize.MEDIUM: ProfessionalConfig(
            model=ModelConfig(
                vocab_size=50000,
                d_model=2560,
                n_layers=24,
                n_heads=20,
                n_kv_heads=5,
                d_ff=10240,
                max_seq_length=3072
            ),
            training=TrainingConfig(
                batch_size=2,
                max_steps=80000,
                learning_rate=3.5e-4
            ),
            data=DataConfig(
                max_seq_length=2048,
                dataset_size=20000
            )
        ),
        
        ModelSize.LARGE: ProfessionalConfig(
            model=ModelConfig(
                vocab_size=65536,
                d_model=4096,
                n_layers=32,
                n_heads=32,
                n_kv_heads=8,
                d_ff=14336,
                max_seq_length=4096
            ),
            training=TrainingConfig(
                batch_size=1,
                gradient_accumulation_steps=64,
                max_steps=100000,
                learning_rate=3e-4
            ),
            data=DataConfig(
                max_seq_length=2048,
                dataset_size=25000
            )
        )
    }
    
    return configs[model_size]

def auto_detect_hardware_config() -> Dict[str, Any]:
    """Auto-detect optimal hardware configuration"""
    import torch
    
    config = {}
    
    # Detect device
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        config['mixed_precision'] = True
        
        # Get GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory_gb >= 40:  # A100, H100 class
            config['batch_size'] = 4
            config['gradient_accumulation_steps'] = 16
        elif gpu_memory_gb >= 24:  # RTX 4090, RTX 6000 class
            config['batch_size'] = 2
            config['gradient_accumulation_steps'] = 32
        elif gpu_memory_gb >= 16:  # RTX 4070 Ti, RTX 3080 class
            config['batch_size'] = 1
            config['gradient_accumulation_steps'] = 64
        else:  # Lower memory GPUs
            config['batch_size'] = 1
            config['gradient_accumulation_steps'] = 128
            config['gradient_checkpointing'] = True
            
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        config['device'] = 'mps'
        config['mixed_precision'] = False  # MPS doesn't support all mixed precision ops
        config['batch_size'] = 1
        config['gradient_accumulation_steps'] = 32
        
    else:
        config['device'] = 'cpu'
        config['mixed_precision'] = False
        config['batch_size'] = 1
        config['gradient_accumulation_steps'] = 64
    
    # Detect CPU cores for data loading
    import multiprocessing
    config['num_workers'] = min(multiprocessing.cpu_count(), 8)
    
    return config

def create_professional_config(model_size: str = "4.9B", 
                             custom_overrides: Optional[Dict] = None) -> ProfessionalConfig:
    """Create a professional configuration with auto-detection"""
    
    # Get preset config
    size_map = {
        "120M": ModelSize.TINY,
        "1.3B": ModelSize.SMALL,
        "2.8B": ModelSize.MEDIUM,
        "4.9B": ModelSize.LARGE
    }
    
    config = get_preset_config(size_map.get(model_size, ModelSize.LARGE))
    
    # Auto-detect hardware optimizations
    hw_config = auto_detect_hardware_config()
    
    # Apply hardware optimizations
    for key, value in hw_config.items():
        if key in ['device', 'num_workers', 'pin_memory']:
            setattr(config.system, key, value)
        else:
            setattr(config.training, key, value)
    
    # Apply custom overrides
    if custom_overrides:
        for section, overrides in custom_overrides.items():
            if section == 'model':
                for key, value in overrides.items():
                    setattr(config.model, key, value)
            elif section == 'training':
                for key, value in overrides.items():
                    setattr(config.training, key, value)
            elif section == 'data':
                for key, value in overrides.items():
                    setattr(config.data, key, value)
            elif section == 'system':
                for key, value in overrides.items():
                    setattr(config.system, key, value)
    
    return config

def main():
    """Demo configuration creation and validation"""
    print("iLLuMinator 4.9B Professional Configuration Demo")
    print("=" * 60)
    
    # Create configurations for different model sizes
    for size in ModelSize:
        config = get_preset_config(size)
        estimated_size = config.get_model_size_estimate()
        
        print(f"\n{size.value} Model Configuration:")
        print(f"  Estimated parameters: {estimated_size}")
        print(f"  Model dimension: {config.model.d_model}")
        print(f"  Layers: {config.model.n_layers}")
        print(f"  Attention heads: {config.model.n_heads}")
        print(f"  Effective batch size: {config.training.effective_batch_size}")
        
        # Save configuration
        config_file = f"config_{size.value.lower()}.json"
        config.save(config_file)
        print(f"  Configuration saved to: {config_file}")
    
    # Demo auto-hardware detection
    print(f"\nAuto-detected hardware configuration:")
    hw_config = auto_detect_hardware_config()
    for key, value in hw_config.items():
        print(f"  {key}: {value}")
    
    # Create professional config with auto-detection
    professional_config = create_professional_config("4.9B")
    professional_config.save("professional_config.json")
    print(f"\nProfessional configuration saved to: professional_config.json")

if __name__ == "__main__":
    main()