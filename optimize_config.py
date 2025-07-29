import json
import torch
import psutil
import GPUtil
import os

def detect_optimal_config():
    """Detect optimal configuration based on system capabilities"""
    
    config = {
        # Base model architecture (4.9B parameters)
        "vocab_size": 50260,
        "d_model": 3584,
        "n_heads": 28,
        "n_layers": 30,
        "d_ff": 14336,
        "max_seq_length": 2048,
        
        # Default training parameters
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 1e-5,
        "num_epochs": 3,
        "warmup_steps": 2000,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        
        # Memory optimizations
        "use_mixed_precision": True,
        "gradient_checkpointing": True,
        "use_flash_attention": False,  # May not be available
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
    
    # System detection
    print("Detecting system configuration...")
    
    # CPU information
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1024**3
    
    print(f"CPU cores: {cpu_count}")
    print(f"System memory: {memory_gb:.1f} GB")
    
    # GPU detection and optimization
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU: {gpu_name}")
        print(f"GPU memory: {gpu_memory_gb:.1f} GB")
        
        # RTX 3070/3080/3090 specific optimizations
        if "RTX 30" in gpu_name or "RTX 40" in gpu_name:
            print("Detected RTX 30xx/40xx series - enabling optimizations")
            config["use_mixed_precision"] = True
            config["gradient_checkpointing"] = True
            
            # Memory-based batch size optimization
            if gpu_memory_gb >= 12:  # RTX 3080/3090
                config["batch_size"] = 2
                config["gradient_accumulation_steps"] = 16
                print("High memory GPU detected - using larger batch size")
            elif gpu_memory_gb >= 8:  # RTX 3070
                config["batch_size"] = 1
                config["gradient_accumulation_steps"] = 32
                print("Medium memory GPU detected - using standard batch size")
            else:  # Lower memory
                config["batch_size"] = 1
                config["gradient_accumulation_steps"] = 64
                config["gradient_checkpointing"] = True
                print("Low memory GPU detected - aggressive memory optimization")
        
        # Try to detect Flash Attention availability
        try:
            import flash_attn
            config["use_flash_attention"] = True
            print("Flash Attention detected - enabling for speed optimization")
        except ImportError:
            config["use_flash_attention"] = False
            print("Flash Attention not available - using standard attention")
    
    else:
        print("No CUDA GPU detected - using CPU optimizations")
        config["use_mixed_precision"] = False
        config["batch_size"] = 1
        config["gradient_accumulation_steps"] = 128  # Larger accumulation for CPU
        config["pin_memory"] = False
    
    # Memory-based optimizations
    if memory_gb < 16:
        print("Low system memory detected - reducing dataloader workers")
        config["dataloader_num_workers"] = 0
    elif memory_gb >= 32:
        print("High system memory detected - increasing dataloader workers")
        config["dataloader_num_workers"] = 4
    else:
        config["dataloader_num_workers"] = 2
    
    # Check for existing training data
    data_files_found = []
    for data_file in config["train_data_files"]:
        if os.path.exists(data_file):
            data_files_found.append(data_file)
            print(f"Found training data: {data_file}")
    
    if not data_files_found:
        print("No training data files found - will use synthetic data")
    
    config["train_data_files"] = data_files_found
    
    # Check for teacher model
    if os.path.exists(config["teacher_model_path"]):
        print(f"Teacher model found: {config['teacher_model_path']}")
        config["use_knowledge_distillation"] = True
    else:
        print("No teacher model found - disabling knowledge distillation")
        config["use_knowledge_distillation"] = False
    
    return config

def create_training_configs():
    """Create multiple training configuration files for different scenarios"""
    
    base_config = detect_optimal_config()
    
    # Configuration variants
    configs = {
        "config_4.9B.json": base_config,
        
        "config_4.9B_fast.json": {
            **base_config,
            "batch_size": min(2, base_config["batch_size"] * 2),
            "gradient_accumulation_steps": max(8, base_config["gradient_accumulation_steps"] // 2),
            "learning_rate": 2e-5,
            "save_steps": 500,
            "logging_steps": 25
        },
        
        "config_4.9B_stable.json": {
            **base_config,
            "learning_rate": 5e-6,
            "warmup_steps": 3000,
            "max_grad_norm": 0.5,
            "weight_decay": 0.02,
            "save_steps": 2000
        },
        
        "config_4.9B_memory_efficient.json": {
            **base_config,
            "batch_size": 1,
            "gradient_accumulation_steps": 64,
            "gradient_checkpointing": True,
            "max_seq_length": 1024,
            "use_mixed_precision": True
        }
    }
    
    # Save configurations
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created configuration: {filename}")
    
    return configs

def print_recommendations():
    """Print training recommendations based on system"""
    
    print("\n" + "="*60)
    print("TRAINING RECOMMENDATIONS")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory_gb >= 12:
            print("🚀 Your GPU has sufficient memory for optimal training:")
            print("  - Use config_4.9B_fast.json for faster training")
            print("  - Enable mixed precision and gradient checkpointing")
            print("  - Consider batch size 2 with gradient accumulation 16")
        
        elif gpu_memory_gb >= 8:
            print("⚡ Your GPU can handle standard training:")
            print("  - Use config_4.9B.json for balanced training")
            print("  - Keep batch size 1 with gradient accumulation 32")
            print("  - Mixed precision is recommended")
        
        else:
            print("💾 Your GPU has limited memory:")
            print("  - Use config_4.9B_memory_efficient.json")
            print("  - Enable all memory optimizations")
            print("  - Consider reducing sequence length to 1024")
    
    else:
        print("🐌 CPU training detected:")
        print("  - Training will be very slow")
        print("  - Consider using the practical 120M model instead")
        print("  - Use config_4.9B_memory_efficient.json if proceeding")
    
    print("\n💡 General recommendations:")
    print("  - Monitor GPU memory usage during training")
    print("  - Use Weights & Biases for experiment tracking")
    print("  - Save checkpoints frequently")
    print("  - Run debug analysis after training")

def main():
    print("iLLuMinator 4.9B Configuration Optimizer")
    print("="*50)
    
    # Create optimized configurations
    configs = create_training_configs()
    
    # Print system analysis
    print(f"\nConfigurations created:")
    for config_name in configs.keys():
        print(f"  - {config_name}")
    
    # Print recommendations
    print_recommendations()
    
    print(f"\n✅ Configuration optimization complete!")
    print(f"Run: python train_4.9B_enhanced.py to start training")

if __name__ == "__main__":
    main()
