# Complete Setup and Launch for 4.9B Parameter Training
# Enterprise-grade setup similar to ChatGPT/Claude/Gemini training

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking Requirements...")
    
    required_packages = [
        "torch",
        "numpy", 
        "tqdm",
        "tiktoken"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
    
    return len(missing_packages) == 0

def check_cuda():
    """Check CUDA availability"""
    print("\n🖥️ Checking CUDA Support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  ✅ VRAM: {gpu_memory:.1f} GB")
            return True
        else:
            print("  ⚠️ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "training_datasets",
        "checkpoints_4.9B", 
        "training_logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")

def check_training_data():
    """Check if training data exists"""
    print("\n📊 Checking Training Data...")
    
    data_files = [
        "training_datasets/enterprise_mixed_dataset.jsonl",
        "training_datasets/sample_dataset.jsonl"
    ]
    
    found_data = False
    for data_file in data_files:
        if Path(data_file).exists():
            size = Path(data_file).stat().st_size
            print(f"  ✅ {data_file} ({size:,} bytes)")
            found_data = True
        else:
            print(f"  ⚠️ {data_file} - Not found")
    
    if not found_data:
        print("  📝 Creating sample training data...")
        create_sample_data()
    
    return True

def create_sample_data():
    """Create high-quality sample training data"""
    sample_data = [
        {"text": "Large language models represent a breakthrough in artificial intelligence, demonstrating remarkable capabilities in natural language understanding, generation, and reasoning through training on vast datasets containing billions of tokens from diverse sources."},
        
        {"text": "The transformer architecture revolutionized sequence modeling by replacing recurrent connections with self-attention mechanisms, enabling parallel processing and better capture of long-range dependencies in text sequences."},
        
        {"text": "Deep learning models learn hierarchical representations through multiple layers of nonlinear transformations, with each layer extracting increasingly abstract features that enable understanding of complex patterns in data."},
        
        {"text": "Natural language processing has evolved from rule-based systems to statistical methods and now to large-scale neural models capable of tasks such as translation, summarization, question answering, and creative writing."},
        
        {"text": "Machine learning optimization techniques including gradient descent, adaptive learning rates, and regularization methods are crucial for training deep neural networks effectively on large-scale datasets."},
        
        {"text": "Artificial intelligence research encompasses computer science, mathematics, cognitive science, and linguistics, with interdisciplinary collaboration driving continuous innovation and breakthrough discoveries in the field."},
        
        {"text": "Training large language models requires substantial computational resources including high-performance GPUs, distributed computing infrastructure, and carefully curated datasets from web text, books, and scientific literature."},
        
        {"text": "The field of computational linguistics combines computational methods with linguistic theory to understand and model human language, contributing foundational knowledge for natural language processing applications."},
        
        {"text": "Modern AI systems demonstrate emergent capabilities that arise from scaling model size and training data, including few-shot learning, reasoning, domain adaptation, and complex problem-solving abilities."},
        
        {"text": "Enterprise applications of large language models span healthcare, finance, education, scientific research, and creative industries, with careful attention to ethical considerations including fairness, transparency, and safety."}
    ]
    
    # Add more comprehensive training examples
    additional_data = [
        {"text": "Neural network architectures have evolved significantly, from simple perceptrons to complex transformer models with billions of parameters. These architectures enable sophisticated pattern recognition and generation capabilities across multiple domains including text, images, and structured data."},
        
        {"text": "The attention mechanism allows models to focus on relevant parts of input sequences when generating outputs, leading to improved performance in tasks requiring long-range dependencies and contextual understanding."},
        
        {"text": "Pre-training on large corpora followed by fine-tuning on specific tasks has become a dominant paradigm in natural language processing, enabling transfer learning and improved performance across diverse applications."},
        
        {"text": "Gradient-based optimization algorithms, including Adam, AdamW, and other adaptive methods, are essential for training deep networks efficiently by automatically adjusting learning rates based on gradient statistics."},
        
        {"text": "Regularization techniques such as dropout, weight decay, and layer normalization help prevent overfitting and improve generalization in large neural networks trained on extensive datasets."},
        
        {"text": "The scaling laws of neural language models demonstrate that performance improves predictably with increases in model size, dataset size, and computational budget, guiding efficient resource allocation in training."},
        
        {"text": "Tokenization strategies significantly impact model performance, with subword methods like Byte-Pair Encoding and SentencePiece enabling efficient representation of diverse vocabularies across multiple languages."},
        
        {"text": "Mixed precision training using 16-bit floating point arithmetic reduces memory usage and increases training speed while maintaining model quality, making it possible to train larger models on available hardware."},
        
        {"text": "Distributed training across multiple GPUs and machines enables scaling to models with hundreds of billions of parameters, requiring sophisticated synchronization and communication strategies."},
        
        {"text": "Evaluation metrics for language models include perplexity, BLEU scores, and task-specific benchmarks, providing quantitative measures of model quality and enabling comparison across different approaches."}
    ]
    
    sample_data.extend(additional_data)
    
    # Write to file
    with open("training_datasets/sample_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(f"{item}\n".replace("'", '"'))
    
    print(f"  ✅ Created sample_dataset.jsonl with {len(sample_data)} examples")

def start_training():
    """Start the 4.9B parameter training"""
    print("\n🚀 Starting 4.9B Parameter Training...")
    print("=" * 60)
    print("This will train a professional model similar to ChatGPT/Claude/Gemini")
    print("Training optimized for RTX 3050 with enterprise-grade datasets")
    print()
    
    # Start training
    try:
        subprocess.run([sys.executable, "train_4_9B_professional.py"])
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")

def main():
    """Main setup and launch function"""
    print("🚀 4.9B Parameter Model - Complete Setup & Training")
    print("=" * 70)
    print("Enterprise-grade AI training similar to ChatGPT, Claude, and Gemini")
    print("Optimized for RTX 3050 8GB VRAM")
    print()
    
    # Run all checks and setup
    if not check_requirements():
        print("❌ Requirements check failed")
        return
    
    cuda_available = check_cuda()
    setup_directories()
    
    if not check_training_data():
        print("❌ Training data setup failed")
        return
    
    print("\n✅ Setup Complete!")
    print(f"🖥️  Using: {'CUDA GPU' if cuda_available else 'CPU'}")
    print("📊 Training data ready")
    print("📁 Directories created")
    print()
    
    # Ask user to start training
    response = input("🎯 Start 4.9B parameter training now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        start_training()
    else:
        print("\n📋 Setup complete. Run 'python train_4_9B_professional.py' to start training.")
        print("📊 Run 'python monitor_training.py' to monitor progress.")

if __name__ == "__main__":
    main()
