# iLLuMinator 4.9B - Professional Language Model Training

A clean, optimized implementation for training a 4.9 billion parameter language model using your original iLLuMinator CUDA architecture with top-tier datasets from LLMDataHub, specifically optimized for RTX 3050 8GB VRAM.

## ✅ What's Working

- **Original Architecture**: Using your complete iLLuMinator CUDA model (4.8B parameters)
- **Training Pipeline**: Professional training script with LLMDataHub datasets
- **Memory Optimization**: RTX 3050 optimized with gradient checkpointing and mixed precision
- **Clean Codebase**: Removed unnecessary files, kept essential components
- **Ready to Train**: The model is now training successfully!

## Features

- **4.9 Billion Parameters**: Your original iLLuMinator CUDA architecture (4.8B parameters)
- **RTX 3050 Optimized**: Memory-efficient training for 8GB VRAM
- **Professional Datasets**: Uses high-quality datasets from LLMDataHub repository
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Gradient Checkpointing**: Memory optimization for large models
- **Professional Logging**: Comprehensive training monitoring and checkpointing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Training

```bash
python train_4_9B_professional.py
```

## Model Architecture

- **Parameters**: ~4.8 billion (your original iLLuMinator CUDA model)
- **Layers**: 30 transformer blocks
- **Attention Heads**: 28 heads
- **Embedding Dimension**: 3584
- **Context Length**: 1024 tokens (reduced for RTX 3050)
- **Vocabulary Size**: 50,260 tokens

## Training Configuration

- **Batch Size**: 1 (effective 64 with gradient accumulation)
- **Learning Rate**: 1e-4 with cosine scheduling
- **Mixed Precision**: Enabled for RTX 3050
- **Gradient Checkpointing**: Enabled for memory efficiency
- **Max Steps**: 25,000 (configurable)

## Datasets

The model uses high-quality datasets from LLMDataHub:

- **OpenOrca**: 4.5M instruction-following examples
- **UltraChat**: 1.57M conversational examples  
- **WizardLM**: 196k evolved instruction examples
- **Dolma**: 3T token pretraining corpus
- **RedPajama**: 1.2T token web corpus

## System Requirements

- **GPU**: NVIDIA RTX 3050 (8GB VRAM) or better
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for checkpoints
- **OS**: Windows 10/11, Linux, or macOS

## Output

- **Checkpoints**: Saved in `checkpoints_4.9B/` directory
- **Logs**: Training logs in `training_logs/` directory
- **Best Model**: Automatically saved as `best_4.9B_model.pt`

## Usage

After training, the model can be loaded and used for:

- Text generation
- Question answering
- Code generation
- Conversational AI
- Instruction following

## Project Structure

```
iLLuMinator-4.9B/
├── train_4_9B_professional.py    # Main training script
├── inference.py                  # Model inference script
├── legacy/                       # Your original CUDA model
│   ├── illuminator_cuda.py       # 4.9B CUDA model
│   ├── illuminator_ai.py         # AI implementation
│   └── cuda_api_server.py        # API server
├── simple_illuminator.py         # Simple AI implementation
├── client.py                     # Client interface
├── chatbot_client.py             # Chatbot client
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Your original iLLuMinator CUDA architecture
- Datasets from [LLMDataHub](https://github.com/Zjh-819/LLMDataHub)
- Optimizations based on RTX 30xx series best practices

## Status

✅ **Training is now working!** Your 4.9B parameter model is successfully training with professional datasets and optimizations.
