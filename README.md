# iLLuMinator: Intelligent RAG-Enhanced Transformer

A sophisticated transformer-based language model with integrated Retrieval-Augmented Generation (RAG) capabilities for intelligent question answering and conversation.

## Overview

iLLuMinator is a custom transformer architecture that combines the power of causal language modeling with intelligent knowledge retrieval. The system can understand queries, retrieve relevant information from its knowledge base, and generate contextually aware responses.

## Features

### Core Architecture
- **Custom Transformer Implementation**: Built from scratch with causal self-attention
- **RAG Integration**: Embedded knowledge retrieval system using FAISS and SentenceTransformers
- **Intelligent Query Understanding**: Intent classification and context processing
- **Adaptive Response Generation**: Temperature-controlled sampling with top-k filtering

### Key Components
- **iLLuMinator Transformer**: The core language model with causal masking
- **Knowledge Retrieval System**: Semantic search over document embeddings
- **Smart Assistant Interface**: Natural language conversation system
- **Training Pipeline**: Custom training loop with conversation datasets

## Project Structure

```
Transformer/
├── model/
│   ├── __init__.py
│   ├── transformer.py          # Core iLLuMinator model
│   └── tokenizer.py           # Text tokenization utilities
├── data/
│   ├── prepare.py             # Data loading and preprocessing
│   └── __pycache__/
├── intelligent_rag.py         # Advanced RAG system
├── smart_assistant.py         # Main assistant interface
├── fixed_assistant.py         # Simplified working assistant
├── train_intelligent.py       # Training script for RAG model
├── train.py                   # Basic training script
├── generate.py               # Text generation utilities
├── rag_system.py             # Original RAG implementation
├── rag_demo.py               # RAG demonstration
├── setup_complete.py         # Complete setup script
├── simple_demo.py            # Simple demo without dependencies
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── illuminator.pth           # Trained model weights
```

## Installation

### Option 1: Full Setup (Recommended)
```bash
python setup_complete.py
# Choose option 1 for full setup
```

This will install all dependencies including:
- PyTorch
- SentenceTransformers  
- FAISS-CPU
- Transformers
- NumPy

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
```

### Option 3: Simple Demo (No Dependencies)
```bash
python setup_complete.py
# Choose option 2 for simple demo
python simple_demo.py
```

## Usage

### Quick Start
```bash
# Train the model
python train.py

# Start the assistant
python fixed_assistant.py
```

### Advanced Usage

#### Training the Enhanced RAG Model
```bash
python train_intelligent.py
```

#### Using the Smart Assistant
```bash
python smart_assistant.py
```

#### Running Demos
```bash
# RAG system demo
python rag_demo.py

# Simple demo (no ML dependencies)
python simple_demo.py
```

## Model Architecture

### iLLuMinator Transformer
- **Vocabulary Size**: Configurable (default: varies by dataset)
- **Context Length**: 512 tokens (expandable)
- **Embedding Dimension**: 256
- **Attention Heads**: 8
- **Transformer Layers**: 6
- **Activation**: GELU
- **Normalization**: LayerNorm (pre-norm)

### Key Features
- **Causal Self-Attention**: Proper masking for autoregressive generation
- **Positional Encoding**: Learned position embeddings
- **Residual Connections**: Skip connections throughout the network
- **Dropout**: Regularization for training stability

## Training

### Basic Training
The model is trained on conversational data using next-token prediction:
```bash
python train.py
```

### Advanced RAG Training
Enhanced training with Q&A pairs and knowledge integration:
```bash
python train_intelligent.py
```

### Training Parameters
- **Learning Rate**: 1e-4 (AdamW optimizer)
- **Batch Size**: 4 (adjustable)
- **Epochs**: 10 (configurable)
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0

## RAG System

### Knowledge Base
The system includes a comprehensive knowledge base covering:
- Artificial Intelligence and Machine Learning
- Computer Science and Programming
- Mathematics and Statistics
- Science and Technology
- General Knowledge

### Retrieval Process
1. **Query Encoding**: Convert user query to embeddings
2. **Semantic Search**: Find relevant documents using FAISS
3. **Context Integration**: Combine retrieved knowledge with query
4. **Response Generation**: Generate contextually aware responses

### Retrieval Parameters
- **Top-K Documents**: 3 (default)
- **Similarity Threshold**: 0.1
- **Max Context Length**: 300 characters per document

## Configuration

### Model Configuration
```python
model = iLLuMinator(
    vocab_size=1000,        # Size of vocabulary
    block_size=512,         # Maximum sequence length
    n_embd=256,            # Embedding dimension
    n_head=8,              # Number of attention heads
    n_layer=6,             # Number of transformer layers
    dropout=0.1            # Dropout rate
)
```

### Generation Parameters
```python
generation_params = {
    "max_tokens": 150,      # Maximum tokens to generate
    "temperature": 0.7,     # Sampling temperature
    "top_k": 40            # Top-k sampling parameter
}
```

## Examples

### Basic Conversation
```
You: What is artificial intelligence?
Assistant: Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction capabilities that enable computers to perform tasks typically requiring human intelligence.

You: How does machine learning work?
Assistant: Machine learning works by training algorithms on data to recognize patterns and make predictions without explicit programming. The system learns from examples and improves its performance over time.
```

### Technical Questions
```
You: Explain neural networks
Assistant: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information through weighted connections and activation functions.

You: What is Python programming?
Assistant: Python is a high-level programming language known for its simplicity and versatility in web development, data science, and AI applications.
```

## Performance

### Model Statistics
- **Parameters**: ~10.6M (configurable)
- **Training Time**: ~10 minutes on CPU (basic model)
- **Inference Speed**: Real-time on modern hardware
- **Memory Usage**: ~100MB (model weights)

### Generation Quality
- **Coherent Responses**: Contextually relevant answers
- **Knowledge Integration**: Effective use of retrieved information
- **Conversation Flow**: Maintains context across interactions

## Troubleshooting

### Common Issues

#### Dimension Mismatch Errors
If you encounter dimension mismatches:
```bash
python fixed_assistant.py  # Uses compatible model loading
```

#### Tokenizer Issues
The system automatically falls back to a simple tokenizer if the main one fails.

#### Memory Issues
Reduce model size or batch size:
```python
model = iLLuMinator(
    vocab_size=1000,
    n_embd=128,    # Reduced from 256
    n_head=4,      # Reduced from 8
    n_layer=3      # Reduced from 6
)
```

### Performance Optimization
- Use GPU if available: `model.to('cuda')`
- Reduce sequence length for faster inference
- Implement model quantization for deployment

## Development

### Adding New Features
1. **Custom Knowledge Sources**: Add documents to the knowledge base
2. **Enhanced Tokenization**: Implement subword tokenization
3. **Multi-turn Conversations**: Extend conversation memory
4. **Domain Specialization**: Fine-tune on domain-specific data

### Testing
```bash
# Run basic tests
python -c "from model.transformer import iLLuMinator; print('Model loads successfully')"

# Test generation
python generate.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this work in your research, please cite:
```
@misc{illuminator2025,
  title={iLLuMinator: Intelligent RAG-Enhanced Transformer},
  author={Anish Paleja},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/Anipaleja/Generative_Transformer}}
}
```

## Acknowledgments

- Inspired by the Transformer architecture from "Attention Is All You Need"
- RAG methodology from "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Built with PyTorch and HuggingFace ecosystem

## Contact

For questions or support, please open an issue on GitHub or contact the maintainer.

---

**Note**: This is an educational and research project. For production use, consider additional optimizations and safety measures.