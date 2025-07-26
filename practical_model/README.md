# iLLuMinator Practical Model (120M Parameters)

This folder contains the practical, lightweight version of iLLuMinator designed to run efficiently on consumer hardware.

## 📊 Model Specifications

- **Parameters**: 124.4 million
- **Architecture**: 12-layer transformer
- **Attention Heads**: 12
- **Hidden Dimension**: 768
- **Context Length**: 1024 tokens
- **Memory Usage**: ~500MB RAM
- **Target Hardware**: CPU, integrated GPUs, low-end hardware

## 🚀 Quick Start

### 1. Test the Model
```bash
python practical_ai.py
```

### 2. Train the Model
```bash
python train_practical.py
```

### 3. Run API Server
```bash
python practical_api_server.py
```

### 4. Interactive Chat
```bash
python interactive_client.py
```

## 📁 Files

- `illuminator_practical.py` - 120M parameter transformer model
- `practical_ai.py` - Complete AI system wrapper
- `train_practical.py` - Training script with conversational data
- `practical_api_server.py` - FastAPI server (port 8001)
- `interactive_client.py` - Command-line chat interface
- `illuminator_practical_weights.pth` - Trained model weights

## ⚡ Performance

- **Generation Speed**: 10-12 tokens/second (CPU)
- **Response Time**: 2-4 seconds for short responses
- **Memory Usage**: ~500MB RAM
- **Training Time**: ~5 minutes for 10 epochs

## 🎯 Use Cases

- **Development & Testing**: Quick prototyping and testing
- **Resource-Constrained Environments**: Laptops, edge devices
- **Educational Purposes**: Learning about transformers
- **Fallback System**: When the large model is unavailable
- **Local AI Assistant**: Personal productivity tool

## 🔧 Configuration

The practical model uses efficient architecture choices:

```python
config = {
    'vocab_size': 50260,
    'd_model': 768,          # Smaller hidden dimension
    'num_layers': 12,        # Fewer layers
    'num_heads': 12,         # Fewer attention heads
    'd_ff': 3072,           # Smaller feed-forward
    'max_seq_length': 1024,  # Shorter context
    'dropout': 0.1,
    'weight_tying': True     # Shared input/output embeddings
}
```

## 📈 Training Data

Currently trained on:
- 20+ conversational examples
- Python programming Q&A
- Basic AI explanations
- Code examples and functions
- Greeting and help responses

## 🚀 API Usage

### Start Server
```bash
python practical_api_server.py
# Server: http://localhost:8001
# Docs: http://localhost:8001/docs
```

### Chat Example
```bash
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "max_tokens": 50}'
```

### Health Check
```bash
curl http://localhost:8001/health
```

## 🎓 Educational Value

This practical model demonstrates:
- Complete transformer implementation
- Training pipeline from scratch
- API server development
- Interactive chat interfaces
- Model optimization techniques

## 🔄 Relationship to Main Model

The practical model serves as:
- **Development Environment**: Test changes quickly
- **Fallback System**: When main model unavailable
- **Educational Tool**: Understand transformer concepts
- **Baseline Comparison**: Performance benchmarking
- **Production Prototype**: Rapid deployment testing

## 💡 Future Improvements

- [ ] Expand training dataset
- [ ] Add more specialized training data
- [ ] Implement conversation memory
- [ ] Add streaming responses
- [ ] Create model variants for specific tasks
- [ ] Optimize inference speed further

## 🔧 Customization

To modify the model:

1. **Change Architecture**: Edit `illuminator_practical.py`
2. **Add Training Data**: Modify `train_practical.py`
3. **Customize API**: Update `practical_api_server.py`
4. **Adjust Parameters**: Configure in respective files

## 🎯 Performance Tips

- **CPU Optimization**: Use Intel MKL or OpenBLAS
- **Memory Efficiency**: Enable gradient checkpointing
- **Batch Processing**: Process multiple requests together
- **Caching**: Implement response caching for common queries
- **Quantization**: Use INT8 for further size reduction

This practical model provides a complete, working AI system that you can run immediately while the larger 4.9B parameter model is being set up on more powerful hardware.
