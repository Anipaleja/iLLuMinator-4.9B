#!/usr/bin/env python3
"""
Example script demonstrating RAG capabilities with iLLuMinator
"""

import torch
from rag_system import RAGSystem
from model.transformer import iLLuMinator
from model.tokenizer import build_tokenizer
from data.prepare import load_data

def setup_rag_demo():
    """Setup a complete RAG demo with sample knowledge base"""
    
    # Sample knowledge base - in practice, this could be loaded from files/database
    knowledge_base = [
        "The Transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It revolutionized natural language processing by using self-attention mechanisms instead of recurrent or convolutional layers.",
        
        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model developed by Google. It uses bidirectional training to understand context from both left and right sides of a token.",
        
        "GPT (Generative Pre-trained Transformer) is an autoregressive language model that generates text by predicting the next token in a sequence. GPT models are trained on large amounts of text data.",
        
        "Attention mechanisms allow neural networks to focus on relevant parts of the input when making predictions. Self-attention computes attention weights between all pairs of positions in a sequence.",
        
        "Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based approaches. It retrieves relevant documents and uses them as context for generating responses.",
        
        "Fine-tuning is the process of adapting a pre-trained model to a specific task by training it on task-specific data with a lower learning rate.",
        
        "Tokenization is the process of breaking down text into smaller units called tokens, which can be words, subwords, or characters. Modern NLP models often use subword tokenization like Byte-Pair Encoding (BPE).",
        
        "The positional encoding in transformers provides information about the position of tokens in a sequence, since the attention mechanism itself is permutation-invariant."
    ]
    
    print("🚀 Setting up RAG system...")
    
    # Initialize RAG system
    rag = RAGSystem(
        documents=knowledge_base,
        top_k=3,
        max_context_length=200
    )
    
    return rag

def demo_retrieval_only(rag: RAGSystem):
    """Demo just the retrieval component"""
    print("\n" + "="*50)
    print("📚 RETRIEVAL DEMO")
    print("="*50)
    
    queries = [
        "What is the transformer architecture?",
        "How does attention work?",
        "What is RAG?",
        "Explain tokenization"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = rag.retrieve(query)
        
        print("📖 Retrieved documents:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. (Score: {score:.3f}) {doc[:100]}...")

def demo_full_rag(rag: RAGSystem):
    """Demo full RAG with generation (requires trained model)"""
    print("\n" + "="*50)
    print("🤖 FULL RAG DEMO (Generation)")
    print("="*50)
    
    # Check if model exists
    try:
        # This would load your trained model
        # rag.load_generator('illuminator.pth', 'data_path_for_vocab')
        print("⚠️  To use generation, uncomment the model loading line above")
        print("⚠️  Make sure you have a trained iLLuMinator model saved")
        return
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("💡 Train your model first using train.py")
        return
    
    # Example queries for generation
    queries = [
        "What is a transformer and how does it work?",
        "Explain the difference between BERT and GPT",
        "How can I use RAG for my chatbot?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        
        # Get full RAG response
        response = rag.chat(query)
        
        print("📖 Retrieved contexts:")
        for i, (doc, score) in enumerate(response['retrieved_documents'], 1):
            print(f"  {i}. (Score: {score:.3f}) {doc[:80]}...")
        
        print(f"\n🤖 Generated answer:")
        print(f"  {response['answer']}")

def demo_context_preparation(rag: RAGSystem):
    """Demo how context is prepared for generation"""
    print("\n" + "="*50)
    print("📝 CONTEXT PREPARATION DEMO")
    print("="*50)
    
    query = "What is the transformer architecture?"
    retrieved_docs = rag.retrieve(query)
    context = rag.prepare_context(retrieved_docs, query)
    
    print(f"🔍 Query: {query}")
    print(f"\n📄 Prepared context for generation:")
    print("-" * 40)
    print(context)
    print("-" * 40)
    print(f"📊 Context length: {len(context)} characters")

def enhanced_generation_demo():
    """Demo the enhanced transformer capabilities"""
    print("\n" + "="*50)
    print("⚡ ENHANCED TRANSFORMER DEMO")
    print("="*50)
    
    # Create a small model for demo
    vocab_size = 1000
    model = iLLuMinator(
        vocab_size=vocab_size,
        block_size=512,
        n_embd=128,  # Smaller for demo
        n_head=4,
        n_layer=2
    )
    
    print(f"✨ Model created with:")
    print(f"   - Block size: {model.block_size}")
    print(f"   - Embedding dim: {model.n_embd}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Demo input
    batch_size, seq_len = 2, 64
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\n🔧 Forward pass demo:")
    print(f"   - Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output represents logits over {vocab_size} vocabulary tokens")

if __name__ == "__main__":
    print("🌟 iLLuMinator RAG System Demo")
    print("=" * 50)
    
    # Setup RAG system
    rag = setup_rag_demo()
    
    # Run demos
    demo_retrieval_only(rag)
    demo_context_preparation(rag)
    enhanced_generation_demo()
    
    print("\n" + "="*50)
    print("✅ Demo complete!")
    print("\n💡 Next steps:")
    print("   1. Train your iLLuMinator model using train.py")
    print("   2. Uncomment model loading in demo_full_rag()")
    print("   3. Try the full RAG generation pipeline")
    print("   4. Experiment with your own documents and queries")
    
    # Uncomment to try full RAG (requires trained model)
    # demo_full_rag(rag)
