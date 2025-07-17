#!/usr/bin/env python3
"""
Setup script for iLLuMinator RAG system
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def verify_installation():
    """Verify that key packages are installed"""
    print("\n🔍 Verifying installation...")
    
    packages = [
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy")
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - Missing")
            all_good = False
    
    return all_good

def run_basic_test():
    """Run a basic test of the system"""
    print("\n🧪 Running basic test...")
    
    try:
        # Test enhanced transformer
        import torch
        from model.transformer import iLLuMinator
        
        model = iLLuMinator(vocab_size=1000, block_size=64, n_embd=32, n_head=2, n_layer=1)
        test_input = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Transformer test passed - Output shape: {output.shape}")
        
        # Test RAG system (retrieval only)
        from rag_system import RAGSystem
        
        docs = ["Test document 1", "Test document 2"]
        rag = RAGSystem(documents=docs)
        results = rag.retrieve("test query")
        
        print(f"✅ RAG retrieval test passed - Retrieved {len(results)} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🌟 iLLuMinator RAG System Setup")
    print("=" * 40)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup failed at verification")
        print("💡 Try running: pip install -r requirements.txt")
        return
    
    # Run basic test
    if not run_basic_test():
        print("❌ Setup failed at testing")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Train your model: python train.py")
    print("2. Run the demo: python rag_demo.py")
    print("3. Try the chatbot: python chatbot.py")
    print("\n💡 For RAG generation, make sure to train your model first!")

if __name__ == "__main__":
    main()
