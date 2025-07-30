# Simple Inference Script for 4.9B Model
# Load and test the trained model using original iLLuMinator architecture

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Import the original CUDA model
sys.path.append('legacy')
from illuminator_cuda import iLLuMinatorCUDA, create_cuda_model

def load_model(checkpoint_path: str):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with original iLLuMinator architecture
    model = create_cuda_model(vocab_size=50260)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, checkpoint.get('config', None)

def generate_text(model, prompt: str, max_length: int = 100, temperature: float = 0.8):
    """Generate text from prompt using original iLLuMinator model"""
    device = next(model.parameters()).device
    
    # Simple tokenization (you can replace with proper tokenizer)
    tokens = [ord(c) % model.config['vocab_size'] for c in prompt[:100]]
    tokens = [model.config['vocab_size'] - 1] + tokens  # BOS token
    
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"Generating text with prompt: '{prompt}'")
    print("Generated text:", end=" ")
    
    with torch.no_grad():
        # Use the model's generate method
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )
        
        # Decode and print
        for token_id in generated_ids[0][len(tokens):]:
            char_code = (token_id.item() - 4) % 256
            if 32 <= char_code <= 126:  # Printable ASCII
                print(chr(char_code), end="", flush=True)
            
            # Stop if we hit EOS
            if token_id.item() == 3:
                break
    
    print("\n")

def main():
    """Main inference function"""
    print("iLLuMinator 4.9B Model Inference")
    print("=" * 40)
    
    # Check for checkpoints
    checkpoint_dir = Path("checkpoints_4.9B")
    if not checkpoint_dir.exists():
        print("No checkpoints found. Please train the model first.")
        return
    
    # Find best checkpoint
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        print("No checkpoint files found in checkpoints_4.9B/")
        return
    
    # Prefer best model, then final, then any checkpoint
    checkpoint_path = None
    for name in ["best_4.9B_model.pt", "final_4.9B_model.pt"]:
        path = checkpoint_dir / name
        if path.exists():
            checkpoint_path = str(path)
            break
    
    if not checkpoint_path:
        checkpoint_path = str(checkpoint_files[0])
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load model
    try:
        model, config = load_model(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Explain quantum computing",
        "Write a Python function to",
        "The benefits of renewable energy",
        "Machine learning is"
    ]
    
    print("\nGenerating text samples:")
    print("-" * 40)
    
    for prompt in test_prompts:
        try:
            generate_text(model, prompt, max_length=50)
            print()
        except Exception as e:
            print(f"Error generating text: {e}")
    
    print("Inference complete!")

if __name__ == "__main__":
    main()
