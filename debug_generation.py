#!/usr/bin/env python3
"""
Debug script to test token generation step by step
"""

import sys
import torch
import torch.nn.functional as F
from illuminator_ai import IlluminatorAI

def debug_generation():
    print("Loading iLLuMinator AI model...")
    
    try:
        # Initialize with fast mode
        ai = IlluminatorAI(fast_mode=True, auto_enhance=False)
        
        print(f"Model loaded successfully!")
        
        # Test prompt
        prompt = "Hello"
        full_prompt = ai._prepare_prompt(prompt)
        print(f"\nFull prompt: '{full_prompt}'")
        
        # Debug tokenization
        input_ids = ai.tokenizer.encode(full_prompt, max_length=512)
        print(f"Input token IDs: {input_ids}")
        print(f"Decoded input: '{ai.tokenizer.decode(input_ids)}'")
        
        input_tensor = torch.tensor([input_ids], device=ai.device)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Try generating one token step by step
        print("\n--- Manual token generation ---")
        generated = input_tensor.clone()
        
        for step in range(5):  # Try generating 5 tokens
            print(f"\nStep {step + 1}:")
            
            with torch.no_grad():
                try:
                    logits = ai.model(generated)
                    print(f"  Logits shape: {logits.shape}")
                    
                    next_token_logits = logits[0, -1, :]
                    print(f"  Next token logits shape: {next_token_logits.shape}")
                    print(f"  Logits min/max: {next_token_logits.min().item():.4f} / {next_token_logits.max().item():.4f}")
                    
                    # Apply temperature
                    temperature = 0.7
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    print(f"  Probs min/max: {probs.min().item():.6f} / {probs.max().item():.6f}")
                    
                    next_token = torch.multinomial(probs, num_samples=1)
                    print(f"  Next token ID: {next_token.item()}")
                    
                    # Decode token
                    if next_token.item() in ai.tokenizer.id_to_token:
                        token_str = ai.tokenizer.id_to_token[next_token.item()]
                        print(f"  Next token: '{token_str}'")
                    else:
                        print(f"  Token ID {next_token.item()} not in vocabulary")
                    
                    # Append token
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    
                    # Decode current generation
                    current_output = ai.tokenizer.decode(generated[0][len(input_ids):].tolist())
                    print(f"  Current output: '{current_output}'")
                    
                except Exception as e:
                    print(f"  Error in step {step + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        # Final output
        final_output = ai.tokenizer.decode(generated[0][len(input_ids):].tolist())
        print(f"\nFinal generated text: '{final_output}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_generation()
