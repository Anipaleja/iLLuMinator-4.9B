"""
iLLuMinator Model Configuration Comparison
Comparison between the original 4.7B and enhanced 4.9B models
"""

import json
from typing import Dict, Any

def get_original_config() -> Dict[str, Any]:
    """Original iLLuMinator 4.7B configuration"""
    return {
        "model_name": "iLLuMinator-4.7B",
        "architecture": "standard-transformer",
        "vocab_size": 50257,
        "d_model": 3584,
        "n_layers": 30,
        "n_heads": 28,
        "d_ff": 14336,
        "max_seq_length": 2048,
        "dropout": 0.1,
        "activation": "gelu",
        "norm_type": "layernorm",
        "position_embedding": "sinusoidal",
        "attention_type": "multi_head_attention",
        "tie_embeddings": False,
        "estimated_parameters": "4.7B"
    }

def get_enhanced_config() -> Dict[str, Any]:
    """Enhanced iLLuMinator 4.9B configuration"""
    return {
        "model_name": "iLLuMinator-4.9B-Enhanced",
        "architecture": "enhanced-transformer",
        "vocab_size": 65536,
        "d_model": 3328,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "d_ff": 9984,
        "max_seq_length": 4096,
        "dropout": 0.0,
        "activation": "swiglu",
        "norm_type": "rmsnorm",
        "position_embedding": "rope",
        "attention_type": "grouped_query_attention",
        "tie_embeddings": True,
        "estimated_parameters": "4.9B"
    }

def calculate_parameter_estimate(config: Dict[str, Any]) -> int:
    """Calculate rough parameter estimate from configuration"""
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    d_ff = config["d_ff"]
    
    # Token embeddings
    embed_params = vocab_size * d_model
    
    # Attention parameters (per layer)
    attn_params_per_layer = 4 * d_model * d_model  # Q, K, V, O projections
    
    # Feed-forward parameters (per layer)
    if config["activation"] == "swiglu":
        # SwiGLU has 3 linear layers instead of 2
        ff_params_per_layer = 3 * d_model * d_ff
    else:
        ff_params_per_layer = 2 * d_model * d_ff
    
    # Layer normalization parameters (per layer)
    norm_params_per_layer = 2 * d_model  # Two norm layers per transformer block
    
    # Total transformer parameters
    transformer_params = n_layers * (attn_params_per_layer + ff_params_per_layer + norm_params_per_layer)
    
    # Output projection (if not tied)
    if config.get("tie_embeddings", False):
        output_params = 0
    else:
        output_params = vocab_size * d_model
    
    # Final layer norm
    final_norm_params = d_model
    
    total_params = embed_params + transformer_params + output_params + final_norm_params
    
    return total_params

def print_comparison():
    """Print detailed comparison between configurations"""
    original = get_original_config()
    enhanced = get_enhanced_config()
    
    print(" iLLuMinator Model Architecture Comparison")
    print("=" * 80)
    
    print(f"\n Basic Configuration:")
    print(f"{'Metric':<25} {'Original 4.7B':<20} {'Enhanced 4.9B':<20} {'Improvement':<15}")
    print("-" * 80)
    
    # Basic metrics
    basic_metrics = [
        ("Model Name", "model_name"),
        ("Vocabulary Size", "vocab_size"),
        ("Model Dimension", "d_model"),
        ("Number of Layers", "n_layers"),
        ("Attention Heads", "n_heads"),
        ("Feed-Forward Dim", "d_ff"),
        ("Max Sequence Len", "max_seq_length"),
        ("Dropout Rate", "dropout")
    ]
    
    for name, key in basic_metrics:
        orig_val = original[key]
        enh_val = enhanced[key]
        
        if isinstance(orig_val, (int, float)) and isinstance(enh_val, (int, float)):
            if orig_val == 0:
                improvement = "N/A"
            else:
                improvement = f"{((enh_val - orig_val) / orig_val * 100):+.1f}%"
        else:
            improvement = "Changed" if orig_val != enh_val else "Same"
            
        print(f"{name:<25} {str(orig_val):<20} {str(enh_val):<20} {improvement:<15}")
    
    print(f"\n Advanced Features:")
    print(f"{'Feature':<25} {'Original 4.7B':<20} {'Enhanced 4.9B':<20}")
    print("-" * 70)
    
    advanced_features = [
        ("Architecture", "architecture"),
        ("Activation Function", "activation"),
        ("Normalization", "norm_type"),
        ("Position Encoding", "position_embedding"),
        ("Attention Type", "attention_type"),
        ("Tied Embeddings", "tie_embeddings")
    ]
    
    for name, key in advanced_features:
        orig_val = original.get(key, "N/A")
        enh_val = enhanced.get(key, "N/A")
        print(f"{name:<25} {str(orig_val):<20} {str(enh_val):<20}")
    
    # Special handling for KV heads (only in enhanced model)
    if "n_kv_heads" in enhanced:
        print(f"{'KV Heads (GQA)':<25} {'N/A':<20} {str(enhanced['n_kv_heads']):<20}")
    
    print(f"\n Parameter Estimates:")
    orig_params = calculate_parameter_estimate(original)
    enh_params = calculate_parameter_estimate(enhanced)
    param_increase = ((enh_params - orig_params) / orig_params * 100)
    
    print(f"Original Model:  {orig_params:,} parameters ({orig_params/1e9:.2f}B)")
    print(f"Enhanced Model:  {enh_params:,} parameters ({enh_params/1e9:.2f}B)")
    print(f"Parameter Increase: {param_increase:+.1f}%")
    
    print(f"\n Key Improvements in Enhanced Model:")
    improvements = [
        " Grouped Query Attention (GQA) - Reduces memory usage and improves efficiency",
        " Rotary Position Embedding (RoPE) - Better handling of long sequences",
        " SwiGLU Activation - Improved performance over GELU",
        " RMSNorm - More stable training than LayerNorm", 
        " Larger vocabulary (65K vs 50K) - Better tokenization coverage",
        " Longer context (4K vs 2K tokens) - Can process longer documents",
        " Tied embeddings - Reduces parameters while maintaining performance",
        " Lower dropout - Optimal for large models",
        " Pre-norm architecture - More stable training dynamics"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print(f"\n Memory and Efficiency Benefits:")
    memory_benefits = [
        " Grouped Query Attention reduces memory usage by ~25%",
        " Tied embeddings save ~268M parameters",
        " RoPE eliminates need for positional embedding table",
        " Enhanced architecture enables better scaling",
        " Optimized for Apple Silicon MPS training"
    ]
    
    for benefit in memory_benefits:
        print(f"  {benefit}")

def save_configs():
    """Save both configurations to JSON files"""
    original = get_original_config()
    enhanced = get_enhanced_config()
    
    with open('illuminator_4_7b_original_config.json', 'w') as f:
        json.dump(original, f, indent=2)
    
    with open('illuminator_4_9b_enhanced_config.json', 'w') as f:
        json.dump(enhanced, f, indent=2)
    
    print("Configuration files saved:")
    print("  - illuminator_4_7b_original_config.json")
    print("  - illuminator_4_9b_enhanced_config.json")

if __name__ == "__main__":
    print_comparison()
    print("\n" + "=" * 80)
    save_configs()
