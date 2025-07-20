"""
Quantum-Enhanced Illuminator Model
Advanced quantum computing integration for next-generation AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import math
import warnings

class QuantumInspiredConfig(PretrainedConfig):
    """
    Quantum-Enhanced Illuminator Configuration with advanced quantum computing principles
    
    Incorporates quantum superposition, entanglement, and interference patterns
    for enhanced representation learning and computational efficiency.
    """
    
    model_type = "quantum_illuminator"
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 4096,
        n_embd: int = 2560,
        n_layer: int = 32,
        n_head: int = 32,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        summary_type: str = "cls_index",
        summary_use_proj: bool = True,
        summary_activation: Optional[str] = None,
        summary_proj_to_labels: bool = True,
        summary_first_dropout: float = 0.1,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
        # Quantum-specific parameters
        quantum_dim: int = 128,
        quantum_layers: int = 4,
        entanglement_strength: float = 0.8,
        superposition_rate: float = 0.3,
        coherence_time: int = 100,
        quantum_noise_level: float = 0.01,
        use_quantum_attention: bool = True,
        use_quantum_gates: bool = True,
        quantum_circuit_depth: int = 6,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        
        # Quantum parameters
        self.quantum_dim = quantum_dim
        self.quantum_layers = quantum_layers
        self.entanglement_strength = entanglement_strength
        self.superposition_rate = superposition_rate
        self.coherence_time = coherence_time
        self.quantum_noise_level = quantum_noise_level
        self.use_quantum_attention = use_quantum_attention
        self.use_quantum_gates = use_quantum_gates
        self.quantum_circuit_depth = quantum_circuit_depth
        
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

class QuantumGate(nn.Module):
    """
    Quantum Gate implementation using classical neural networks
    
    Simulates quantum gate operations including Pauli-X, Pauli-Y, Pauli-Z,
    Hadamard, and CNOT gates for quantum-inspired computations.
    """
    
    def __init__(self, dim: int, gate_type: str = "hadamard"):
        super().__init__()
        self.dim = dim
        self.gate_type = gate_type
        
        # Initialize quantum gate matrices
        if gate_type == "hadamard":
            self.gate_matrix = nn.Parameter(
                torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
            )
        elif gate_type == "pauli_x":
            self.gate_matrix = nn.Parameter(
                torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            )
        elif gate_type == "pauli_y":
            self.gate_matrix = nn.Parameter(
                torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex64)
            )
        elif gate_type == "pauli_z":
            self.gate_matrix = nn.Parameter(
                torch.tensor([[1.0, 0.0], [0.0, -1.0]])
            )
        else:
            # Learnable quantum gate
            self.gate_matrix = nn.Parameter(torch.randn(2, 2))
        
        self.phase_rotation = nn.Parameter(torch.zeros(dim))
        self.amplitude_scaling = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum gate transformation to input tensor
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Quantum-transformed tensor
        """
        batch_dims = x.shape[:-1]
        x = x.reshape(-1, self.dim)
        
        # Apply quantum superposition
        x_real = x * torch.cos(self.phase_rotation)
        x_imag = x * torch.sin(self.phase_rotation)
        
        # Quantum gate operation simulation
        if self.gate_type in ["hadamard", "pauli_x", "pauli_z"]:
            # Real-valued gates
            gate_real = self.gate_matrix.real if hasattr(self.gate_matrix, 'real') else self.gate_matrix
            x_transformed = torch.matmul(
                torch.stack([x_real, x_imag], dim=-1), 
                gate_real.expand(x.size(0), -1, -1)
            )
            x = x_transformed.sum(dim=-1) * self.amplitude_scaling
        else:
            # Complex-valued operations
            x = x * self.amplitude_scaling
        
        return x.reshape(*batch_dims, self.dim)

class QuantumEntanglementLayer(nn.Module):
    """
    Quantum Entanglement Layer implementing non-local correlations
    
    Creates entangled states between different positions in the sequence,
    enabling long-range dependencies and quantum interference effects.
    """
    
    def __init__(self, hidden_size: int, num_entangled_pairs: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_entangled_pairs = num_entangled_pairs
        
        # Entanglement creation networks
        self.entanglement_creator = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_entangled_pairs)
        ])
        
        # Bell state preparation
        self.bell_state_prep = nn.Linear(hidden_size * 2, hidden_size)
        
        # Measurement operators
        self.measurement_ops = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(4)  # |00⟩, |01⟩, |10⟩, |11⟩
        ])
        
        # Quantum decoherence simulation
        self.decoherence_rate = nn.Parameter(torch.tensor(0.01))
    
    def create_entangled_pairs(self, x: torch.Tensor) -> torch.Tensor:
        """Create quantum entangled pairs across sequence positions"""
        batch_size, seq_len, hidden_size = x.shape
        
        entangled_states = []
        for i in range(min(self.num_entangled_pairs, seq_len // 2)):
            # Select pairs of positions
            pos1, pos2 = 2 * i, 2 * i + 1
            if pos2 >= seq_len:
                break
                
            state1 = x[:, pos1, :]  # Shape: (batch_size, hidden_size)
            state2 = x[:, pos2, :]
            
            # Create entangled state |ψ⟩ = α|00⟩ + β|11⟩
            entangled_pair = self.entanglement_creator[i](state1) + self.entanglement_creator[i](state2)
            
            # Bell state preparation
            bell_input = torch.cat([state1, state2], dim=-1)
            bell_state = self.bell_state_prep(bell_input)
            
            # Superposition of entangled states
            entangled_states.append(bell_state)
        
        if entangled_states:
            return torch.stack(entangled_states, dim=1)  # (batch_size, num_pairs, hidden_size)
        return torch.zeros(batch_size, 1, hidden_size, device=x.device)
    
    def quantum_measurement(self, entangled_states: torch.Tensor) -> torch.Tensor:
        """Perform quantum measurement on entangled states"""
        measurements = []
        for op in self.measurement_ops:
            measurement = op(entangled_states).squeeze(-1)
            measurements.append(measurement)
        
        # Quantum probability amplitudes
        measurement_probs = F.softmax(torch.stack(measurements, dim=-1), dim=-1)
        
        # Expected measurement outcome
        return torch.sum(measurement_probs.unsqueeze(-1) * entangled_states.unsqueeze(-2), dim=-2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum entanglement to input sequence"""
        entangled_pairs = self.create_entangled_pairs(x)
        measured_states = self.quantum_measurement(entangled_pairs)
        
        # Decoherence simulation
        decoherence_noise = torch.randn_like(measured_states) * self.decoherence_rate
        measured_states = measured_states + decoherence_noise
        
        # Broadcast back to original sequence length
        batch_size, seq_len, hidden_size = x.shape
        if measured_states.size(1) < seq_len:
            # Pad or repeat to match sequence length
            padding_needed = seq_len - measured_states.size(1)
            padding = measured_states[:, -1:, :].repeat(1, padding_needed, 1)
            measured_states = torch.cat([measured_states, padding], dim=1)
        elif measured_states.size(1) > seq_len:
            measured_states = measured_states[:, :seq_len, :]
        
        return measured_states

class QuantumAttention(nn.Module):
    """
    Quantum-Enhanced Multi-Head Attention with superposition and interference
    
    Implements quantum superposition in attention patterns and quantum
    interference effects for enhanced representation learning.
    """
    
    def __init__(self, config: QuantumInspiredConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard attention components
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Quantum components
        self.quantum_gates = nn.ModuleList([
            QuantumGate(self.head_dim, gate_type="hadamard"),
            QuantumGate(self.head_dim, gate_type="pauli_x"),
            QuantumGate(self.head_dim, gate_type="pauli_z"),
        ])
        
        # Quantum superposition parameters
        self.superposition_weights = nn.Parameter(torch.ones(self.num_heads))
        self.quantum_phase = nn.Parameter(torch.zeros(self.num_heads))
        
        # Quantum interference modeling
        self.interference_strength = nn.Parameter(torch.tensor(config.entanglement_strength))
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
    
    def quantum_superposition_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        """Apply quantum superposition to attention computation"""
        
        # Standard attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Quantum superposition of attention patterns
        quantum_queries = []
        for i, gate in enumerate(self.quantum_gates):
            # Apply quantum gate to queries
            q_transformed = gate(query.reshape(-1, query.size(-1)))
            q_transformed = q_transformed.reshape_as(query)
            
            # Compute quantum attention scores
            quantum_attn = torch.matmul(q_transformed, key.transpose(-2, -1)) * self.scale
            quantum_queries.append(quantum_attn)
        
        # Quantum interference between different attention patterns
        if len(quantum_queries) > 1:
            interference_pattern = torch.zeros_like(attn_scores)
            for i in range(len(quantum_queries)):
                for j in range(i + 1, len(quantum_queries)):
                    # Quantum interference term
                    interference = (quantum_queries[i] * quantum_queries[j]).real \
                                 if hasattr(quantum_queries[i], 'real') else quantum_queries[i] * quantum_queries[j]
                    interference_pattern += interference * self.interference_strength
            
            attn_scores = attn_scores + interference_pattern / len(quantum_queries)
        
        return attn_scores
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        
        batch_size, seq_length, embed_dim = hidden_states.size()
        
        # Compute Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply quantum superposition attention
        attn_scores = self.quantum_superposition_attention(query, key, value)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax with quantum phase modulation
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply quantum phase to attention probabilities
        phase_modulation = torch.cos(self.quantum_phase).view(1, -1, 1, 1)
        attn_probs = attn_probs * phase_modulation
        
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        
        # Compute attention output
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, embed_dim)
        
        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_probs,)
        
        return outputs

class QuantumMLP(nn.Module):
    """
    Quantum-Enhanced MLP with quantum nonlinearities and superposition
    """
    
    def __init__(self, config: QuantumInspiredConfig):
        super().__init__()
        self.config = config
        embed_dim = config.n_embd
        
        self.c_fc = nn.Linear(embed_dim, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, embed_dim)
        self.act = self._get_activation_fn(config.activation_function)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # Quantum components
        self.quantum_nonlinearity = QuantumGate(config.n_inner, gate_type="hadamard")
        self.superposition_mixer = nn.Linear(config.n_inner, config.n_inner)
        
        # Quantum superposition strength
        self.superposition_strength = nn.Parameter(torch.tensor(config.superposition_rate))
    
    def _get_activation_fn(self, activation_function: str):
        if activation_function == "gelu":
            return F.gelu
        elif activation_function == "gelu_new":
            return lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        elif activation_function == "relu":
            return F.relu
        elif activation_function == "silu":
            return F.silu
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Standard MLP forward pass
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        
        # Quantum superposition enhancement
        quantum_component = self.quantum_nonlinearity(hidden_states)
        superposition_component = self.superposition_mixer(quantum_component)
        
        # Quantum interference between classical and quantum components
        hidden_states = (
            (1 - self.superposition_strength) * hidden_states +
            self.superposition_strength * superposition_component
        )
        
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

class QuantumTransformerBlock(nn.Module):
    """
    Quantum-Enhanced Transformer Block with quantum attention and MLP
    """
    
    def __init__(self, config: QuantumInspiredConfig):
        super().__init__()
        self.config = config
        hidden_size = config.n_embd
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        
        # Quantum attention
        self.attn = QuantumAttention(config)
        
        # Quantum MLP
        self.mlp = QuantumMLP(config)
        
        # Quantum entanglement layer
        if config.use_quantum_attention:
            self.quantum_entanglement = QuantumEntanglementLayer(
                hidden_size, 
                num_entangled_pairs=min(16, config.n_positions // 4)
            )
        else:
            self.quantum_entanglement = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Quantum attention
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        
        attn_output = attn_outputs[0]
        
        # Apply quantum entanglement if enabled
        if self.quantum_entanglement is not None:
            quantum_enhancement = self.quantum_entanglement(attn_output)
            attn_output = 0.8 * attn_output + 0.2 * quantum_enhancement
        
        # First residual connection
        hidden_states = attn_output + residual
        
        # MLP block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = feed_forward_hidden_states + residual
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += attn_outputs[1:]
        
        return outputs

class QuantumIlluminatorModel(PreTrainedModel):
    """
    Quantum-Enhanced Illuminator Model with advanced quantum computing integration
    
    Features:
    - Quantum superposition in attention mechanisms
    - Quantum entanglement between sequence positions
    - Quantum gates for nonlinear transformations
    - Quantum interference effects
    - Decoherence simulation for robustness
    """
    
    config_class = QuantumInspiredConfig
    
    def __init__(self, config: QuantumInspiredConfig):
        super().__init__(config)
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Quantum-enhanced transformer blocks
        self.h = nn.ModuleList([
            QuantumTransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Quantum circuit simulation components
        if config.use_quantum_gates:
            self.quantum_circuit = nn.ModuleList([
                QuantumGate(config.n_embd // 4, gate_type="hadamard"),
                QuantumGate(config.n_embd // 4, gate_type="pauli_x"),
                QuantumGate(config.n_embd // 4, gate_type="pauli_z"),
                QuantumGate(config.n_embd // 4, gate_type="learnable")
            ])
        
        # Initialize weights
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        # Input processing
        if input_ids is not None:
            batch_size, seq_length = input_ids.size()
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.size()[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Apply quantum circuit simulation if enabled
        if hasattr(self, 'quantum_circuit') and self.config.use_quantum_gates:
            # Split hidden states for quantum processing
            chunks = torch.chunk(hidden_states, len(self.quantum_circuit), dim=-1)
            quantum_processed = []
            for chunk, gate in zip(chunks, self.quantum_circuit):
                quantum_processed.append(gate(chunk))
            hidden_states = torch.cat(quantum_processed, dim=-1)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        
        # Transformer blocks
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
            
            hidden_states = outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)
        
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (lm_logits,)
            if output_attentions:
                output += (all_attentions,)
            if output_hidden_states:
                output += (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    def num_parameters(self, only_trainable: bool = True) -> int:
        """Calculate total number of parameters"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

# Register the model
QuantumInspiredConfig.register_for_auto_class()
QuantumIlluminatorModel.register_for_auto_class("AutoModel")
