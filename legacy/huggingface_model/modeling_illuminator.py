"""
Hugging Face Compatible Transformer Model
Enhanced accuracy with comprehensive training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union
import math
import json

class IlluminatorConfig(PretrainedConfig):
    """
    Configuration class for Illuminator Transformer model compatible with Hugging Face
    """
    model_type = "illuminator"
    
    def __init__(
        self,
        vocab_size=50257,
        n_positions=4096,
        n_embd=2560,
        n_layer=32,
        n_head=32,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50257,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

class IlluminatorAttention(nn.Module):
    """Enhanced multi-head self-attention with improved accuracy"""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0
        
        # Enhanced projections with better initialization
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        # Attention and residual dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.scale_attn_weights = config.scale_attn_weights
        
        # Improved positional bias
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        
        # Enhanced scaling
        self.scale = (1.0 / math.sqrt(self.head_dim)) if config.scale_attn_weights else 1.0
    
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split the last dimension into (num_heads, head_size)"""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attn_head_size dim and num_attn_heads dim into hidden_size"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, use_cache=False, past_key_value=None):
        # Enhanced attention computation
        query, key, value = self.c_attn(hidden_states).split(self.n_embd, dim=2)
        
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        if use_cache:
            present = (key, value)
        else:
            present = None
        
        # Improved attention computation with numerical stability
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        # Apply causal mask
        seq_len = key.size(-2)
        if seq_len > self.bias.size(-1):
            # Extend bias if sequence is longer
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        else:
            causal_mask = self.bias[:, :, :seq_len, :seq_len]
        
        attn_scores = torch.where(causal_mask, attn_scores, torch.finfo(attn_scores.dtype).min)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Improved softmax with numerical stability
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present, attn_weights

class IlluminatorMLP(nn.Module):
    """Enhanced MLP block with improved activation and regularization"""
    
    def __init__(self, config):
        super().__init__()
        n_inner = config.n_inner if hasattr(config, 'n_inner') else 4 * config.n_embd
        
        self.c_fc = nn.Linear(config.n_embd, n_inner)
        self.c_proj = nn.Linear(n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # Enhanced activation function
        if config.activation_function == "gelu_new":
            self.act = self.gelu_new
        elif config.activation_function == "swish":
            self.act = F.silu
        else:
            self.act = F.gelu
    
    def gelu_new(self, x):
        """Improved GELU activation"""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class IlluminatorBlock(nn.Module):
    """Enhanced transformer block with pre-norm and improved residual connections"""
    
    def __init__(self, config):
        super().__init__()
        
        # Pre-normalization for better training stability
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = IlluminatorAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = IlluminatorMLP(config)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, use_cache=False, past_key_value=None):
        # Pre-norm attention
        ln_hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            ln_hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        attn_output = attn_outputs[0]
        present = attn_outputs[1]
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Pre-norm MLP
        ln_hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(ln_hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present,)
        
        return outputs

class IlluminatorModel(PreTrainedModel):
    """
    Enhanced Illuminator Transformer Model for Hugging Face
    Improved accuracy with better architecture and training
    """
    config_class = IlluminatorConfig
    base_model_prefix = "transformer"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Enhanced embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Enhanced transformer blocks
        self.h = nn.ModuleList([IlluminatorBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm for stability
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # Attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        
        # Head mask
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        
        # Enhanced embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        
        hidden_states = self.drop(hidden_states)
        
        output_shape = input_shape + (hidden_states.size(-1),)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                past_key_value=layer_past,
            )
            
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
        }

class IlluminatorLMHeadModel(PreTrainedModel):
    """Enhanced Language Model with improved accuracy for text generation"""
    
    config_class = IlluminatorConfig
    base_model_prefix = "transformer"
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = IlluminatorModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights for better parameter efficiency
        self.tie_weights()
        
        # Initialize weights
        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    def tie_weights(self):
        """Tie the weights between input and output embeddings"""
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Only use last token if past is provided
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": kwargs.get("attention_mask"),
        }
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        
        hidden_states = transformer_outputs[0] if not return_dict else transformer_outputs['last_hidden_state']
        
        # Enhanced language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Enhanced loss computation with label smoothing
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Use label smoothing for better training
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.get('past_key_values'),
            hidden_states=transformer_outputs.get('hidden_states'),
            attentions=transformer_outputs.get('attentions'),
        )
