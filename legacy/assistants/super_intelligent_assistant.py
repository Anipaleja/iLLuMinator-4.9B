import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from typing import Optional, Tuple, List, Dict, Any

class AdvancedRotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) like in LLaMA for better position understanding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute rotation matrices for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        self._update_cache(seq_len, x.device)
        
        cos = self._cos_cached[:seq_len, :]
        sin = self._sin_cached[:seq_len, :]
        
        return cos, sin

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Rotate
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return rotated

class AdvancedMultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with RoPE, better scaling, and optimizations"""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        
        # Multi-head projections with better initialization
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Rotary embeddings
        if self.use_rope:
            self.rotary_emb = AdvancedRotaryEmbedding(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.flash_attn = hasattr(F, 'scaled_dot_product_attention')  # PyTorch 2.0+ optimization
        
        # Initialize weights with better scaling
        self._init_weights()
    
    def _init_weights(self):
        # Xavier/Glorot initialization for better convergence
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        
        # Apply rotary embeddings
        if self.use_rope:
            cos, sin = self.rotary_emb(q, T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        # Efficient attention computation
        if self.flash_attn and mask is None:
            # Use PyTorch's optimized attention (Flash Attention when available)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0)
        else:
            # Manual attention with causal masking
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            
            # Apply causal mask
            if mask is None:
                mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v  # (B, nh, T, hd)
        
        # Reassemble and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.o_proj(y)
        
        return y

class AdvancedFeedForward(nn.Module):
    """Enhanced Feed-Forward Network with SwiGLU activation (like LLaMA)"""
    
    def __init__(self, n_embd: int, dropout: float = 0.1, ff_mult: float = 8/3):
        super().__init__()
        
        # Calculate hidden dimension (LLaMA uses 8/3 * n_embd, rounded to nearest multiple of 256)
        hidden_dim = int(ff_mult * n_embd)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Round to nearest 256
        
        # SwiGLU: split linear layer for gating
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with proper scaling
        for module in [self.gate_proj, self.up_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1/math.sqrt(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))  # SiLU activation (Swish)
        up = self.up_proj(x)
        hidden = gate * up  # Element-wise multiplication
        
        output = self.down_proj(hidden)
        return self.dropout(output)

class AdvancedTransformerBlock(nn.Module):
    """Enhanced Transformer Block with pre-norm, better residuals, and optimizations"""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        
        # Pre-layer normalization (like in modern transformers)
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = AdvancedMultiHeadAttention(n_embd, n_head, dropout)
        
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.mlp = AdvancedFeedForward(n_embd, dropout)
        
        # Learnable residual scaling (helps with training stability)
        self.ls1 = nn.Parameter(torch.ones(n_embd) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(n_embd) * 0.1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm with learnable scaling
        x = x + self.ls1 * self.attn(self.ln1(x), mask)
        x = x + self.ls2 * self.mlp(self.ln2(x))
        return x

class IntelligentTokenizer:
    """Advanced tokenizer with better vocabulary and subword handling"""
    
    def __init__(self):
        self.setup_enhanced_vocabulary()
    
    def setup_enhanced_vocabulary(self):
        """Create a comprehensive vocabulary for better understanding"""
        
        # Special tokens
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]
        
        # Common English words (high frequency)
        common_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might", "must", "shall",
            "this", "that", "these", "those", "what", "when", "where", "why", "how", "who", "which",
            "not", "no", "yes", "all", "any", "some", "many", "much", "more", "most", "few", "little",
            "one", "two", "three", "first", "second", "last", "next", "other", "same", "different",
            "big", "small", "large", "great", "good", "bad", "best", "better", "new", "old", "young",
            "long", "short", "high", "low", "right", "wrong", "true", "false", "real", "simple", "hard"
        ]
        
        # Technical and AI vocabulary
        tech_vocab = [
            # AI/ML Terms
            "artificial", "intelligence", "machine", "learning", "deep", "neural", "network", "algorithm",
            "model", "training", "data", "dataset", "feature", "prediction", "classification", "regression",
            "supervised", "unsupervised", "reinforcement", "attention", "transformer", "embedding",
            "gradient", "optimization", "backpropagation", "epoch", "batch", "loss", "accuracy",
            "overfitting", "underfitting", "validation", "testing", "cross", "validation",
            
            # Programming
            "programming", "code", "software", "computer", "system", "function", "method", "class",
            "variable", "parameter", "argument", "return", "loop", "condition", "if", "else", "while",
            "for", "import", "from", "def", "class", "try", "except", "error", "debug",
            
            # Libraries and frameworks
            "python", "pytorch", "tensorflow", "numpy", "pandas", "matplotlib", "scikit", "learn",
            "transformers", "huggingface", "openai", "chatgpt", "gpt", "bert", "llama",
            
            # Science and math
            "mathematics", "statistics", "probability", "calculus", "linear", "algebra", "matrix",
            "vector", "dimension", "space", "function", "equation", "formula", "theorem",
            "analysis", "research", "experiment", "hypothesis", "theory", "scientific"
        ]
        
        # Action and descriptive words
        action_words = [
            "create", "build", "make", "develop", "design", "implement", "write", "read", "understand",
            "learn", "teach", "explain", "describe", "analyze", "process", "compute", "calculate",
            "predict", "generate", "produce", "output", "input", "transform", "convert", "change",
            "improve", "optimize", "enhance", "increase", "decrease", "reduce", "maximize", "minimize"
        ]
        
        # Domain-specific terms
        domain_terms = [
            "natural", "language", "processing", "nlp", "computer", "vision", "speech", "recognition",
            "robotics", "automation", "cognitive", "semantic", "syntactic", "linguistic", "grammar",
            "syntax", "vocabulary", "word", "sentence", "paragraph", "document", "text", "corpus",
            "token", "tokenization", "embedding", "representation", "encoding", "decoding"
        ]
        
        # Numbers and quantifiers
        numbers = [str(i) for i in range(100)] + ["hundred", "thousand", "million", "billion"]
        
        # Punctuation and symbols
        punctuation = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_", "+", "=", "*", "/", "%", "@", "#", "$"]
        
        # Combine all vocabularies
        all_tokens = (special_tokens + common_words + tech_vocab + action_words + 
                     domain_terms + numbers + punctuation)
        
        # Remove duplicates and create mappings
        self.vocab = list(set(all_tokens))
        self.vocab.sort()  # Sort for consistency
        
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        print(f"Intelligent tokenizer initialized with {self.vocab_size} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Enhanced encoding with better word handling"""
        # Normalize text
        text = text.lower().strip()
        
        # Handle punctuation
        for punct in ".,!?:;()[]{}":
            text = text.replace(punct, f" {punct} ")
        
        # Split into tokens
        tokens = text.split()
        
        # Convert to IDs with intelligent fallback
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Try to find partial matches
                found = False
                for vocab_token in self.vocab:
                    if len(vocab_token) > 3 and vocab_token in token:
                        token_ids.append(self.token_to_id[vocab_token])
                        found = True
                        break
                
                if not found:
                    token_ids.append(self.token_to_id["<UNK>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]:
                    tokens.append(token)
        
        # Join and clean up
        text = " ".join(tokens)
        
        # Fix punctuation spacing
        for punct in ".,!?:;":
            text = text.replace(f" {punct}", punct)
        for punct in "()[]{}":
            text = text.replace("( ", "(").replace(" )", ")")
            text = text.replace("[ ", "[").replace(" ]", "]")
            text = text.replace("{ ", "{").replace(" }", "}")
        
        return text.strip()

class SuperIntelligentTransformer(nn.Module):
    """Advanced transformer model with LLaMA-level architecture and intelligence"""
    
    def __init__(self, vocab_size: int, block_size: int = 2048, n_embd: int = 768, 
                 n_head: int = 12, n_layer: int = 12, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        
        # Enhanced embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-5)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Tie embeddings (weight sharing like in modern LLMs)
        self.head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output layer
        with torch.no_grad():
            self.head.weight *= 0.5  # Scale down output layer
    
    def _init_weights(self, module):
        """Improved weight initialization"""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings
            nn.init.xavier_uniform_(module.weight, gain=1.0)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm initialization
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional loss calculation"""
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Token embeddings
        x = self.token_embedding(idx)  # (B, T, n_embd)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output logits
        logits = self.head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        """Advanced generation with multiple sampling strategies"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Focus on last token, apply temperature
            
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Apply the mask
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

class SuperSmartAssistant:
    """The most intelligent assistant implementation"""
    
    def __init__(self, model_path: str = "illuminator.pth"):
        print("🧠 Super Intelligent Assistant")
        print("=" * 60)
        print("LLaMA-level intelligence with advanced reasoning!")
        print("Type 'quit' to exit")
        print("=" * 60)
        
        self.tokenizer = IntelligentTokenizer()
        self.setup_knowledge_base()
        self.load_model(model_path)
        
        print("\n🚀 Super Intelligent Assistant ready!")
        print("Ask me anything - I can handle complex reasoning!")
    
    def setup_knowledge_base(self):
        """Comprehensive knowledge base with detailed explanations"""
        self.knowledge = {
            # AI and Machine Learning
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think, learn, and problem-solve like humans. It encompasses machine learning, natural language processing, computer vision, robotics, and expert systems. AI systems can perceive their environment, process information, learn from experience, and make decisions to achieve specific goals.",
            
            "machine learning": "Machine Learning is a subset of AI that enables computers to automatically learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Key types include supervised learning (with labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through rewards and penalties).",
            
            "deep learning": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep networks) to model and understand complex patterns in data. It excels at tasks like image recognition, natural language processing, and speech recognition. Deep learning automatically learns hierarchical representations of data, from simple features to complex concepts.",
            
            "neural networks": "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections and activation functions. Information flows from input layer through hidden layers to output layer, with weights adjusted during training to minimize errors and improve performance.",
            
            "transformer": "Transformers are a neural network architecture that uses self-attention mechanisms to process sequential data in parallel rather than sequentially. They excel at natural language processing tasks and form the basis of models like GPT, BERT, and T5. The key innovation is the attention mechanism that allows the model to focus on relevant parts of the input when generating outputs.",
            
            "attention mechanism": "Attention mechanisms allow neural networks to focus on relevant parts of the input when making predictions. In transformers, self-attention computes relationships between all positions in a sequence, enabling the model to understand context and dependencies regardless of distance. This parallel processing makes transformers more efficient than recurrent networks.",
            
            # Programming and Technology
            "python": "Python is a high-level, interpreted programming language known for its simplicity, readability, and versatility. It's widely used in web development, data science, machine learning, automation, and scientific computing. Python's extensive library ecosystem, including NumPy, Pandas, PyTorch, and TensorFlow, makes it the preferred language for AI development.",
            
            "pytorch": "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides dynamic computation graphs, automatic differentiation, and GPU acceleration. PyTorch is popular for research due to its flexibility, ease of debugging, and pythonic interface. It's widely used for deep learning, computer vision, and natural language processing.",
            
            "tensorflow": "TensorFlow is an open-source machine learning framework developed by Google. It provides comprehensive tools for building and deploying machine learning models at scale. TensorFlow supports both eager execution and graph execution, offers robust production deployment capabilities, and includes TensorBoard for visualization.",
            
            "algorithm": "An algorithm is a step-by-step procedure or set of rules for solving a problem or completing a task. In computer science, algorithms define how to process data, make decisions, and achieve desired outcomes. They are fundamental to programming and vary in efficiency, complexity, and application domain.",
            
            # Data Science
            "data science": "Data Science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, machine learning, domain expertise, and programming to solve complex problems and inform decision-making across various industries.",
            
            "big data": "Big Data refers to extremely large, complex datasets that require specialized tools and techniques to store, process, and analyze effectively. It's characterized by the 4 Vs: Volume (large amounts), Velocity (high speed), Variety (different types), and Veracity (data quality). Technologies like Hadoop, Spark, and cloud computing enable big data processing.",
            
            # Advanced Concepts
            "reinforcement learning": "Reinforcement Learning is a machine learning paradigm where agents learn optimal actions through trial-and-error interactions with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward over time. It's used in game playing, robotics, autonomous vehicles, and recommendation systems.",
            
            "natural language processing": "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language. It involves text processing, sentiment analysis, machine translation, question answering, and language generation. Modern NLP uses deep learning and transformer architectures for state-of-the-art performance.",
            
            "computer vision": "Computer Vision is an AI field that trains computers to interpret and understand visual information from the world. It involves image processing, object detection, facial recognition, and scene understanding. Applications include autonomous vehicles, medical imaging, surveillance, and augmented reality.",
            
            "llama": "LLaMA (Large Language Model Meta AI) is a family of foundation language models developed by Meta. These models are designed to be more efficient and performant than previous large language models, using techniques like rotary positional embeddings and SwiGLU activation functions. LLaMA models demonstrate strong performance across various natural language tasks."
        }
        
        # Create search index for better matching
        self.create_search_index()
    
    def create_search_index(self):
        """Create search index for intelligent knowledge retrieval"""
        self.search_index = {}
        
        for topic, description in self.knowledge.items():
            # Index by topic and key terms
            words = topic.split() + description.lower().split()
            for word in words:
                if len(word) > 3:  # Only index meaningful words
                    if word not in self.search_index:
                        self.search_index[word] = []
                    if topic not in self.search_index[word]:
                        self.search_index[word].append(topic)
    
    def find_relevant_knowledge(self, query: str) -> List[Tuple[str, str, float]]:
        """Advanced knowledge retrieval with relevance scoring"""
        query_words = query.lower().split()
        topic_scores = {}
        
        for word in query_words:
            if word in self.search_index:
                for topic in self.search_index[word]:
                    if topic not in topic_scores:
                        topic_scores[topic] = 0
                    topic_scores[topic] += 1
        
        # Boost exact matches
        for topic in self.knowledge:
            if topic in query.lower():
                topic_scores[topic] = topic_scores.get(topic, 0) + 10
        
        # Sort by relevance
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(topic, self.knowledge[topic], score) for topic, score in sorted_topics[:3]]
    
    def load_model(self, model_path: str):
        """Load the intelligent model"""
        try:
            # Create advanced model
            self.model = SuperIntelligentTransformer(
                vocab_size=self.tokenizer.vocab_size,
                block_size=1024,  # Larger context
                n_embd=768,       # Larger embeddings
                n_head=12,        # More attention heads
                n_layer=12,       # Deeper network
                dropout=0.1
            )
            
            print(f"Created advanced model with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
            
            # Try to load existing weights if available
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                # Only load compatible weights
                compatible_weights = {}
                for key, value in state_dict.items():
                    if key in self.model.state_dict():
                        if value.shape == self.model.state_dict()[key].shape:
                            compatible_weights[key] = value
                
                if compatible_weights:
                    self.model.load_state_dict(compatible_weights, strict=False)
                    print(f"Loaded {len(compatible_weights)} compatible weight tensors")
            
            self.model.eval()
            
        except Exception as e:
            print(f"Model loading error: {e}")
            # Create minimal fallback
            self.model = None
    
    def generate_intelligent_response(self, query: str) -> str:
        """Generate highly intelligent responses using advanced reasoning"""
        
        # Find relevant knowledge
        relevant_knowledge = self.find_relevant_knowledge(query)
        
        if not relevant_knowledge:
            return self.handle_unknown_query(query)
        
        # Get the best match
        best_topic, best_description, score = relevant_knowledge[0]
        
        # Analyze question type for intelligent response formatting
        query_lower = query.lower()
        
        # Question type analysis
        if any(phrase in query_lower for phrase in ['what is', 'what are', 'define', 'definition']):
            return self.format_definition_response(best_topic, best_description, query)
        
        elif any(phrase in query_lower for phrase in ['how does', 'how do', 'how can', 'how to']):
            return self.format_how_response(best_topic, best_description, query)
        
        elif any(phrase in query_lower for phrase in ['why', 'why is', 'why are', 'why do']):
            return self.format_why_response(best_topic, best_description, query)
        
        elif any(phrase in query_lower for phrase in ['when', 'where', 'who', 'which']):
            return self.format_context_response(best_topic, best_description, query)
        
        elif any(phrase in query_lower for phrase in ['compare', 'difference', 'vs', 'versus']):
            return self.format_comparison_response(query, relevant_knowledge)
        
        elif any(phrase in query_lower for phrase in ['example', 'examples', 'show me', 'demonstrate']):
            return self.format_example_response(best_topic, best_description, query)
        
        else:
            return self.format_general_response(best_topic, best_description, query)
    
    def format_definition_response(self, topic: str, description: str, query: str) -> str:
        """Format definition-style responses"""
        return f"{topic.title()} is {description.split('.')[0].lower()}. {' '.join(description.split('.')[1:3])}"
    
    def format_how_response(self, topic: str, description: str, query: str) -> str:
        """Format how-to responses"""
        sentences = description.split('.')
        mechanism = ""
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['uses', 'works', 'process', 'method', 'by']):
                mechanism = sentence.strip()
                break
        
        if mechanism:
            return f"{topic.title()} works by {mechanism.lower()}. {' '.join(sentences[1:2])}"
        else:
            return f"{topic.title()} operates through {description.split('.')[0].lower()}. {' '.join(sentences[1:2])}"
    
    def format_why_response(self, topic: str, description: str, query: str) -> str:
        """Format why-style responses"""
        sentences = description.split('.')
        benefit_sentence = ""
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['enable', 'allow', 'provide', 'offer', 'benefit']):
                benefit_sentence = sentence.strip()
                break
        
        if benefit_sentence:
            return f"{topic.title()} is important because {benefit_sentence.lower()}. {' '.join(sentences[:2])}"
        else:
            return f"{topic.title()} is valuable because {description.split('.')[0].lower()}. {' '.join(sentences[1:2])}"
    
    def format_context_response(self, topic: str, description: str, query: str) -> str:
        """Format contextual responses"""
        return f"Regarding {topic}, {description.split('.')[0].lower()}. {' '.join(description.split('.')[1:3])}"
    
    def format_comparison_response(self, query: str, relevant_knowledge: List) -> str:
        """Format comparison responses"""
        if len(relevant_knowledge) >= 2:
            topic1, desc1, _ = relevant_knowledge[0]
            topic2, desc2, _ = relevant_knowledge[1]
            
            return f"{topic1.title()} {desc1.split('.')[0].lower()}, while {topic2} {desc2.split('.')[0].lower()}. Both are important in their respective domains."
        else:
            topic, description, _ = relevant_knowledge[0]
            return f"When comparing {topic}, {description.split('.')[0].lower()}. {' '.join(description.split('.')[1:2])}"
    
    def format_example_response(self, topic: str, description: str, query: str) -> str:
        """Format example-focused responses"""
        sentences = description.split('.')
        example_info = ""
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['include', 'such as', 'example', 'like', 'used in']):
                example_info = sentence.strip()
                break
        
        if example_info:
            return f"For {topic}, {example_info.lower()}. {description.split('.')[0]} {' '.join(sentences[1:2])}"
        else:
            return f"{topic.title()} can be demonstrated through {description.split('.')[0].lower()}. {' '.join(sentences[1:3])}"
    
    def format_general_response(self, topic: str, description: str, query: str) -> str:
        """Format general responses"""
        return f"{description.split('.')[0]}. {' '.join(description.split('.')[1:3])}"
    
    def handle_unknown_query(self, query: str) -> str:
        """Handle queries outside the knowledge base"""
        suggestions = [
            "artificial intelligence", "machine learning", "deep learning", 
            "neural networks", "python programming", "data science"
        ]
        
        return f"I don't have specific information about '{query}'. However, I can help you with topics like {', '.join(suggestions[:3])}. Could you ask about one of these areas?"
    
    def run(self):
        """Main interaction loop with advanced conversation handling"""
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n🎯 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("🤖 Assistant: Thank you for using the Super Intelligent Assistant! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Add to conversation history
                conversation_history.append(f"User: {user_input}")
                
                print("🤔 Processing your query with advanced reasoning...")
                
                # Generate intelligent response
                response = self.generate_intelligent_response(user_input)
                
                # Add response to history
                conversation_history.append(f"Assistant: {response}")
                
                # Keep conversation history manageable
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-8:]
                
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Assistant: Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🤖 Assistant: I encountered an error. Please try rephrasing your question.")

def main():
    """Initialize and run the super intelligent assistant"""
    import os
    assistant = SuperSmartAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
