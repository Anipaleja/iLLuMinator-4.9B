import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class IntelligentRAGTransformer(nn.Module):
    """
    Smart RAG-integrated Transformer that can answer any query by:
    1. Understanding the query intent
    2. Retrieving relevant knowledge automatically
    3. Generating contextually aware responses
    """
    
    def __init__(self, 
                 vocab_size: int,
                 block_size: int = 1024,
                 n_embd: int = 512,
                 n_head: int = 16,
                 n_layer: int = 12,
                 dropout: float = 0.1,
                 knowledge_base: Optional[List[str]] = None,
                 retriever_model: str = 'all-MiniLM-L6-v2'):
        super().__init__()
        
        self.block_size = block_size
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        
        # Core transformer components
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced transformer layers with knowledge integration
        self.layers = nn.ModuleList([
            KnowledgeAwareDecoderBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        # Query understanding and intent detection
        self.query_encoder = QueryEncoder(n_embd)
        self.intent_classifier = IntentClassifier(n_embd)
        
        # Knowledge retrieval system
        self.knowledge_retriever = None
        if knowledge_base:
            self.setup_knowledge_base(knowledge_base, retriever_model)
        
        # Response quality enhancer
        self.response_enhancer = ResponseEnhancer(n_embd)
        
        # Special tokens for different query types
        self.special_tokens = {
            'CONTEXT_START': vocab_size - 5,
            'CONTEXT_END': vocab_size - 4,
            'QUERY_START': vocab_size - 3,
            'ANSWER_START': vocab_size - 2,
            'EOS': vocab_size - 1
        }
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def setup_knowledge_base(self, documents: List[str], retriever_model: str):
        """Setup the integrated knowledge retrieval system"""
        print("🧠 Setting up intelligent knowledge base...")
        
        self.knowledge_retriever = IntelligentRetriever(
            documents=documents,
            model_name=retriever_model,
            embedding_dim=self.n_embd
        )
        
        # Add knowledge integration layer
        self.knowledge_integrator = KnowledgeIntegrator(self.n_embd)
        
        # Add embedding projection layer to match dimensions
        retriever_dim = self.knowledge_retriever.get_embedding_dim()
        if retriever_dim != self.n_embd:
            self.embedding_projection = nn.Linear(retriever_dim, self.n_embd)
        else:
            self.embedding_projection = nn.Identity()
    
    def understand_query(self, query_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Understand query intent and extract key information
        
        Returns:
            query_embedding: Rich representation of the query
            intent_scores: Classification of query type/intent
        """
        # Get embeddings for query tokens
        query_emb = self.token_embedding(query_tokens)
        pos = torch.arange(query_tokens.size(1), device=query_tokens.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        query_repr = query_emb + pos_emb
        
        # Encode query understanding
        query_embedding = self.query_encoder(query_repr)
        intent_scores = self.intent_classifier(query_embedding)
        
        return query_embedding, intent_scores
    
    def retrieve_smart_context(self, query_embedding: torch.Tensor, query_text: str, top_k: int = 5) -> torch.Tensor:
        """
        Intelligently retrieve and integrate relevant knowledge
        """
        if self.knowledge_retriever is None:
            return torch.zeros(1, 1, self.n_embd, device=query_embedding.device)
        
        # Get relevant documents
        relevant_docs = self.knowledge_retriever.retrieve_smart(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Convert to embeddings and integrate
        context_embeddings = self.knowledge_retriever.docs_to_embeddings(relevant_docs)
        
        # Project embeddings to match model dimension
        if hasattr(self, 'embedding_projection'):
            context_embeddings = self.embedding_projection(context_embeddings)
        
        integrated_context = self.knowledge_integrator(query_embedding, context_embeddings)
        
        return integrated_context
    
    def forward(self, idx: torch.Tensor, query_text: Optional[str] = None) -> torch.Tensor:
        """
        Enhanced forward pass with intelligent context integration
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Get token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # If this is a query, enhance with retrieved knowledge
        if query_text and hasattr(self, 'knowledge_retriever'):
            query_embedding, intent_scores = self.understand_query(idx)
            context_embeddings = self.retrieve_smart_context(query_embedding, query_text)
            
            # Integrate context into the representation
            x = self.integrate_knowledge_into_sequence(x, context_embeddings, intent_scores)
        
        # Apply transformer layers with knowledge awareness
        for layer in self.layers:
            x = layer(x, context_aware=query_text is not None)
        
        # Final processing
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def integrate_knowledge_into_sequence(self, x: torch.Tensor, context: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """
        Smartly integrate retrieved knowledge into the sequence representation
        """
        B, T, C = x.size()
        
        # Create attention weights based on intent and relevance
        intent_weights = F.softmax(intent, dim=-1)
        
        # Broadcast context and blend with sequence
        if context.size(1) > 0:
            # Use cross-attention to integrate context
            context_attended = F.scaled_dot_product_attention(
                x, context, context,
                attn_mask=None,
                dropout_p=0.1 if self.training else 0.0
            )
            
            # Weighted combination based on intent
            alpha = intent_weights.mean(dim=-1, keepdim=True).unsqueeze(-1)
            x = (1 - alpha) * x + alpha * context_attended
        
        return x
    
    def chat(self, query: str, max_tokens: int = 200, temperature: float = 0.7, top_k: int = 50) -> str:
        """
        Main chat interface - input any query and get intelligent response
        
        Args:
            query: Natural language query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated response string
        """
        self.eval()
        
        # Tokenize query (you'll need to implement this based on your tokenizer)
        query_tokens = self._tokenize_query(query)
        
        # Start with query tokens
        input_ids = query_tokens.unsqueeze(0)
        
        # Add answer start token
        answer_token = torch.tensor([[self.special_tokens['ANSWER_START']]], 
                                  device=input_ids.device)
        input_ids = torch.cat([input_ids, answer_token], dim=1)
        
        # Generate response
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if input_ids.size(1) >= self.block_size:
                    # Keep recent context
                    input_ids = input_ids[:, -self.block_size:]
                
                # Forward pass with query context
                logits = self.forward(input_ids, query_text=query)
                
                # Get next token logits
                next_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end of response
                if next_token.item() == self.special_tokens['EOS']:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode response
        response = self._detokenize(generated_tokens)
        
        # Enhance response quality
        enhanced_response = self._enhance_response(response, query)
        
        return enhanced_response
    
    def _tokenize_query(self, query: str) -> torch.Tensor:
        """Tokenize input query - implement based on your tokenizer"""
        # Placeholder - replace with your actual tokenizer
        # For now, return dummy tokens
        tokens = [i % (self.vocab_size - 10) for i in range(min(len(query.split()), 50))]
        return torch.tensor(tokens)
    
    def _detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text - implement based on your tokenizer"""
        # Placeholder - replace with your actual detokenizer
        return " ".join([f"token_{tid}" for tid in token_ids])
    
    def _enhance_response(self, response: str, query: str) -> str:
        """Post-process response for better quality"""
        # Apply response enhancement
        if hasattr(self, 'response_enhancer'):
            # This would use the response enhancer module
            pass
        
        # Basic post-processing
        response = response.strip()
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response


class QueryEncoder(nn.Module):
    """Encodes queries to understand intent and extract key information"""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_embd, nhead=8, batch_first=True),
            num_layers=2
        )
        self.pooler = nn.Linear(n_embd, n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        # Pool to get single representation
        pooled = encoded.mean(dim=1)
        return torch.tanh(self.pooler(pooled))


class IntentClassifier(nn.Module):
    """Classifies query intent for better response generation"""
    
    def __init__(self, n_embd: int, num_intents: int = 8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_embd // 2, num_intents)
        )
        
        # Intent types: factual, creative, analytical, conversational, etc.
    
    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(query_embedding)


class KnowledgeAwareDecoderBlock(nn.Module):
    """Enhanced decoder block that can incorporate external knowledge"""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
        
        # Knowledge integration components
        self.knowledge_gate = nn.Linear(n_embd, 1)
        self.ln_knowledge = nn.LayerNorm(n_embd)
    
    def forward(self, x: torch.Tensor, context_aware: bool = False) -> torch.Tensor:
        # Standard transformer processing
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        
        # Knowledge-aware processing if context is available
        if context_aware:
            # Apply knowledge gating
            knowledge_weights = torch.sigmoid(self.knowledge_gate(x))
            x = x * knowledge_weights + self.ln_knowledge(x) * (1 - knowledge_weights)
        
        # Feed-forward
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Enhanced causal self-attention with knowledge integration capabilities"""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # QKV computation
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """Enhanced MLP with knowledge integration"""
    
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class IntelligentRetriever:
    """Smart knowledge retrieval system"""
    
    def __init__(self, documents: List[str], model_name: str, embedding_dim: int):
        self.documents = documents
        self.embedding_dim = embedding_dim
        
        # Initialize retriever
        self.retriever = SentenceTransformer(model_name)
        self.retriever_embedding_dim = self.retriever.get_sentence_embedding_dimension()
        
        # Build search index
        self.doc_embeddings = self.retriever.encode(documents)
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings.astype('float32'))
    
    def get_embedding_dim(self):
        """Get the actual embedding dimension from the retriever"""
        return self.retriever_embedding_dim
    
    def retrieve_smart(self, query_text: str, query_embedding: torch.Tensor, top_k: int = 5) -> List[str]:
        """Smart retrieval using both semantic and neural query understanding"""
        
        # Semantic retrieval
        query_vec = self.retriever.encode([query_text])
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec.astype('float32'), top_k)
        
        # Get relevant documents
        relevant_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > 0.1:  # Lower threshold for more results
                relevant_docs.append(self.documents[idx])
        
        return relevant_docs
    
    def docs_to_embeddings(self, docs: List[str]) -> torch.Tensor:
        """Convert documents to embeddings for integration"""
        if not docs:
            return torch.zeros(1, 1, self.retriever_embedding_dim)
        
        doc_embeddings = self.retriever.encode(docs)
        return torch.tensor(doc_embeddings, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


class KnowledgeIntegrator(nn.Module):
    """Integrates retrieved knowledge with query understanding"""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(n_embd, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(n_embd)
        self.gate = nn.Linear(n_embd * 2, n_embd)
    
    def forward(self, query_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        if context_emb.size(1) == 0:
            return torch.zeros_like(query_emb).unsqueeze(1)
        
        # Cross-attention between query and retrieved knowledge
        attended, _ = self.cross_attention(
            query_emb.unsqueeze(1), context_emb, context_emb
        )
        
        # Gated integration
        combined = torch.cat([query_emb.unsqueeze(1), attended], dim=-1)
        gated = torch.sigmoid(self.gate(combined))
        
        return self.norm(attended * gated)


class ResponseEnhancer(nn.Module):
    """Enhances response quality and coherence"""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.enhancer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_embd, nhead=8, batch_first=True),
            num_layers=2
        )
    
    def forward(self, response_emb: torch.Tensor) -> torch.Tensor:
        return self.enhancer(response_emb)
