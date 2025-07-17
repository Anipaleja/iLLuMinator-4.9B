import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from model.transformer import iLLuMinator
from model.tokenizer import build_tokenizer, encode, decode


class RAGSystem:
    """
    Retrieval-Augmented Generation system using iLLuMinator transformer
    """
    
    def __init__(self, 
                 documents: List[str],
                 model_path: str = None,
                 retriever_model: str = 'all-MiniLM-L6-v2',
                 top_k: int = 3,
                 max_context_length: int = 300):
        
        self.documents = documents
        self.top_k = top_k
        self.max_context_length = max_context_length
        
        # Initialize retriever (sentence transformer for semantic search)
        print("Loading retriever model...")
        self.retriever = SentenceTransformer(retriever_model)
        
        # Build document embeddings and FAISS index
        print("Building document index...")
        self.doc_embeddings = self.retriever.encode(documents)
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])  # Inner product similarity
        faiss.normalize_L2(self.doc_embeddings)  # Normalize for cosine similarity
        self.index.add(self.doc_embeddings.astype('float32'))
        
        # Initialize tokenizer and generator
        self.stoi = None
        self.itos = None
        self.generator = None
        
        if model_path:
            self.load_generator(model_path)
    
    def load_generator(self, model_path: str, vocab_data_path: str = None):
        """Load the pre-trained iLLuMinator model"""
        print("Loading generator model...")
        
        # Load tokenizer (you'll need to save/load your tokenizer)
        if vocab_data_path:
            from data.prepare import load_data
            text = load_data(vocab_data_path)
            self.stoi, self.itos = build_tokenizer(text)
        
        # Initialize and load model
        vocab_size = len(self.stoi) if self.stoi else 1000  # fallback
        self.generator = iLLuMinator(
            vocab_size=vocab_size,
            block_size=512,
            n_embd=256,
            n_head=8,
            n_layer=6
        )
        
        if model_path:
            self.generator.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.generator.eval()
    
    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant documents for the query
        
        Returns:
            List of (document, score) tuples
        """
        # Encode query
        query_embedding = self.retriever.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        scores, indices = self.index.search(query_embedding.astype('float32'), self.top_k)
        
        # Return documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def prepare_context(self, retrieved_docs: List[Tuple[str, float]], query: str) -> str:
        """
        Prepare context string from retrieved documents and query
        """
        context_parts = []
        context_parts.append("Context:")
        
        for i, (doc, score) in enumerate(retrieved_docs):
            # Truncate long documents
            if len(doc) > self.max_context_length:
                doc = doc[:self.max_context_length] + "..."
            context_parts.append(f"[{i+1}] {doc}")
        
        context_parts.append(f"\nQuestion: {query}")
        context_parts.append("Answer:")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, 
                       query: str, 
                       max_new_tokens: int = 100,
                       temperature: float = 0.7,
                       top_k: int = 40) -> str:
        """
        Generate answer using RAG: retrieve relevant docs, then generate
        """
        if not self.generator or not self.stoi:
            raise ValueError("Generator model not loaded. Call load_generator() first.")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Prepare context
        context_text = self.prepare_context(retrieved_docs, query)
        print(f"Context length: {len(context_text)} characters")
        
        # Step 3: Tokenize context
        context_tokens = encode(context_text, self.stoi)
        
        # Ensure context fits in model's context window
        if len(context_tokens) > self.generator.block_size - max_new_tokens:
            # Truncate from the beginning, keeping the question and answer prompt
            excess = len(context_tokens) - (self.generator.block_size - max_new_tokens)
            context_tokens = context_tokens[excess:]
        
        # Step 4: Generate response
        input_tensor = torch.tensor([context_tokens], dtype=torch.long)
        
        with torch.no_grad():
            # Generate tokens one by one
            for _ in range(max_new_tokens):
                if input_tensor.size(1) >= self.generator.block_size:
                    # Sliding window
                    input_tensor = input_tensor[:, -self.generator.block_size:]
                
                logits = self.generator(input_tensor)
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                
                # Stop if we hit end token (you might want to define this)
                # if next_token.item() == self.stoi.get('<|endoftext|>', -1):
                #     break
        
        # Decode the full sequence and extract just the answer
        full_text = decode(input_tensor[0].tolist(), self.itos)
        
        # Extract answer part (everything after "Answer:")
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text[len(context_text):].strip()
        
        return answer
    
    def chat(self, query: str) -> dict:
        """
        Complete RAG chat function that returns both retrieved docs and answer
        """
        retrieved_docs = self.retrieve(query)
        answer = self.generate_answer(query)
        
        return {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'answer': answer,
            'context_used': self.prepare_context(retrieved_docs, query)
        }


# Example usage and testing
if __name__ == "__main__":
    # Sample documents for testing
    sample_documents = [
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and AI.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data. It has revolutionized computer vision and natural language processing.",
        "Transformers are a type of neural network architecture that has become the foundation for large language models like GPT and BERT."
    ]
    
    # Initialize RAG system
    rag = RAGSystem(documents=sample_documents)
    
    # Test retrieval
    query = "What is machine learning?"
    results = rag.retrieve(query)
    print(f"Query: {query}")
    print("Retrieved documents:")
    for doc, score in results:
        print(f"Score: {score:.3f} - {doc[:100]}...")
    
    # Test context preparation
    context = rag.prepare_context(results, query)
    print(f"\nPrepared context:\n{context}")
    
    print("\nRAG System ready! Load your trained model with:")
    print("rag.load_generator('illuminator.pth', 'path/to/vocab/data')")
