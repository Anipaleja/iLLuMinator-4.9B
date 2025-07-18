#!/usr/bin/env python3
"""
Fixed Smart Assistant that works with your trained model
"""

import torch
import json
import os
import re
from typing import List, Dict, Any
from model.transformer import iLLuMinator
from model.tokenizer import build_tokenizer, encode, decode
from data.prepare import load_data
import faiss
import numpy as np

class FixedSmartAssistant:
    """
    Fixed Smart Assistant that properly integrates with your transformer
    """
    
    def __init__(self, model_path: str = "intelligent_rag_model.pth"):
        print("Initializing Fixed Smart Assistant...")
        
        # Load tokenizer
        self.setup_tokenizer()
        
        # Load model
        self.setup_model(model_path)
        
        # Setup knowledge base
        self.setup_knowledge_base()
        
        # Conversation history
        self.conversation_history = []
        
        print("Smart Assistant ready!")
        print("Ask me anything!")
    
    def setup_tokenizer(self):
        """Setup tokenizer from your existing data"""
        try:
            text = load_data()
            self.stoi, self.itos = build_tokenizer(text)
            self.vocab_size = len(self.stoi)
            print(f"Loaded tokenizer with vocab size: {self.vocab_size}")
        except Exception as e:
            print(f"Tokenizer error: {e}")
            # Create a simple fallback tokenizer
            self.create_simple_tokenizer()
    
    def create_simple_tokenizer(self):
        """Create a simple word-based tokenizer"""
        print("Creating simple tokenizer...")
        
        # Basic vocabulary
        words = [
            "hello", "hi", "what", "is", "how", "why", "the", "a", "an", "and", "or", "but",
            "artificial", "intelligence", "machine", "learning", "deep", "neural", "network",
            "computer", "science", "technology", "programming", "python", "data", "algorithm",
            "question", "answer", "help", "explain", "understand", "know", "think", "learn",
            "work", "works", "working", "system", "model", "train", "training", "language",
            "natural", "process", "processing", "information", "knowledge", "smart", "intelligent",
            "thank", "thanks", "please", "yes", "no", "can", "could", "would", "should", "will"
        ]
        
        # Add numbers and punctuation
        for i in range(100):
            words.append(str(i))
        
        for char in ".,!?;:()[]{}\"'-":
            words.append(char)
        
        # Add special tokens
        special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
        words.extend(special_tokens)
        
        # Create mappings
        self.stoi = {word: i for i, word in enumerate(words)}
        self.itos = {i: word for i, word in enumerate(words)}
        self.vocab_size = len(words)
        
        print(f"Simple tokenizer created with {self.vocab_size} tokens")
    
    def setup_model(self, model_path: str):
        """Setup the transformer model"""
        print("Loading model...")
        
        # Create model with correct vocab size
        self.model = iLLuMinator(
            vocab_size=self.vocab_size,
            block_size=512,
            n_embd=256,
            n_head=8,
            n_layer=6
        )
        
        # Try to load basic model weights first
        basic_model_path = "illuminator.pth"
        loaded_weights = False
        
        if os.path.exists(basic_model_path):
            try:
                state_dict = torch.load(basic_model_path, map_location='cpu')
                # Only load core transformer weights
                model_dict = self.model.state_dict()
                filtered_dict = {}
                
                for k, v in state_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                
                if filtered_dict:
                    model_dict.update(filtered_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"Loaded weights from {basic_model_path}")
                    loaded_weights = True
            except Exception as e:
                print(f"Could not load basic model: {e}")
        
        # If no basic model, try the intelligent model but filter aggressively
        if not loaded_weights and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                model_dict = self.model.state_dict()
                filtered_dict = {}
                
                # Only load core transformer components
                core_prefixes = ['token_embedding', 'position_embedding', 'layers', 'ln_f', 'head']
                
                for k, v in state_dict.items():
                    if any(k.startswith(prefix) for prefix in core_prefixes):
                        if k in model_dict and v.shape == model_dict[k].shape:
                            filtered_dict[k] = v
                
                if filtered_dict:
                    model_dict.update(filtered_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"Loaded core weights from {model_path}")
                    loaded_weights = True
            except Exception as e:
                print(f"Could not load intelligent model: {e}")
        
        if not loaded_weights:
            print("Using randomly initialized weights")
        
        self.model.eval()
    
    def setup_knowledge_base(self):
        """Setup simple knowledge base with vector search"""
        print("Setting up knowledge base...")
        
        # Load knowledge from demo file or create default
        try:
            with open("demo_knowledge.json", "r") as f:
                self.knowledge = json.load(f)
        except:
            self.knowledge = [
                "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
                "Machine Learning allows computers to learn from data without explicit programming.",
                "Deep Learning uses neural networks with multiple layers to learn complex patterns.",
                "Natural Language Processing helps computers understand and generate human language.",
                "Python is a popular programming language for AI and data science applications.",
                "Neural networks are inspired by how the human brain processes information.",
                "Computer vision enables machines to interpret and understand visual information.",
                "Algorithms are step-by-step procedures for solving problems or performing tasks."
            ]
        
        print(f"Loaded {len(self.knowledge)} knowledge entries")
    
    def simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        # Clean and split text
        text = text.lower().strip()
        words = re.findall(r'\w+|[.,!?;]', text)
        
        # Convert to token IDs
        tokens = []
        for word in words:
            if word in self.stoi:
                tokens.append(self.stoi[word])
            else:
                tokens.append(self.stoi.get("<UNK>", 0))
        
        return tokens
    
    def simple_detokenize(self, token_ids: List[int]) -> str:
        """Simple detokenization"""
        words = []
        for token_id in token_ids:
            if token_id in self.itos:
                words.append(self.itos[token_id])
        
        # Join words and clean up
        text = " ".join(words)
        text = re.sub(r' ([.,!?;])', r'\1', text)  # Remove spaces before punctuation
        return text.strip()
    
    def find_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """Find relevant knowledge using simple keyword matching"""
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_knowledge = []
        for knowledge_item in self.knowledge:
            knowledge_words = set(re.findall(r'\w+', knowledge_item.lower()))
            
            # Calculate overlap score
            overlap = len(query_words.intersection(knowledge_words))
            if overlap > 0:
                # Boost score for exact phrase matches
                for word in query_words:
                    if word in knowledge_item.lower():
                        overlap += 0.5
                
                scored_knowledge.append((knowledge_item, overlap))
        
        # Sort by relevance and return top results
        scored_knowledge.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_knowledge[:top_k]]
    
    def generate_response(self, query: str, max_tokens: int = 50) -> str:
        """Generate response using the transformer model"""
        try:
            # Find relevant knowledge
            relevant_knowledge = self.find_relevant_knowledge(query)
            
            # Create a simple prompt format
            if relevant_knowledge:
                context = f"Knowledge: {relevant_knowledge[0][:100]}... "
            else:
                context = ""
            
            prompt = f"{context}Question: {query} Answer:"
            
            # Tokenize with length limit
            input_tokens = self.simple_tokenize(prompt)
            
            # Limit input length
            max_input = self.model.block_size - max_tokens - 10
            if len(input_tokens) > max_input:
                input_tokens = input_tokens[-max_input:]
            
            # Convert to tensor
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            
            # Generate response tokens
            generated_tokens = input_tokens.copy()
            answer_tokens = []
            
            with torch.no_grad():
                for i in range(max_tokens):
                    # Prepare current sequence
                    current_seq = generated_tokens[-self.model.block_size:]
                    current_tensor = torch.tensor([current_seq], dtype=torch.long)
                    
                    # Forward pass
                    logits = self.model(current_tensor)
                    
                    # Get next token probabilities with temperature
                    next_logits = logits[0, -1, :] / 0.7  # Lower temperature for more focused responses
                    
                    # Apply top-k filtering
                    top_k = 20
                    if top_k > 0:
                        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        next_logits[next_logits < v[-1]] = float('-inf')
                    
                    next_probs = torch.softmax(next_logits, dim=-1)
                    
                    # Sample next token
                    next_token = torch.multinomial(next_probs, num_samples=1).item()
                    
                    # Add to sequences
                    generated_tokens.append(next_token)
                    answer_tokens.append(next_token)
                    
                    # Stop on padding or repeated tokens
                    if next_token == self.stoi.get("<PAD>", 0):
                        break
                    
                    # Stop if we're generating repetitive content
                    if len(answer_tokens) >= 3 and all(t == answer_tokens[-1] for t in answer_tokens[-3:]):
                        break
            
            # Decode the answer
            if answer_tokens:
                answer = self.simple_detokenize(answer_tokens)
                answer = answer.strip()
                
                # Clean up the response
                answer = re.sub(r'\s+', ' ', answer)
                answer = re.sub(r'<[^>]*>', '', answer)  # Remove special tokens
                
                # Ensure we have a meaningful response
                if len(answer) > 2 and not answer.replace(' ', '').replace('.', '').replace(',', '') == '':
                    # Make it a proper sentence
                    if not answer.endswith(('.', '!', '?')):
                        answer += '.'
                    return answer.capitalize()
            
            # Fall back to knowledge-based response
            return self.fallback_response(query, relevant_knowledge)
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self.fallback_response(query, self.find_relevant_knowledge(query))
    
    def fallback_response(self, query: str, relevant_knowledge: List[str] = None) -> str:
        """Fallback response when generation fails"""
        query_lower = query.lower()
        
        # Greeting responses
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm your AI assistant. How can I help you today?"
        
        # Help responses
        if any(word in query_lower for word in ['help', 'assist']):
            return "I can help answer questions about technology, AI, programming, and more. What would you like to know?"
        
        # Thank you responses
        if any(word in query_lower for word in ['thank', 'thanks']):
            return "You're welcome! Feel free to ask me anything else."
        
        # Use relevant knowledge if available
        if relevant_knowledge:
            return f"Based on what I know: {relevant_knowledge[0]}"
        
        # Default response
        return "That's an interesting question! I'd be happy to help if you could provide more context or ask about a specific topic."
    
    def chat(self, query: str) -> str:
        """Main chat interface"""
        if not query.strip():
            return "Please ask me a question!"
        
        print(f"Processing: {query}")
        
        # Generate response
        response = self.generate_response(query)
        
        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "response": response
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response


def main():
    """Interactive chat with the fixed assistant"""
    print("Fixed Smart Assistant")
    print("=" * 50)
    print("Now with proper dimension handling!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    # Initialize assistant
    assistant = FixedSmartAssistant()
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            # Get response
            print("Processing:", query)
            response = assistant.chat(query)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
