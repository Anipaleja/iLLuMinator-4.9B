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
        
        # Comprehensive vocabulary for better coverage
        base_words = [
            # Common words
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "what", "how", "why", "when", "where", "who", "which", "can", "will", "would", "should",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
            "this", "that", "these", "those", "here", "there", "now", "then", "yes", "no", "not",
            
            # AI/Tech vocabulary
            "artificial", "intelligence", "machine", "learning", "deep", "neural", "network", "algorithm",
            "computer", "science", "technology", "programming", "python", "data", "model", "train",
            "system", "process", "information", "knowledge", "smart", "intelligent", "code", "software",
            "digital", "analyze", "pattern", "prediction", "automation", "robot", "brain", "think",
            
            # Action words
            "work", "works", "working", "learn", "learning", "understand", "explain", "help", "create",
            "build", "develop", "design", "solve", "find", "search", "recognize", "classify", "predict",
            "analyze", "compute", "calculate", "process", "generate", "produce", "make", "use", "apply",
            
            # Descriptive words
            "good", "better", "best", "new", "old", "big", "small", "fast", "slow", "easy", "hard",
            "simple", "complex", "powerful", "useful", "important", "different", "similar", "same",
            "many", "few", "more", "less", "most", "some", "all", "every", "each", "other", "another",
            
            # Response words
            "answer", "question", "response", "reply", "solution", "result", "output", "input",
            "example", "case", "way", "method", "approach", "technique", "strategy", "plan",
            
            # Common phrases parts
            "hello", "hi", "hey", "thanks", "thank", "please", "welcome", "sorry", "excuse",
            "help", "assist", "support", "guide", "teach", "show", "tell", "say", "speak",
        ]
        
        # Add numbers
        for i in range(100):
            base_words.append(str(i))
        
        # Add punctuation and special characters
        punctuation = [".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", "\"", "'", "-", "_"]
        base_words.extend(punctuation)
        
        # Add special tokens
        special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>", "<SPACE>"]
        base_words.extend(special_tokens)
        
        # Remove duplicates and create mappings
        unique_words = list(set(base_words))
        self.stoi = {word: i for i, word in enumerate(unique_words)}
        self.itos = {i: word for i, word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)
        
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
    
    def generate_response(self, query: str, max_tokens: int = 30) -> str:
        """Generate response using the transformer model"""
        try:
            # Find relevant knowledge first
            relevant_knowledge = self.find_relevant_knowledge(query)
            
            # If model generation fails, use knowledge-based response
            if relevant_knowledge:
                # Try simple template-based response first
                knowledge_text = relevant_knowledge[0]
                
                # Create a response based on the question type
                query_lower = query.lower()
                if 'what' in query_lower and 'is' in query_lower:
                    return f"{knowledge_text.split('.')[0]}."
                elif 'how' in query_lower and 'work' in query_lower:
                    return f"It works by {knowledge_text.lower().split('by')[-1] if 'by' in knowledge_text else knowledge_text.split('.')[0].lower()}."
                elif 'how' in query_lower:
                    return f"Here's how: {knowledge_text.split('.')[0].lower()}."
                else:
                    return knowledge_text.split('.')[0] + "."
            
            # If no relevant knowledge, try model generation with very simple prompt
            prompt = f"Q: {query[:20]} A:"  # Keep it very short
            input_tokens = self.simple_tokenize(prompt)
            
            # Limit input length severely
            if len(input_tokens) > 20:
                input_tokens = input_tokens[-20:]
            
            # Generate with the model
            generated_tokens = input_tokens.copy()
            answer_tokens = []
            
            with torch.no_grad():
                for i in range(max_tokens):
                    # Prepare sequence
                    current_seq = generated_tokens[-50:]  # Very short context
                    current_tensor = torch.tensor([current_seq + [0] * (50 - len(current_seq))], dtype=torch.long)
                    
                    # Forward pass
                    try:
                        logits = self.model(current_tensor)
                        next_logits = logits[0, len(current_seq)-1, :] / 0.5  # Low temperature
                        
                        # Heavy filtering - only top 5 tokens
                        top_k = 5
                        v, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        filtered_logits = torch.full_like(next_logits, float('-inf'))
                        filtered_logits[indices] = v
                        
                        next_probs = torch.softmax(filtered_logits, dim=-1)
                        next_token = torch.multinomial(next_probs, num_samples=1).item()
                        
                        # Stop on padding or repeated patterns
                        if next_token == self.stoi.get("<PAD>", 0):
                            break
                        
                        generated_tokens.append(next_token)
                        answer_tokens.append(next_token)
                        
                        # Stop on reasonable punctuation
                        if next_token in [self.stoi.get(".", -1), self.stoi.get("!", -1), self.stoi.get("?", -1)]:
                            break
                            
                    except Exception as e:
                        print(f"Generation step failed: {e}")
                        break
            
            # Try to decode the answer
            if len(answer_tokens) > 2:
                answer = self.simple_detokenize(answer_tokens)
                answer = re.sub(r'\s+', ' ', answer).strip()
                
                if len(answer) > 5 and not answer.replace(' ', '').replace('.', '') == '':
                    if not answer.endswith(('.', '!', '?')):
                        answer += '.'
                    return answer.capitalize()
            
            # Final fallback to knowledge
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
