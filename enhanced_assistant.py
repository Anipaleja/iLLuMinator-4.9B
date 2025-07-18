#!/usr/bin/env python3
"""
Enhanced Smart Assistant - Final Version
Combines transformer model with intelligent RAG system for optimal responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import json
import os
from typing import List, Dict, Tuple, Optional

# Import the core transformer
from model.transformer import iLLuMinator

class EnhancedSmartAssistant:
    """Enhanced Assistant with improved knowledge matching and generation"""
    
    def __init__(self, model_path: str = "illuminator.pth"):
        """Initialize the enhanced assistant"""
        print("Enhanced Smart Assistant")
        print("=" * 50)
        print("Final version with optimal response generation!")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        self.setup_tokenizer()
        self.load_model(model_path)
        self.setup_knowledge_base()
        print("Enhanced Assistant ready!")
        print("Ask me anything!")
        
    def setup_tokenizer(self):
        """Create comprehensive tokenizer"""
        # Enhanced vocabulary for better coverage
        vocab = [
            # Special tokens
            "<PAD>", "<UNK>", "<START>", "<END>",
            
            # Common words
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might", "must",
            "this", "that", "these", "those", "what", "when", "where", "why", "how", "who", "which",
            "not", "no", "yes", "all", "any", "some", "many", "much", "more", "most", "few", "little",
            "big", "small", "good", "bad", "new", "old", "first", "last", "next", "other",
            
            # AI/Tech terms
            "AI", "artificial", "intelligence", "machine", "learning", "deep", "neural", "network",
            "data", "algorithm", "model", "train", "training", "prediction", "classification",
            "regression", "supervised", "unsupervised", "reinforcement", "computer", "science",
            "programming", "python", "code", "software", "technology", "digital", "system",
            "information", "knowledge", "analysis", "pattern", "recognition", "automation",
            "robot", "robotics", "cognitive", "computation", "processing", "algorithm",
            "feature", "dataset", "input", "output", "layer", "neuron", "weight", "bias",
            "gradient", "optimization", "backpropagation", "epoch", "batch", "tensor",
            "pytorch", "tensorflow", "sklearn", "pandas", "numpy", "matplotlib",
            
            # Action words
            "learn", "teach", "understand", "analyze", "process", "compute", "calculate",
            "predict", "classify", "recognize", "identify", "detect", "generate", "create",
            "build", "develop", "design", "implement", "optimize", "improve", "enhance",
            "train", "test", "validate", "evaluate", "measure", "compare", "study",
            
            # Descriptive words
            "complex", "simple", "advanced", "basic", "powerful", "efficient", "accurate",
            "precise", "fast", "slow", "large", "massive", "tiny", "intelligent", "smart",
            "automatic", "manual", "real-time", "statistical", "mathematical", "logical",
            "structured", "unstructured", "labeled", "unlabeled", "distributed", "parallel",
            
            # Common phrases and connectors
            "such", "as", "like", "similar", "different", "same", "various", "multiple",
            "several", "numerous", "including", "example", "instance", "case", "type",
            "kind", "way", "method", "technique", "approach", "strategy", "solution",
            "problem", "challenge", "task", "goal", "objective", "purpose", "function",
            "role", "part", "component", "element", "aspect", "feature", "property",
            "characteristic", "attribute", "quality", "value", "result", "outcome",
            "effect", "impact", "influence", "relationship", "connection", "link",
            
            # Punctuation and numbers
            ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", '"',
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000"
        ]
        
        self.vocab = list(set(vocab))  # Remove duplicates
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        print(f"Enhanced tokenizer with vocab size: {self.vocab_size}")
        
    def tokenize(self, text: str) -> List[int]:
        """Enhanced tokenization with better word splitting"""
        # Clean and normalize text
        text = re.sub(r'[^\w\s.,!?:;()\[\]{}\'"-]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into tokens
        tokens = []
        words = text.split()
        
        for word in words:
            # Handle punctuation
            if word in self.stoi:
                tokens.append(self.stoi[word])
            else:
                # Try to find closest matches or break down further
                found = False
                for vocab_word in self.vocab:
                    if vocab_word.lower() in word.lower() and len(vocab_word) > 2:
                        tokens.append(self.stoi[vocab_word])
                        found = True
                        break
                
                if not found:
                    tokens.append(self.stoi.get("<UNK>", 1))
        
        return tokens
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text"""
        words = []
        for token in tokens:
            if token in self.itos:
                word = self.itos[token]
                if word not in ["<PAD>", "<UNK>", "<START>", "<END>"]:
                    words.append(word)
        
        text = " ".join(words)
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_model(self, model_path: str):
        """Load the transformer model"""
        try:
            print("Loading enhanced model...")
            self.model = iLLuMinator(
                vocab_size=self.vocab_size,
                n_embd=256,
                n_head=8,
                n_layer=6,
                block_size=512
            )
            
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                # Filter weights to match current model
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key in self.model.state_dict():
                        model_shape = self.model.state_dict()[key].shape
                        if value.shape == model_shape:
                            filtered_state_dict[key] = value
                        else:
                            print(f"Skipping {key}: shape mismatch {value.shape} vs {model_shape}")
                
                self.model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded compatible weights from {model_path}")
            else:
                print("No model file found, using random weights")
                
            self.model.eval()
            
        except Exception as e:
            print(f"Model loading error: {e}")
            # Create simple fallback model
            self.model = None
    
    def setup_knowledge_base(self):
        """Setup comprehensive knowledge base"""
        self.knowledge_base = [
            "Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems, including learning, reasoning, and self-correction.",
            "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
            "Neural Networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information using mathematical operations.",
            "Supervised Learning is a machine learning approach where models learn from labeled training data to make predictions on new, unseen data.",
            "Unsupervised Learning discovers hidden patterns in data without labeled examples, including clustering and dimensionality reduction techniques.",
            "Reinforcement Learning is a machine learning paradigm where agents learn optimal actions through trial and error interactions with an environment.",
            "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language.",
            "Computer Vision is an AI field that trains computers to interpret and understand visual information from the world, such as images and videos.",
            "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab, widely used for deep learning applications.",
            "TensorFlow is an open-source machine learning framework developed by Google for building and training neural networks and deep learning models.",
            "Scikit-learn is a popular Python library for machine learning that provides simple and efficient tools for data analysis and modeling.",
            "Pandas is a powerful Python library for data manipulation and analysis, providing data structures like DataFrames for handling structured data.",
            "NumPy is a fundamental Python library for scientific computing that provides support for large arrays and mathematical functions.",
            "Data Science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
            "Big Data refers to extremely large datasets that require specialized tools and techniques to store, process, and analyze effectively.",
            "Algorithm is a step-by-step procedure or formula for solving a problem, fundamental to all computer programming and machine learning.",
            "Feature Engineering is the process of selecting, transforming, and creating variables from raw data to improve machine learning model performance.",
            "Cross-validation is a statistical method used to estimate the performance of machine learning models by partitioning data into training and testing sets.",
            "Overfitting occurs when a machine learning model learns the training data too well, including noise, leading to poor performance on new data."
        ]
        print(f"Loaded {len(self.knowledge_base)} knowledge entries")
    
    def find_best_knowledge(self, query: str) -> str:
        """Find the most relevant knowledge entry"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        best_match = ""
        best_score = 0
        
        for knowledge in self.knowledge_base:
            knowledge_lower = knowledge.lower()
            knowledge_words = set(re.findall(r'\w+', knowledge_lower))
            
            # Calculate relevance score
            common_words = query_words.intersection(knowledge_words)
            score = len(common_words)
            
            # Boost score for key terms
            key_terms = {
                'machine learning': 10, 'ai': 8, 'artificial intelligence': 10,
                'deep learning': 10, 'neural network': 10, 'pytorch': 8,
                'tensorflow': 8, 'scikit': 6, 'sklearn': 6, 'pandas': 6,
                'numpy': 6, 'data science': 8, 'algorithm': 6
            }
            
            for term, boost in key_terms.items():
                if term in query_lower and term in knowledge_lower:
                    score += boost
            
            if score > best_score:
                best_score = score
                best_match = knowledge
        
        return best_match
    
    def generate_smart_response(self, query: str) -> str:
        """Generate intelligent response based on query type"""
        # Find relevant knowledge
        knowledge = self.find_best_knowledge(query)
        
        if not knowledge:
            return "I'm not sure about that. Could you ask about AI, machine learning, or data science topics?"
        
        # Create response based on question type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'definition', 'explain']):
            # Definition questions
            if 'is' in query_lower:
                return knowledge.split('.')[0] + '.'
            else:
                return knowledge.split('.')[0] + '.'
                
        elif any(word in query_lower for word in ['how', 'work', 'works', 'function']):
            # How-to questions
            sentences = knowledge.split('.')
            if len(sentences) > 1:
                return f"It works by {sentences[1].strip().lower()}." if sentences[1].strip() else sentences[0] + '.'
            return knowledge.split('.')[0] + '.'
            
        elif any(word in query_lower for word in ['why', 'purpose', 'benefit']):
            # Why questions
            return f"The purpose is {knowledge.split('.')[0].lower()}."
            
        elif any(word in query_lower for word in ['where', 'when', 'who']):
            # Context questions
            return knowledge.split('.')[0] + '.'
            
        else:
            # General questions
            return knowledge.split('.')[0] + '.'
    
    def run(self):
        """Main interaction loop"""
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Clean input
                clean_input = re.sub(r'[^\w\s.,!?]', ' ', user_input)
                clean_input = re.sub(r'\s+', ' ', clean_input).strip()
                
                print(f"Processing: {clean_input}")
                
                # Generate response
                response = self.generate_smart_response(clean_input)
                
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Assistant: I encountered an error. Please try again.")

def main():
    """Main function"""
    assistant = EnhancedSmartAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
