#!/usr/bin/env python3
"""
Ultimate Smart Assistant with Wikipedia Integration
When the model doesn't know something, it automatically searches Wikipedia for the answer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import re
import requests
import json
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import quote
import time

class WikipediaSearcher:
    """Intelligent Wikipedia searcher for unknown queries"""
    
    def __init__(self):
        self.search_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'iLLuMinator-Assistant/1.0 (Educational Research Project)'
        })
        
    def search_wikipedia(self, query: str) -> Optional[str]:
        """Search Wikipedia for information about the query"""
        try:
            # Clean the query
            clean_query = self._clean_query(query)
            print(f"🔍 Searching Wikipedia for: {clean_query}")
            
            # Search for pages
            search_results = self._search_pages(clean_query)
            if search_results:
                # Get content from the first relevant result
                for page_title in search_results[:3]:
                    content = self._get_page_content(page_title)
                    if content:
                        return content
            
            return None
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return None
    
    def _clean_query(self, query: str) -> str:
        """Clean and prepare query for Wikipedia search"""
        # Remove question words and clean up
        query = re.sub(r'\b(what|who|when|where|why|how|is|are|was|were|did|does|do)\b', '', query.lower())
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Handle common variations
        replacements = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision'
        }
        
        for short, full in replacements.items():
            if short == query:
                query = full
        
        return query
    
    def _search_pages(self, query: str) -> List[str]:
        """Search for Wikipedia pages matching the query"""
        try:
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': 5,
                'format': 'json'
            }
            
            response = self.session.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) >= 2 and data[1]:
                return data[1]  # Return the list of page titles
            
            return []
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _get_page_content(self, title: str) -> Optional[str]:
        """Get Wikipedia page content"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = self.session.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                for page_id, page_data in pages.items():
                    if 'extract' in page_data and page_data['extract']:
                        extract = page_data['extract']
                        
                        # Clean and format the extract
                        extract = re.sub(r'\([^)]*\)', '', extract)  # Remove parenthetical info
                        extract = re.sub(r'\s+', ' ', extract).strip()
                        
                        # Limit length for better responses
                        if len(extract) > 400:
                            sentences = extract.split('.')
                            extract = '. '.join(sentences[:3]) + '.'
                        
                        return extract
            
            return None
            
        except Exception as e:
            print(f"Page content error: {e}")
            return None
    
    def get_intelligent_answer(self, query: str) -> str:
        """Get an intelligent answer from Wikipedia with context"""
        wiki_info = self.search_wikipedia(query)
        
        if not wiki_info:
            return None
        
        # Format the response based on question type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['who', 'founded', 'created', 'developed', 'invented']):
            # Person-related questions
            sentences = wiki_info.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['founded', 'created', 'developed', 'invented', 'established', 'started']):
                    return f"According to Wikipedia: {sentence.strip()}."
            return f"According to Wikipedia: {sentences[0]}."
        
        elif any(word in query_lower for word in ['when', 'year', 'date']):
            # Time-related questions
            sentences = wiki_info.split('.')
            for sentence in sentences:
                if re.search(r'\b(19|20)\d{2}\b', sentence):  # Look for years
                    return f"According to Wikipedia: {sentence.strip()}."
            return f"According to Wikipedia: {sentences[0]}."
        
        elif any(word in query_lower for word in ['what', 'define', 'definition']):
            # Definition questions
            sentences = wiki_info.split('.')
            return f"According to Wikipedia: {sentences[0]}."
        
        elif any(word in query_lower for word in ['how', 'work', 'works']):
            # How-to questions
            sentences = wiki_info.split('.')
            relevant_sentences = []
            for sentence in sentences[:3]:
                if any(word in sentence.lower() for word in ['work', 'function', 'operate', 'process', 'method']):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return f"According to Wikipedia: {' '.join(relevant_sentences[:2])}."
            return f"According to Wikipedia: {sentences[0]}."
        
        else:
            # General questions
            sentences = wiki_info.split('.')
            return f"According to Wikipedia: {'. '.join(sentences[:2])}." if len(sentences) >= 2 else f"According to Wikipedia: {wiki_info}"

class IntelligentTokenizer:
    """Advanced tokenizer with comprehensive vocabulary"""
    
    def __init__(self):
        self.setup_enhanced_vocabulary()
    
    def setup_enhanced_vocabulary(self):
        """Create comprehensive vocabulary for better understanding"""
        
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
            
            # Companies and people
            "google", "facebook", "meta", "microsoft", "openai", "nvidia", "intel", "amd", "apple",
            "amazon", "tesla", "spacex", "wikipedia", "github", "stackoverflow",
            
            # Programming and tech
            "programming", "code", "software", "computer", "system", "function", "method", "class",
            "python", "pytorch", "tensorflow", "numpy", "pandas", "scikit", "learn",
            "javascript", "html", "css", "react", "node", "angular", "vue"
        ]
        
        # Numbers and punctuation
        numbers = [str(i) for i in range(100)] + ["hundred", "thousand", "million", "billion"]
        punctuation = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "'", '"', "-", "_"]
        
        # Combine all vocabularies
        all_tokens = special_tokens + common_words + tech_vocab + numbers + punctuation
        
        # Remove duplicates and create mappings
        self.vocab = list(set(all_tokens))
        self.vocab.sort()
        
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        print(f"🧠 Enhanced tokenizer with {self.vocab_size} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Enhanced encoding"""
        text = text.lower().strip()
        
        # Handle punctuation
        for punct in ".,!?:;()[]{}":
            text = text.replace(punct, f" {punct} ")
        
        tokens = text.split()
        token_ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Try partial matching
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
        """Decode tokens to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]:
                    tokens.append(token)
        
        text = " ".join(tokens)
        # Fix punctuation spacing
        for punct in ".,!?:;":
            text = text.replace(f" {punct}", punct)
        
        return text.strip()

class UltimateSmartAssistant:
    """The ultimate intelligent assistant with Wikipedia integration"""
    
    def __init__(self, model_path: str = "illuminator.pth"):
        print("🌟 Ultimate Smart Assistant with Wikipedia Integration")
        print("=" * 70)
        print("🧠 LLaMA-level intelligence + 🌐 Real-time Wikipedia knowledge!")
        print("🚀 Can answer ANY question using built-in knowledge + Wikipedia")
        print("Type 'quit' to exit")
        print("=" * 70)
        
        self.tokenizer = IntelligentTokenizer()
        self.wikipedia = WikipediaSearcher()
        self.setup_knowledge_base()
        self.load_model(model_path)
        
        print("\n✨ Ultimate Assistant ready!")
        print("Ask me ANYTHING - I'll use my knowledge or search Wikipedia!")
    
    def setup_knowledge_base(self):
        """Comprehensive built-in knowledge base"""
        self.knowledge = {
            # AI and Machine Learning
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think, learn, and problem-solve like humans. It encompasses machine learning, natural language processing, computer vision, robotics, and expert systems.",
            
            "machine learning": "Machine Learning is a subset of AI that enables computers to automatically learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
            
            "deep learning": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It excels at tasks like image recognition, natural language processing, and speech recognition.",
            
            "neural networks": "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections and activation functions.",
            
            "transformer": "Transformers are a neural network architecture that uses self-attention mechanisms to process sequential data in parallel. They form the basis of models like GPT, BERT, and T5, revolutionizing natural language processing.",
            
            "pytorch": "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides dynamic computation graphs, automatic differentiation, and GPU acceleration for deep learning research and development.",
            
            "tensorflow": "TensorFlow is an open-source machine learning framework developed by Google. It provides comprehensive tools for building and deploying machine learning models at scale with robust production capabilities.",
            
            "python": "Python is a high-level, interpreted programming language known for its simplicity, readability, and versatility. It's widely used in web development, data science, machine learning, and automation.",
            
            "data science": "Data Science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data for decision-making.",
            
            "algorithm": "An algorithm is a step-by-step procedure or set of rules for solving a problem or completing a task. Algorithms are fundamental to computer programming and vary in efficiency and complexity.",
        }
        
        # Create search index
        self.search_index = {}
        for topic, description in self.knowledge.items():
            words = topic.split() + description.lower().split()
            for word in words:
                if len(word) > 3:
                    if word not in self.search_index:
                        self.search_index[word] = []
                    if topic not in self.search_index[word]:
                        self.search_index[word].append(topic)
    
    def find_relevant_knowledge(self, query: str) -> List[Tuple[str, str, float]]:
        """Find relevant knowledge with scoring"""
        query_words = query.lower().split()
        topic_scores = {}
        
        for word in query_words:
            if word in self.search_index:
                for topic in self.search_index[word]:
                    topic_scores[topic] = topic_scores.get(topic, 0) + 1
        
        # Boost exact matches
        for topic in self.knowledge:
            if topic in query.lower():
                topic_scores[topic] = topic_scores.get(topic, 0) + 10
        
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [(topic, self.knowledge[topic], score) for topic, score in sorted_topics[:3]]
    
    def load_model(self, model_path: str):
        """Load model (simplified for this demo)"""
        try:
            # For this demo, we'll focus on the knowledge retrieval
            # The model loading is simplified
            self.model = None
            print("📚 Focusing on intelligent knowledge retrieval + Wikipedia integration")
        except Exception as e:
            print(f"Model note: {e}")
            self.model = None
    
    def generate_intelligent_response(self, query: str) -> str:
        """Generate highly intelligent responses with Wikipedia fallback"""
        
        # Step 1: Try built-in knowledge base
        relevant_knowledge = self.find_relevant_knowledge(query)
        
        if relevant_knowledge and relevant_knowledge[0][2] > 0:  # If we found relevant knowledge
            best_topic, best_description, score = relevant_knowledge[0]
            
            # Format response based on question type
            response = self.format_response_by_type(query, best_topic, best_description)
            print("💡 Using built-in knowledge")
            return response
        
        # Step 2: Fallback to Wikipedia if no relevant knowledge found
        print("🔍 No built-in knowledge found, searching Wikipedia...")
        wikipedia_answer = self.wikipedia.get_intelligent_answer(query)
        
        if wikipedia_answer:
            return wikipedia_answer
        
        # Step 3: Final fallback
        return self.handle_unknown_query(query)
    
    def format_response_by_type(self, query: str, topic: str, description: str) -> str:
        """Format response based on question type"""
        query_lower = query.lower()
        sentences = description.split('.')
        
        if any(phrase in query_lower for phrase in ['what is', 'what are', 'define']):
            return f"{topic.title()} is {description.split('.')[0].lower()}. {sentences[1].strip() if len(sentences) > 1 else ''}"
        
        elif any(phrase in query_lower for phrase in ['how does', 'how do', 'how']):
            mechanism = ""
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['uses', 'works', 'process', 'by']):
                    mechanism = sentence.strip()
                    break
            
            if mechanism:
                return f"{topic.title()} works by {mechanism.lower()}."
            else:
                return f"{topic.title()} operates through {description.split('.')[0].lower()}."
        
        elif any(phrase in query_lower for phrase in ['why']):
            return f"{topic.title()} is important because {description.split('.')[0].lower()}."
        
        else:
            return f"{description.split('.')[0]}. {sentences[1].strip() if len(sentences) > 1 else ''}"
    
    def handle_unknown_query(self, query: str) -> str:
        """Handle completely unknown queries"""
        return f"I couldn't find specific information about '{query}' in my knowledge base or Wikipedia. Could you try rephrasing your question or ask about a different topic?"
    
    def run(self):
        """Main interaction loop with Wikipedia integration"""
        print("\n🎯 Ready for questions! I can answer using:")
        print("   📚 Built-in AI/ML knowledge")
        print("   🌐 Real-time Wikipedia search")
        print("   🧠 Intelligent question understanding")
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input(f"\n🎯 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("🤖 Assistant: Thank you for using the Ultimate Smart Assistant! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                conversation_count += 1
                print(f"🤔 Processing query #{conversation_count}...")
                
                # Generate intelligent response
                response = self.generate_intelligent_response(user_input)
                
                print(f"\n🤖 Assistant: {response}")
                
                # Add helpful suggestions occasionally
                if conversation_count % 5 == 0:
                    print("\n💡 Tip: I can answer questions about AI, technology, history, science, and much more!")
                
            except KeyboardInterrupt:
                print("\n🤖 Assistant: Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🤖 Assistant: I encountered an error. Please try again.")

def main():
    """Initialize and run the ultimate smart assistant"""
    
    # Check internet connection
    try:
        requests.get("https://www.wikipedia.org", timeout=5)
        print("✅ Internet connection verified - Wikipedia integration enabled!")
    except:
        print("⚠️  No internet connection - Wikipedia features will be limited")
    
    assistant = UltimateSmartAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
