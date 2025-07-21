#!/usr/bin/env python3
"""
Ultimate Smart Assistant with Working Wikipedia Integration
When the model doesn't know something, it automatically searches Wikipedia for the answer
"""

import os
import re
import requests
import json
from typing import Optional, List, Dict, Any

class WorkingWikipediaSearcher:
    """Working Wikipedia searcher for unknown queries"""
    
    def __init__(self):
        self.search_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'iLLuMinator-Assistant/1.0 (Educational Research Project)'
        })
    
    def search_and_answer(self, query: str) -> Optional[str]:
        """Search Wikipedia and provide intelligent answer"""
        try:
            # Clean the query for better search
            search_term = self._extract_search_term(query)
            print(f"🔍 Searching Wikipedia for: {search_term}")
            
            # Search for pages
            page_titles = self._search_pages(search_term)
            
            if page_titles:
                # Get content from the best matching page
                for title in page_titles[:2]:  # Try first 2 results
                    content = self._get_page_content(title)
                    if content:
                        # Format answer based on question type
                        return self._format_answer(query, content, title)
            
            return None
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return None
    
    def _extract_search_term(self, query: str) -> str:
        """Extract the main search term from the query"""
        # Remove question words
        query = re.sub(r'\b(what|who|when|where|why|how|is|are|was|were|did|does|do|founded|created)\b', '', query.lower())
        
        # Clean punctuation
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # If query is too short, keep original
        if len(query) < 3:
            return query
        
        # Extract key terms
        words = query.split()
        
        # Prioritize company names, proper nouns, etc.
        important_words = []
        for word in words:
            if len(word) > 2 and word not in ['the', 'and', 'for', 'with', 'from']:
                important_words.append(word)
        
        # Return the most relevant term
        if important_words:
            # If we have Nvidia, Tesla, etc., prioritize those
            for word in important_words:
                if word.lower() in ['nvidia', 'tesla', 'google', 'microsoft', 'apple', 'facebook', 'meta']:
                    return word.capitalize()
            return important_words[0]
        
        return query
    
    def _search_pages(self, search_term: str) -> List[str]:
        """Search for Wikipedia pages"""
        try:
            params = {
                'action': 'opensearch',
                'search': search_term,
                'limit': 5,
                'format': 'json'
            }
            
            response = self.session.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # OpenSearch returns [query, [titles], [descriptions], [urls]]
            if len(data) >= 2 and data[1]:
                return data[1]
            
            return []
            
        except Exception as e:
            print(f"Search pages error: {e}")
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
                        
                        # Clean the extract
                        extract = re.sub(r'\s+', ' ', extract).strip()
                        
                        return extract
            
            return None
            
        except Exception as e:
            print(f"Get page content error: {e}")
            return None
    
    def _format_answer(self, query: str, content: str, title: str) -> str:
        """Format the answer based on question type"""
        query_lower = query.lower()
        sentences = content.split('. ')
        
        # Limit to first few sentences for concise answers
        if len(sentences) > 3:
            content_summary = '. '.join(sentences[:3]) + '.'
        else:
            content_summary = content
        
        # Format based on question type
        if any(word in query_lower for word in ['who', 'founded', 'created', 'established', 'started']):
            # Look for founding information
            for sentence in sentences[:5]:
                if any(word in sentence.lower() for word in ['founded', 'established', 'created', 'started', 'co-founded']):
                    return f"According to Wikipedia: {sentence.strip()}."
            
            # Fallback to general info
            return f"According to Wikipedia: {sentences[0]}." if sentences else f"According to Wikipedia: {content_summary}"
        
        elif any(word in query_lower for word in ['when', 'year', 'date']):
            # Look for dates
            for sentence in sentences[:5]:
                if re.search(r'\b(19|20)\d{2}\b', sentence):
                    return f"According to Wikipedia: {sentence.strip()}."
            
            return f"According to Wikipedia: {sentences[0]}." if sentences else f"According to Wikipedia: {content_summary}"
        
        elif any(word in query_lower for word in ['what', 'define', 'definition']):
            # Definition
            return f"According to Wikipedia: {sentences[0]}." if sentences else f"According to Wikipedia: {content_summary}"
        
        else:
            # General answer
            return f"According to Wikipedia: {content_summary}"

class FinalSmartAssistant:
    """The final intelligent assistant with working Wikipedia integration"""
    
    def __init__(self):
        print("🌟 Final Smart Assistant with Wikipedia Integration")
        print("=" * 65)
        print("🧠 Built-in knowledge + 🌐 Real-time Wikipedia search!")
        print("🚀 Can answer ANY question!")
        print("Type 'quit' to exit")
        print("=" * 65)
        
        self.wikipedia = WorkingWikipediaSearcher()
        self.setup_knowledge_base()
        
        print("\n✨ Final Smart Assistant ready!")
        print("Ask me ANYTHING!")
    
    def setup_knowledge_base(self):
        """Comprehensive built-in knowledge base"""
        self.knowledge = {
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It includes machine learning, natural language processing, computer vision, and robotics.",
            
            "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions.",
            
            "deep learning": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model complex patterns in data. It excels at image recognition, natural language processing, and speech recognition.",
            
            "neural networks": "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information through weighted connections and activation functions.",
            
            "transformer": "Transformers are neural network architectures that use self-attention mechanisms to process sequential data in parallel. They form the basis of models like GPT, BERT, and T5.",
            
            "pytorch": "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides dynamic computation graphs and automatic differentiation for deep learning.",
            
            "tensorflow": "TensorFlow is an open-source machine learning framework developed by Google. It provides comprehensive tools for building and deploying machine learning models at scale.",
            
            "python": "Python is a high-level programming language known for its simplicity and versatility. It's widely used in web development, data science, machine learning, and automation.",
            
            "data science": "Data Science is an interdisciplinary field that uses scientific methods and algorithms to extract knowledge and insights from structured and unstructured data.",
            
            "algorithm": "An algorithm is a step-by-step procedure for solving a problem or completing a task. Algorithms are fundamental to computer programming and problem-solving."
        }
        
        print(f"📚 Loaded {len(self.knowledge)} built-in knowledge topics")
    
    def find_knowledge(self, query: str) -> Optional[str]:
        """Find relevant knowledge in the built-in database"""
        query_lower = query.lower()
        
        # Direct topic matching
        for topic, description in self.knowledge.items():
            if topic in query_lower:
                return self._format_knowledge_response(query, topic, description)
        
        # Keyword matching
        query_words = set(query_lower.split())
        for topic, description in self.knowledge.items():
            topic_words = set(topic.split())
            if topic_words.intersection(query_words):
                return self._format_knowledge_response(query, topic, description)
        
        return None
    
    def _format_knowledge_response(self, query: str, topic: str, description: str) -> str:
        """Format built-in knowledge response"""
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ['what is', 'what are', 'define']):
            return f"{topic.title()} is {description.lower()}"
        elif any(phrase in query_lower for phrase in ['how does', 'how do']):
            return f"{topic.title()} works by {description.lower()}"
        else:
            return f"{description}"
    
    def generate_response(self, query: str) -> str:
        """Generate intelligent response with Wikipedia fallback"""
        
        # Step 1: Check built-in knowledge
        knowledge_response = self.find_knowledge(query)
        if knowledge_response:
            print("💡 Using built-in knowledge")
            return knowledge_response
        
        # Step 2: Search Wikipedia
        print("🔍 Searching Wikipedia...")
        wikipedia_response = self.wikipedia.search_and_answer(query)
        if wikipedia_response:
            return wikipedia_response
        
        # Step 3: Helpful fallback
        return f"I couldn't find specific information about '{query}'. Try asking about AI, technology, companies, or historical topics - I can search Wikipedia for almost anything!"
    
    def run(self):
        """Main interaction loop"""
        print("\n🎯 Ready! I can answer using:")
        print("   📚 Built-in AI/ML knowledge")
        print("   🌐 Wikipedia search for everything else")
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input(f"\n🎯 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("🤖 Assistant: Thank you for using the Final Smart Assistant! Goodbye!")
                    break
                
                if not user_input:
                    print("🤖 Assistant: Please ask me a question!")
                    continue
                
                conversation_count += 1
                print(f"🤔 Processing question #{conversation_count}...")
                
                # Generate response
                response = self.generate_response(user_input)
                
                print(f"\n🤖 Assistant: {response}")
                
                # Helpful tips
                if conversation_count % 5 == 0:
                    print("\n💡 Tip: I can answer questions about companies, people, history, science, and technology!")
                
            except KeyboardInterrupt:
                print("\n🤖 Assistant: Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🤖 Assistant: Sorry, I encountered an error. Please try again.")

def main():
    """Run the final smart assistant"""
    
    # Test internet connection
    try:
        response = requests.get("https://en.wikipedia.org", timeout=5)
        if response.status_code == 200:
            print("✅ Wikipedia connection verified!")
        else:
            print("⚠️  Wikipedia connection issue - some features may be limited")
    except:
        print("⚠️  No internet connection - Wikipedia features disabled")
    
    assistant = FinalSmartAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
