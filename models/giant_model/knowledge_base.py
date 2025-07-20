"""
Comprehensive Knowledge Base for Giant Transformer
Extensive built-in knowledge to minimize Wikipedia dependency
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class KnowledgeEntry:
    """Structure for knowledge entries"""
    topic: str
    description: str
    details: str
    related_topics: List[str]
    keywords: List[str]
    category: str

class ComprehensiveKnowledgeBase:
    """Massive knowledge base for the giant model"""
    
    def __init__(self):
        self.knowledge_entries: Dict[str, KnowledgeEntry] = {}
        self.category_index: Dict[str, List[str]] = {}
        self.keyword_index: Dict[str, List[str]] = {}
        self.setup_comprehensive_knowledge()
        print(f"🧠 Loaded comprehensive knowledge base with {len(self.knowledge_entries)} entries")
    
    def setup_comprehensive_knowledge(self):
        """Setup massive knowledge base covering all major topics"""
        
        # AI and Machine Learning
        ai_ml_knowledge = [
            {
                "topic": "artificial intelligence",
                "description": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think, learn, and problem-solve like humans.",
                "details": "AI encompasses various subfields including machine learning, natural language processing, computer vision, robotics, expert systems, and neural networks. It aims to create systems that can perform tasks that typically require human intelligence such as visual perception, speech recognition, decision-making, and language translation. Modern AI uses techniques like deep learning, reinforcement learning, and neural networks to achieve human-like performance in specific domains.",
                "related_topics": ["machine learning", "deep learning", "neural networks", "computer vision", "natural language processing"],
                "keywords": ["ai", "artificial", "intelligence", "machine", "learning", "automation", "cognitive"],
                "category": "AI/ML"
            },
            {
                "topic": "machine learning",
                "description": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.",
                "details": "Machine learning algorithms build mathematical models based on training data to make predictions or decisions. There are three main types: supervised learning (learning with labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through rewards and penalties). Common algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. Applications include recommendation systems, fraud detection, image recognition, and predictive analytics.",
                "related_topics": ["supervised learning", "unsupervised learning", "reinforcement learning", "deep learning", "neural networks"],
                "keywords": ["machine", "learning", "ml", "algorithm", "training", "prediction", "model"],
                "category": "AI/ML"
            },
            {
                "topic": "deep learning",
                "description": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
                "details": "Deep learning architectures include feedforward neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), transformers, and generative adversarial networks (GANs). These models excel at tasks like image recognition, natural language processing, speech recognition, and game playing. Key innovations include backpropagation for training, dropout for regularization, batch normalization for stability, and attention mechanisms for focusing on relevant information.",
                "related_topics": ["neural networks", "cnn", "rnn", "transformer", "backpropagation"],
                "keywords": ["deep", "learning", "neural", "network", "layers", "cnn", "rnn", "transformer"],
                "category": "AI/ML"
            },
            {
                "topic": "neural networks",
                "description": "Neural Networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections.",
                "details": "Neural networks consist of layers of neurons (nodes) connected by weighted edges. Information flows from input layer through hidden layers to output layer. Each neuron applies an activation function to the weighted sum of its inputs. Training involves adjusting weights through algorithms like gradient descent and backpropagation to minimize error. Types include perceptrons, multilayer perceptrons, convolutional networks, recurrent networks, and transformer networks.",
                "related_topics": ["perceptron", "backpropagation", "gradient descent", "activation function", "deep learning"],
                "keywords": ["neural", "network", "neuron", "layer", "weight", "activation", "gradient"],
                "category": "AI/ML"
            },
            {
                "topic": "transformer",
                "description": "Transformers are neural network architectures that use self-attention mechanisms to process sequential data in parallel, revolutionizing natural language processing.",
                "details": "Introduced in 'Attention Is All You Need', transformers use self-attention to relate different positions in a sequence. Key components include multi-head attention, positional encoding, layer normalization, and feed-forward networks. Transformers enable parallel processing of sequences, leading to faster training and better performance. They form the basis of models like BERT, GPT, T5, and modern language models. Applications include machine translation, text summarization, question answering, and text generation.",
                "related_topics": ["attention mechanism", "bert", "gpt", "self-attention", "positional encoding"],
                "keywords": ["transformer", "attention", "bert", "gpt", "language", "model", "nlp"],
                "category": "AI/ML"
            }
        ]
        
        # Technology and Programming
        tech_knowledge = [
            {
                "topic": "python",
                "description": "Python is a high-level, interpreted programming language known for its simplicity, readability, and versatility.",
                "details": "Created by Guido van Rossum in 1991, Python emphasizes code readability and simplicity. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python has extensive standard libraries and third-party packages available through PyPI. It's widely used in web development (Django, Flask), data science (Pandas, NumPy), machine learning (TensorFlow, PyTorch), automation, and scientific computing. Key features include dynamic typing, garbage collection, and cross-platform compatibility.",
                "related_topics": ["programming", "django", "flask", "pandas", "numpy", "pip"],
                "keywords": ["python", "programming", "language", "code", "script", "development"],
                "category": "Programming"
            },
            {
                "topic": "pytorch",
                "description": "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab, providing dynamic computation graphs and automatic differentiation.",
                "details": "PyTorch offers a dynamic computational graph that builds on-the-fly, making it intuitive for research and debugging. Key features include tensor operations with GPU acceleration, automatic differentiation through autograd, neural network modules through torch.nn, and distributed training capabilities. It's popular in research due to its flexibility and pythonic interface. PyTorch includes TorchVision for computer vision, TorchText for NLP, and TorchAudio for audio processing.",
                "related_topics": ["machine learning", "tensor", "gpu", "neural networks", "facebook"],
                "keywords": ["pytorch", "torch", "tensor", "autograd", "facebook", "ai", "research"],
                "category": "AI/ML"
            },
            {
                "topic": "tensorflow",
                "description": "TensorFlow is an open-source machine learning framework developed by Google for building and deploying machine learning models at scale.",
                "details": "TensorFlow provides comprehensive tools for machine learning workflow from research to production. It supports both eager execution for immediate iteration and graph execution for performance. Key components include Keras for high-level APIs, TensorFlow Serving for model deployment, TensorFlow Lite for mobile/embedded devices, and TensorBoard for visualization. It offers robust distributed training, quantization for efficiency, and deployment across various platforms including cloud, mobile, and edge devices.",
                "related_topics": ["google", "keras", "machine learning", "deployment", "tensorboard"],
                "keywords": ["tensorflow", "tf", "google", "keras", "machine", "learning", "deployment"],
                "category": "AI/ML"
            }
        ]
        
        # Companies and Organizations
        company_knowledge = [
            {
                "topic": "nvidia",
                "description": "NVIDIA Corporation is an American multinational technology company known for designing graphics processing units (GPUs) and system on chip units (SoCs).",
                "details": "Founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem in Santa Clara, California. Initially focused on graphics cards for gaming, NVIDIA became a leader in AI and machine learning acceleration through CUDA parallel computing platform. Key products include GeForce gaming GPUs, Quadro professional graphics, Tesla data center GPUs, and Tegra mobile processors. The company has been crucial in the AI revolution, providing hardware that accelerates deep learning training and inference.",
                "related_topics": ["gpu", "cuda", "jensen huang", "graphics", "ai acceleration"],
                "keywords": ["nvidia", "gpu", "graphics", "cuda", "jensen", "huang", "ai", "acceleration"],
                "category": "Technology Companies"
            },
            {
                "topic": "google",
                "description": "Google LLC is an American multinational technology company specializing in Internet-related services and products.",
                "details": "Founded in 1998 by Larry Page and Sergey Brin while PhD students at Stanford University. Started as a search engine, Google has expanded to offer email (Gmail), cloud computing (Google Cloud), mobile operating system (Android), web browser (Chrome), and numerous other services. Google is a subsidiary of Alphabet Inc. and generates revenue primarily through advertising. The company has made significant contributions to AI research through DeepMind acquisition and development of TensorFlow, BERT, and other AI technologies.",
                "related_topics": ["search engine", "larry page", "sergey brin", "alphabet", "android"],
                "keywords": ["google", "search", "larry", "page", "sergey", "brin", "alphabet"],
                "category": "Technology Companies"
            },
            {
                "topic": "microsoft",
                "description": "Microsoft Corporation is an American multinational technology corporation producing computer software, consumer electronics, and personal computers.",
                "details": "Founded in 1975 by Bill Gates and Paul Allen in Albuquerque, New Mexico. Microsoft developed MS-DOS and later Windows operating system, becoming the dominant PC software company. Key products include Windows OS, Microsoft Office suite, Azure cloud platform, Xbox gaming console, and Surface devices. The company has invested heavily in AI through partnerships with OpenAI and development of Cortana, Azure AI services, and integration of AI across its product portfolio.",
                "related_topics": ["windows", "bill gates", "paul allen", "office", "azure"],
                "keywords": ["microsoft", "windows", "bill", "gates", "office", "azure", "xbox"],
                "category": "Technology Companies"
            }
        ]
        
        # Science and Mathematics
        science_knowledge = [
            {
                "topic": "mathematics",
                "description": "Mathematics is the study of numbers, shapes, patterns, and logical reasoning, forming the foundation of scientific understanding.",
                "details": "Mathematics encompasses various branches including arithmetic, algebra, geometry, calculus, statistics, and discrete mathematics. It provides the language and tools for describing natural phenomena, solving problems, and making predictions. Mathematical concepts are essential in computer science, physics, engineering, economics, and many other fields. Key areas include number theory, linear algebra, differential equations, topology, and mathematical logic.",
                "related_topics": ["algebra", "geometry", "calculus", "statistics", "logic"],
                "keywords": ["mathematics", "math", "numbers", "equations", "algebra", "geometry"],
                "category": "Science"
            },
            {
                "topic": "physics",
                "description": "Physics is the fundamental science that studies matter, energy, and their interactions in the universe.",
                "details": "Physics seeks to understand how the universe behaves through observation, experimentation, and mathematical modeling. Major branches include classical mechanics, thermodynamics, electromagnetism, quantum mechanics, and relativity. Physics has led to numerous technological advances including electricity, electronics, lasers, nuclear energy, and modern computing. Key principles include conservation laws, wave-particle duality, uncertainty principle, and the fundamental forces of nature.",
                "related_topics": ["mechanics", "thermodynamics", "quantum", "relativity", "electromagnetism"],
                "keywords": ["physics", "matter", "energy", "quantum", "relativity", "mechanics"],
                "category": "Science"
            }
        ]
        
        # History and Culture
        history_knowledge = [
            {
                "topic": "world war ii",
                "description": "World War II was a global war that lasted from 1939 to 1945, involving most of the world's nations and resulting in 70-85 million fatalities.",
                "details": "WWII was the deadliest conflict in human history, fought between the Axis powers (Germany, Italy, Japan) and the Allied powers (Britain, Soviet Union, United States, China, and others). Key events include the invasion of Poland, Battle of Britain, attack on Pearl Harbor, Holocaust, D-Day invasion, and atomic bombings of Hiroshima and Nagasaki. The war ended with the surrender of Germany in May 1945 and Japan in September 1945, leading to major geopolitical changes and the establishment of the United Nations.",
                "related_topics": ["hitler", "holocaust", "pearl harbor", "dday", "hiroshima"],
                "keywords": ["world", "war", "wwii", "hitler", "allies", "axis", "holocaust"],
                "category": "History"
            }
        ]
        
        # Combine all knowledge
        all_knowledge = (ai_ml_knowledge + tech_knowledge + company_knowledge + 
                        science_knowledge + history_knowledge)
        
        # Process and index all knowledge
        for entry_data in all_knowledge:
            entry = KnowledgeEntry(**entry_data)
            self.knowledge_entries[entry.topic] = entry
            
            # Index by category
            if entry.category not in self.category_index:
                self.category_index[entry.category] = []
            self.category_index[entry.category].append(entry.topic)
            
            # Index by keywords
            for keyword in entry.keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(entry.topic)
    
    def search_knowledge(self, query: str) -> List[Tuple[str, KnowledgeEntry, float]]:
        """Search knowledge base and return ranked results"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        results = []
        
        for topic, entry in self.knowledge_entries.items():
            score = 0
            
            # Exact topic match (highest score)
            if topic in query_lower:
                score += 100
            
            # Keyword matching
            for keyword in entry.keywords:
                if keyword in query_lower:
                    score += 10
            
            # Word overlap scoring
            topic_words = set(re.findall(r'\w+', topic.lower()))
            desc_words = set(re.findall(r'\w+', entry.description.lower()))
            all_entry_words = topic_words.union(desc_words)
            
            overlap = query_words.intersection(all_entry_words)
            score += len(overlap) * 5
            
            # Partial matching
            for query_word in query_words:
                for entry_word in all_entry_words:
                    if query_word in entry_word or entry_word in query_word:
                        score += 2
            
            if score > 0:
                results.append((topic, entry, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def get_detailed_answer(self, query: str) -> Optional[str]:
        """Get a detailed answer from the knowledge base"""
        results = self.search_knowledge(query)
        
        if not results:
            return None
        
        # Get the best match
        topic, entry, score = results[0]
        
        # Format response based on question type
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ['what is', 'what are', 'define', 'definition']):
            return f"{entry.description} {entry.details[:200]}..."
        
        elif any(phrase in query_lower for phrase in ['who founded', 'who created', 'who started']):
            # Look for founding information in details
            details_lower = entry.details.lower()
            sentences = entry.details.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['founded', 'created', 'started', 'established']):
                    return f"According to my knowledge: {sentence.strip()}."
            return f"{entry.description} {entry.details[:150]}..."
        
        elif any(phrase in query_lower for phrase in ['when', 'year', 'date']):
            # Look for dates in details
            details = entry.details
            date_match = re.search(r'\b(19|20)\d{2}\b', details)
            if date_match:
                # Find sentence containing the date
                sentences = details.split('.')
                for sentence in sentences:
                    if date_match.group() in sentence:
                        return f"According to my knowledge: {sentence.strip()}."
            return f"{entry.description} {entry.details[:150]}..."
        
        elif any(phrase in query_lower for phrase in ['how', 'how does', 'how do']):
            return f"{entry.description} {entry.details[:200]}..."
        
        else:
            return f"{entry.description} {entry.details[:200]}..."
    
    def get_related_topics(self, topic: str) -> List[str]:
        """Get related topics for a given topic"""
        if topic in self.knowledge_entries:
            return self.knowledge_entries[topic].related_topics
        return []
    
    def get_category_topics(self, category: str) -> List[str]:
        """Get all topics in a category"""
        return self.category_index.get(category, [])

# Test the knowledge base
if __name__ == "__main__":
    kb = ComprehensiveKnowledgeBase()
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "Who founded NVIDIA?",
        "When was Google founded?",
        "How does machine learning work?",
        "What is PyTorch?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        answer = kb.get_detailed_answer(query)
        if answer:
            print(f"Answer: {answer}")
        else:
            print("No answer found in knowledge base")
