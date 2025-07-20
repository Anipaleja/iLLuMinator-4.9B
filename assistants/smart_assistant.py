#!/usr/bin/env python3
"""
Intelligent AI Assistant - Just ask anything!
"""

import torch
import json
import os
from typing import List, Dict, Any
from intelligent_rag import IntelligentRAGTransformer
from model.tokenizer import build_tokenizer, encode, decode
from data.prepare import load_data

class SmartAssistant:
    """
    Intelligent AI Assistant that can answer any question by:
    1. Understanding what you're asking
    2. Finding relevant information automatically
    3. Generating smart, contextual responses
    """
    
    def __init__(self, 
                 model_path: str = None,
                 knowledge_sources: List[str] = None,
                 config_path: str = "assistant_config.json"):
        
        self.config_path = config_path
        self.load_or_create_config()
        
        # Initialize knowledge base
        self.knowledge_base = self.load_knowledge_base(knowledge_sources)
        
        # Setup tokenizer
        self.setup_tokenizer()
        
        # Initialize intelligent model
        self.setup_model(model_path)
        
        # Conversation history for context
        self.conversation_history = []
        
        print("🤖 Smart Assistant initialized!")
        print("💡 I can help with any question - just ask!")
    
    def load_or_create_config(self):
        """Load or create configuration"""
        default_config = {
            "model_params": {
                "block_size": 1024,
                "n_embd": 512,
                "n_head": 16,
                "n_layer": 12,
                "dropout": 0.1
            },
            "generation_params": {
                "max_tokens": 200,
                "temperature": 0.7,
                "top_k": 50
            },
            "knowledge_sources": [
                "wikipedia_articles",
                "documentation",
                "books",
                "papers"
            ]
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def load_knowledge_base(self, sources: List[str] = None) -> List[str]:
        """Load comprehensive knowledge base"""
        knowledge = []
        
        # Default knowledge base with diverse topics
        default_knowledge = [
            # Technology & AI
            "Artificial Intelligence is the simulation of human intelligence by machines. It includes machine learning, deep learning, natural language processing, computer vision, and robotics.",
            "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to find patterns in data.",
            "Deep Learning uses artificial neural networks with multiple layers to model complex patterns. It has revolutionized computer vision, natural language processing, and speech recognition.",
            "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information.",
            "The Transformer architecture revolutionized NLP with self-attention mechanisms. It forms the basis of models like GPT, BERT, and T5.",
            
            # Science & Mathematics
            "Quantum Computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways impossible for classical computers.",
            "Calculus is the mathematical study of continuous change. It includes differential calculus (derivatives) and integral calculus (integrals).",
            "Linear Algebra deals with vector spaces and linear mappings between them. It's fundamental to machine learning, computer graphics, and quantum mechanics.",
            "Statistics is the discipline of collecting, analyzing, interpreting, and presenting data. It's essential for data science and scientific research.",
            "Physics is the natural science that studies matter, motion, energy, and their interactions. It includes mechanics, thermodynamics, electromagnetism, and quantum physics.",
            
            # Programming & Software
            "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, and AI.",
            "JavaScript is a programming language primarily used for web development. It enables interactive web pages and is essential for modern web applications.",
            "Git is a distributed version control system for tracking changes in source code during software development. It enables collaboration among developers.",
            "Software Engineering is the systematic approach to designing, developing, and maintaining software systems. It includes requirements analysis, design, testing, and deployment.",
            "Data Structures are ways of organizing and storing data efficiently. Common types include arrays, linked lists, stacks, queues, trees, and graphs.",
            
            # History & Culture
            "The Renaissance was a period of European cultural, artistic, political, and economic rebirth following the Middle Ages, roughly from the 14th to 17th centuries.",
            "World War II was a global war from 1939 to 1945 involving most of the world's nations. It was the deadliest conflict in human history.",
            "The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States from about 1760 to 1840.",
            "Ancient civilizations like Egypt, Mesopotamia, Greece, and Rome laid the foundations for modern society through innovations in government, philosophy, and technology.",
            
            # Health & Biology
            "DNA (Deoxyribonucleic Acid) is the hereditary material in humans and almost all organisms. It contains the instructions for building and maintaining life.",
            "The human brain contains approximately 86 billion neurons connected by trillions of synapses. It controls thought, memory, emotion, and bodily functions.",
            "Vaccines work by training the immune system to recognize and fight specific diseases without causing the disease itself.",
            "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen, forming the basis of most food chains.",
            
            # Economics & Business
            "Economics is the social science that studies how societies allocate scarce resources. It includes microeconomics and macroeconomics.",
            "Supply and demand is the fundamental economic model describing the relationship between the availability of a commodity and the desire for it.",
            "Entrepreneurship involves creating and running a business while taking financial risks in hope of profit. It drives innovation and economic growth.",
            "Cryptocurrency is a digital or virtual currency secured by cryptography. Bitcoin was the first decentralized cryptocurrency, created in 2009.",
            
            # Arts & Literature
            "Literature encompasses written works, especially those considered to have artistic or intellectual value. It includes poetry, novels, drama, and essays.",
            "Classical music refers to Western art music composed from roughly the 11th century to the present, including periods like Baroque, Classical, and Romantic.",
            "Visual arts include painting, sculpture, drawing, printmaking, and photography. They communicate ideas and emotions through visual elements.",
            
            # Philosophy & Psychology
            "Philosophy is the study of fundamental questions about existence, knowledge, values, reason, mind, and language. Major branches include ethics, logic, and metaphysics.",
            "Psychology is the scientific study of mind and behavior. It includes cognitive psychology, behavioral psychology, and developmental psychology.",
            "Critical thinking involves analyzing and evaluating information objectively to form well-reasoned judgments and decisions.",
            
            # Environment & Climate
            "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the Industrial Revolution.",
            "Renewable energy sources like solar, wind, hydro, and geothermal power provide sustainable alternatives to fossil fuels.",
            "Biodiversity refers to the variety of life on Earth, including diversity of species, ecosystems, and genetic variation within species.",
            
            # Current Technology Trends
            "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software.",
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records linked and secured using cryptography.",
            "Internet of Things (IoT) refers to the network of physical devices embedded with sensors, software, and connectivity to exchange data.",
            "Cybersecurity protects digital systems, networks, and data from digital attacks, unauthorized access, and data breaches."
        ]
        
        knowledge.extend(default_knowledge)
        
        # Add custom knowledge sources if provided
        if sources:
            for source in sources:
                if os.path.isfile(source):
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Split into chunks for better retrieval
                        chunks = self.chunk_text(content, max_length=500)
                        knowledge.extend(chunks)
        
        print(f"📚 Loaded {len(knowledge)} knowledge entries")
        return knowledge
    
    def chunk_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into manageable chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def setup_tokenizer(self):
        """Setup tokenizer"""
        try:
            # Load existing data for tokenizer
            text = load_data()
            self.stoi, self.itos = build_tokenizer(text)
            self.vocab_size = len(self.stoi)
            print(f"🔤 Tokenizer ready with vocabulary size: {self.vocab_size}")
        except:
            # Fallback to basic tokenizer
            print("⚠️  Using fallback tokenizer")
            self.vocab_size = 10000
            self.stoi = {f"token_{i}": i for i in range(self.vocab_size)}
            self.itos = {i: f"token_{i}" for i in range(self.vocab_size)}
    
    def setup_model(self, model_path: str = None):
        """Initialize the intelligent model"""
        print("🧠 Setting up intelligent model...")
        
        # Create model with knowledge base
        self.model = IntelligentRAGTransformer(
            vocab_size=self.vocab_size,
            knowledge_base=self.knowledge_base,
            **self.config["model_params"]
        )
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                # Load compatible weights
                model_dict = self.model.state_dict()
                filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict)
                print(f"✅ Loaded pre-trained weights from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model weights: {e}")
                print("🔄 Using randomly initialized weights")
        
        self.model.eval()
    
    def ask(self, question: str) -> str:
        """
        Main interface - ask any question and get an intelligent response
        """
        if not question.strip():
            return "I'm here to help! Please ask me anything."
        
        print(f"\n🤔 Thinking about: {question}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": question})
        
        # Generate response using intelligent model
        try:
            response = self.model.chat(
                query=question,
                **self.config["generation_params"]
            )
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            response = self.fallback_response(question)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def fallback_response(self, question: str) -> str:
        """Fallback response when main model fails"""
        # Simple keyword-based responses
        keywords_responses = {
            "hello": "Hello! I'm your AI assistant. How can I help you today?",
            "help": "I can answer questions on a wide range of topics including science, technology, history, and more. Just ask!",
            "thanks": "You're welcome! Feel free to ask me anything else.",
            "python": "Python is a versatile programming language great for beginners and experts alike. What would you like to know about it?",
            "ai": "Artificial Intelligence is fascinating! It involves creating machines that can think and learn. What aspect interests you?",
            "how": "That's a great question! I'd be happy to explain how something works. Could you be more specific?",
            "what": "I can help explain concepts, definitions, and provide information. What would you like to know about?",
            "why": "Understanding the 'why' behind things is important! I can help explain reasons and causes."
        }
        
        question_lower = question.lower()
        for keyword, response in keywords_responses.items():
            if keyword in question_lower:
                return response
        
        return "That's an interesting question! While I'm still learning, I'd be happy to help if you could rephrase or ask about a specific topic I might know about."
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary = "Conversation Summary:\n"
        for i, entry in enumerate(self.conversation_history[-6:], 1):  # Last 6 entries
            role = "You" if entry["role"] == "user" else "Assistant"
            content = entry["content"][:100] + "..." if len(entry["content"]) > 100 else entry["content"]
            summary += f"{i}. {role}: {content}\n"
        
        return summary
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared.")
    
    def save_conversation(self, filename: str = None):
        """Save conversation to file"""
        if not filename:
            filename = f"conversation_{len(self.conversation_history)}_messages.json"
        
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"💾 Conversation saved to {filename}")


def main():
    """Interactive chat interface"""
    print("🌟 Welcome to the Intelligent AI Assistant!")
    print("=" * 50)
    print("I can answer questions on any topic!")
    print("Type 'quit' to exit, 'clear' to clear history, 'summary' for conversation summary")
    print("=" * 50)
    
    # Initialize assistant
    assistant = SmartAssistant()
    
    while True:
        try:
            # Get user input
            question = input("\n💬 You: ").strip()
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting!")
                break
            elif question.lower() == 'clear':
                assistant.clear_conversation()
                continue
            elif question.lower() == 'summary':
                print(assistant.get_conversation_summary())
                continue
            elif question.lower() == 'save':
                assistant.save_conversation()
                continue
            
            # Get response
            response = assistant.ask(question)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Let's try that again!")


if __name__ == "__main__":
    main()
