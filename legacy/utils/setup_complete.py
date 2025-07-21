#!/usr/bin/env python3
"""
Complete Setup and Demo for Intelligent AI Assistant
"""

import os
import sys
import subprocess
import json

def install_requirements():
    """Install all required packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "torch",
        "sentence-transformers", 
        "faiss-cpu",
        "numpy",
        "transformers"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def setup_demo_data():
    """Setup demo data and configuration"""
    print("🔧 Setting up demo configuration...")
    
    # Create demo knowledge base
    demo_knowledge = [
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction.",
        "Machine Learning is a subset of AI that enables computers to learn automatically from data without being explicitly programmed.",
        "Deep Learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural Language Processing (NLP) helps computers understand, interpret, and generate human language in a valuable way.",
        "Computer Vision enables machines to interpret and understand visual information from the world, like images and videos.",
        "Python is a high-level programming language known for its simplicity and versatility in web development, data science, and AI.",
        "JavaScript is the programming language of the web, enabling interactive web pages and modern web applications.",
        "Git is a distributed version control system that tracks changes in source code during software development.",
        "The Internet works through interconnected networks using protocols like TCP/IP to route data between computers globally.",
        "Climate change refers to long-term shifts in global temperatures caused primarily by human activities and greenhouse gas emissions.",
        "Renewable energy comes from naturally replenishing sources like solar, wind, hydroelectric, and geothermal power.",
        "Quantum computing uses quantum mechanical phenomena to process information in ways impossible for classical computers.",
        "DNA contains the genetic instructions for the development and function of all known living organisms.",
        "The human brain has approximately 86 billion neurons that communicate through electrical and chemical signals.",
        "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
    ]
    
    # Save demo knowledge base
    with open("demo_knowledge.json", "w") as f:
        json.dump(demo_knowledge, f, indent=2)
    
    # Create assistant configuration
    config = {
        "model_params": {
            "block_size": 512,
            "n_embd": 256,
            "n_head": 8, 
            "n_layer": 4,
            "dropout": 0.1
        },
        "generation_params": {
            "max_tokens": 150,
            "temperature": 0.8,
            "top_k": 40
        },
        "demo_mode": True
    }
    
    with open("assistant_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Demo configuration created!")

def create_simple_tokenizer():
    """Create a simple tokenizer for demo purposes"""
    print("🔤 Setting up demo tokenizer...")
    
    # Create a basic vocabulary
    vocab = {}
    vocab_list = []
    
    # Common words and tokens
    common_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "what", "how", "why", "when", "where", "who", "which", "can", "will", "would", "should",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
        "this", "that", "these", "those", "here", "there", "now", "then", "yes", "no", "not",
        "computer", "science", "technology", "artificial", "intelligence", "machine", "learning",
        "neural", "network", "deep", "python", "programming", "data", "algorithm", "software",
        "hello", "help", "thanks", "please", "question", "answer", "explain", "understand"
    ]
    
    # Add common words
    for i, word in enumerate(common_words):
        vocab[word] = i
        vocab_list.append(word)
    
    # Add numbers and punctuation
    for i in range(100):
        token = str(i)
        vocab[token] = len(vocab_list)
        vocab_list.append(token)
    
    for punct in ".,!?;:()[]{}\"'-":
        vocab[punct] = len(vocab_list)
        vocab_list.append(punct)
    
    # Add special tokens
    special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>", "<CONTEXT>", "<QUERY>", "<ANSWER>"]
    for token in special_tokens:
        vocab[token] = len(vocab_list)
        vocab_list.append(token)
    
    # Pad to reasonable size
    while len(vocab_list) < 1000:
        token = f"<TOKEN_{len(vocab_list)}>"
        vocab[token] = len(vocab_list)
        vocab_list.append(token)
    
    # Save tokenizer
    tokenizer_data = {
        "vocab": vocab,
        "vocab_list": vocab_list,
        "vocab_size": len(vocab_list)
    }
    
    with open("demo_tokenizer.json", "w") as f:
        json.dump(tokenizer_data, f, indent=2)
    
    print(f"✅ Demo tokenizer created with {len(vocab_list)} tokens!")

def create_simple_demo():
    """Create a simplified demo that works without complex dependencies"""
    print("🎯 Creating simplified demo...")
    
    demo_code = '''#!/usr/bin/env python3
"""
Simplified AI Assistant Demo
"""

import json
import random
import re

class SimpleAssistant:
    """Simplified AI assistant for demo purposes"""
    
    def __init__(self):
        self.load_knowledge()
        self.conversation_history = []
        print("🤖 Simple AI Assistant ready!")
        print("💡 I can answer questions about technology, science, and more!")
    
    def load_knowledge(self):
        """Load knowledge base"""
        try:
            with open("demo_knowledge.json", "r") as f:
                self.knowledge = json.load(f)
        except:
            self.knowledge = [
                "I'm a demo AI assistant ready to help with questions!",
                "Artificial Intelligence helps computers think and learn like humans.",
                "Machine Learning allows computers to learn from data automatically.",
                "Programming languages like Python make it easy to build AI systems."
            ]
    
    def find_relevant_info(self, question):
        """Find relevant information from knowledge base"""
        question_words = set(re.findall(r'\\w+', question.lower()))
        
        scored_knowledge = []
        for knowledge_item in self.knowledge:
            knowledge_words = set(re.findall(r'\\w+', knowledge_item.lower()))
            
            # Simple relevance scoring
            overlap = len(question_words.intersection(knowledge_words))
            if overlap > 0:
                scored_knowledge.append((knowledge_item, overlap))
        
        # Sort by relevance score
        scored_knowledge.sort(key=lambda x: x[1], reverse=True)
        
        # Return most relevant items
        return [item[0] for item in scored_knowledge[:2]]
    
    def generate_response(self, question):
        """Generate response based on question"""
        question_lower = question.lower()
        
        # Greeting responses
        if any(word in question_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm your AI assistant. How can I help you today?"
        
        # Help responses
        if any(word in question_lower for word in ['help', 'assist']):
            return "I can help answer questions about technology, science, programming, and general knowledge. Just ask!"
        
        # Thank you responses
        if any(word in question_lower for word in ['thank', 'thanks']):
            return "You're welcome! Feel free to ask me anything else."
        
        # Find relevant knowledge
        relevant_info = self.find_relevant_info(question)
        
        if relevant_info:
            # Use relevant knowledge to form response
            base_response = relevant_info[0]
            
            # Add contextual framing
            if 'what' in question_lower:
                response = f"Based on what I know: {base_response}"
            elif 'how' in question_lower:
                response = f"Here's how it works: {base_response}"
            elif 'why' in question_lower:
                response = f"The reason is: {base_response}"
            else:
                response = f"Let me explain: {base_response}"
            
            # Add additional context if available
            if len(relevant_info) > 1:
                response += f" Additionally, {relevant_info[1].lower()}"
            
            return response
        
        # Default responses for unknown questions
        default_responses = [
            "That's an interesting question! Let me think about that.",
            "I'd need more context to give you a complete answer, but I'm happy to help explore the topic.",
            "That's a great question! Can you provide more details so I can give you a better answer?",
            "I want to make sure I give you accurate information. Could you rephrase or be more specific?"
        ]
        
        return random.choice(default_responses)
    
    def chat(self):
        """Main chat interface"""
        print("\\n" + "="*50)
        print("🌟 Welcome to the AI Assistant Demo!")
        print("Type 'quit' to exit, 'clear' for new conversation")
        print("="*50)
        
        while True:
            try:
                question = input("\\n💬 You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thanks for trying the demo!")
                    break
                
                if question.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ Conversation cleared!")
                    continue
                
                # Generate and display response
                response = self.generate_response(question)
                print(f"\\n🤖 Assistant: {response}")
                
                # Track conversation
                self.conversation_history.append({
                    "question": question,
                    "response": response
                })
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    assistant = SimpleAssistant()
    assistant.chat()
'''
    
    with open("simple_demo.py", "w") as f:
        f.write(demo_code)
    
    print("✅ Simple demo created!")

def main():
    """Main setup function"""
    print("🚀 Intelligent AI Assistant Setup")
    print("=" * 50)
    
    # Check if user wants full setup or simple demo
    print("Choose setup option:")
    print("1. Full setup (installs PyTorch, FAISS, etc.) - Recommended")
    print("2. Simple demo (no dependencies) - Quick start")
    
    while True:
        choice = input("\\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    if choice == '1':
        # Full setup
        print("\\n🔧 Setting up full intelligent assistant...")
        
        if not install_requirements():
            print("❌ Setup failed. Try the simple demo instead.")
            return
        
        setup_demo_data()
        create_simple_tokenizer()
        
        print("\\n🎉 Full setup complete!")
        print("\\n📚 Next steps:")
        print("1. Train the model: python train_intelligent.py")
        print("2. Run the assistant: python smart_assistant.py")
        print("3. Or try the simple demo: python simple_demo.py")
        
    else:
        # Simple demo
        print("\\n🎯 Setting up simple demo...")
        setup_demo_data()
        create_simple_demo()
        
        print("\\n🎉 Simple demo ready!")
        print("\\n🚀 Run the demo: python simple_demo.py")
    
    print("\\n💡 The assistant can answer questions about:")
    print("   • Technology and AI")
    print("   • Programming and software")
    print("   • Science and mathematics")
    print("   • General knowledge")
    print("\\n✨ Just ask anything and get intelligent responses!")

if __name__ == "__main__":
    main()
