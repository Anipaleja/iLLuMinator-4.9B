#!/usr/bin/env python3
"""
Training script for the Intelligent RAG Transformer
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from typing import List, Tuple
from intelligent_rag import IntelligentRAGTransformer
from model.tokenizer import build_tokenizer, encode, decode
from data.prepare import load_data

class ConversationDataset(Dataset):
    """Dataset for training conversational AI with Q&A pairs"""
    
    def __init__(self, 
                 conversations: List[dict],
                 tokenizer_stoi: dict,
                 block_size: int = 512):
        
        self.conversations = conversations
        self.stoi = tokenizer_stoi
        self.block_size = block_size
        self.data = self.prepare_training_data()
    
    def prepare_training_data(self) -> List[Tuple[List[int], List[int]]]:
        """Convert conversations to training examples"""
        training_examples = []
        
        for conv in self.conversations:
            if 'question' in conv and 'answer' in conv:
                # Create input sequence: question + answer_start_token
                question_tokens = encode(conv['question'], self.stoi)
                answer_tokens = encode(conv['answer'], self.stoi)
                
                # Combine: [question] [ANSWER_START] [answer] [EOS]
                full_sequence = question_tokens + [len(self.stoi) - 2] + answer_tokens + [len(self.stoi) - 1]
                
                if len(full_sequence) <= self.block_size:
                    # Input: everything except last token
                    # Target: everything except first token (shifted by 1)
                    input_seq = full_sequence[:-1]
                    target_seq = full_sequence[1:]
                    
                    # Pad if necessary
                    while len(input_seq) < self.block_size:
                        input_seq.append(0)  # Padding token
                        target_seq.append(0)
                    
                    training_examples.append((input_seq[:self.block_size], target_seq[:self.block_size]))
        
        return training_examples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def create_synthetic_training_data() -> List[dict]:
    """Create synthetic Q&A pairs for training"""
    training_data = [
        # Technology questions
        {"question": "What is artificial intelligence?", 
         "answer": "Artificial intelligence is the simulation of human intelligence by machines, enabling them to think, learn, and make decisions."},
        
        {"question": "How does machine learning work?", 
         "answer": "Machine learning works by training algorithms on data to recognize patterns and make predictions without explicit programming."},
        
        {"question": "What is a neural network?", 
         "answer": "A neural network is a computing system inspired by biological brains, consisting of interconnected nodes that process information."},
        
        {"question": "Explain deep learning", 
         "answer": "Deep learning uses multi-layered neural networks to automatically learn complex patterns from large amounts of data."},
        
        # Science questions
        {"question": "What is quantum computing?", 
         "answer": "Quantum computing uses quantum mechanical phenomena like superposition to process information exponentially faster than classical computers."},
        
        {"question": "How does photosynthesis work?", 
         "answer": "Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen, providing energy for plants and producing oxygen for life."},
        
        {"question": "What is DNA?", 
         "answer": "DNA is the hereditary material that contains genetic instructions for building and maintaining all living organisms."},
        
        # Programming questions
        {"question": "What is Python programming?", 
         "answer": "Python is a high-level, readable programming language widely used for web development, data science, AI, and automation."},
        
        {"question": "How do you debug code?", 
         "answer": "Debugging involves identifying and fixing errors through techniques like print statements, debuggers, unit tests, and code review."},
        
        {"question": "What are data structures?", 
         "answer": "Data structures are ways to organize and store data efficiently, including arrays, lists, trees, graphs, and hash tables."},
        
        # General knowledge
        {"question": "What causes climate change?", 
         "answer": "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes."},
        
        {"question": "How does the internet work?", 
         "answer": "The internet works through interconnected networks that use protocols like TCP/IP to route data packets between computers worldwide."},
        
        {"question": "What is renewable energy?", 
         "answer": "Renewable energy comes from naturally replenishing sources like solar, wind, water, and geothermal that don't deplete over time."},
        
        # Conversational
        {"question": "Hello, how are you?", 
         "answer": "Hello! I'm doing well and ready to help answer your questions. How can I assist you today?"},
        
        {"question": "Can you help me learn?", 
         "answer": "Absolutely! I'm here to help you learn about any topic. What would you like to explore or understand better?"},
        
        {"question": "Thank you for your help", 
         "answer": "You're very welcome! I'm glad I could help. Feel free to ask me anything else you're curious about."},
        
        # Math and logic
        {"question": "What is calculus used for?", 
         "answer": "Calculus is used to study rates of change, areas under curves, optimization problems, and modeling continuous phenomena in physics and engineering."},
        
        {"question": "How do you solve problems?", 
         "answer": "Problem-solving involves understanding the problem, breaking it into smaller parts, exploring solutions, implementing them, and verifying results."},
        
        # Creative and philosophical
        {"question": "What makes art meaningful?", 
         "answer": "Art becomes meaningful through emotional expression, cultural significance, technical skill, and its ability to communicate ideas and experiences."},
        
        {"question": "Why is learning important?", 
         "answer": "Learning is important because it develops critical thinking, adapts us to change, expands opportunities, and helps us understand ourselves and the world."}
    ]
    
    return training_data


def train_intelligent_model(
    model: IntelligentRAGTransformer,
    train_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    device: str = 'cpu'
):
    """Train the intelligent RAG model"""
    
    model.to(device)
    model.train()
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"🚀 Starting training for {num_epochs} epochs on {device}")
    print(f"📊 Training data: {len(train_loader.dataset)} examples")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss (only on non-padding tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=0  # Ignore padding tokens
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress reporting
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        scheduler.step()
    
    print("🎉 Training completed!")
    return model


def main():
    """Main training function"""
    print("🧠 Training Intelligent RAG Transformer")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load tokenizer
    try:
        text = load_data()
        stoi, itos = build_tokenizer(text)
        vocab_size = len(stoi)
        print(f"📚 Loaded tokenizer with vocab size: {vocab_size}")
    except:
        # Fallback tokenizer
        print("⚠️  Using synthetic tokenizer")
        vocab_size = 10000
        stoi = {f"token_{i}": i for i in range(vocab_size)}
        itos = {i: f"token_{i}" for i in range(vocab_size)}
    
    # Create training data
    print("📝 Creating training data...")
    training_conversations = create_synthetic_training_data()
    
    # Add some Q&A from knowledge base
    knowledge_qa = [
        {"question": "What is the transformer architecture?", 
         "answer": "The Transformer architecture uses self-attention mechanisms to process sequences in parallel, revolutionizing natural language processing."},
        
        {"question": "How does attention work in transformers?", 
         "answer": "Attention allows the model to focus on relevant parts of the input when making predictions by computing weighted relationships between all positions."},
        
        {"question": "What is retrieval-augmented generation?", 
         "answer": "RAG combines retrieval of relevant documents with text generation, allowing models to access external knowledge for better responses."}
    ]
    
    training_conversations.extend(knowledge_qa)
    
    # Create dataset and dataloader
    dataset = ConversationDataset(training_conversations, stoi, block_size=256)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"📊 Created dataset with {len(dataset)} training examples")
    
    # Initialize model with knowledge base
    knowledge_base = [
        "The transformer architecture revolutionized NLP with self-attention mechanisms instead of recurrence.",
        "Machine learning enables computers to learn patterns from data without explicit programming.",
        "Python is a versatile programming language popular for AI, web development, and data science.",
        "Artificial intelligence aims to create machines that can think, learn, and solve problems like humans.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns automatically."
    ]
    
    model = IntelligentRAGTransformer(
        vocab_size=vocab_size,
        block_size=256,  # Smaller for training
        n_embd=256,
        n_head=8,
        n_layer=6,
        knowledge_base=knowledge_base
    )
    
    print(f"🏗️  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    trained_model = train_intelligent_model(
        model=model,
        train_loader=train_loader,
        num_epochs=10,
        learning_rate=1e-4,
        device=device
    )
    
    # Save the trained model
    model_path = "intelligent_rag_model.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Save configuration
    config = {
        "vocab_size": vocab_size,
        "model_params": {
            "block_size": 256,
            "n_embd": 256,
            "n_head": 8,
            "n_layer": 6
        },
        "training_info": {
            "num_epochs": 10,
            "learning_rate": 1e-4,
            "training_examples": len(dataset)
        }
    }
    
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training complete! You can now use the smart assistant.")
    print("🚀 Run: python smart_assistant.py")


if __name__ == "__main__":
    main()
