"""
Professional Demonstration Script for iLLuMinator AI
Showcases advanced code generation and intelligent conversation capabilities
"""

import time
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from illuminator_ai import IlluminatorAI

def print_header(title: str):
    """Print formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"{title.center(60)}")
    print(f"{'=' * 60}")

def print_demo_section(title: str, description: str):
    """Print demo section with description"""
    print(f"\n{'-' * 50}")
    print(f"🔹 {title}")
    print(f"{description}")
    print(f"{'-' * 50}")

def simulate_typing(text: str, delay: float = 0.02):
    """Simulate typing effect for demonstrations"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def demo_code_generation(ai: IlluminatorAI):
    """Demonstrate advanced code generation capabilities"""
    print_demo_section(
        "Code Generation Capabilities",
        "iLLuMinator AI can generate professional-quality code in multiple languages"
    )
    
    code_examples = [
        {
            "description": "Create a Python class for a REST API client with error handling",
            "language": "python"
        },
        {
            "description": "Write a JavaScript function to debounce user input",
            "language": "javascript"
        },
        {
            "description": "Implement a binary search algorithm with comprehensive comments",
            "language": "python"
        }
    ]
    
    for i, example in enumerate(code_examples, 1):
        print(f"\n📝 Example {i}: {example['description']}")
        print("\nGenerating code...")
        
        start_time = time.time()
        code = ai.generate_code(example['description'], example['language'])
        generation_time = time.time() - start_time
        
        print(f"\n✅ Generated in {generation_time:.2f} seconds:")
        print(f"```{example['language']}")
        print(code)
        print("```")
        
        time.sleep(1)  # Brief pause between examples

def demo_technical_discussion(ai: IlluminatorAI):
    """Demonstrate intelligent technical conversation"""
    print_demo_section(
        "Technical Discussion Capabilities",
        "Engage in sophisticated technical conversations with contextual understanding"
    )
    
    technical_questions = [
        "What are the key differences between SQL and NoSQL databases?",
        "Explain the concept of microservices architecture and its benefits",
        "How does machine learning differ from traditional programming approaches?",
        "What are the main principles of clean code according to Robert Martin?"
    ]
    
    for i, question in enumerate(technical_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("\nProcessing response...")
        
        start_time = time.time()
        response = ai.chat(question)
        response_time = time.time() - start_time
        
        print(f"\n🤖 iLLuMinator AI ({response_time:.2f}s):")
        simulate_typing(response, delay=0.01)
        
        time.sleep(1)

def demo_context_awareness(ai: IlluminatorAI):
    """Demonstrate context-aware conversation capabilities"""
    print_demo_section(
        "Context-Aware Conversation",
        "Maintaining context across multiple exchanges for intelligent follow-up responses"
    )
    
    conversation_flow = [
        "What is Docker and why is it useful?",
        "How does it differ from virtual machines?",
        "What are some best practices when using it in production?",
        "Can you show me a simple Dockerfile example?"
    ]
    
    for i, message in enumerate(conversation_flow, 1):
        print(f"\n👤 User: {message}")
        
        start_time = time.time()
        response = ai.chat(message)
        response_time = time.time() - start_time
        
        print(f"\n🤖 iLLuMinator AI ({response_time:.2f}s):")
        simulate_typing(response, delay=0.01)
        
        if i < len(conversation_flow):
            print("\n⏳ Building context for next response...")
            time.sleep(1)

def demo_model_capabilities(ai: IlluminatorAI):
    """Show model specifications and capabilities"""
    print_demo_section(
        "Model Architecture & Specifications",
        "Technical details about the iLLuMinator AI model"
    )
    
    # Get model parameters
    total_params = sum(p.numel() for p in ai.model.parameters())
    trainable_params = sum(p.numel() for p in ai.model.parameters() if p.requires_grad)
    
    specs = f"""
🏗️  Model Architecture: Advanced Transformer with Multi-Head Attention
📊 Total Parameters: {total_params:,}
🔧 Trainable Parameters: {trainable_params:,}
🧠 Model Dimension: {ai.model.d_model}
🔄 Transformer Layers: {ai.model.n_layers}
👁️  Attention Heads: {ai.model.n_heads}
📝 Vocabulary Size: {ai.model.vocab_size:,}
📏 Max Sequence Length: {ai.model.max_seq_len}
💾 Device: {ai.device}
"""
    
    simulate_typing(specs.strip(), delay=0.005)

def interactive_demo(ai: IlluminatorAI):
    """Interactive demonstration mode"""
    print_demo_section(
        "Interactive Mode",
        "Try the AI assistant yourself with custom queries"
    )
    
    print("\n🎯 Try these example commands:")
    print("• 'code: Create a FastAPI endpoint for user authentication'")
    print("• 'What is the difference between REST and GraphQL?'")
    print("• 'Explain design patterns in software engineering'")
    print("• Type 'demo_end' to return to main demo")
    
    while True:
        try:
            user_input = input("\n👤 Your input: ").strip()
            
            if user_input.lower() == 'demo_end':
                break
            
            if not user_input:
                continue
                
            start_time = time.time()
            
            if user_input.lower().startswith('code:'):
                description = user_input[5:].strip()
                response = ai.generate_code(description)
            else:
                response = ai.chat(user_input)
            
            response_time = time.time() - start_time
            
            print(f"\n🤖 iLLuMinator AI ({response_time:.2f}s):")
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nReturning to main demo...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main demonstration function"""
    print_header("iLLuMinator AI Professional Demonstration")
    
    print("""
🚀 Welcome to the iLLuMinator AI Professional Demonstration

This demo showcases the advanced capabilities of iLLuMinator AI:
• Professional code generation in multiple languages
• Intelligent technical discussions with context awareness
• Clean, production-ready output without unnecessary formatting
• Minimal dependencies while maintaining high performance

Initializing iLLuMinator AI...
    """)
    
    try:
        # Initialize AI
        print("⏳ Loading model and initializing components...")
        ai = IlluminatorAI()
        print("✅ Initialization complete!\n")
        
        # Run demonstration sections
        demo_model_capabilities(ai)
        
        demo_code_generation(ai)
        
        demo_technical_discussion(ai)
        
        demo_context_awareness(ai)
        
        # Interactive mode
        while True:
            print(f"\n{'=' * 60}")
            print("Demo Options:")
            print("1. Interactive Mode - Try it yourself")
            print("2. Restart Demo")
            print("3. Exit")
            print(f"{'=' * 60}")
            
            choice = input("\nSelect an option (1-3): ").strip()
            
            if choice == '1':
                interactive_demo(ai)
            elif choice == '2':
                ai.clear_conversation()
                main()
                break
            elif choice == '3':
                print("\n🎉 Thank you for trying iLLuMinator AI!")
                print("For more information, see README_PROFESSIONAL.md")
                break
            else:
                print("❌ Invalid choice. Please select 1, 2, or 3.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thank you for trying iLLuMinator AI!")
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        print("\nPlease ensure you have PyTorch installed:")
        print("pip install torch>=2.0.0 numpy>=1.21.0")

if __name__ == "__main__":
    main()
