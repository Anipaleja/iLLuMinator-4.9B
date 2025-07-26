#!/usr/bin/env python3
"""
Enhanced Test Script for Practical iLLuMinator
Tests the enhanced trained model with comprehensive evaluation
"""

import torch
import os
from typing import List, Dict
import json
from datetime import datetime

from illuminator_practical import iLLuMinatorPractical
from tokenizer import iLLuMinatorTokenizer

class EnhancedModelTester:
    """Test the enhanced trained model"""
    
    def __init__(self, model_path: str = "illuminator_practical_enhanced_best.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        self.tokenizer = iLLuMinatorTokenizer()
        
        # Load model
        print(f"🧠 Loading enhanced model from {model_path}...")
        self.model = self._load_model()
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def _load_model(self):
        """Load the trained model"""
        # Initialize model
        model = iLLuMinatorPractical(vocab_size=len(self.tokenizer))
        
        # Load checkpoint
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📊 Training history loaded: {len(checkpoint.get('train_losses', []))} epochs")
        else:
            print(f"⚠️ Model file not found: {self.model_path}")
            print("🔄 Using untrained model for demonstration")
        
        model.to(self.device)
        model.eval()
        return model
    
    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate a response to a prompt"""
        # Encode input
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_length=min(len(input_ids) + max_length, 200),
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.tokenizer.pad_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        
        # Extract just the response part
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif "AI:" in response:
            response = response.split("AI:")[-1].strip()
        elif "A:" in response:
            response = response.split("A:")[-1].strip()
        
        return response
    
    def run_comprehensive_test(self):
        """Run comprehensive tests on various categories"""
        
        test_categories = {
            "Greetings": [
                "Human: Hello\nAssistant:",
                "User: Hi there!\nAI:",
                "Human: Good morning\nAssistant:",
                "User: How are you?\nAI:",
            ],
            
            "Programming Questions": [
                "Human: What is Python?\nAssistant:",
                "User: How do I create a function?\nAI:",
                "Human: Explain variables in Python\nAssistant:",
                "User: What are Python lists?\nAI:",
                "Human: How do loops work?\nAssistant:",
            ],
            
            "AI & Technology": [
                "Human: What is artificial intelligence?\nAssistant:",
                "User: Explain machine learning\nAI:",
                "Human: What is deep learning?\nAssistant:",
                "User: How do neural networks work?\nAI:",
            ],
            
            "Problem Solving": [
                "Human: How do I debug code?\nAssistant:",
                "User: What makes good code?\nAI:",
                "Human: Best practices for programming?\nAssistant:",
                "User: How to learn programming?\nAI:",
            ]
        }
        
        print("🧪 Running Comprehensive Test Suite")
        print("=" * 60)
        
        results = {}
        
        for category, prompts in test_categories.items():
            print(f"\n🔍 Testing Category: {category}")
            print("-" * 40)
            
            category_results = []
            
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{i}. {prompt.split(':')[1].split('\\n')[0].strip()}")
                
                # Test with different temperatures
                for temp in [0.5, 0.7, 0.9]:
                    response = self.generate_response(prompt, temperature=temp)
                    
                    if temp == 0.7:  # Show main response
                        print(f"   🤖: {response}")
                    
                    category_results.append({
                        'prompt': prompt,
                        'temperature': temp,
                        'response': response,
                        'response_length': len(response.split())
                    })
            
            results[category] = category_results
        
        # Generate analysis
        self._analyze_results(results)
        
        return results
    
    def _analyze_results(self, results: Dict):
        """Analyze and display test results"""
        print(f"\n📊 Test Analysis")
        print("=" * 40)
        
        total_tests = sum(len(category_results) for category_results in results.values())
        
        print(f"📈 Total tests run: {total_tests}")
        
        # Response length analysis
        all_lengths = []
        for category_results in results.values():
            all_lengths.extend([r['response_length'] for r in category_results])
        
        if all_lengths:
            avg_length = sum(all_lengths) / len(all_lengths)
            print(f"📏 Average response length: {avg_length:.1f} words")
            print(f"📐 Response length range: {min(all_lengths)} - {max(all_lengths)} words")
        
        # Category performance
        print(f"\n📋 Category breakdown:")
        for category, category_results in results.items():
            temp_07_results = [r for r in category_results if r['temperature'] == 0.7]
            avg_cat_length = sum(r['response_length'] for r in temp_07_results) / len(temp_07_results)
            print(f"  {category}: {len(temp_07_results)} tests, avg {avg_cat_length:.1f} words")
    
    def interactive_chat(self):
        """Interactive chat mode"""
        print(f"\n💬 Interactive Chat Mode")
        print("=" * 40)
        print("Type 'quit' or 'exit' to end the chat")
        print("Type 'help' for commands")
        
        conversation_history = []
        
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("Commands:")
                print("  'temp X' - Set temperature (0.1-1.0)")
                print("  'clear' - Clear conversation history")
                print("  'save' - Save conversation")
                print("  'quit' - Exit chat")
                continue
            
            if user_input.lower().startswith('temp '):
                try:
                    temp = float(user_input.split()[1])
                    self.current_temp = max(0.1, min(1.0, temp))
                    print(f"🌡️ Temperature set to {self.current_temp}")
                except:
                    print("❌ Invalid temperature. Use 0.1-1.0")
                continue
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation history cleared")
                continue
            
            if user_input.lower() == 'save':
                self._save_conversation(conversation_history)
                continue
            
            if not user_input:
                continue
            
            # Generate response
            prompt = f"Human: {user_input}\nAssistant:"
            response = self.generate_response(
                prompt, 
                temperature=getattr(self, 'current_temp', 0.7)
            )
            
            print(f"🤖 iLLuMinator: {response}")
            
            # Save to history
            conversation_history.append({
                'user': user_input,
                'assistant': response,
                'timestamp': datetime.now().isoformat()
            })
    
    def _save_conversation(self, history: List[Dict]):
        """Save conversation history"""
        if not history:
            print("📝 No conversation to save")
            return
        
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"💾 Conversation saved to {filename}")
    
    def quick_test(self):
        """Quick functionality test"""
        print(f"\n⚡ Quick Test")
        print("=" * 30)
        
        quick_prompts = [
            "Human: Hello\nAssistant:",
            "User: What is Python?\nAI:",
            "Human: How do I create a function?\nAssistant:",
        ]
        
        for prompt in quick_prompts:
            question = prompt.split(':')[1].split('\\n')[0].strip()
            response = self.generate_response(prompt, temperature=0.7)
            print(f"Q: {question}")
            print(f"A: {response}\n")

def main():
    """Main test function"""
    print("🎯 iLLuMinator Enhanced Model Testing")
    print("=" * 50)
    
    # Try to load best model first, fallback to regular
    model_paths = [
        "illuminator_practical_improved_best.pth",
        "illuminator_practical_improved.pth",
        "illuminator_practical_enhanced_best.pth",
        "illuminator_practical_enhanced.pth", 
        "illuminator_practical_weights.pth"
    ]
    
    tester = None
    for path in model_paths:
        if os.path.exists(path):
            print(f"📁 Found model: {path}")
            tester = EnhancedModelTester(path)
            break
    
    if not tester:
        print("⚠️ No trained model found. Please run training first.")
        print("Available options:")
        print("  python train_enhanced.py  # Enhanced training")
        print("  python train_practical.py  # Basic training")
        return
    
    while True:
        print(f"\n🎮 Choose test mode:")
        print("1. Quick Test (3 samples)")
        print("2. Comprehensive Test (all categories)")
        print("3. Interactive Chat")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            tester.quick_test()
        
        elif choice == '2':
            tester.run_comprehensive_test()
        
        elif choice == '3':
            tester.interactive_chat()
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
