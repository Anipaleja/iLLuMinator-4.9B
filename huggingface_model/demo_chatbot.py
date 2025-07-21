#!/usr/bin/env python3
"""
Quick demo of the Quantum Enterprise AI Chatbot
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from quantum_chatbot import QuantumChatbot

def demo_chatbot():
    """Run a quick demo of the chatbot"""
    print("🎬 QUANTUM CHATBOT DEMO")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = QuantumChatbot()
    
    if not chatbot.initialize_assistant():
        print("❌ Failed to initialize chatbot")
        return
    
    print("\n🤖 Chatbot initialized successfully!")
    
    # Demo queries
    demo_queries = [
        ("Tell me about quantum computing", "conversational"),
        ("Create a Python function to calculate fibonacci", "code generation"),
        ("Write a short poem about AI", "creative writing")
    ]
    
    print("\n🎯 Running demo queries...")
    print("-" * 30)
    
    for i, (query, mode) in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print(f"   Mode: {mode}")
        
        # Process query
        response = chatbot.process_user_input(query)
        
        if response:
            # Show preview of response
            preview = response[:200].replace('\n', ' ').strip()
            print(f"   Response: {preview}...")
            print(f"   Length: {len(response)} characters")
        else:
            print("   No response received")
    
    # Show final status
    chatbot.show_status()
    
    # Cleanup
    chatbot.shutdown()
    
    print("\n🎊 Demo completed successfully!")
    print("\nTo start interactive chatbot, run: python quantum_chatbot.py")

if __name__ == "__main__":
    demo_chatbot()
