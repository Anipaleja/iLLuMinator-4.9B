#!/usr/bin/env python3
"""
Quantum Enterprise AI Chatbot
============================

Interactive chatbot interface for the Quantum Enterprise AI Assistant
Features:
- Natural conversation interface
- Multiple query type support
- Real-time response generation
- Clean terminal UI
"""

import os
import sys
import time
from typing import Optional
from quantum_enterprise_assistant import QuantumEnhancedAssistant, QueryType

class QuantumChatbot:
    """
    Interactive chatbot interface for Quantum Enterprise AI Assistant
    """
    
    def __init__(self):
        self.assistant = None
        self.conversation_history = []
        self.current_mode = QueryType.CONVERSATIONAL
        
    def initialize_assistant(self):
        """Initialize the AI assistant"""
        print("🚀 Initializing Quantum Enterprise AI Assistant...")
        try:
            self.assistant = QuantumEnhancedAssistant(
                enable_quantum=True,
                enable_mongodb=True
            )
            print("✅ Assistant initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize assistant: {e}")
            print("🔄 Falling back to basic mode...")
            try:
                self.assistant = QuantumEnhancedAssistant(
                    enable_quantum=False,
                    enable_mongodb=False
                )
                print("✅ Assistant initialized in basic mode!")
                return True
            except Exception as e2:
                print(f"❌ Failed to initialize in basic mode: {e2}")
                return False
    
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*60)
        print("🤖 QUANTUM ENTERPRISE AI CHATBOT")
        print("="*60)
        print("Welcome! I'm your Quantum-Enhanced AI Assistant.")
        print("I can help with:")
        print("  • Code generation in multiple languages")
        print("  • Creative writing and storytelling")
        print("  • Technical documentation")
        print("  • Data science and machine learning")
        print("  • Cybersecurity best practices")
        print("  • Web development guidance")
        print("  • Educational content creation")
        print("  • Quantum computing concepts")
        print("\nType '/help' for commands or just start chatting!")
        print("="*60)
    
    def print_help(self):
        """Print help information"""
        print("\n📋 AVAILABLE COMMANDS:")
        print("  /help          - Show this help message")
        print("  /mode          - Change conversation mode")
        print("  /history       - Show conversation history")
        print("  /clear         - Clear conversation history")
        print("  /status        - Show system status")
        print("  /quit or /exit - Exit the chatbot")
        print("\n🎯 CONVERSATION MODES:")
        print("  1. Conversational  - Natural chat (default)")
        print("  2. Code Generation - Programming help")
        print("  3. Creative Writing - Stories and creative content")
        print("  4. Technical Docs  - Documentation and guides")
        print("  5. Data Science    - ML and analytics help")
        print("  6. Cybersecurity   - Security best practices")
        print("  7. Web Development - Frontend/backend development")
        print("  8. Educational     - Tutorials and learning")
        print("  9. Quantum Enhanced - Advanced quantum processing")
    
    def change_mode(self):
        """Change conversation mode"""
        print("\n🎯 SELECT CONVERSATION MODE:")
        modes = [
            (QueryType.CONVERSATIONAL, "Conversational - Natural chat"),
            (QueryType.CODE_GENERATION, "Code Generation - Programming help"),
            (QueryType.CREATIVE_WRITING, "Creative Writing - Stories and content"),
            (QueryType.TECHNICAL_DOCUMENTATION, "Technical Docs - Documentation"),
            (QueryType.DATA_SCIENCE, "Data Science - ML and analytics"),
            (QueryType.CYBERSECURITY, "Cybersecurity - Security practices"),
            (QueryType.WEB_DEVELOPMENT, "Web Development - Frontend/backend"),
            (QueryType.TUTORIAL_CREATION, "Educational - Tutorials and learning"),
            (QueryType.QUANTUM_ENHANCED, "Quantum Enhanced - Advanced processing")
        ]
        
        for i, (mode, description) in enumerate(modes, 1):
            current = " (current)" if mode == self.current_mode else ""
            print(f"  {i}. {description}{current}")
        
        try:
            choice = input("\nEnter mode number (1-9): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(modes):
                self.current_mode = modes[int(choice)-1][0]
                print(f"✅ Mode changed to: {self.current_mode.value}")
            else:
                print("❌ Invalid choice. Mode unchanged.")
        except Exception as e:
            print(f"❌ Error changing mode: {e}")
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("\n📝 No conversation history yet.")
            return
        
        print(f"\n📝 CONVERSATION HISTORY ({len(self.conversation_history)} exchanges):")
        print("-" * 50)
        
        for i, (user_msg, ai_response, mode) in enumerate(self.conversation_history[-5:], 1):
            print(f"\n{i}. You ({mode.value}): {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")
            print(f"   AI: {ai_response[:150]}{'...' if len(ai_response) > 150 else ''}")
    
    def show_status(self):
        """Show system status"""
        print("\n📊 SYSTEM STATUS:")
        print(f"  🤖 Assistant Status: {'✅ Active' if self.assistant else '❌ Not initialized'}")
        print(f"  🎯 Current Mode: {self.current_mode.value}")
        print(f"  💬 Conversations: {len(self.conversation_history)}")
        
        if self.assistant:
            try:
                # Try to get performance analytics
                analytics = self.assistant.get_performance_analytics()
                print(f"  ⏱️  Average Response Time: {analytics.get('average_processing_time_ms', 'N/A')}ms")
                print(f"  📈 Total Queries: {analytics.get('total_conversations', len(self.conversation_history))}")
            except:
                print("  📊 Analytics: Not available")
    
    def process_user_input(self, user_input: str) -> Optional[str]:
        """Process user input and get AI response"""
        if not self.assistant:
            return "❌ Assistant not initialized. Please restart the chatbot."
        
        try:
            print("🤔 Thinking...", end="", flush=True)
            start_time = time.time()
            
            # Get response from assistant
            response = self.assistant.process_query(user_input, self.current_mode)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            # Store in history
            self.conversation_history.append((user_input, response, self.current_mode))
            
            print(f"\r✅ Response ready ({processing_time:.0f}ms)")
            return response
            
        except Exception as e:
            error_msg = f"❌ Error processing your request: {e}"
            self.conversation_history.append((user_input, error_msg, self.current_mode))
            return error_msg
    
    def run_chat_loop(self):
        """Main chat loop"""
        if not self.initialize_assistant():
            print("❌ Failed to initialize assistant. Exiting...")
            return
        
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{self.current_mode.value}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['/quit', '/exit']:
                    print("👋 Goodbye! Thanks for using Quantum Enterprise AI Chatbot!")
                    break
                elif user_input.lower() == '/help':
                    self.print_help()
                    continue
                elif user_input.lower() == '/mode':
                    self.change_mode()
                    continue
                elif user_input.lower() == '/history':
                    self.show_history()
                    continue
                elif user_input.lower() == '/clear':
                    self.conversation_history.clear()
                    print("✅ Conversation history cleared!")
                    continue
                elif user_input.lower() == '/status':
                    self.show_status()
                    continue
                
                # Process regular input
                response = self.process_user_input(user_input)
                
                if response:
                    print(f"\n🤖 AI Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for using Quantum Enterprise AI Chatbot!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Please try again or type /quit to exit.")
    
    def shutdown(self):
        """Cleanup and shutdown"""
        if self.assistant:
            try:
                self.assistant.shutdown()
            except:
                pass

def main():
    """Main function to run the chatbot"""
    try:
        chatbot = QuantumChatbot()
        chatbot.run_chat_loop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        try:
            chatbot.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()
