"""
Enhanced Chatbot Client for iLLuMinator AI API
Comprehensive testing of code generation and AI responses
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Optional

class IlluminatorChatbotClient:
    """Professional chatbot client for iLLuMinator AI API"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.conversation_history = []
        self.session = requests.Session()
        
    def check_connection(self) -> bool:
        """Check if API server is running"""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"Connected to iLLuMinator AI API")
                print(f"Model Status: {health_data.get('status', 'unknown')}")
                print(f"Model Parameters: {health_data.get('model_parameters', 0):,}")
                print(f"Uptime: {health_data.get('uptime', 0):.2f} seconds")
                return True
            else:
                print(f"API health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed model information"""
        try:
            response = self.session.get(f"{self.api_base_url}/model/info", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get model info: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Model info request failed: {e}")
            return None
    
    def chat(self, message: str, temperature: float = 0.7, max_tokens: int = 200) -> Optional[str]:
        """Send chat message to AI"""
        try:
            payload = {
                "message": message,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.api_base_url}/chat", json=payload, timeout=30)
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get("response", "")
                
                # Store in conversation history
                self.conversation_history.append({
                    "type": "chat",
                    "user_message": message,
                    "ai_response": ai_response,
                    "response_time": data.get("response_time", 0),
                    "request_time": request_time,
                    "tokens_generated": data.get("tokens_generated", 0)
                })
                
                return ai_response
            else:
                print(f"Chat request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Chat request error: {e}")
            return None
    
    def generate_code(self, description: str, language: str = "python", 
                     temperature: float = 0.3, max_tokens: int = 400) -> Optional[str]:
        """Generate code based on description"""
        try:
            payload = {
                "description": description,
                "language": language,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.api_base_url}/code", json=payload, timeout=30)
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                code_response = data.get("response", "")
                
                # Store in conversation history
                self.conversation_history.append({
                    "type": "code",
                    "description": description,
                    "language": language,
                    "code_response": code_response,
                    "response_time": data.get("response_time", 0),
                    "request_time": request_time,
                    "tokens_generated": data.get("tokens_generated", 0)
                })
                
                return code_response
            else:
                print(f"Code generation failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Code generation error: {e}")
            return None
    
    def clear_conversation(self) -> bool:
        """Clear conversation history on server"""
        try:
            response = self.session.delete(f"{self.api_base_url}/conversation", timeout=10)
            if response.status_code == 200:
                self.conversation_history.clear()
                return True
            else:
                print(f"Failed to clear conversation: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Clear conversation error: {e}")
            return False
    
    def run_tests(self):
        """Run comprehensive tests of the API"""
        print("\n" + "=" * 60)
        print("Running Comprehensive Tests")
        print("=" * 60)
        
        # Test 1: Basic Chat
        print("\n1. Testing Basic Chat...")
        response = self.chat("What is artificial intelligence?")
        if response:
            print(f"AI Response: {response[:100]}...")
            print("Basic chat test: PASSED")
        else:
            print("Basic chat test: FAILED")
        
        # Test 2: Code Generation - Python
        print("\n2. Testing Python Code Generation...")
        code = self.generate_code("Create a function to calculate factorial", "python")
        if code:
            print(f"Generated Python Code:\n{code}")
            print("Python code generation test: PASSED")
        else:
            print("Python code generation test: FAILED")
        
        # Test 3: Code Generation - JavaScript
        print("\n3. Testing JavaScript Code Generation...")
        js_code = self.generate_code("Create a function to validate email addresses", "javascript")
        if js_code:
            print(f"Generated JavaScript Code:\n{js_code}")
            print("JavaScript code generation test: PASSED")
        else:
            print("JavaScript code generation test: FAILED")
        
        # Test 4: Technical Discussion
        print("\n4. Testing Technical Discussion...")
        tech_response = self.chat("Explain the differences between REST and GraphQL APIs")
        if tech_response:
            print(f"Technical Response: {tech_response[:150]}...")
            print("Technical discussion test: PASSED")
        else:
            print("Technical discussion test: FAILED")
        
        # Test 5: Complex Code Generation
        print("\n5. Testing Complex Code Generation...")
        complex_code = self.generate_code(
            "Create a Python class for a basic REST API client with error handling and retry logic",
            "python"
        )
        if complex_code:
            print(f"Generated Complex Code:\n{complex_code}")
            print("Complex code generation test: PASSED")
        else:
            print("Complex code generation test: FAILED")
        
        # Test Results Summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print summary of all tests performed"""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        total_requests = len(self.conversation_history)
        successful_requests = sum(1 for h in self.conversation_history 
                                if h.get("ai_response") or h.get("code_response"))
        
        avg_response_time = sum(h.get("response_time", 0) for h in self.conversation_history) / max(total_requests, 1)
        total_tokens = sum(h.get("tokens_generated", 0) for h in self.conversation_history)
        
        print(f"Total Requests: {total_requests}")
        print(f"Successful Requests: {successful_requests}")
        print(f"Success Rate: {(successful_requests/max(total_requests,1)*100):.1f}%")
        print(f"Average Response Time: {avg_response_time:.2f} seconds")
        print(f"Total Tokens Generated: {total_tokens}")
        
        # Show breakdown by type
        chat_requests = [h for h in self.conversation_history if h.get("type") == "chat"]
        code_requests = [h for h in self.conversation_history if h.get("type") == "code"]
        
        print(f"\nBreakdown:")
        print(f"  Chat Requests: {len(chat_requests)}")
        print(f"  Code Generation Requests: {len(code_requests)}")
    
    def interactive_mode(self):
        """Interactive chat mode"""
        print("\n" + "=" * 60)
        print("Interactive Mode - iLLuMinator AI Chatbot")
        print("=" * 60)
        print("Commands:")
        print("  'code: <description>' - Generate code")
        print("  'lang <language>' - Set programming language (default: python)")
        print("  'temp <value>' - Set temperature (0.1-1.0)")
        print("  'clear' - Clear conversation")
        print("  'stats' - Show conversation statistics")
        print("  'quit' - Exit")
        print("=" * 60)
        
        current_language = "python"
        current_temperature = 0.7
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Thank you for using iLLuMinator AI!")
                    break
                
                elif user_input.lower() == 'clear':
                    if self.clear_conversation():
                        print("Conversation cleared.")
                    continue
                
                elif user_input.lower() == 'stats':
                    self.print_test_summary()
                    continue
                
                elif user_input.lower().startswith('lang '):
                    current_language = user_input[5:].strip()
                    print(f"Programming language set to: {current_language}")
                    continue
                
                elif user_input.lower().startswith('temp '):
                    try:
                        current_temperature = float(user_input[5:].strip())
                        current_temperature = max(0.1, min(1.0, current_temperature))
                        print(f"Temperature set to: {current_temperature}")
                    except ValueError:
                        print("Invalid temperature. Please provide a number between 0.1 and 1.0")
                    continue
                
                elif user_input.lower().startswith('code:'):
                    # Code generation
                    description = user_input[5:].strip()
                    if description:
                        print(f"Generating {current_language} code...")
                        start_time = time.time()
                        
                        code = self.generate_code(description, current_language, current_temperature)
                        
                        if code:
                            print(f"\niLLuMinator AI ({time.time() - start_time:.2f}s):")
                            print(f"```{current_language}")
                            print(code)
                            print("```")
                        else:
                            print("Code generation failed.")
                    else:
                        print("Please provide a code description.")
                
                else:
                    # Regular chat
                    print("Processing...")
                    start_time = time.time()
                    
                    response = self.chat(user_input, current_temperature)
                    
                    if response:
                        print(f"\niLLuMinator AI ({time.time() - start_time:.2f}s):")
                        print(response)
                    else:
                        print("Failed to get response.")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function"""
    print("iLLuMinator AI Chatbot Client")
    print("Connecting to API server...")
    
    client = IlluminatorChatbotClient()
    
    if not client.check_connection():
        print("\nFailed to connect to API server.")
        print("Please ensure the API server is running:")
        print("python api_server.py")
        return
    
    # Show model info
    model_info = client.get_model_info()
    if model_info:
        print(f"\nModel: {model_info.get('model_name', 'Unknown')}")
        print(f"Version: {model_info.get('version', 'Unknown')}")
        print(f"Parameters: {model_info.get('total_parameters', 0):,}")
    
    # Choose mode
    print("\nSelect mode:")
    print("1. Run comprehensive tests")
    print("2. Interactive chat mode")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            client.run_tests()
        else:
            client.interactive_mode()
    
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
