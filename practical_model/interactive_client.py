#!/usr/bin/env python3
"""
Interactive Client for Practical iLLuMinator API
Simple command-line interface for chatting with the AI
"""

import requests
import json
import time
from typing import Dict, Any

class PracticaliLLuMinatorClient:
    """Client for the practical iLLuMinator API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Test connection
        if self.check_health():
            print("Connected to iLLuMinator Practical API")
        else:
            print("Failed to connect to API")
    
    def check_health(self) -> bool:
        """Check if the API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def chat(self, message: str, max_tokens: int = 80) -> Dict[str, Any]:
        """Send chat message"""
        try:
            response = self.session.post(
                f"{self.base_url}/chat",
                json={
                    "message": message,
                    "max_tokens": max_tokens,
                    "temperature": 0.8
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def get_completion(self, prompt: str, max_tokens: int = 60) -> Dict[str, Any]:
        """Get text completion"""
        try:
            response = self.session.post(
                f"{self.base_url}/completion",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Could not get model info"}
        except Exception as e:
            return {"error": str(e)}

def main():
    """Interactive chat interface"""
    print("🤖 iLLuMinator Practical AI - Interactive Chat")
    print("=" * 50)
    print("Commands:")
    print("  /info     - Show model information")
    print("  /complete - Text completion mode")
    print("  /quit     - Exit")
    print("  /help     - Show this help")
    print("=" * 50)
    
    client = PracticaliLLuMinatorClient()
    
    if not client.check_health():
        print("API server not available. Please start it with:")
        print("   python practical_api_server.py")
        return
    
    # Show model info
    info = client.get_model_info()
    if "error" not in info:
        print(f"Model: {info.get('model_type', 'Unknown')}")
        print(f"Parameters: {info.get('parameters', 'Unknown')}")
        print()
    
    print("Start chatting! (type /quit to exit)")
    print("-" * 30)
    
    while True:
        try:
            user_input = input("\n You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("Goodbye!")
                break
            
            elif user_input.lower() in ['/help', 'help']:
                print("\nCommands:")
                print("  /info     - Show model information") 
                print("  /complete - Text completion mode")
                print("  /quit     - Exit")
                continue
            
            elif user_input.lower() == '/info':
                info = client.get_model_info()
                if "error" not in info:
                    print("\n Model Information:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"Error: {info['error']}")
                continue
            
            elif user_input.lower() == '/complete':
                prompt = input("Enter prompt for completion: ").strip()
                if prompt:
                    print("Completing...", end="", flush=True)
                    result = client.get_completion(prompt)
                    
                    if "error" not in result:
                        print(f"\r Completion: {result['completion']}")
                        print(f"Generated in {result['generation_time']}s")
                    else:
                        print(f"\r Error: {result['error']}")
                continue
            
            # Regular chat
            print("🤖 iLLuMinator: ", end="", flush=True)
            
            start_time = time.time()
            result = client.chat(user_input)
            end_time = time.time()
            
            if "error" not in result:
                print(f"{result['response']}")
                print(f"Response time: {result['generation_time']}s")
            else:
                print(f"Error: {result['error']}")
                print(f"Total time: {end_time - start_time:.3f}s")
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Unexpected error: {e}")

if __name__ == "__main__":
    main()
