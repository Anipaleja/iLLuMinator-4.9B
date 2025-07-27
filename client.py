#!/usr/bin/env python3
"""
Interactive client for the iLLuMinator AI API
"""

import requests
import json
import time

class IlluminatorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def chat(self, message):
        """Send a chat message to the AI"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                headers={"Content-Type": "application/json"},
                json={"message": message}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def generate_code(self, description, language="python"):
        """Generate code"""
        try:
            response = requests.post(
                f"{self.base_url}/code",
                headers={"Content-Type": "application/json"},
                json={"description": description, "language": language}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}

def main():
    print("iLLuMinator AI Interactive Client")
    print("=====================================")
    
    client = IlluminatorClient()
    
    # Test connection
    print("Testing connection...")
    info = client.get_model_info()
    if "error" in info:
        print(f"Connection failed: {info['error']}")
        return
    
    print(f"Connected to {info['model_name']} v{info['version']}")
    print(f"Parameters: {info['total_parameters']}")
    print(f"Architecture: {info['architecture']}")
    print()
    
    # Interactive chat loop
    print("Type 'quit' to exit, '/code <description>' for code generation, '/info' for model info")
    print("----------------------------------------")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.startswith('/code '):
                # Code generation
                description = user_input[6:]  # Remove '/code '
                print("Generating code...")
                
                result = client.generate_code(description)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"iLLuMinator: \n{result['response']}")
                    print(f"Response time: {result['response_time']:.3f}s")
            
            elif user_input == '/info':
                # Model info
                info = client.get_model_info()
                if "error" in info:
                    print(f"Error: {info['error']}")
                else:
                    print("Model Information:")
                    for key, value in info.items():
                        print(f"   {key}: {value}")
            
            else:
                # Regular chat
                print("Thinking...")
                
                result = client.chat(user_input)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"iLLuMinator: {result['response']}")
                    print(f"Response time: {result['response_time']:.3f}s | Tokens: {result['tokens_generated']}")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
