"""
Simple Chatbot Client for iLLuMinator AI API
Test code generation and AI responses
"""

import requests
import time
import json

# API configuration
API_BASE_URL = "http://localhost:8000"

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Simple Chatbot Client for iLLuMinator AI")
    print("Connecting to API at: {}".format(API_BASE_URL))
    print("=" * 60)
    
    def interact_with_api():
        """Interact with the iLLuMinator AI API"""
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting...")
                break
            elif user_input.lower().startswith('code:'):
                # Code generation request
                description = user_input[5:].strip()
                payload = {
                    'description': description,
                    'language': 'python',  # You can change this to any supported language
                }
                response = requests.post(f"{API_BASE_URL}/code", json=payload)
            else:
                # Regular chat request
                payload = {
                    'message': user_input,
                }
                response = requests.post(f"{API_BASE_URL}/chat", json=payload)
            
            if response.ok:
                result = response.json()
                ai_response = result.get('response', 'No response')
                print(f"iLLuMinator AI: {ai_response}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
            
    # Check if server is running
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.ok:
            print("Connection successful.")
            interact_with_api()
        else:
            print("Failed to connect to the API server.")
            print(f"Status Code: {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to the API server. Make sure it is running.")
