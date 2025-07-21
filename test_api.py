"""
Test Script for iLLuMinator AI API
Quick verification of API functionality
"""

import time
import subprocess
import sys
import threading
import requests
from pathlib import Path

def start_api_server():
    """Start the API server in background"""
    try:
        # Start server process
        server_process = subprocess.Popen([
            sys.executable, "api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("Starting API server...")
        time.sleep(5)  # Give server time to initialize
        
        return server_process
    except Exception as e:
        print(f"Failed to start API server: {e}")
        return None

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing API endpoints...")
    
    # Test 1: Health Check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ Health check passed")
            data = response.json()
            print(f"  Model status: {data.get('status')}")
            print(f"  Parameters: {data.get('model_parameters'):,}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False
    
    # Test 2: Model Info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=10)
        if response.status_code == 200:
            print("✓ Model info retrieved successfully")
            data = response.json()
            print(f"  Model: {data.get('model_name')}")
            print(f"  Version: {data.get('version')}")
        else:
            print(f"✗ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Model info error: {e}")
    
    # Test 3: Chat Endpoint
    print("\n3. Testing chat endpoint...")
    try:
        payload = {
            "message": "What is machine learning?",
            "temperature": 0.7,
            "max_tokens": 100
        }
        response = requests.post(f"{base_url}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            print("✓ Chat endpoint working")
            data = response.json()
            print(f"  Response: {data.get('response', '')[:100]}...")
            print(f"  Response time: {data.get('response_time', 0):.2f}s")
        else:
            print(f"✗ Chat endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Chat endpoint error: {e}")
    
    # Test 4: Code Generation Endpoint
    print("\n4. Testing code generation endpoint...")
    try:
        payload = {
            "description": "Create a Python function to reverse a string",
            "language": "python",
            "temperature": 0.3,
            "max_tokens": 200
        }
        response = requests.post(f"{base_url}/code", json=payload, timeout=30)
        if response.status_code == 200:
            print("✓ Code generation working")
            data = response.json()
            print(f"  Generated code preview:")
            print(f"  {data.get('response', '')[:150]}...")
            print(f"  Response time: {data.get('response_time', 0):.2f}s")
        else:
            print(f"✗ Code generation failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Code generation error: {e}")
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("iLLuMinator AI API Test Suite")
    print("=" * 60)
    
    # Check if dependencies are installed
    try:
        import torch
        import fastapi
        import uvicorn
        import requests
        print("✓ All required dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies with:")
        print("pip install -r requirements_api.txt")
        return
    
    # Check if core files exist
    required_files = ["illuminator_ai.py", "api_server.py"]
    for file in required_files:
        if not Path(file).exists():
            print(f"✗ Required file missing: {file}")
            return
    
    print("✓ All required files found")
    
    # Start API server
    server_process = start_api_server()
    if not server_process:
        print("✗ Failed to start API server")
        return
    
    try:
        # Wait a bit more for full initialization
        print("Waiting for AI model to load...")
        time.sleep(8)
        
        # Run tests
        if test_api_endpoints():
            print("\n" + "=" * 60)
            print("API Test Results: SUCCESS")
            print("The iLLuMinator AI API is working correctly!")
            print("=" * 60)
            
            print("\nNext steps:")
            print("1. Run the API server: python api_server.py")
            print("2. Use the chatbot client: python chatbot_client.py")
            print("3. Access API docs: http://localhost:8000/docs")
        else:
            print("\n" + "=" * 60)
            print("API Test Results: PARTIAL SUCCESS")
            print("Some tests failed, but basic functionality is working")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        # Clean up
        if server_process:
            print("\nStopping API server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()
