#!/usr/bin/env python3
"""
Quick Test for iLLuMinator AI API
Verify the API is working correctly
"""

import requests
import time
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing iLLuMinator AI API...")
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy")
            print(f"  Model parameters: {data.get('model_parameters'):,}")
            print(f"  Uptime: {data.get('uptime', 0):.1f} seconds")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False
    
    # Test 2: Chat
    print("\n2. Testing Chat...")
    try:
        payload = {
            "message": "What is Python programming?",
            "temperature": 0.7,
            "max_tokens": 100
        }
        start_time = time.time()
        response = requests.post(f"{base_url}/chat", json=payload, timeout=60)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Chat working")
            print(f"  Response time: {request_time:.2f}s")
            print(f"  AI Response: {data.get('response', '')[:100]}...")
        else:
            print(f"✗ Chat failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Chat error: {e}")
        return False
    
    # Test 3: Code Generation
    print("\n3. Testing Code Generation...")
    try:
        payload = {
            "description": "Create a Python function to check if a number is prime",
            "language": "python",
            "temperature": 0.3,
            "max_tokens": 200
        }
        start_time = time.time()
        response = requests.post(f"{base_url}/code", json=payload, timeout=60)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Code generation working")
            print(f"  Response time: {request_time:.2f}s")
            print(f"  Generated Code:")
            print(f"  {data.get('response', '')[:200]}...")
        else:
            print(f"✗ Code generation failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Code generation error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("=" * 50)
    print("Quick API Test for iLLuMinator AI")
    print("=" * 50)
    
    if test_api():
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED")
        print("iLLuMinator AI API is working correctly!")
        print("=" * 50)
        
        print("\nYou can now:")
        print("1. Use the chatbot client: python chatbot_client.py")
        print("2. Access API docs: http://localhost:8000/docs")
        print("3. Make direct API calls to the endpoints")
    else:
        print("\n" + "=" * 50)
        print("❌ SOME TESTS FAILED")
        print("Check if the API server is running:")
        print("python api_server.py")
        print("=" * 50)

if __name__ == "__main__":
    main()
