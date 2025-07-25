#!/usr/bin/env python3
"""
Enhanced iLLuMinator AI Test Client
Test the improved web search and intelligent summarization
"""

import requests
import json
import time

def test_illuminator():
    """Test various types of queries with the enhanced system"""
    
    base_url = "http://localhost:8000"
    
    # Test queries of different types
    test_queries = [
        # Simple conversation
        ("Hello, how are you?", "Conversational"),
        
        # Programming questions  
        ("How do I create a Python class?", "Programming"),
        
        # Factual questions that should trigger web search
        ("What is artificial intelligence?", "Definition/Factual"),
        ("Who is the current CEO of Apple?", "Current Facts"),
        ("What are the latest developments in quantum computing?", "Current Events"),
        
        # Mixed questions
        ("Explain how machine learning works", "Explanation"),
        ("What is the weather like today?", "Current Info"),
    ]
    
    print("🤖 Testing Enhanced iLLuMinator AI System")
    print("=" * 50)
    
    for query, query_type in test_queries:
        print(f"\n📝 Query Type: {query_type}")
        print(f"❓ Question: {query}")
        print("-" * 30)
        
        try:
            # Send request
            start_time = time.time()
            response = requests.post(
                f"{base_url}/chat",
                json={"message": query},
                timeout=15
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Response ({end_time-start_time:.2f}s):")
                print(f"   {data['response']}")
                print(f"   Tokens: {data.get('tokens_generated', 'N/A')}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)  # Brief pause between requests
    
    print(f"\n{'='*50}")
    print("✅ Testing completed!")

if __name__ == "__main__":
    test_illuminator()
