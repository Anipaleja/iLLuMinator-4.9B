#!/usr/bin/env python3
"""
Complete Setup and Enhancement Script for iLLuMinator AI
Fetches web data, enhances the model, and starts the optimized system
"""

import sys
import os
import time
import argparse
from pathlib import Path

def setup_environment():
    """Setup the complete environment for iLLuMinator AI"""
    print("🚀 iLLuMinator AI Complete Setup & Enhancement")
    print("=" * 60)
    
    # Step 1: Generate comprehensive web data
    print("📡 Step 1: Fetching comprehensive web training data...")
    try:
        from web_data_fetcher import WebDataFetcher
        fetcher = WebDataFetcher()
        fetcher.compile_web_data("web_training_data.json")
        print("✅ Web data compilation complete!")
    except Exception as e:
        print(f"⚠️  Web data generation failed: {e}")
    
    # Step 2: Generate enhanced training data
    print("\n📚 Step 2: Generating enhanced training data...")
    try:
        from enhanced_training_data_fetcher import TrainingDataFetcher
        fetcher = TrainingDataFetcher()
        fetcher.compile_training_data("enhanced_training_data.json")
        print("✅ Enhanced training data complete!")
    except Exception as e:
        print(f"⚠️  Enhanced data generation failed: {e}")
    
    # Step 3: Initialize AI with all enhancements
    print("\n🧠 Step 3: Initializing enhanced iLLuMinator AI...")
    try:
        from illuminator_ai import IlluminatorAI
        ai = IlluminatorAI(fast_mode=True, auto_enhance=True)
        
        # Fetch fresh web data
        ai.fetch_and_integrate_web_data()
        
        print("✅ iLLuMinator AI enhanced and ready!")
        return ai
    except Exception as e:
        print(f"❌ AI initialization failed: {e}")
        return None

def run_interactive_demo(ai):
    """Run an interactive demo of the enhanced AI"""
    print("\n🎯 Interactive Demo - Enhanced iLLuMinator AI")
    print("=" * 60)
    
    demo_questions = [
        "What is Python?",
        "How do I optimize Python code?",
        "Write a FastAPI example",
        "Explain machine learning",
        "What are JavaScript best practices?",
        "How to design scalable databases?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n🤖 Demo {i}/{len(demo_questions)}: {question}")
        print("-" * 40)
        
        start_time = time.time()
        response = ai.chat(question)
        response_time = time.time() - start_time
        
        print(f"📝 Response: {response}")
        print(f"⏱️  Response time: {response_time:.2f}s")
        print(f"📊 Knowledge base entries: {len(ai.web_knowledge_base)}")
        
        # Brief pause between questions
        time.sleep(1)

def start_enhanced_api_server(port=8000):
    """Start the API server with all enhancements"""
    print(f"\n🌐 Starting Enhanced API Server on port {port}...")
    
    try:
        import uvicorn
        from api_server import app
        
        print(f"📡 API Documentation: http://localhost:{port}/docs")
        print(f"🏥 Health Check: http://localhost:{port}/health")
        print(f"💬 Chat Endpoint: http://localhost:{port}/chat")
        print("🚀 Server starting with all enhancements...")
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")

def run_performance_benchmark():
    """Run a comprehensive performance benchmark"""
    print("\n⚡ Performance Benchmark")
    print("=" * 60)
    
    try:
        from performance_monitor import benchmark_model
        benchmark_model()
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def main():
    """Main function with enhanced options"""
    parser = argparse.ArgumentParser(description="iLLuMinator AI Complete Enhancement Script")
    parser.add_argument("--mode", choices=[
        "setup", "demo", "api", "benchmark", "interactive", "all"
    ], default="all", help="Mode to run")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--skip-setup", action="store_true", help="Skip initial setup")
    
    args = parser.parse_args()
    
    # Initialize AI instance
    ai = None
    
    if not args.skip_setup and args.mode in ["setup", "demo", "all"]:
        ai = setup_environment()
        if not ai:
            print("❌ Setup failed. Exiting.")
            return
    
    if args.mode == "setup":
        print("\n✅ Setup complete! Run with --mode demo to test or --mode api to start server.")
        
    elif args.mode == "demo":
        if not ai:
            print("🧠 Loading iLLuMinator AI for demo...")
            from illuminator_ai import IlluminatorAI
            ai = IlluminatorAI(fast_mode=True, auto_enhance=True)
        
        run_interactive_demo(ai)
        
    elif args.mode == "api":
        start_enhanced_api_server(args.port)
        
    elif args.mode == "benchmark":
        run_performance_benchmark()
        
    elif args.mode == "interactive":
        if not ai:
            from illuminator_ai import IlluminatorAI
            ai = IlluminatorAI(fast_mode=True, auto_enhance=True)
        
        print("\n💬 Interactive Chat Mode")
        print("Type 'quit' to exit, 'demo' for demo questions")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'demo':
                    run_interactive_demo(ai)
                    continue
                
                if not user_input:
                    continue
                
                start_time = time.time()
                response = ai.chat(user_input)
                response_time = time.time() - start_time
                
                print(f"\n🤖 iLLuMinator AI: {response}")
                print(f"⏱️  ({response_time:.2f}s)")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.mode == "all":
        if ai:
            print("\n🎯 Running demo...")
            run_interactive_demo(ai)
            
            print("\n🌐 Starting API server...")
            start_enhanced_api_server(args.port)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode selection
        print("🚀 iLLuMinator AI Enhancement Suite")
        print("=" * 40)
        print("1. Complete Setup & Demo")
        print("2. Interactive Chat")
        print("3. Start API Server")
        print("4. Performance Benchmark")
        print("5. Setup Only")
        
        try:
            choice = input("\n📝 Enter choice (1-5): ").strip()
            modes = ["all", "interactive", "api", "benchmark", "setup"]
            
            if choice.isdigit() and 1 <= int(choice) <= 5:
                sys.argv.extend(["--mode", modes[int(choice) - 1]])
            else:
                print("❌ Invalid choice. Running complete setup...")
                sys.argv.extend(["--mode", "all"])
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            sys.exit(0)
    
    main()
