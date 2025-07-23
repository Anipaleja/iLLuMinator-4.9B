#!/usr/bin/env python3
"""
Optimized Startup Script for iLLuMinator AI
Provides multiple optimization options and performance improvements
"""

import sys
import os
import time
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'fastapi',
        'uvicorn',
        'requests',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def optimize_environment():
    """Set environment variables for better performance"""
    optimizations = {
        'PYTHONUNBUFFERED': '1',  # Ensure output is not buffered
        'TOKENIZERS_PARALLELISM': 'false',  # Avoid tokenizer warnings
        'OMP_NUM_THREADS': '4',  # Limit OpenMP threads
        'MKL_NUM_THREADS': '4',  # Limit MKL threads
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
    
    print("Environment optimized for performance")

def setup_model_cache():
    """Setup model caching for faster subsequent loads"""
    cache_dir = Path("model_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Set PyTorch cache directory
    os.environ['TORCH_HOME'] = str(cache_dir)
    print(f"Model cache directory: {cache_dir}")

def start_api_server(fast_mode=True, port=8000):
    """Start the API server with optimizations"""
    print(f"Starting iLLuMinator AI API Server...")
    print(f"Fast mode: {fast_mode}")
    print(f"Port: {port}")
    
    try:
        import uvicorn
        from api_server import app
        
        # Configure uvicorn for performance
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,  # Disable reload for production
            log_level="info",
            access_log=False,  # Disable access logs for performance
            workers=1  # Single worker for memory efficiency
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except ImportError as e:
        print(f"Failed to start server: {e}")
        print("Make sure FastAPI and Uvicorn are installed:")
        print("pip install fastapi uvicorn")
    except Exception as e:
        print(f"Server error: {e}")

def start_interactive_chat():
    """Start interactive chat interface"""
    print("Starting iLLuMinator AI Interactive Chat...")
    
    try:
        from illuminator_ai import IlluminatorAI
        
        # Initialize with fast mode
        ai = IlluminatorAI(fast_mode=True)
        
        print("\\n" + "="*60)
        print("iLLuMinator AI - Interactive Chat")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("Type 'help' for available commands")
        print("="*60)
        
        while True:
            try:
                user_input = input("\\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    ai.clear_conversation()
                    print("Conversation history cleared.")
                    continue
                
                if user_input.lower() == 'help':
                    print("Available commands:")
                    print("  quit/exit/bye - End conversation")
                    print("  clear - Clear conversation history")
                    print("  help - Show this help message")
                    continue
                
                # Generate response with timing
                start_time = time.time()
                response = ai.chat(user_input)
                response_time = time.time() - start_time
                
                print(f"\\nAI: {response}")
                print(f"\\n(Response time: {response_time:.2f}s)")
                
            except KeyboardInterrupt:
                print("\\n\\nGoodbye!")
                break
            except Exception as e:
                print(f"\\nError: {e}")
                print("Please try again.")
    
    except Exception as e:
        print(f"Failed to start chat: {e}")

def run_performance_test():
    """Run a quick performance test"""
    print("Running iLLuMinator AI Performance Test...")
    
    try:
        from performance_monitor import benchmark_model
        benchmark_model()
    except Exception as e:
        print(f"Performance test failed: {e}")

def generate_training_data():
    """Generate enhanced training data"""
    print("Generating enhanced training data...")
    
    try:
        from enhanced_training_data_fetcher import TrainingDataFetcher
        
        fetcher = TrainingDataFetcher()
        fetcher.compile_training_data()
        
        print("Training data generated successfully!")
        
    except Exception as e:
        print(f"Failed to generate training data: {e}")

def run_quick_training():
    """Run quick training to improve the model"""
    print("Running quick training session...")
    
    try:
        from quick_train_enhanced import quick_training_session
        quick_training_session()
    except Exception as e:
        print(f"Training failed: {e}")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="iLLuMinator AI Startup Script")
    parser.add_argument("mode", choices=[
        "api", "chat", "test", "train-data", "train", "optimize"
    ], help="Mode to run")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--slow", action="store_true", help="Use slow mode (full model)")
    
    args = parser.parse_args()
    
    print("iLLuMinator AI - Optimized Startup Script")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Apply optimizations
    optimize_environment()
    setup_model_cache()
    
    # Run selected mode
    if args.mode == "api":
        start_api_server(fast_mode=not args.slow, port=args.port)
    elif args.mode == "chat":
        start_interactive_chat()
    elif args.mode == "test":
        run_performance_test()
    elif args.mode == "train-data":
        generate_training_data()
    elif args.mode == "train":
        run_quick_training()
    elif args.mode == "optimize":
        print("Environment optimized. You can now run other modes.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode selection
        print("iLLuMinator AI - Select Mode:")
        print("1. Start API Server")
        print("2. Interactive Chat")
        print("3. Performance Test")
        print("4. Generate Training Data")
        print("5. Quick Training")
        print("6. Optimize Environment Only")
        
        try:
            choice = input("\\nEnter choice (1-6): ").strip()
            modes = ["api", "chat", "test", "train-data", "train", "optimize"]
            
            if choice.isdigit() and 1 <= int(choice) <= 6:
                sys.argv.append(modes[int(choice) - 1])
            else:
                print("Invalid choice. Defaulting to chat mode.")
                sys.argv.append("chat")
        except KeyboardInterrupt:
            print("\\nExiting...")
            sys.exit(0)
    
    main()
