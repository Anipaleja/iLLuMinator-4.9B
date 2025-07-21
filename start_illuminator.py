#!/usr/bin/env python3
"""
iLLuMinator AI Startup Script
Easy way to start the API server and client
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Please install with: pip install -r requirements_api.txt")
        return False
    
    return True

def start_server():
    """Start the API server"""
    print("Starting iLLuMinator AI API Server...")
    try:
        server_process = subprocess.Popen([
            sys.executable, "api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to initialize
        time.sleep(8)  # Give model time to load
        
        return server_process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

def start_client():
    """Start the chatbot client"""
    print("Starting Chatbot Client...")
    try:
        subprocess.run([sys.executable, "chatbot_client.py"])
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Client error: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("iLLuMinator AI - Professional Language Model System")
    print("=" * 60)
    
    # Check if in correct directory
    if not Path("illuminator_ai.py").exists():
        print("Error: illuminator_ai.py not found")
        print("Please run this script from the Transformer directory")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\nSelect startup mode:")
    print("1. Start API server only")
    print("2. Start API server and client together")
    print("3. Start client only (server must be running)")
    print("4. Run quick test")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Server only
            server_process = start_server()
            if server_process:
                try:
                    print("\nAPI Server is running. Press Ctrl+C to stop.")
                    server_process.wait()
                except KeyboardInterrupt:
                    print("\nStopping server...")
                    server_process.terminate()
                    server_process.wait()
            
        elif choice == "2":
            # Server and client
            server_process = start_server()
            if server_process:
                try:
                    # Start client in the main thread
                    start_client()
                except KeyboardInterrupt:
                    print("\nStopping...")
                finally:
                    print("Stopping server...")
                    server_process.terminate()
                    server_process.wait()
            
        elif choice == "3":
            # Client only
            start_client()
            
        elif choice == "4":
            # Quick test
            print("Running quick test...")
            try:
                subprocess.run([sys.executable, "test_api.py"])
            except Exception as e:
                print(f"Test failed: {e}")
        
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
