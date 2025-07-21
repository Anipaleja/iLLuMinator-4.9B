#!/usr/bin/env python3
"""
iLLuMinator AI - Main Entry Point
Professional artificial intelligence assistant with advanced capabilities

Usage:
    python main.py              # Interactive mode
    python main.py --demo        # Professional demonstration
    python main.py --help        # Show help information
"""

import sys
import argparse
from pathlib import Path

def print_banner():
    """Display professional banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                            iLLuMinator AI                                    ║
║                   Professional Language Model System                         ║
║                                                                              ║
║  Advanced AI assistant with code generation and intelligent conversation     ║
║  Designed for professional software development and technical support        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - Ready")
    except ImportError:
        missing_deps.append("torch>=2.0.0")
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__} - Ready")
    except ImportError:
        missing_deps.append("numpy>=1.21.0")
    
    if missing_deps:
        print("\n❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\nPlease install with: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All dependencies satisfied")
    return True

def run_interactive_mode():
    """Run the AI in interactive mode"""
    try:
        from illuminator_ai import main as ai_main
        print("\n🚀 Starting Interactive Mode...")
        ai_main()
    except ImportError as e:
        print(f"❌ Failed to import iLLuMinator AI: {e}")
        print("Please ensure illuminator_ai.py is in the current directory")
    except Exception as e:
        print(f"❌ Error running interactive mode: {e}")

def run_demo_mode():
    """Run the professional demonstration"""
    try:
        from demo_professional import main as demo_main
        print("\n🎯 Starting Professional Demonstration...")
        demo_main()
    except ImportError as e:
        print(f"❌ Failed to import demo: {e}")
        print("Please ensure demo_professional.py is in the current directory")
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")

def show_help():
    """Show comprehensive help information"""
    help_text = """
iLLuMinator AI - Professional Language Model System

OVERVIEW:
    iLLuMinator AI is a sophisticated artificial intelligence assistant designed
    for professional software development and technical support. It provides
    advanced code generation capabilities and intelligent conversation.

USAGE:
    python main.py                 # Interactive mode (default)
    python main.py --demo          # Professional demonstration
    python main.py --interactive   # Interactive mode (explicit)
    python main.py --help          # Show this help

FEATURES:
    • Advanced code generation in multiple programming languages
    • Intelligent technical discussions with context awareness
    • Professional-quality output without unnecessary formatting
    • Minimal dependencies for maximum portability
    • Clean, scalable transformer-based architecture

REQUIREMENTS:
    • Python 3.7+
    • PyTorch 2.0+
    • NumPy 1.21+

COMMANDS IN INTERACTIVE MODE:
    code: <description>           # Generate code based on description
    <message>                     # Chat with the AI assistant
    clear                         # Clear conversation history
    save <filename>              # Save conversation to file
    quit                         # Exit the application

EXAMPLES:
    Interactive Usage:
        You: code: Create a FastAPI endpoint for user registration
        You: What are the benefits of microservices architecture?
        You: Explain how to optimize database queries

    Demo Mode:
        python main.py --demo     # Run comprehensive demonstration

PROJECT STRUCTURE:
    main.py                      # Main entry point (this file)
    illuminator_ai.py           # Core AI implementation
    demo_professional.py        # Professional demonstration
    requirements_clean.txt      # Essential dependencies
    README_PROFESSIONAL.md      # Comprehensive documentation
    legacy/                     # Previous implementations
    docs/                       # Additional documentation

SUPPORT:
    For detailed documentation, see README_PROFESSIONAL.md
    For technical support, refer to the inline documentation
    """
    print(help_text)

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="iLLuMinator AI - Professional Language Model System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run professional demonstration mode"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode (default)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="iLLuMinator AI v2.0 Professional"
    )
    
    args = parser.parse_args()
    
    # Show banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Determine mode
    if args.demo:
        run_demo_mode()
    else:
        # Default to interactive mode
        run_interactive_mode()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using iLLuMinator AI!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)
