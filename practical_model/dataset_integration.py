#!/usr/bin/env python3
"""
Enhanced Dataset Integration for iLLuMinator Training
Based on top-tier datasets from LLMDataHub repository
"""

import torch
import requests
import json
import os
from typing import List, Dict, Optional
from urllib.parse import urlparse
import gzip
from pathlib import Path
from datasets import load_dataset
import random
from transformers import AutoTokenizer

class LLMDatasetDownloader:
    """Download and prepare datasets from LLMDataHub recommendations"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_alpaca_dataset(self) -> List[Dict]:
        """Download Stanford Alpaca dataset (52K instruction-following examples)"""
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        filepath = self.data_dir / "alpaca_data.json"
        
        if not filepath.exists():
            print("Downloading Stanford Alpaca dataset...")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'w') as f:
                json.dump(response.json(), f, indent=2)
            print(f"Downloaded Alpaca dataset to {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def download_dolly_dataset(self) -> List[Dict]:
        """Download Databricks Dolly 15K dataset"""
        # Note: This would typically use the Hugging Face datasets library
        # For now, we'll create a sample structure
        print("Dolly 15K dataset would be downloaded from Hugging Face")
        return []
    
    def download_openorca_sample(self) -> List[Dict]:
        """Download a sample of OpenOrca dataset (4.5M completions)"""
        print("OpenOrca dataset would be downloaded from Hugging Face")
        # This is a massive dataset, typically requires HF datasets library
        return []
    
    def get_code_datasets(self) -> List[Dict]:
        """Get programming-focused datasets"""
        code_examples = [
            {
                "instruction": "Write a Python function to calculate factorial",
                "input": "",
                "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\n# Example usage\nprint(factorial(5))  # Output: 120"
            },
            {
                "instruction": "Create a Python class for a simple calculator",
                "input": "",
                "output": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b != 0:\n            return a / b\n        else:\n            return 'Error: Division by zero'\n\n# Example usage\ncalc = Calculator()\nprint(calc.add(5, 3))  # Output: 8"
            },
            {
                "instruction": "Implement binary search algorithm",
                "input": "",
                "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1  # Target not found\n\n# Example usage\narr = [1, 3, 5, 7, 9, 11]\nresult = binary_search(arr, 7)\nprint(f\"Index: {result}\")  # Output: Index: 3"
            }
        ]
        return code_examples
    
    def get_conversation_datasets(self) -> List[Dict]:
        """Get conversational AI datasets"""
        conversations = [
            {
                "instruction": "Explain machine learning to a beginner",
                "input": "",
                "output": "Machine learning is like teaching a computer to recognize patterns and make predictions, similar to how humans learn from experience. Instead of programming specific rules, we show the computer lots of examples and let it figure out the patterns on its own. For instance, by showing it thousands of photos labeled 'cat' or 'dog', it learns to distinguish between cats and dogs in new photos it hasn't seen before."
            },
            {
                "instruction": "How do I start learning programming?",
                "input": "",
                "output": "Here's a beginner-friendly approach to learning programming:\n\n1. Choose a beginner-friendly language like Python\n2. Start with basic concepts: variables, loops, and functions\n3. Practice with small projects like calculators or simple games\n4. Use online resources like Codecademy, freeCodeCamp, or Python.org\n5. Join programming communities for support\n6. Build projects that interest you to stay motivated\n7. Don't rush - focus on understanding concepts thoroughly\n\nRemember, programming is about problem-solving, so practice regularly and be patient with yourself!"
            },
            {
                "instruction": "What is the difference between AI and machine learning?",
                "input": "",
                "output": "AI (Artificial Intelligence) and Machine Learning are related but distinct concepts:\n\n**Artificial Intelligence (AI):**\n- Broader concept of creating intelligent machines\n- Includes any technique that enables computers to mimic human intelligence\n- Can include rule-based systems, expert systems, and machine learning\n\n**Machine Learning (ML):**\n- A subset of AI\n- Focuses on algorithms that learn and improve from data\n- Systems get better at tasks through experience without being explicitly programmed\n\nThink of AI as the goal (making machines smart) and ML as one of the main methods to achieve that goal (learning from data). Other AI approaches include symbolic reasoning, expert systems, and rule-based programming."
            }
        ]
        return conversations
    
    def get_scientific_datasets(self) -> List[Dict]:
        """Get science and mathematics focused datasets"""
        science_data = [
            {
                "instruction": "Explain photosynthesis in simple terms",
                "input": "",
                "output": "Photosynthesis is how plants make their own food using sunlight. Here's the simple process:\n\n1. Plants absorb sunlight through their leaves\n2. They take in carbon dioxide from the air through tiny pores\n3. They absorb water through their roots\n4. Using chlorophyll (the green stuff in leaves), they combine these ingredients\n5. This creates glucose (sugar) for food and releases oxygen as a byproduct\n\nThe basic equation: Carbon Dioxide + Water + Sunlight → Glucose + Oxygen\n\nThis process is crucial for life on Earth because it produces the oxygen we breathe and forms the base of most food chains."
            },
            {
                "instruction": "What is the theory of relativity?",
                "input": "",
                "output": "Einstein's theory of relativity consists of two parts:\n\n**Special Relativity (1905):**\n- Time and space are connected (spacetime)\n- Nothing can travel faster than light\n- Time passes differently for objects moving at different speeds\n- Mass and energy are equivalent (E=mc²)\n\n**General Relativity (1915):**\n- Gravity isn't a force, but the curvature of spacetime\n- Massive objects bend spacetime, creating what we experience as gravity\n- Explains phenomena like black holes and gravitational time dilation\n\nKey insight: Space and time aren't fixed - they're flexible and affected by motion and gravity. This revolutionized our understanding of the universe and led to technologies like GPS satellites."
            }
        ]
        return science_data

class EnhancedDatasetProcessor:
    """Process datasets for optimal training"""
    
    def __init__(self):
        self.downloader = LLMDatasetDownloader()
    
    def format_instruction_data(self, data: List[Dict]) -> List[str]:
        """Format instruction-following data for training"""
        formatted_texts = []
        
        for item in data:
            if 'instruction' in item:
                instruction = item['instruction']
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                if input_text:
                    text = f"Human: {instruction}\nInput: {input_text}\nAssistant: {output_text}"
                else:
                    text = f"Human: {instruction}\nAssistant: {output_text}"
                
                formatted_texts.append(text)
        
        return formatted_texts
    
    def create_comprehensive_dataset(self) -> List[str]:
        """Create a comprehensive training dataset from multiple sources"""
        all_texts = []
        
        print("Building comprehensive training dataset...")
        
        # Add code-focused examples
        code_data = self.downloader.get_code_datasets()
        all_texts.extend(self.format_instruction_data(code_data))
        print(f"Added {len(code_data)} code examples")
        
        # Add conversational examples
        conv_data = self.downloader.get_conversation_datasets()
        all_texts.extend(self.format_instruction_data(conv_data))
        print(f"Added {len(conv_data)} conversation examples")
        
        # Add scientific examples
        sci_data = self.downloader.get_scientific_datasets()
        all_texts.extend(self.format_instruction_data(sci_data))
        print(f"Added {len(sci_data)} scientific examples")
        
        # Try to download Alpaca dataset
        try:
            alpaca_data = self.downloader.download_alpaca_dataset()
            # Sample a subset for memory efficiency
            alpaca_sample = alpaca_data[:1000] if len(alpaca_data) > 1000 else alpaca_data
            all_texts.extend(self.format_instruction_data(alpaca_sample))
            print(f"Added {len(alpaca_sample)} Alpaca examples")
        except Exception as e:
            print(f"Could not download Alpaca dataset: {e}")
        
        print(f"Total dataset size: {len(all_texts)} examples")
        return all_texts
    
    def save_dataset(self, texts: List[str], filename: str = "comprehensive_training_data.json"):
        """Save dataset to file"""
        filepath = Path("datasets") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(texts, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
        return filepath

def create_training_datasets():
    """Create enhanced training datasets based on LLMDataHub recommendations"""
    
    print("Creating Enhanced Training Datasets")
    print("Based on LLMDataHub recommendations")
    print("=" * 50)
    
    processor = EnhancedDatasetProcessor()
    
    # Create comprehensive dataset
    texts = processor.create_comprehensive_dataset()
    
    # Save dataset
    dataset_path = processor.save_dataset(texts)
    
    # Create a smaller focused dataset for quick training
    focused_texts = texts[:100]  # First 100 examples
    focused_path = processor.save_dataset(focused_texts, "focused_training_data.json")
    
    print("\nDataset Creation Complete!")
    print(f"Comprehensive dataset: {dataset_path} ({len(texts)} examples)")
    print(f"Focused dataset: {focused_path} ({len(focused_texts)} examples)")
    
    return texts

if __name__ == "__main__":
    create_training_datasets()
