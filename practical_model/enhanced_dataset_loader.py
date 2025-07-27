#!/usr/bin/env python3
"""
Enhanced Dataset Loader for iLLuMinator Training
Uses the best datasets from LLMDataHub for optimal training
"""

import torch
import json
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PremiumDatasetLoader:
    """Load and prepare premium datasets for training"""
    
    def __init__(self, data_dir: str = "datasets", max_samples_per_dataset: int = 5000):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_samples = max_samples_per_dataset
        
    def load_alpaca_dataset(self) -> List[Dict]:
        """Load Stanford Alpaca dataset (52K instruction-following examples)"""
        try:
            from datasets import load_dataset
            logger.info("Loading Stanford Alpaca dataset...")
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            
            # Convert to our format and sample
            data = []
            for item in dataset:
                formatted_item = {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"]
                }
                data.append(formatted_item)
                if len(data) >= self.max_samples:
                    break
                    
            logger.info(f"Loaded {len(data)} Alpaca examples")
            return data
            
        except Exception as e:
            logger.warning(f"Could not load Alpaca dataset via HuggingFace: {e}")
            return self._get_alpaca_fallback()
    
    def _get_alpaca_fallback(self) -> List[Dict]:
        """Fallback Alpaca-style examples"""
        return [
            {
                "instruction": "Write a Python function to calculate the fibonacci sequence",
                "input": "",
                "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# More efficient iterative version:\ndef fibonacci_iterative(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
            },
            {
                "instruction": "Explain the concept of machine learning in simple terms",
                "input": "",
                "output": "Machine learning is like teaching a computer to recognize patterns and make predictions based on examples. Instead of programming specific rules, we show the computer lots of data and let it learn patterns on its own. For example, by showing it thousands of photos of cats and dogs with labels, it learns to tell the difference between cats and dogs in new photos it hasn't seen before."
            },
            {
                "instruction": "Create a Python class for a basic bank account",
                "input": "",
                "output": "class BankAccount:\n    def __init__(self, account_holder, initial_balance=0):\n        self.account_holder = account_holder\n        self.balance = initial_balance\n        self.transaction_history = []\n    \n    def deposit(self, amount):\n        if amount > 0:\n            self.balance += amount\n            self.transaction_history.append(f'Deposited ${amount}')\n            return True\n        return False\n    \n    def withdraw(self, amount):\n        if 0 < amount <= self.balance:\n            self.balance -= amount\n            self.transaction_history.append(f'Withdrew ${amount}')\n            return True\n        return False\n    \n    def get_balance(self):\n        return self.balance\n    \n    def get_history(self):\n        return self.transaction_history"
            }
        ]
    
    def load_dolly_dataset(self) -> List[Dict]:
        """Load Databricks Dolly 15K dataset"""
        try:
            from datasets import load_dataset
            logger.info("Loading Databricks Dolly 15K dataset...")
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
            
            data = []
            for item in dataset:
                formatted_item = {
                    "instruction": item["instruction"],
                    "input": item.get("context", ""),
                    "output": item["response"]
                }
                data.append(formatted_item)
                if len(data) >= self.max_samples:
                    break
                    
            logger.info(f"Loaded {len(data)} Dolly examples")
            return data
            
        except Exception as e:
            logger.warning(f"Could not load Dolly dataset: {e}")
            return self._get_dolly_fallback()
    
    def _get_dolly_fallback(self) -> List[Dict]:
        """Fallback Dolly-style examples"""
        return [
            {
                "instruction": "What are the benefits of renewable energy?",
                "input": "",
                "output": "Renewable energy offers several key benefits:\n\n1. Environmental: Reduces greenhouse gas emissions and air pollution\n2. Economic: Creates jobs and reduces dependence on fossil fuel imports\n3. Sustainability: Sources like solar and wind are inexhaustible\n4. Energy security: Reduces reliance on volatile fossil fuel markets\n5. Health: Cleaner air leads to better public health outcomes\n6. Innovation: Drives technological advancement and efficiency improvements\n\nThese factors make renewable energy crucial for a sustainable future."
            },
            {
                "instruction": "Explain how to start a small business",
                "input": "",
                "output": "Starting a small business involves several key steps:\n\n1. Develop a business idea and validate market demand\n2. Create a comprehensive business plan\n3. Secure funding through savings, loans, or investors\n4. Choose a business structure (LLC, corporation, etc.)\n5. Register your business and obtain necessary licenses\n6. Set up accounting and tax systems\n7. Establish your brand and online presence\n8. Launch your product or service\n9. Focus on customer acquisition and retention\n10. Monitor performance and adapt as needed\n\nSuccess requires dedication, planning, and willingness to learn from challenges."
            }
        ]
    
    def load_openorca_sample(self) -> List[Dict]:
        """Load a sample of OpenOrca dataset"""
        try:
            from datasets import load_dataset
            logger.info("Loading OpenOrca dataset sample...")
            # Load a smaller subset due to size
            dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            
            data = []
            for i, item in enumerate(dataset):
                if i >= self.max_samples:
                    break
                    
                formatted_item = {
                    "instruction": item["question"],
                    "input": "",
                    "output": item["response"]
                }
                data.append(formatted_item)
                
            logger.info(f"Loaded {len(data)} OpenOrca examples")
            return data
            
        except Exception as e:
            logger.warning(f"Could not load OpenOrca dataset: {e}")
            return self._get_openorca_fallback()
    
    def _get_openorca_fallback(self) -> List[Dict]:
        """Fallback OpenOrca-style examples"""
        return [
            {
                "instruction": "Write a Python function to check if a number is prime",
                "input": "",
                "output": "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    \n    # Check odd divisors up to sqrt(n)\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n\n# Example usage:\nprint(is_prime(17))  # True\nprint(is_prime(15))  # False"
            },
            {
                "instruction": "Explain the water cycle",
                "input": "",
                "output": "The water cycle is the continuous movement of water on, above, and below Earth's surface. It consists of several key processes:\n\n1. **Evaporation**: Heat from the sun causes water from oceans, lakes, and rivers to turn into water vapor\n2. **Transpiration**: Plants release water vapor through their leaves\n3. **Condensation**: Water vapor rises, cools, and forms clouds\n4. **Precipitation**: Water falls back to Earth as rain, snow, or hail\n5. **Collection**: Water flows into rivers, lakes, and oceans, or soaks into groundwater\n\nThis cycle is essential for distributing fresh water across the planet and supporting all life."
            }
        ]
    
    def load_code_dataset(self) -> List[Dict]:
        """Load programming-focused dataset"""
        return [
            {
                "instruction": "Create a Python function to implement binary search",
                "input": "",
                "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1  # Target not found\n\n# Example usage:\nsorted_array = [1, 3, 5, 7, 9, 11, 13]\nresult = binary_search(sorted_array, 7)\nprint(f\"Index: {result}\")  # Output: Index: 3"
            },
            {
                "instruction": "Write a Python class to implement a stack data structure",
                "input": "",
                "output": "class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        \"\"\"Add an item to the top of the stack\"\"\"\n        self.items.append(item)\n    \n    def pop(self):\n        \"\"\"Remove and return the top item\"\"\"\n        if self.is_empty():\n            raise IndexError(\"Stack is empty\")\n        return self.items.pop()\n    \n    def peek(self):\n        \"\"\"Return the top item without removing it\"\"\"\n        if self.is_empty():\n            raise IndexError(\"Stack is empty\")\n        return self.items[-1]\n    \n    def is_empty(self):\n        \"\"\"Check if the stack is empty\"\"\"\n        return len(self.items) == 0\n    \n    def size(self):\n        \"\"\"Return the number of items in the stack\"\"\"\n        return len(self.items)\n\n# Example usage:\nstack = Stack()\nstack.push(1)\nstack.push(2)\nprint(stack.pop())  # Output: 2"
            },
            {
                "instruction": "Implement a simple calculator in Python",
                "input": "",
                "output": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError(\"Cannot divide by zero\")\n        return a / b\n    \n    def power(self, a, b):\n        return a ** b\n    \n    def sqrt(self, a):\n        if a < 0:\n            raise ValueError(\"Cannot take square root of negative number\")\n        return a ** 0.5\n\n# Example usage:\ncalc = Calculator()\nprint(calc.add(5, 3))      # 8\nprint(calc.multiply(4, 7)) # 28\nprint(calc.divide(10, 2))  # 5.0"
            }
        ]
    
    def load_scientific_dataset(self) -> List[Dict]:
        """Load science and mathematics focused dataset"""
        return [
            {
                "instruction": "Explain photosynthesis in detail",
                "input": "",
                "output": "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in two main stages:\n\n**Light-Dependent Reactions (in thylakoids):**\n1. Chlorophyll absorbs light energy\n2. Water molecules are split, releasing oxygen\n3. Energy is used to produce ATP and NADPH\n\n**Light-Independent Reactions (Calvin Cycle in stroma):**\n1. CO₂ is fixed into organic molecules\n2. ATP and NADPH provide energy\n3. Glucose is produced as the final product\n\n**Overall equation:** 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\nThis process is crucial as it produces oxygen and serves as the foundation of most food chains on Earth."
            },
            {
                "instruction": "What is Newton's Second Law of Motion?",
                "input": "",
                "output": "Newton's Second Law of Motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.\n\n**Mathematical formula:** F = ma\nWhere:\n- F = net force (in Newtons)\n- m = mass (in kilograms)\n- a = acceleration (in m/s²)\n\n**Key implications:**\n1. More force produces greater acceleration\n2. Greater mass produces less acceleration for the same force\n3. Force and acceleration are in the same direction\n\n**Example:** A 10 kg object with 20 N of force applied will accelerate at 2 m/s² (20 ÷ 10 = 2)\n\nThis law is fundamental to understanding motion and is used in engineering, physics, and everyday applications."
            }
        ]
    
    def create_comprehensive_dataset(self) -> List[str]:
        """Create a comprehensive training dataset from all sources"""
        all_data = []
        
        logger.info("Creating comprehensive dataset from multiple sources...")
        
        # Load datasets
        datasets = [
            ("Alpaca", self.load_alpaca_dataset()),
            ("Dolly", self.load_dolly_dataset()),
            ("OpenOrca", self.load_openorca_sample()),
            ("Code", self.load_code_dataset()),
            ("Scientific", self.load_scientific_dataset())
        ]
        
        # Format all data
        formatted_texts = []
        for name, dataset in datasets:
            logger.info(f"Processing {name} dataset: {len(dataset)} examples")
            
            for item in dataset:
                instruction = item["instruction"]
                input_text = item.get("input", "")
                output_text = item["output"]
                
                if input_text:
                    text = f"Human: {instruction}\nContext: {input_text}\nAssistant: {output_text}"
                else:
                    text = f"Human: {instruction}\nAssistant: {output_text}"
                
                formatted_texts.append(text)
        
        # Shuffle for better training
        random.shuffle(formatted_texts)
        
        logger.info(f"Created comprehensive dataset with {len(formatted_texts)} examples")
        return formatted_texts
    
    def save_dataset(self, texts: List[str], filename: str) -> Path:
        """Save dataset to file"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {filepath}")
        return filepath

class DatasetProcessor:
    """Process datasets for different model sizes"""
    
    def __init__(self):
        self.loader = PremiumDatasetLoader()
    
    def create_dataset_for_120M_model(self) -> Tuple[List[str], Path]:
        """Create optimized dataset for 120M parameter model"""
        logger.info("Creating dataset optimized for 120M parameter model...")
        
        # Use smaller dataset for faster training
        self.loader.max_samples = 1000  # Smaller for 120M model
        texts = self.loader.create_comprehensive_dataset()
        
        # Keep most relevant examples (first N)
        optimized_texts = texts[:2000]  # 2K examples for 120M model
        
        filepath = self.loader.save_dataset(optimized_texts, "dataset_120M_optimized.json")
        logger.info(f"120M model dataset: {len(optimized_texts)} examples")
        
        return optimized_texts, filepath
    
    def create_dataset_for_4_7B_model(self) -> Tuple[List[str], Path]:
        """Create comprehensive dataset for 4.7B parameter model"""
        logger.info("Creating dataset optimized for 4.7B parameter model...")
        
        # Use larger dataset for better training
        self.loader.max_samples = 3000  # Larger for 4.7B model
        texts = self.loader.create_comprehensive_dataset()
        
        # Use full dataset for larger model
        comprehensive_texts = texts  # Up to 15K examples
        
        filepath = self.loader.save_dataset(comprehensive_texts, "dataset_4_7B_comprehensive.json")
        logger.info(f"4.7B model dataset: {len(comprehensive_texts)} examples")
        
        return comprehensive_texts, filepath

def main():
    """Main function to create datasets for both models"""
    processor = DatasetProcessor()
    
    print("Creating Premium Datasets for iLLuMinator Models")
    print("=" * 60)
    
    # Create dataset for 120M model
    texts_120M, path_120M = processor.create_dataset_for_120M_model()
    
    # Create dataset for 4.7B model  
    texts_4_7B, path_4_7B = processor.create_dataset_for_4_7B_model()
    
    print("\nDataset Creation Complete!")
    print(f"120M Model Dataset: {path_120M} ({len(texts_120M)} examples)")
    print(f"4.7B Model Dataset: {path_4_7B} ({len(texts_4_7B)} examples)")
    
    return {
        "120M": {"texts": texts_120M, "path": path_120M},
        "4_7B": {"texts": texts_4_7B, "path": path_4_7B}
    }

if __name__ == "__main__":
    main()
