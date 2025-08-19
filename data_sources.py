#!/usr/bin/env python3
"""
Professional Training Data Sources for iLLuMinator 4.9B
Integrates high-quality datasets used by leading AI research organizations
"""

import json
import os
from typing import List, Dict, Optional, Iterator
from pathlib import Path
import requests
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalDatasetLoader:
    """
    Loads and processes professional-grade training datasets
    similar to those used by Gemini, LLaMA, Mistral, and OpenAI models
    """
    
    def __init__(self, cache_dir: str = "datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_openorca_dataset(self, subset_size: Optional[int] = 10000) -> List[Dict]:
        """
        Load OpenOrca dataset - high-quality instruction following data
        Used by leading open-source models
        """
        try:
            logger.info("Loading OpenOrca dataset...")
            dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            
            examples = []
            for i, example in enumerate(dataset):
                if subset_size and i >= subset_size:
                    break
                    
                formatted_example = {
                    "instruction": example.get("question", ""),
                    "input": "",
                    "output": example.get("response", ""),
                    "source": "OpenOrca",
                    "quality_score": 0.9
                }
                examples.append(formatted_example)
                
            logger.info(f"Loaded {len(examples)} OpenOrca examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load OpenOrca: {e}")
            return []
    
    def load_ultrachat_dataset(self, subset_size: Optional[int] = 5000) -> List[Dict]:
        """
        Load UltraChat dataset - multi-turn conversational data
        High-quality dialogue training data
        """
        try:
            logger.info("Loading UltraChat dataset...")
            dataset = load_dataset("stingning/ultrachat", split="train", streaming=True)
            
            examples = []
            for i, example in enumerate(dataset):
                if subset_size and i >= subset_size:
                    break
                
                # Process multi-turn conversation
                conversation = example.get("data", [])
                if len(conversation) >= 2:
                    human_msg = conversation[0] if conversation[0] else ""
                    assistant_msg = conversation[1] if len(conversation) > 1 else ""
                    
                    formatted_example = {
                        "instruction": human_msg,
                        "input": "",
                        "output": assistant_msg,
                        "source": "UltraChat",
                        "quality_score": 0.85
                    }
                    examples.append(formatted_example)
                    
            logger.info(f"Loaded {len(examples)} UltraChat examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load UltraChat: {e}")
            return []
    
    def load_code_alpaca_dataset(self, subset_size: Optional[int] = 5000) -> List[Dict]:
        """
        Load Code Alpaca dataset - programming instruction data
        Essential for code generation capabilities
        """
        try:
            logger.info("Loading Code Alpaca dataset...")
            dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            
            examples = []
            for i, example in enumerate(dataset):
                if subset_size and i >= subset_size:
                    break
                    
                formatted_example = {
                    "instruction": example.get("instruction", ""),
                    "input": example.get("input", ""),
                    "output": example.get("output", ""),
                    "source": "CodeAlpaca",
                    "quality_score": 0.88
                }
                examples.append(formatted_example)
                
            logger.info(f"Loaded {len(examples)} Code Alpaca examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load Code Alpaca: {e}")
            return []
    
    def load_wizardlm_dataset(self, subset_size: Optional[int] = 3000) -> List[Dict]:
        """
        Load WizardLM dataset - evolved instruction following
        Complex reasoning and multi-step problem solving
        """
        try:
            logger.info("Loading WizardLM dataset...")
            dataset = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
            
            examples = []
            for i, example in enumerate(dataset):
                if subset_size and i >= subset_size:
                    break
                    
                formatted_example = {
                    "instruction": example.get("instruction", ""),
                    "input": "",
                    "output": example.get("output", ""),
                    "source": "WizardLM",
                    "quality_score": 0.92
                }
                examples.append(formatted_example)
                
            logger.info(f"Loaded {len(examples)} WizardLM examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load WizardLM: {e}")
            return []
    
    def load_math_dataset(self, subset_size: Optional[int] = 2000) -> List[Dict]:
        """
        Load mathematical reasoning dataset
        Important for quantitative reasoning capabilities
        """
        try:
            logger.info("Loading Math dataset...")
            dataset = load_dataset("hendrycks/competition_math", split="train")
            
            examples = []
            for i, example in enumerate(dataset):
                if subset_size and i >= subset_size:
                    break
                    
                formatted_example = {
                    "instruction": f"Solve this {example.get('type', 'math')} problem: {example.get('problem', '')}",
                    "input": "",
                    "output": example.get("solution", ""),
                    "source": "Competition Math",
                    "quality_score": 0.95
                }
                examples.append(formatted_example)
                
            logger.info(f"Loaded {len(examples)} Math examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load Math dataset: {e}")
            return []
    
    def load_scientific_papers_dataset(self, subset_size: Optional[int] = 1000) -> List[Dict]:
        """
        Load scientific papers dataset for academic knowledge
        Based on arXiv abstracts and papers
        """
        try:
            logger.info("Loading Scientific Papers dataset...")
            dataset = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
            
            examples = []
            for i, example in enumerate(dataset):
                if subset_size and i >= subset_size:
                    break
                
                abstract = example.get("abstract", "")
                article = example.get("article", "")
                
                if abstract and len(abstract) > 100:
                    formatted_example = {
                        "instruction": "Explain this scientific concept in simple terms",
                        "input": abstract[:500],  # Truncate for memory efficiency
                        "output": f"This research explores {abstract[:200]}...",
                        "source": "ArXiv Papers",
                        "quality_score": 0.90
                    }
                    examples.append(formatted_example)
                    
            logger.info(f"Loaded {len(examples)} Scientific Papers examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load Scientific Papers: {e}")
            return []
    
    def create_comprehensive_dataset(self, total_size: int = 25000) -> List[Dict]:
        """
        Create a comprehensive training dataset combining multiple sources
        Balances different types of knowledge and capabilities
        """
        logger.info("Creating comprehensive professional training dataset...")
        
        all_examples = []
        
        # Load datasets with balanced proportions
        datasets_config = [
            ("OpenOrca", self.load_openorca_dataset, 0.4),  # 40% instruction following
            ("UltraChat", self.load_ultrachat_dataset, 0.2),  # 20% conversation
            ("CodeAlpaca", self.load_code_alpaca_dataset, 0.2),  # 20% programming
            ("WizardLM", self.load_wizardlm_dataset, 0.1),  # 10% complex reasoning
            ("Math", self.load_math_dataset, 0.05),  # 5% mathematics
            ("Scientific", self.load_scientific_papers_dataset, 0.05),  # 5% scientific
        ]
        
        for dataset_name, loader_func, proportion in datasets_config:
            subset_size = int(total_size * proportion)
            try:
                examples = loader_func(subset_size)
                all_examples.extend(examples)
                logger.info(f"Added {len(examples)} examples from {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
        
        # Shuffle the combined dataset
        import random
        random.shuffle(all_examples)
        
        logger.info(f"Created comprehensive dataset with {len(all_examples)} total examples")
        return all_examples
    
    def save_dataset(self, examples: List[Dict], filename: str = "professional_training_data.json"):
        """Save dataset to file for reuse"""
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {filepath}")
        return filepath
    
    def load_saved_dataset(self, filename: str = "professional_training_data.json") -> List[Dict]:
        """Load previously saved dataset"""
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            logger.info(f"Loaded {len(examples)} examples from {filepath}")
            return examples
        else:
            logger.warning(f"No saved dataset found at {filepath}")
            return []

def format_for_training(examples: List[Dict], format_type: str = "alpaca") -> List[str]:
    """
    Format examples for training in different styles
    """
    formatted_texts = []
    
    for example in examples:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        
        if format_type == "alpaca":
            if input_text:
                text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        
        elif format_type == "chat":
            if input_text:
                text = f"Human: {instruction}\nContext: {input_text}\nAssistant: {output_text}"
            else:
                text = f"Human: {instruction}\nAssistant: {output_text}"
        
        elif format_type == "simple":
            text = f"Question: {instruction}\nAnswer: {output_text}"
        
        else:
            text = f"{instruction}\n{output_text}"
        
        formatted_texts.append(text)
    
    return formatted_texts

def main():
    """Main function to create and save professional training dataset"""
    loader = ProfessionalDatasetLoader()
    
    # Check if dataset already exists
    existing_dataset = loader.load_saved_dataset()
    
    if existing_dataset:
        logger.info(f"Using existing dataset with {len(existing_dataset)} examples")
        examples = existing_dataset
    else:
        # Create new comprehensive dataset
        examples = loader.create_comprehensive_dataset(total_size=25000)
        
        # Save for future use
        loader.save_dataset(examples)
    
    # Format for training
    formatted_texts = format_for_training(examples, format_type="alpaca")
    
    # Save formatted texts
    with open(loader.cache_dir / "formatted_training_texts.json", 'w') as f:
        json.dump(formatted_texts, f, indent=2)
    
    logger.info(f"Training dataset ready with {len(formatted_texts)} formatted examples")
    
    # Print sample statistics
    sources = {}
    quality_scores = []
    
    for example in examples:
        source = example.get("source", "Unknown")
        sources[source] = sources.get(source, 0) + 1
        quality_scores.append(example.get("quality_score", 0.5))
    
    print("\nDataset Statistics:")
    print(f"Total examples: {len(examples)}")
    print(f"Average quality score: {sum(quality_scores) / len(quality_scores):.3f}")
    print("\nSource distribution:")
    for source, count in sources.items():
        print(f"  {source}: {count} examples ({count/len(examples)*100:.1f}%)")

if __name__ == "__main__":
    main()