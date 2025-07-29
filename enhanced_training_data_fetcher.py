# Enhanced Training Data Fetcher for Professional 4.9B Model
# Designed to use high-quality datasets similar to Claude, GPT, and Gemini

import os
import json
import requests
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import hashlib

class EnterpriseDatasetManager:
    """Professional dataset manager for training large language models"""
    
    def __init__(self, data_dir: str = "training_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'dataset_preparation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # High-quality dataset sources (legal and ethical)
        self.dataset_sources = {
            "openwebtext": {
                "description": "High-quality web text similar to GPT-2 training data",
                "size_gb": 12.0,
                "url": "https://huggingface.co/datasets/openwebtext",
                "quality": "high"
            },
            "c4": {
                "description": "Colossal Clean Crawled Corpus (T5 training data)",
                "size_gb": 305.0,
                "url": "https://huggingface.co/datasets/c4",
                "quality": "very_high"
            },
            "pile": {
                "description": "The Pile - Diverse text dataset",
                "size_gb": 825.0,
                "url": "https://huggingface.co/datasets/EleutherAI/pile",
                "quality": "very_high"
            },
            "wikipedia": {
                "description": "Wikipedia dumps in multiple languages",
                "size_gb": 20.0,
                "url": "https://huggingface.co/datasets/wikipedia",
                "quality": "high"
            },
            "bookcorpus": {
                "description": "Collection of over 11,000 books",
                "size_gb": 4.5,
                "url": "https://huggingface.co/datasets/bookcorpus",
                "quality": "very_high"
            },
            "common_crawl": {
                "description": "Filtered Common Crawl data",
                "size_gb": 100.0,
                "url": "https://huggingface.co/datasets/common_crawl",
                "quality": "medium"
            },
            "github_code": {
                "description": "High-quality code repositories",
                "size_gb": 150.0,
                "url": "https://huggingface.co/datasets/codeparrot/github-code",
                "quality": "high"
            },
            "arxiv": {
                "description": "Academic papers from arXiv",
                "size_gb": 10.0,
                "url": "https://huggingface.co/datasets/arxiv_dataset",
                "quality": "very_high"
            }
        }
    
    def install_requirements(self):
        """Install required packages for dataset processing"""
        self.logger.info("Installing required packages...")
        
        packages = [
            "datasets>=2.14.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "huggingface_hub>=0.16.0",
            "tokenizers>=0.13.0",
            "numpy>=1.24.0",
            "tqdm>=4.65.0",
            "requests>=2.31.0",
            "beautifulsoup4>=4.12.0",
            "nltk>=3.8.0",
            "spacy>=3.6.0"
        ]
        
        import subprocess
        import sys
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                self.logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"⚠️ Failed to install {package}: {e}")
    
    def download_dataset(self, dataset_name: str, sample_size: int = 100000) -> bool:
        """Download and prepare a specific dataset"""
        
        if dataset_name not in self.dataset_sources:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.dataset_sources[dataset_name]
        self.logger.info(f"📥 Downloading {dataset_name}: {dataset_info['description']}")
        
        try:
            from datasets import load_dataset
            
            # Load dataset with Hugging Face datasets library
            if dataset_name == "openwebtext":
                dataset = load_dataset("openwebtext", split=f"train[:{sample_size}]")
            elif dataset_name == "c4":
                dataset = load_dataset("c4", "en", split=f"train[:{sample_size}]")
            elif dataset_name == "pile":
                dataset = load_dataset("EleutherAI/pile", split=f"train[:{sample_size}]")
            elif dataset_name == "wikipedia":
                dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{sample_size}]")
            elif dataset_name == "bookcorpus":
                dataset = load_dataset("bookcorpus", split=f"train[:{sample_size}]")
            elif dataset_name == "github_code":
                dataset = load_dataset("codeparrot/github-code", split=f"train[:{sample_size}]")
            elif dataset_name == "arxiv":
                dataset = load_dataset("arxiv_dataset", split=f"train[:{sample_size}]")
            else:
                self.logger.error(f"Dataset {dataset_name} not implemented yet")
                return False
            
            # Save processed dataset
            output_file = self.data_dir / f"{dataset_name}_processed.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, item in enumerate(dataset):
                    # Extract text based on dataset structure
                    text = ""
                    if 'text' in item:
                        text = item['text']
                    elif 'content' in item:
                        text = item['content']
                    elif 'article' in item:
                        text = item['article']
                    elif 'code' in item:
                        text = item['code']
                    
                    if text and len(text.strip()) > 100:  # Filter short texts
                        json.dump({"text": text.strip()}, f)
                        f.write('\n')
                    
                    if (i + 1) % 1000 == 0:
                        self.logger.info(f"  Processed {i + 1} samples...")
            
            self.logger.info(f"✅ {dataset_name} downloaded and saved to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    def create_mixed_dataset(self, target_size_gb: float = 20.0) -> str:
        """Create a mixed dataset similar to what large AI companies use"""
        
        self.logger.info(f"🎯 Creating mixed enterprise dataset (target: {target_size_gb} GB)")
        
        # Optimal mix for 4.9B parameter model (similar to GPT-3 training data)
        dataset_mix = [
            ("c4", 0.30),           # 30% - Clean web text
            ("openwebtext", 0.20),  # 20% - High-quality web content
            ("wikipedia", 0.15),     # 15% - Encyclopedia knowledge
            ("bookcorpus", 0.15),   # 15% - Literature and books
            ("arxiv", 0.10),        # 10% - Academic/scientific content
            ("github_code", 0.10)   # 10% - Programming knowledge
        ]
        
        # Calculate sample sizes based on available storage and quality
        total_samples = int(target_size_gb * 1024 * 1024 * 100)  # Rough estimation
        
        mixed_dataset = []
        
        for dataset_name, ratio in dataset_mix:
            sample_size = int(total_samples * ratio)
            self.logger.info(f"📊 Processing {dataset_name} ({ratio*100:.0f}%): {sample_size:,} samples")
            
            if self.download_dataset(dataset_name, sample_size):
                # Load processed data
                dataset_file = self.data_dir / f"{dataset_name}_processed.jsonl"
                if dataset_file.exists():
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if 'text' in data and len(data['text']) > 200:
                                    mixed_dataset.append({
                                        "text": data['text'],
                                        "source": dataset_name,
                                        "quality": self.dataset_sources[dataset_name]["quality"]
                                    })
                            except json.JSONDecodeError:
                                continue
        
        # Shuffle and save mixed dataset
        import random
        random.shuffle(mixed_dataset)
        
        output_file = self.data_dir / "enterprise_mixed_dataset.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in mixed_dataset:
                json.dump(item, f)
                f.write('\n')
        
        # Create summary
        summary = {
            "total_samples": len(mixed_dataset),
            "estimated_size_gb": len(mixed_dataset) * 2000 / (1024**3),  # Rough estimate
            "dataset_composition": {name: ratio for name, ratio in dataset_mix},
            "creation_date": datetime.now().isoformat(),
            "quality_distribution": {}
        }
        
        # Calculate quality distribution
        for item in mixed_dataset:
            quality = item["quality"]
            summary["quality_distribution"][quality] = summary["quality_distribution"].get(quality, 0) + 1
        
        with open(self.data_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"🎉 Mixed dataset created: {len(mixed_dataset):,} samples")
        self.logger.info(f"📁 Saved to: {output_file}")
        
        return str(output_file)
    
    def optimize_for_4_9b_model(self) -> Dict[str, Any]:
        """Create optimized dataset configuration for 4.9B parameter model"""
        
        self.logger.info("⚙️ Optimizing dataset for 4.9B parameter model training")
        
        # Professional optimization settings
        config = {
            "model_architecture": {
                "parameters": "4.9B",
                "context_length": 2048,
                "vocabulary_size": 50257,  # GPT-3 vocabulary
                "architecture": "transformer_decoder"
            },
            "training_data": {
                "total_tokens": 300_000_000_000,  # 300B tokens (similar to GPT-3)
                "batch_size": 1,  # RTX 3050 optimization
                "gradient_accumulation_steps": 32,
                "sequence_length": 2048,
                "preprocessing": {
                    "tokenizer": "tiktoken_gpt3",
                    "filtering": {
                        "min_length": 200,
                        "max_length": 8192,
                        "remove_duplicates": True,
                        "quality_threshold": 0.7
                    }
                }
            },
            "optimization": {
                "hardware": "RTX_3050_8GB",
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "memory_efficient_attention": True,
                "compile_model": True
            },
            "data_quality": {
                "deduplication": True,
                "content_filtering": True,
                "language_detection": True,
                "toxic_content_removal": True,
                "privacy_filtering": True
            }
        }
        
        # Save configuration
        config_file = self.data_dir / "enterprise_training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"📋 Training configuration saved: {config_file}")
        
        return config

def main():
    print("🚀 Enterprise Dataset Manager for 4.9B Parameter Model")
    print("=" * 60)
    print("Creating training data similar to Claude, GPT, and Gemini...")
    print()
    
    # Initialize dataset manager
    manager = EnterpriseDatasetManager()
    
    # Install requirements
    manager.install_requirements()
    
    # Create enterprise-grade mixed dataset
    dataset_file = manager.create_mixed_dataset(target_size_gb=15.0)  # Reasonable for RTX 3050
    
    # Optimize for 4.9B model
    config = manager.optimize_for_4_9b_model()
    
    print("\n" + "=" * 60)
    print("🎉 ENTERPRISE DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"📁 Dataset file: {dataset_file}")
    print(f"📊 Total samples: {config.get('estimated_samples', 'Processing...')}")
    print(f"🎯 Optimized for: RTX 3050 with 4.9B parameter model")
    print(f"🔥 Training data quality: Enterprise-grade")
    print("\nReady for professional model training!")

if __name__ == "__main__":
    main()
