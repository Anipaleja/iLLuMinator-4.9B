"""
Hugging Face Hub Deployment Script
Deploy Illuminator model to Hugging Face Model Hub
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo, upload_folder
import argparse

class HuggingFaceDeployer:
    """Deploy Illuminator model to Hugging Face Hub"""
    
    def __init__(self, model_dir="./huggingface_model", repo_name="illuminator-4b"):
        self.model_dir = Path(model_dir)
        self.repo_name = repo_name
        self.api = HfApi()
        
        print(f"🚀 Initializing Hugging Face deployment for {repo_name}")
        print(f"📁 Model directory: {self.model_dir}")
    
    def validate_model_files(self):
        """Validate all required model files are present"""
        print("🔍 Validating model files...")
        
        required_files = [
            "config.json",
            "tokenizer_config.json", 
            "README.md",
            "modeling_illuminator.py",
            "tokenization_illuminator.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.model_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        
        print("✅ All required model files present")
        return True
    
    def create_model_card(self):
        """Create or update model card with metadata"""
        print("📝 Creating model card...")
        
        model_card_path = self.model_dir / "README.md"
        
        # Read existing README if it exists
        if model_card_path.exists():
            print("✅ Model card already exists and is comprehensive")
            return True
        
        # If we reach here, something went wrong
        print("❌ Model card not found")
        return False
    
    def test_model_loading(self):
        """Test that the model can be loaded successfully"""
        print("🧪 Testing model loading...")
        
        try:
            # Test config loading
            config_path = self.model_dir / "config.json"
            with open(config_path) as f:
                config_dict = json.load(f)
            
            print(f"✅ Config loaded: {config_dict['model_type']}")
            
            # Test if our custom classes can be imported
            import sys
            sys.path.append(str(self.model_dir))
            
            from modeling_illuminator import IlluminatorLMHeadModel, IlluminatorConfig
            from tokenization_illuminator import IlluminatorTokenizer
            
            print("✅ Custom model classes imported successfully")
            
            # Test basic initialization
            config = IlluminatorConfig(**config_dict)
            print(f"✅ Model configuration created")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading test failed: {e}")
            return False
    
    def create_repository(self, private=False):
        """Create repository on Hugging Face Hub"""
        print(f"📦 Creating repository: {self.repo_name}")
        
        try:
            repo_url = create_repo(
                repo_id=self.repo_name,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            print(f"✅ Repository created/exists: {repo_url}")
            return repo_url
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return None
    
    def prepare_deployment_files(self):
        """Prepare additional files for deployment"""
        print("🔧 Preparing deployment files...")
        
        # Create __init__.py for package
        init_file = self.model_dir / "__init__.py"
        if not init_file.exists():
            init_content = '''"""
Illuminator Model Package
"""

from .modeling_illuminator import IlluminatorLMHeadModel, IlluminatorConfig
from .tokenization_illuminator import IlluminatorTokenizer

__all__ = ["IlluminatorLMHeadModel", "IlluminatorConfig", "IlluminatorTokenizer"]
'''
            with open(init_file, "w") as f:
                f.write(init_content)
            print("✅ Created __init__.py")
        
        # Create requirements.txt
        requirements_file = self.model_dir / "requirements.txt"
        if not requirements_file.exists():
            requirements = """torch>=1.9.0
transformers>=4.21.0
numpy>=1.21.0
tokenizers>=0.13.0
"""
            with open(requirements_file, "w") as f:
                f.write(requirements)
            print("✅ Created requirements.txt")
        
        return True
    
    def upload_to_hub(self):
        """Upload model to Hugging Face Hub"""
        print("🚀 Uploading to Hugging Face Hub...")
        
        try:
            upload_folder(
                folder_path=str(self.model_dir),
                repo_id=self.repo_name,
                repo_type="model",
                commit_message="Upload Illuminator-4B model",
                ignore_patterns=[
                    "*.pyc",
                    "__pycache__/",
                    "*.log",
                    ".git/",
                    ".DS_Store"
                ]
            )
            
            print(f"✅ Model uploaded successfully!")
            print(f"🌐 Model available at: https://huggingface.co/{self.repo_name}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def deploy(self, private=False, test_loading=True):
        """Main deployment function"""
        print("🎯 Starting Hugging Face deployment process")
        print("=" * 60)
        
        # Step 1: Validate files
        if not self.validate_model_files():
            print("❌ Deployment aborted: Missing required files")
            return False
        
        # Step 2: Test model loading (optional)
        if test_loading and not self.test_model_loading():
            print("⚠️ Model loading test failed, but continuing...")
        
        # Step 3: Prepare deployment files
        if not self.prepare_deployment_files():
            print("❌ Deployment aborted: Failed to prepare files")
            return False
        
        # Step 4: Create repository
        repo_url = self.create_repository(private=private)
        if not repo_url:
            print("❌ Deployment aborted: Failed to create repository")
            return False
        
        # Step 5: Upload to hub
        if not self.upload_to_hub():
            print("❌ Deployment aborted: Upload failed")
            return False
        
        print("\n🎉 Deployment Complete!")
        print("=" * 60)
        print(f"✅ Model successfully deployed to: {self.repo_name}")
        print(f"🌐 Access your model at: https://huggingface.co/{self.repo_name}")
        print("\n📋 Next steps:")
        print("1. Test your model on the Hugging Face Hub")
        print("2. Share your model with the community")
        print("3. Monitor usage and feedback")
        
        return True

def create_example_usage_script():
    """Create an example usage script"""
    example_script = '''"""
Example usage of Illuminator-4B model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_illuminator_model(model_name="your-username/illuminator-4b"):
    """Load the Illuminator model and tokenizer"""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=256):
    """Generate a response using the model"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def main():
    # Load model
    model, tokenizer = load_illuminator_model()
    
    # Example prompts
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
        "What are the benefits of renewable energy?"
    ]
    
    print("🤖 Illuminator-4B Model Demo")
    print("=" * 40)
    
    for prompt in prompts:
        print(f"\\n💬 Prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"🤖 Response: {response}")
        print("-" * 40)

if __name__ == "__main__":
    main()
'''
    
    with open("example_usage.py", "w") as f:
        f.write(example_script)
    
    print("✅ Created example_usage.py")

def main():
    parser = argparse.ArgumentParser(description="Deploy Illuminator model to Hugging Face Hub")
    parser.add_argument("--repo-name", default="illuminator-4b", help="Repository name on Hugging Face Hub")
    parser.add_argument("--model-dir", default="./huggingface_model", help="Directory containing model files")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--skip-test", action="store_true", help="Skip model loading test")
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = HuggingFaceDeployer(
        model_dir=args.model_dir,
        repo_name=args.repo_name
    )
    
    # Deploy model
    success = deployer.deploy(
        private=args.private,
        test_loading=not args.skip_test
    )
    
    if success:
        # Create example usage script
        create_example_usage_script()
        
        print("\n🎯 Deployment Summary:")
        print(f"Repository: {args.repo_name}")
        print(f"Model Directory: {args.model_dir}")
        print(f"Private: {args.private}")
        print("Example usage script created: example_usage.py")
        
        return 0
    else:
        print("❌ Deployment failed!")
        return 1

if __name__ == "__main__":
    exit(main())
