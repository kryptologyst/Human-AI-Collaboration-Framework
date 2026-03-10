#!/usr/bin/env python3
"""Setup script for Human-AI Collaboration Framework."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def setup_pre_commit():
    """Set up pre-commit hooks."""
    commands = [
        ("pip install pre-commit", "Installing pre-commit"),
        ("pre-commit install", "Setting up pre-commit hooks"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️ {description} failed, but continuing...")
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "logs",
        "results",
        "assets",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True


def run_tests():
    """Run basic tests to verify installation."""
    if not run_command("python -c \"import torch; print('PyTorch version:', torch.__version__)\"", "Testing PyTorch"):
        return False
    
    if not run_command("python -c \"import transformers; print('Transformers version:', transformers.__version__)\"", "Testing Transformers"):
        return False
    
    if not run_command("python -c \"import streamlit; print('Streamlit version:', streamlit.__version__)\"", "Testing Streamlit"):
        return False
    
    # Test basic functionality
    test_script = """
import sys
sys.path.append('src')
from human_ai_collaboration import set_seed, get_device, SyntheticDataGenerator

# Test basic functionality
set_seed(42)
device = get_device()
print(f'Device: {device}')

generator = SyntheticDataGenerator()
texts, labels, names = generator.generate_text_classification_data(n_samples=5, n_classes=3)
print(f'Generated {len(texts)} samples with {len(names)} classes')
print('✅ Basic functionality test passed')
"""
    
    if not run_command(f"python -c \"{test_script}\"", "Testing basic functionality"):
        return False
    
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Human-AI Collaboration Framework")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Set up pre-commit (optional)
    print("\n🔧 Setting up development tools...")
    setup_pre_commit()
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_tests():
        print("❌ Tests failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: streamlit run demo/app.py")
    print("2. Train a model: python scripts/train.py --config configs/default.yaml")
    print("3. Evaluate a model: python scripts/evaluate.py --model_path checkpoints/model.pth")
    print("\nFor more information, see README.md")
    print("=" * 50)


if __name__ == "__main__":
    main()
