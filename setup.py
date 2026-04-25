#!/usr/bin/env python3
"""
Setup Script for MedExRAG - Medical Expert X-ray RAG Analysis System
Checks dependencies, downloads models, and initializes the system
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")


def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("✗ Python 3.11+ required")
        print("  Please upgrade Python")
        return False

    print("✓ Python version OK")
    return True


def check_cuda():
    """Check CUDA availability"""
    print_header("Checking CUDA")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠️  CUDA not available")
            print("  The pipeline will run on CPU (slower)")
            return True  # Not critical, can run on CPU

    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return True


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")

    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print(f"✗ Requirements file not found: {requirements_file}")
        return False

    print(f"Installing from: {requirements_file}")
    print("This may take several minutes...")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print("\n✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        return False


def download_model():
    """Download VLM model"""
    print_header("Downloading VLM Model")

    print("Downloading Qwen2-VL-2B-Instruct...")
    print("This is a large model (~4GB) and may take time.")

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_name = "Qwen/Qwen2-VL-2B-Instruct"

        print(f"Model: {model_name}")
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print("Downloading model weights...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Don't load to GPU during download
        )

        print("✓ Model downloaded successfully")
        return True

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("  You can download it later when running the pipeline")
        return True  # Non-critical, can download later


def download_embeddings():
    """Download medical domain embeddings"""
    print_header("Downloading Medical Embeddings")

    print("Downloading BioBERT/PubMedBERT for medical literature...")

    try:
        from sentence_transformers import SentenceTransformer

        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

        print(f"Model: {model_name}")
        model = SentenceTransformer(model_name)

        print("✓ Embeddings downloaded successfully")
        return True

    except Exception as e:
        print(f"✗ Error downloading embeddings: {e}")
        print("  You can download them later when running the pipeline")
        return True  # Non-critical


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")

    directories = [
        "./data/medical_kb",
        "./data/medical_literature",
        "./data/xray_images",
        "./data/analysis_results",
        "./metrics",
    ]

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}")
        else:
            print(f"  Exists: {directory}")

    return True


def test_imports():
    """Test critical imports"""
    print_header("Testing Imports")

    critical_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("docling", "DocLing"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("PIL", "Pillow"),
        ("pydicom", "PyDICOM"),
    ]

    all_ok = True

    for module_name, display_name in critical_imports:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - Not installed")
            all_ok = False

    if all_ok:
        print("\n✓ All critical imports OK")
    else:
        print("\n⚠️  Some imports failed. Run: pip install -r requirements.txt")

    return all_ok


def create_env_file():
    """Create example .env file"""
    print_header("Creating Environment File")

    env_file = Path(".env")

    if env_file.exists():
        print("  .env file already exists")
        return True

    env_content = """# MedExRAG Configuration

# Knowledge Base
MEDICAL_KB_PATH=./data/medical_kb

# Literature Directory (for auto-ingestion)
LITERATURE_DIR=./data/medical_literature

# Model Settings
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
EMBEDDING_MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

# Retrieval Settings
RAG_K_SOURCES=5
RAG_SCORE_THRESHOLD=0.6

# Logging
LOG_LEVEL=INFO
"""

    with open(env_file, 'w') as f:
        f.write(env_content)

    print(f"✓ Created: {env_file}")
    print("  Edit this file to customize settings")

    return True


def print_next_steps():
    """Print next steps for user"""
    print_header("Setup Complete!")

    print("Next steps:")
    print()
    print("1. Add medical literature PDFs:")
    print("   - Place PDF files in ./data/medical_literature/")
    print("   - Good sources: ACR guidelines, research papers, textbooks")
    print()
    print("2. Ingest the literature:")
    print("   PYTHONPATH=src python -m medexrag.cli ingest ./data/medical_literature")
    print()
    print("3. Analyze an X-ray:")
    print("   PYTHONPATH=src python -m medexrag.cli analyze chest_xray.dcm --rag")
    print()
    print("4. Try the examples:")
    print("   PYTHONPATH=src python examples/example_usage.py")
    print()
    print("5. Run the web UI:")
    print("   PYTHONPATH=src streamlit run src/medexrag/app.py")
    print()
    print("Documentation:")
    print("  - Architecture:   docs/architecture/RAG_ARCHITECTURE.md")
    print("  - RAG Guide:      docs/guides/RAG_GUIDE.md")
    print("  - VLM Guide:      docs/guides/VLM_GUIDE.md")
    print("  - Docker Setup:   docs/deployment/DOCKER_SETUP.md")
    print("  - Troubleshoot:   docs/TROUBLESHOOTING.md")


def main():
    """Run setup"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║   MedExRAG - Medical Expert X-ray RAG Analysis System         ║
║   Qwen2-VL + DocLing + LangChain + LangGraph                 ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    response = input("Install dependencies from requirements.txt? (y/n): ")
    if response.lower() == 'y':
        if not install_dependencies():
            print("\n⚠️  Dependency installation failed")
            print("You can try manually: pip install -r requirements.txt")

    # Check CUDA
    check_cuda()

    # Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed. Dependencies may not be installed correctly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Download models
    response = input("\nDownload VLM model now? (~4GB, may take time) (y/n): ")
    if response.lower() == 'y':
        download_model()

    response = input("\nDownload medical embeddings now? (~400MB) (y/n): ")
    if response.lower() == 'y':
        download_embeddings()

    # Create directories
    create_directories()

    # Create .env file
    create_env_file()

    # Print next steps
    print_next_steps()

    print("\n" + "="*60)
    print("Setup script complete!")
    print("="*60)


if __name__ == "__main__":
    main()
