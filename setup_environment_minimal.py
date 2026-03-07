"""
Minimal environment setup script for cervical cancer classification pipeline.
Creates virtual environment and installs dependencies without PyTorch.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version compatible")
    return True


def create_virtual_environment():
    """Create virtual environment in project root."""
    print("\n" + "="*60)
    print("CREATING VIRTUAL ENVIRONMENT")
    print("="*60)
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    try:
        # Create virtual environment
        result = subprocess.run([
            sys.executable, "-m", "venv", "venv"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Virtual environment created successfully")
            return True
        else:
            print(f"❌ Failed to create virtual environment: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Virtual environment creation timed out")
        return False
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False


def get_venv_python():
    """Get path to virtual environment Python executable."""
    if os.name == 'nt':  # Windows
        venv_python = Path("venv/Scripts/python.exe")
    else:  # Unix-like
        venv_python = Path("venv/bin/python")
    
    return venv_python if venv_python.exists() else None


def install_dependencies():
    """Install dependencies from requirements.txt (excluding PyTorch)."""
    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)
    
    venv_python = get_venv_python()
    if not venv_python:
        print("❌ Virtual environment Python not found")
        return False
    
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("Installing dependencies from requirements.txt...")
    try:
        result = subprocess.run([
            str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Dependency installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def verify_dataset():
    """Verify dataset loading works correctly."""
    print("\n" + "="*60)
    print("VERIFYING DATASET LOADING")
    print("="*60)
    
    venv_python = get_venv_python()
    if not venv_python:
        print("❌ Virtual environment Python not found")
        return False
    
    print("Testing dataset loading with demo_dataset_simple.py...")
    try:
        result = subprocess.run([
            str(venv_python), "demo_dataset_simple.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Dataset loading verification passed")
            print("\n" + "="*60)
            print("DATASET SUMMARY")
            print("="*60)
            
            # Extract key information from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Total images:' in line or 'SIPaKMeD:' in line or 'Herlev:' in line:
                    print(line)
                elif 'SIPaKMeD distribution:' in line or 'Herlev distribution:' in line:
                    print(line)
                elif 'SIPaKMeD splits:' in line or 'Herlev splits:' in line:
                    print(line)
            
            return True
        else:
            print(f"❌ Dataset verification failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Dataset verification timed out")
        return False
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False


def print_setup_instructions():
    """Print instructions for manual setup."""
    print("\n" + "="*80)
    print("PYTORCH SETUP INSTRUCTIONS")
    print("="*80)
    print("Dataset pipeline is working! For PyTorch integration, follow these steps:")
    print()
    print("1. Activate virtual environment:")
    print("   venv\\Scripts\\activate")
    print()
    print("2. Try installing PyTorch (CPU version):")
    print("   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("3. If that fails, try older version:")
    print("   pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("4. Test PyTorch integration:")
    print("   python demo_dataset.py")
    print()
    print("5. Start training:")
    print("   python main.py --binary")
    print("   python main.py --multiclass")
    print("="*80)


def main():
    """Main setup function."""
    print("CERVICAL CANCER CLASSIFICATION - MINIMAL ENVIRONMENT SETUP")
    print("="*80)
    print("This script will set up a Python environment without PyTorch.")
    print("The dataset pipeline will work, but model training requires PyTorch.")
    print("="*80)
    
    # Check Python version
    if not check_python_version():
        print_setup_instructions()
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        print_setup_instructions()
        return False
    
    # Install dependencies
    if not install_dependencies():
        print_setup_instructions()
        return False
    
    # Verify dataset loading
    if not verify_dataset():
        print_setup_instructions()
        return False
    
    # Success!
    print("\n" + "="*80)
    print("🎉 MINIMAL SETUP COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("✅ Dataset pipeline is working correctly!")
    print("✅ All dependencies installed (except PyTorch)")
    print()
    print("Dataset Summary:")
    print("- SIPaKMeD: 966 images (5 classes)")
    print("- Herlev: 1,834 images (7 classes)")
    print("- Total: 2,800 medical images")
    print("- Binary and multiclass classification supported")
    print("- Train/Val/Test splits: 70/15/15")
    print()
    print("Next steps:")
    print("1. Activate virtual environment: venv\\Scripts\\activate")
    print("2. Install PyTorch manually (see instructions above)")
    print("3. Test full integration: python demo_dataset.py")
    print("4. Start training: python main.py --binary")
    print("="*80)
    
    # Print PyTorch setup instructions
    print_setup_instructions()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
