"""
Environment setup script for cervical cancer classification pipeline.
Creates virtual environment and installs compatible PyTorch and dependencies.
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
    """Install dependencies from requirements.txt."""
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


def install_pytorch():
    """Install CPU-compatible PyTorch."""
    print("\n" + "="*60)
    print("INSTALLING PYTORCH (CPU VERSION)")
    print("="*60)
    
    venv_python = get_venv_python()
    if not venv_python:
        print("❌ Virtual environment Python not found")
        return False
    
    print("Installing CPU-compatible PyTorch...")
    try:
        # Try installing PyTorch 2.1.0 with CPU (known to work with Windows)
        result = subprocess.run([
            str(venv_python), "-m", "pip", "install", 
            "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ PyTorch installed successfully")
            return True
        else:
            print(f"❌ Failed to install PyTorch: {result.stderr}")
            # Try fallback to older version
            print("Trying fallback to PyTorch 2.0.1...")
            result2 = subprocess.run([
                str(venv_python), "-m", "pip", "install", 
                "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], capture_output=True, text=True, timeout=600)
            
            if result2.returncode == 0:
                print("✅ PyTorch 2.0.1 installed successfully (fallback)")
                return True
            else:
                print(f"❌ Fallback also failed: {result2.stderr}")
                return False
            
    except subprocess.TimeoutExpired:
        print("❌ PyTorch installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing PyTorch: {e}")
        return False


def test_pytorch_import():
    """Test if PyTorch can be imported successfully."""
    print("\n" + "="*60)
    print("TESTING PYTORCH IMPORT")
    print("="*60)
    
    venv_python = get_venv_python()
    if not venv_python:
        print("❌ Virtual environment Python not found")
        return False
    
    print("Testing PyTorch import...")
    try:
        # Test import in virtual environment
        result = subprocess.run([
            str(venv_python), "-c", 
            "import torch; print(f'PyTorch version: {torch.__version__}'); "
            "import torchvision; print(f'TorchVision version: {torchvision.__version__}'); "
            "print('✅ PyTorch imports successfully')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ PyTorch import failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ PyTorch import test timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing PyTorch import: {e}")
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
    
    # First try PyTorch-free demo
    print("Testing dataset loading with demo_dataset_simple.py...")
    try:
        result = subprocess.run([
            str(venv_python), "demo_dataset_simple.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Dataset loading verification passed (PyTorch-free)")
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
            
            # Try PyTorch demo if possible
            print("\nTesting PyTorch integration...")
            try:
                result2 = subprocess.run([
                    str(venv_python), "demo_dataset.py"
                ], capture_output=True, text=True, timeout=120)
                
                if result2.returncode == 0:
                    print("✅ PyTorch integration also working")
                else:
                    print("⚠️  PyTorch integration not working, but PyTorch-free version works")
            except:
                print("⚠️  PyTorch integration test failed, but PyTorch-free version works")
            
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
    print("MANUAL SETUP INSTRUCTIONS")
    print("="*80)
    print("If automatic setup fails, follow these steps:")
    print()
    print("1. Create virtual environment:")
    print("   python -m venv venv")
    print()
    print("2. Activate virtual environment:")
    print("   venv\\Scripts\\activate")
    print()
    print("3. Install PyTorch (CPU version):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("4. Install other dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("5. Test the setup:")
    print("   python demo_dataset.py")
    print()
    print("6. Start training:")
    print("   python main.py --binary")
    print("   python main.py --multiclass")
    print("="*80)


def main():
    """Main setup function."""
    print("CERVICAL CANCER CLASSIFICATION - ENVIRONMENT SETUP")
    print("="*80)
    print("This script will set up a proper Python environment for the project.")
    print("="*80)
    
    # Check Python version
    if not check_python_version():
        print_setup_instructions()
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        print_setup_instructions()
        return False
    
    # Install PyTorch first (most critical)
    if not install_pytorch():
        print_setup_instructions()
        return False
    
    # Test PyTorch import
    if not test_pytorch_import():
        print_setup_instructions()
        return False
    
    # Install other dependencies
    if not install_dependencies():
        print_setup_instructions()
        return False
    
    # Verify dataset loading
    if not verify_dataset():
        print_setup_instructions()
        return False
    
    # Success!
    print("\n" + "="*80)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Your environment is ready for cervical cancer classification!")
    print()
    print("Next steps:")
    print("1. Activate virtual environment: venv\\Scripts\\activate")
    print("2. Run demo: python demo_dataset.py")
    print("3. Start training: python main.py --binary")
    print()
    print("Available models:")
    print("- EfficientNet-B0")
    print("- ResNet50")
    print("- Swin Transformer")
    print()
    print("Classification modes:")
    print("--binary: Normal vs Abnormal")
    print("--multiclass: Original class labels")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
