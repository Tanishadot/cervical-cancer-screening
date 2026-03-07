"""
Diagnostic script to find Kaggle CLI installation on Windows.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def find_kaggle_executable():
    """Find Kaggle executable on Windows."""
    print("Searching for Kaggle CLI...")
    
    # Try PATH first
    kaggle_cmd = shutil.which("kaggle")
    if kaggle_cmd:
        print(f"✅ Found in PATH: {kaggle_cmd}")
        return kaggle_cmd
    
    # Try common Python installation paths
    python_paths = [
        sys.prefix,
        os.path.join(sys.prefix, "Scripts"),
        os.path.join(sys.prefix, "bin"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Packages", "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0", "LocalCache", "local-packages", "Python311", "Scripts"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python311", "Scripts"),
        os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python311", "Scripts")
    ]
    
    print("Checking common Python installation paths...")
    for path in python_paths:
        kaggle_exe = os.path.join(path, "kaggle.exe")
        kaggle_py = os.path.join(path, "kaggle")
        
        if os.path.exists(kaggle_exe):
            print(f"✅ Found kaggle.exe: {kaggle_exe}")
            return kaggle_exe
        
        if os.path.exists(kaggle_py):
            print(f"✅ Found kaggle: {kaggle_py}")
            return kaggle_py
        
        print(f"   Checked: {path}")
    
    # Search in user's AppData more broadly
    print("Searching in AppData...")
    appdata_paths = [
        os.path.expanduser("~/AppData/Local"),
        os.path.expanduser("~/AppData/Roaming")
    ]
    
    for appdata in appdata_paths:
        try:
            for root, dirs, files in os.walk(appdata):
                for file in files:
                    if file.lower() in ["kaggle.exe", "kaggle"]:
                        full_path = os.path.join(root, file)
                        print(f"✅ Found: {full_path}")
                        return full_path
        except (PermissionError, OSError):
            continue
    
    print("❌ Kaggle CLI not found")
    return None

def test_kaggle_command(kaggle_path):
    """Test if Kaggle command works."""
    print(f"\nTesting Kaggle command: {kaggle_path}")
    
    try:
        result = subprocess.run(
            [kaggle_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✅ Kaggle CLI working: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Kaggle CLI failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing Kaggle: {e}")
        return False

def install_kaggle():
    """Install Kaggle CLI."""
    print("\nInstalling Kaggle CLI...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Kaggle CLI installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def main():
    """Main diagnostic function."""
    print("="*60)
    print("KAGGLE CLI DIAGNOSTIC TOOL")
    print("="*60)
    
    # Find Kaggle
    kaggle_path = find_kaggle_executable()
    
    if kaggle_path:
        # Test it
        if test_kaggle_command(kaggle_path):
            print(f"\n✅ SUCCESS: Use this command: {kaggle_path}")
            print("You can now run: python setup_dataset.py")
        else:
            print("\n❌ Kaggle found but not working properly")
            print("Try reinstalling:")
            install_kaggle()
    else:
        print("\n❌ Kaggle CLI not found")
        print("Installing Kaggle CLI...")
        if install_kaggle():
            print("\n✅ Installation completed. Running search again...")
            kaggle_path = find_kaggle_executable()
            if kaggle_path:
                test_kaggle_command(kaggle_path)
            else:
                print("❌ Still not found. You may need to restart your terminal or add to PATH manually.")
        else:
            print("❌ Installation failed. Please install manually:")
            print("pip install kaggle")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
