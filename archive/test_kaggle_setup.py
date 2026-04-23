"""
Test script to verify Kaggle downloader functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.kaggle_downloader import KaggleDownloader


def test_kaggle_downloader():
    """Test Kaggle downloader functionality."""
    print("="*60)
    print("TESTING KAGGLE DOWNLOADER")
    print("="*60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create downloader instance
        downloader = KaggleDownloader(temp_dir)
        
        # Test 1: Check Kaggle CLI
        print("\n1. Testing Kaggle CLI check...")
        cli_available = downloader.check_kaggle_cli()
        print(f"   Kaggle CLI available: {cli_available}")
        
        if not cli_available:
            print("   Note: Kaggle CLI will be installed during setup")
        
        # Test 2: Check API key
        print("\n2. Testing API key verification...")
        api_key_valid = downloader.verify_api_key()
        print(f"   API key valid: {api_key_valid}")
        
        if not api_key_valid:
            print("   Note: API key instructions will be shown during setup")
        
        # Test 3: Check directory structure
        print("\n3. Testing directory structure...")
        datasets_dir = Path(temp_dir) / "datasets"
        print(f"   Datasets directory: {datasets_dir}")
        print(f"   Datasets directory exists: {datasets_dir.exists()}")
        
        # Test 4: Check project structure
        print("\n4. Testing project structure...")
        project_root = Path(temp_dir)
        print(f"   Project root: {project_root}")
        print(f"   Project root exists: {project_root.exists()}")
        
        print("\n" + "="*60)
        print("KAGGLE DOWNLOADER TEST COMPLETED")
        print("="*60)
        
        if cli_available and api_key_valid:
            print("✅ Ready to download datasets!")
            print("   Run: python setup_dataset.py")
        else:
            print("⚠️  Setup required:")
            if not cli_available:
                print("   - Kaggle CLI will be installed automatically")
            if not api_key_valid:
                print("   - Please configure Kaggle API key")
                print("   - Instructions will be shown during setup")
        
        print("\nNext steps:")
        print("1. Configure Kaggle API key if needed")
        print("2. Run: python setup_dataset.py")
        print("3. Verify dataset structure: python setup_dataset.py --verify")


def test_dataset_setup_manager():
    """Test dataset setup manager."""
    print("\n" + "="*60)
    print("TESTING DATASET SETUP MANAGER")
    print("="*60)
    
    try:
        from setup_dataset import DatasetSetupManager
        
        # Create manager instance
        manager = DatasetSetupManager()
        
        print("✅ DatasetSetupManager imported successfully")
        
        # Test project structure
        print(f"   Project root: {manager.project_root}")
        print(f"   Datasets directory: {manager.project_root / 'datasets'}")
        
        # Test dataset info
        print("\nTesting dataset info display...")
        manager.show_dataset_info()
        
        print("\n✅ DatasetSetupManager test completed")
        
    except ImportError as e:
        print(f"❌ Failed to import DatasetSetupManager: {e}")
    except Exception as e:
        print(f"❌ DatasetSetupManager test failed: {e}")


def main():
    """Main test function."""
    print("KAGGLE DATASET SETUP TESTS")
    print("="*60)
    
    try:
        test_kaggle_downloader()
        test_dataset_setup_manager()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        print("\nTo download datasets, run:")
        print("  python setup_dataset.py")
        print("\nFor help:")
        print("  python setup_dataset.py --help")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
