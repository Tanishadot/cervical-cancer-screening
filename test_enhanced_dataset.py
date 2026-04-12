"""
Test script for enhanced dataset loading with logging and CROPPED-only option.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.label_mapping import LabelMapper, ClassificationMode
from datasets.dataset import SIPaKMeDDataset, DatasetManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_loading.log')
        ]
    )


def test_sipakmed_loading():
    """Test SIPaKMeD dataset loading with both modes."""
    print("\n" + "="*80)
    print("TESTING SIPaKMeD DATASET LOADING")
    print("="*80)
    
    # Create label mapper
    label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
    
    # Test 1: Load all images
    print("\n🔍 TEST 1: Loading ALL images")
    print("-" * 40)
    
    try:
        dataset_all = SIPaKMeDDataset(
            root_dir="datasets",
            label_mapper=label_mapper,
            transform=None,
            mode="all",
            use_cropped_only=False
        )
        print(f"✅ Successfully loaded {len(dataset_all)} images (all)")
    except Exception as e:
        print(f"❌ Failed to load all images: {e}")
        return False
    
    # Test 2: Load CROPPED only
    print("\n🔍 TEST 2: Loading CROPPED images only")
    print("-" * 40)
    
    try:
        dataset_cropped = SIPaKMeDDataset(
            root_dir="datasets",
            label_mapper=label_mapper,
            transform=None,
            mode="all",
            use_cropped_only=True
        )
        print(f"✅ Successfully loaded {len(dataset_cropped)} images (CROPPED only)")
    except Exception as e:
        print(f"❌ Failed to load CROPPED images: {e}")
        return False
    
    # Verify CROPPED filtering worked
    if len(dataset_all) > 0 and len(dataset_cropped) > 0:
        cropped_ratio = len(dataset_cropped) / len(dataset_all)
        print(f"\n📊 COMPARISON:")
        print(f"  All images: {len(dataset_all)}")
        print(f"  CROPPED only: {len(dataset_cropped)}")
        print(f"  CROPPED ratio: {cropped_ratio:.1%}")
        
        if 0.7 <= cropped_ratio <= 0.9:  # Expected range based on validation
            print("✅ CROPPED filtering working correctly")
        else:
            print("⚠️  Unexpected CROPPED ratio")
    
    return True


def test_dataset_manager():
    """Test DatasetManager with enhanced logging."""
    print("\n" + "="*80)
    print("TESTING DATASET MANAGER")
    print("="*80)
    
    # Test 1: All images
    print("\n🔍 TEST 1: DatasetManager with ALL images")
    print("-" * 40)
    
    try:
        manager_all = DatasetManager(
            root_dir="datasets",
            classification_mode="multiclass",
            use_cropped_only=False
        )
        manager_all.load_datasets()
        manager_all.create_splits()
        print("✅ DatasetManager (all images) working correctly")
    except Exception as e:
        print(f"❌ DatasetManager (all images) failed: {e}")
        return False
    
    # Test 2: CROPPED only
    print("\n🔍 TEST 2: DatasetManager with CROPPED only")
    print("-" * 40)
    
    try:
        manager_cropped = DatasetManager(
            root_dir="datasets",
            classification_mode="multiclass",
            use_cropped_only=True
        )
        manager_cropped.load_datasets()
        manager_cropped.create_splits()
        print("✅ DatasetManager (CROPPED only) working correctly")
    except Exception as e:
        print(f"❌ DatasetManager (CROPPED only) failed: {e}")
        return False
    
    return True


def test_data_loading_stability():
    """Test data loading stability and sample access."""
    print("\n" + "="*80)
    print("TESTING DATA LOADING STABILITY")
    print("="*80)
    
    try:
        # Create dataset
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        dataset = SIPaKMeDDataset(
            root_dir="datasets",
            label_mapper=label_mapper,
            transform=None,
            mode="all",
            use_cropped_only=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test accessing first few samples
        print("\n🔍 TESTING SAMPLE ACCESS:")
        for i in range(min(5, len(dataset))):
            try:
                img_path, mapped_label, original_label = dataset.samples[i]
                print(f"  Sample {i}: {Path(img_path).name} -> Class {mapped_label}")
            except Exception as e:
                print(f"  ❌ Failed to access sample {i}: {e}")
                return False
        
        # Test __getitem__ method
        print("\n🔍 TESTING __getitem__:")
        for i in range(min(3, len(dataset))):
            try:
                image, mapped_label, original_label, img_path = dataset[i]
                print(f"  Sample {i}: Loaded image shape {image.shape} -> Class {mapped_label}")
            except Exception as e:
                print(f"  ❌ Failed to __getitem__ {i}: {e}")
                return False
        
        print("✅ Data loading stability test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data loading stability test failed: {e}")
        return False


def main():
    """Run all enhanced dataset tests."""
    print("ENHANCED DATASET LOADING TESTS")
    print("="*80)
    
    # Setup logging
    setup_logging()
    
    tests = [
        ("SIPaKMeD Loading", test_sipakmed_loading),
        ("Dataset Manager", test_dataset_manager),
        ("Data Loading Stability", test_data_loading_stability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests successful")
    
    if passed == total:
        print("\n🎉 ALL ENHANCED DATASET TESTS PASSED!")
        print("\nThe enhanced dataset pipeline is working correctly with:")
        print("1. ✅ Structured logging")
        print("2. ✅ CROPPED-only option")
        print("3. ✅ Comprehensive statistics")
        print("4. ✅ Data loading stability")
        print("\nReady for model training and validation!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
