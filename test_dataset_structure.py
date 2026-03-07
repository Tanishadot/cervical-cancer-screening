"""
Simple test script to verify dataset structure without PyTorch dependencies.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.label_mapping import LabelMapper, DatasetType, ClassificationMode


def test_dataset_structure():
    """Test that datasets are in the expected structure."""
    print("="*80)
    print("TESTING DATASET STRUCTURE")
    print("="*80)
    
    datasets_root = Path("datasets")
    
    if not datasets_root.exists():
        print("❌ Datasets directory not found")
        return False
    
    # Check for archive folders
    sipakmed_archive = datasets_root / "sipakmed" / "archive"
    herlev_archive = datasets_root / "herlev" / "archive (1)"
    
    print(f"Datasets directory: {datasets_root}")
    print(f"SIPaKMeD archive exists: {sipakmed_archive.exists()}")
    print(f"Herlev archive exists: {herlev_archive.exists()}")
    
    # Expected SIPaKMeD folders
    expected_sipakmed = [
        "im_Superficial-Intermediate",
        "im_Parabasal",
        "im_Koilocytotic",
        "im_Metaplastic",
        "im_Dyskeratotic"
    ]
    
    # Expected Herlev folders
    expected_herlev = [
        "normal_superficiel",
        "normal_intermediate",
        "normal_columnar",
        "light_dysplastic",
        "moderate_dysplastic",
        "severe_dysplastic",
        "carcinoma_in_situ"
    ]
    
    results = {}
    
    # Test SIPaKMeD
    if sipakmed_archive.exists():
        print(f"\n--- SIPaKMeD Archive Structure ---")
        sipakmed_results = {}
        
        for folder in expected_sipakmed:
            folder_path = sipakmed_archive / folder
            exists = folder_path.exists()
            sipakmed_results[folder] = exists
            
            if exists:
                # Check for nested structure
                nested_folder = folder_path / folder
                if nested_folder.exists():
                    image_count = len([f for f in nested_folder.iterdir() 
                                     if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
                    print(f"  {folder}: ✅ {image_count} images (nested)")
                    sipakmed_results[f"{folder}_count"] = image_count
                else:
                    image_count = len([f for f in folder_path.iterdir() 
                                     if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
                    print(f"  {folder}: ✅ {image_count} images")
                    sipakmed_results[f"{folder}_count"] = image_count
            else:
                print(f"  {folder}: ❌ Not found")
        
        results["sipakmed"] = sipakmed_results
    
    # Test Herlev
    if herlev_archive.exists():
        print(f"\n--- Herlev Archive Structure ---")
        herlev_results = {}
        
        for folder in expected_herlev:
            folder_path = herlev_archive / folder
            exists = folder_path.exists()
            herlev_results[folder] = exists
            
            if exists:
                image_count = len([f for f in folder_path.iterdir() 
                                 if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
                print(f"  {folder}: ✅ {image_count} images")
                herlev_results[f"{folder}_count"] = image_count
            else:
                print(f"  {folder}: ❌ Not found")
        
        results["herlev"] = herlev_results
    
    return results


def test_label_mapping():
    """Test label mapping functionality."""
    print("\n" + "="*80)
    print("TESTING LABEL MAPPING")
    print("="*80)
    
    try:
        # Test binary classification
        print("\n--- Binary Classification ---")
        binary_mapper = LabelMapper(ClassificationMode.BINARY)
        
        print(f"Number of classes: {binary_mapper.get_num_classes()}")
        print(f"Class names: {binary_mapper.get_class_names()}")
        
        # Test SIPaKMeD mapping
        sipakmed_classes = binary_mapper.get_dataset_classes(DatasetType.SIPAKMED)
        print(f"SIPaKMeD classes: {sipakmed_classes}")
        
        for class_id, class_name in sipakmed_classes.items():
            mapped_label = binary_mapper.map_label(class_id, DatasetType.SIPAKMED)
            print(f"  {class_name} ({class_id}) -> {mapped_label}")
        
        # Test Herlev mapping
        herlev_classes = binary_mapper.get_dataset_classes(DatasetType.HERLEV)
        print(f"Herlev classes: {herlev_classes}")
        
        for class_id, class_name in herlev_classes.items():
            mapped_label = binary_mapper.map_label(class_id, DatasetType.HERLEV)
            print(f"  {class_name} ({class_id}) -> {mapped_label}")
        
        print("✅ Binary label mapping working")
        
        # Test multiclass classification
        print("\n--- Multiclass Classification ---")
        multiclass_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        
        print(f"SIPaKMeD classes: {multiclass_mapper.get_num_classes(DatasetType.SIPAKMED)}")
        print(f"Herlev classes: {multiclass_mapper.get_num_classes(DatasetType.HERLEV)}")
        print(f"SIPaKMeD class names: {multiclass_mapper.get_class_names(DatasetType.SIPAKMED)}")
        print(f"Herlev class names: {multiclass_mapper.get_class_names(DatasetType.HERLEV)}")
        
        print("✅ Multiclass label mapping working")
        
        return True
        
    except Exception as e:
        print(f"❌ Label mapping test failed: {e}")
        return False


def test_dataset_loading_basic():
    """Test basic dataset loading without PyTorch."""
    print("\n" + "="*80)
    print("TESTING BASIC DATASET LOADING")
    print("="*80)
    
    try:
        # Test binary classification
        print("\n--- Binary Classification Loading ---")
        binary_mapper = LabelMapper(ClassificationMode.BINARY)
        
        # Simulate dataset loading logic
        datasets_root = Path("datasets")
        
        # Test SIPaKMeD
        sipakmed_dir = datasets_root / "sipakmed" / "archive"
        if sipakmed_dir.exists():
            sipakmed_classes = binary_mapper.get_dataset_classes(DatasetType.SIPAKMED)
            samples = []
            
            # Archive folder mapping
            archive_mapping = {
                0: "im_Superficial-Intermediate",
                1: "im_Parabasal",
                2: "im_Koilocytotic",
                3: "im_Metaplastic",
                4: "im_Dyskeratotic"
            }
            
            for class_id, class_name in sipakmed_classes.items():
                archive_folder = archive_mapping.get(class_id)
                if archive_folder:
                    class_dir = sipakmed_dir / archive_folder
                    if class_dir.exists():
                        # Check for nested structure
                        nested_dir = class_dir / archive_folder
                        if nested_dir.exists():
                            image_files = [f for f in nested_dir.iterdir() 
                                         if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']]
                        else:
                            image_files = [f for f in class_dir.iterdir() 
                                         if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']]
                        
                        for img_file in image_files:
                            mapped_label = binary_mapper.map_label(class_id, DatasetType.SIPAKMED)
                            samples.append((str(img_file), mapped_label, class_id))
            
            print(f"SIPaKMeD samples found: {len(samples)}")
            
            # Count distribution
            distribution = defaultdict(int)
            for _, label, _ in samples:
                distribution[label] += 1
            
            print(f"Class distribution: {dict(distribution)}")
            print("✅ SIPaKMeD loading simulation successful")
        
        # Test Herlev
        herlev_dir = datasets_root / "herlev" / "archive (1)"
        if herlev_dir.exists():
            herlev_classes = binary_mapper.get_dataset_classes(DatasetType.HERLEV)
            samples = []
            
            # Archive folder mapping
            archive_mapping = {
                0: "normal_superficiel",
                1: "normal_intermediate",
                2: "normal_columnar",
                3: "light_dysplastic",
                4: "moderate_dysplastic",
                5: "severe_dysplastic",
                6: "carcinoma_in_situ"
            }
            
            for class_id, class_name in herlev_classes.items():
                archive_folder = archive_mapping.get(class_id)
                if archive_folder:
                    class_dir = herlev_dir / archive_folder
                    if class_dir.exists():
                        image_files = [f for f in class_dir.iterdir() 
                                     if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']]
                        
                        for img_file in image_files:
                            mapped_label = binary_mapper.map_label(class_id, DatasetType.HERLEV)
                            samples.append((str(img_file), mapped_label, class_id))
            
            print(f"Herlev samples found: {len(samples)}")
            
            # Count distribution
            distribution = defaultdict(int)
            for _, label, _ in samples:
                distribution[label] += 1
            
            print(f"Class distribution: {dict(distribution)}")
            print("✅ Herlev loading simulation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic dataset loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("DATASET STRUCTURE AND LOGIC TEST")
    print("="*80)
    
    tests = [
        ("Dataset Structure", test_dataset_structure),
        ("Label Mapping", test_label_mapping),
        ("Basic Dataset Loading", test_dataset_loading_basic)
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
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 DATASET STRUCTURE TESTS PASSED!")
        print("\nNext steps:")
        print("1. Fix PyTorch installation if needed")
        print("2. Run: python test_dataset.py")
        print("3. Start training: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the dataset structure.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
