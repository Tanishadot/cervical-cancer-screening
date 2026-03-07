"""
Demonstration script showing dataset loading and management capabilities.
This script works without PyTorch to show the dataset structure and statistics.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset import DatasetManager
from utils.label_mapping import LabelMapper, DatasetType, ClassificationMode


def demonstrate_binary_classification():
    """Demonstrate binary classification dataset loading."""
    print("="*80)
    print("BINARY CLASSIFICATION DEMONSTRATION")
    print("="*80)
    
    try:
        # Create dataset manager
        manager = DatasetManager(
            root_dir="datasets",
            classification_mode="binary",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Load datasets
        manager.load_datasets()
        
        # Create splits
        manager.create_splits()
        
        # Print summary
        manager.print_summary()
        
        # Get detailed statistics
        summary = manager.get_dataset_summary()
        
        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)
        
        for dataset_name, info in summary["datasets"].items():
            print(f"\n{dataset_name.upper()} DATASET:")
            print(f"  Total samples: {info['total_samples']}")
            print(f"  Splits: Train={info['train_samples']}, Val={info['val_samples']}, Test={info['test_samples']}")
            
            print(f"  Binary distribution:")
            for class_id, count in info["class_distribution"].items():
                class_name = summary["class_names"][class_id]
                percentage = (count / info["total_samples"]) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Binary classification demo failed: {e}")
        return False


def demonstrate_multiclass_classification():
    """Demonstrate multiclass classification dataset loading."""
    print("\n" + "="*80)
    print("MULTICLASS CLASSIFICATION DEMONSTRATION")
    print("="*80)
    
    try:
        # Create dataset manager
        manager = DatasetManager(
            root_dir="datasets",
            classification_mode="multiclass",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Load datasets
        manager.load_datasets()
        
        # Create splits
        manager.create_splits()
        
        # Print summary
        manager.print_summary()
        
        # Get detailed statistics
        summary = manager.get_dataset_summary()
        
        print("\n" + "="*60)
        print("MULTICLASS STATISTICS")
        print("="*60)
        
        for dataset_name, info in summary["datasets"].items():
            print(f"\n{dataset_name.upper()} DATASET:")
            print(f"  Total samples: {info['total_samples']}")
            print(f"  Splits: Train={info['train_samples']}, Val={info['val_samples']}, Test={info['test_samples']}")
            
            print(f"  Multiclass distribution:")
            for class_id, count in info["class_distribution"].items():
                if dataset_name == "sipakmed":
                    class_names = summary["class_names"]  # SIPaKMeD names
                else:
                    # For Herlev in multiclass, we need to get Herlev names
                    herlev_mapper = LabelMapper(ClassificationMode.MULTICLASS)
                    class_names = herlev_mapper.get_class_names(DatasetType.HERLEV)
                
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                    percentage = (count / info["total_samples"]) * 100
                    print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiclass classification demo failed: {e}")
        return False


def demonstrate_cross_dataset_compatibility():
    """Demonstrate cross-dataset label mapping."""
    print("\n" + "="*80)
    print("CROSS-DATASET COMPATIBILITY DEMONSTRATION")
    print("="*80)
    
    try:
        # Create binary mapper
        binary_mapper = LabelMapper(ClassificationMode.BINARY)
        
        print("Binary Classification Label Mapping:")
        print("="*50)
        
        # SIPaKMeD mapping
        print("\nSIPaKMeD → Binary:")
        sipakmed_classes = binary_mapper.get_dataset_classes(DatasetType.SIPAKMED)
        for class_id, class_name in sipakmed_classes.items():
            mapped_label = binary_mapper.map_label(class_id, DatasetType.SIPAKMED)
            binary_label = "NORMAL" if mapped_label == 0 else "ABNORMAL"
            print(f"  {class_name} → {binary_label}")
        
        # Herlev mapping
        print("\nHerlev → Binary:")
        herlev_classes = binary_mapper.get_dataset_classes(DatasetType.HERLEV)
        for class_id, class_name in herlev_classes.items():
            mapped_label = binary_mapper.map_label(class_id, DatasetType.HERLEV)
            binary_label = "NORMAL" if mapped_label == 0 else "ABNORMAL"
            print(f"  {class_name} → {binary_label}")
        
        print("\n✅ Cross-dataset mapping working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Cross-dataset demo failed: {e}")
        return False


def show_dataset_statistics():
    """Show comprehensive dataset statistics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATASET STATISTICS")
    print("="*80)
    
    try:
        # Binary classification
        binary_manager = DatasetManager(
            root_dir="datasets",
            classification_mode="binary",
            random_seed=42
        )
        binary_manager.load_datasets()
        binary_manager.create_splits()
        
        # Multiclass classification
        multiclass_manager = DatasetManager(
            root_dir="datasets",
            classification_mode="multiclass",
            random_seed=42
        )
        multiclass_manager.load_datasets()
        multiclass_manager.create_splits()
        
        # Get summaries
        binary_summary = binary_manager.get_dataset_summary()
        multiclass_summary = multiclass_manager.get_dataset_summary()
        
        print("DATASET OVERVIEW:")
        print("="*50)
        
        total_binary_samples = sum(info["total_samples"] for info in binary_summary["datasets"].values())
        total_multiclass_samples = sum(info["total_samples"] for info in multiclass_summary["datasets"].values())
        
        print(f"Total images available: {total_binary_samples}")
        print(f"Datasets: SIPaKMeD, Herlev")
        print(f"Classification modes: Binary (2 classes), Multiclass (5-7 classes)")
        print(f"Image formats: BMP, PNG, JPG, JPEG, TIF, TIFF")
        
        print("\nBINARY CLASSIFICATION:")
        print("-" * 30)
        for dataset_name, info in binary_summary["datasets"].items():
            normal_count = info["class_distribution"].get(0, 0)
            abnormal_count = info["class_distribution"].get(1, 0)
            print(f"{dataset_name}: {normal_count} normal, {abnormal_count} abnormal")
        
        print("\nMULTICLASS CLASSIFICATION:")
        print("-" * 30)
        for dataset_name, info in multiclass_summary["datasets"].items():
            print(f"{dataset_name}: {len(info['class_distribution'])} classes, {info['total_samples']} images")
        
        print("\nSPLIT RATIOS (70/15/15):")
        print("-" * 30)
        for dataset_name, info in binary_summary["datasets"].items():
            train_pct = (info['train_samples'] / info['total_samples']) * 100
            val_pct = (info['val_samples'] / info['total_samples']) * 100
            test_pct = (info['test_samples'] / info['total_samples']) * 100
            print(f"{dataset_name}: Train {train_pct:.1f}%, Val {val_pct:.1f}%, Test {test_pct:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics demo failed: {e}")
        return False


def main():
    """Run all demonstrations."""
    print("CERVICAL CANCER DATASET DEMONSTRATION")
    print("="*80)
    print("This demo shows the dataset loading and management capabilities")
    print("without requiring PyTorch installation.")
    print("="*80)
    
    demos = [
        ("Binary Classification", demonstrate_binary_classification),
        ("Multiclass Classification", demonstrate_multiclass_classification),
        ("Cross-Dataset Compatibility", demonstrate_cross_dataset_compatibility),
        ("Dataset Statistics", show_dataset_statistics)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            results[demo_name] = demo_func()
        except Exception as e:
            print(f"❌ {demo_name} crashed: {e}")
            results[demo_name] = False
    
    # Print summary
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for demo_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{demo_name:<30}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} demonstrations successful")
    
    if passed == total:
        print("\n🎉 ALL DEMONSTRATIONS PASSED!")
        print("\nThe dataset pipeline is working correctly and ready for:")
        print("1. Model training with EfficientNet-B0, ResNet50, or Swin Transformer")
        print("2. Binary and multiclass classification")
        print("3. Cross-dataset evaluation")
        print("4. Explainable AI analysis")
        print("\nNext steps:")
        print("- Install PyTorch: pip install torch torchvision")
        print("- Run full test: python test_dataset.py")
        print("- Start training: python main.py")
    else:
        print("\n⚠️  Some demonstrations failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
