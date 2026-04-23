"""
Test script to verify dataset loading and data pipeline functionality.
Tests both binary and multiclass classification with different models.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset import DatasetManager
from preprocessing.transforms import get_train_transforms, get_val_transforms
from models.model_factory import create_model


def test_dataset_loading():
    """Test dataset loading and basic functionality."""
    print("="*80)
    print("TESTING DATASET LOADING")
    print("="*80)
    
    # Test both binary and multiclass
    for mode in ["binary", "multiclass"]:
        print(f"\n--- Testing {mode.upper()} Classification ---")
        
        try:
            # Create dataset manager
            manager = DatasetManager(
                root_dir="datasets",
                classification_mode=mode,
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
            
            print(f"✅ {mode.upper()} dataset loading successful")
            
        except Exception as e:
            print(f"❌ {mode.upper()} dataset loading failed: {e}")
            return False
    
    return True


def test_data_loaders():
    """Test data loader creation and batch generation."""
    print("\n" + "="*80)
    print("TESTING DATA LOADERS")
    print("="*80)
    
    try:
        # Create dataset manager
        manager = DatasetManager(
            root_dir="datasets",
            classification_mode="binary",  # Test binary first
            random_seed=42
        )
        
        # Load and split
        manager.load_datasets()
        manager.create_splits()
        
        # Get transforms
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        # Create data loaders
        data_loaders = manager.create_data_loaders(
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=16,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        print(f"Created {len(data_loaders)} data loaders")
        
        # Test each data loader
        for loader_name, loader in data_loaders.items():
            print(f"\n--- Testing {loader_name} ---")
            
            # Get one batch
            batch = next(iter(loader))
            images, labels, original_labels, paths = batch
            
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"Image dtype: {images.dtype}")
            print(f"Label dtype: {labels.dtype}")
            
            # Check image values
            if torch.is_tensor(images):
                print(f"Image value range: [{images.min().item():.3f}, {images.max().item():.3f}]")
            
            print(f"✅ {loader_name} working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False


def test_model_compatibility():
    """Test model compatibility with dataset."""
    print("\n" + "="*80)
    print("TESTING MODEL COMPATIBILITY")
    print("="*80)
    
    models_to_test = [
        ("efficientnet_b0", "EfficientNet-B0"),
        ("resnet50", "ResNet50"),
        ("swin_transformer", "Swin Transformer")
    ]
    
    try:
        # Create dataset manager
        manager = DatasetManager(
            root_dir="datasets",
            classification_mode="binary",
            random_seed=42
        )
        
        # Load and split
        manager.load_datasets()
        manager.create_splits()
        
        # Get transforms and data loaders
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        data_loaders = manager.create_data_loaders(
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=4,  # Small batch for testing
            num_workers=0
        )
        
        # Get a sample batch
        sample_loader = list(data_loaders.values())[0]
        sample_batch = next(iter(sample_loader))
        sample_images, sample_labels, _, _ = sample_batch
        
        print(f"Sample batch shape: {sample_images.shape}")
        print(f"Sample labels: {sample_labels}")
        
        # Test each model
        for model_name, display_name in models_to_test:
            print(f"\n--- Testing {display_name} ---")
            
            try:
                # Create model
                model = create_model(
                    model_name=model_name,
                    num_classes=2,  # Binary classification
                    pretrained=False  # Don't need pretrained for testing
                )
                
                # Set to eval mode
                model.eval()
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(sample_images)
                
                print(f"Output shape: {outputs.shape}")
                print(f"Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                print(f"Predictions: {predictions}")
                print(f"True labels: {sample_labels}")
                
                print(f"✅ {display_name} working correctly")
                
            except Exception as e:
                print(f"❌ {display_name} failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False


def test_multiclass_models():
    """Test models with multiclass classification."""
    print("\n" + "="*80)
    print("TESTING MULTICLASS MODELS")
    print("="*80)
    
    try:
        # Create dataset manager for multiclass
        manager = DatasetManager(
            root_dir="datasets",
            classification_mode="multiclass",
            random_seed=42
        )
        
        # Load and split
        manager.load_datasets()
        manager.create_splits()
        
        # Get transforms and data loaders
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        data_loaders = manager.create_data_loaders(
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=4,
            num_workers=0
        )
        
        # Get sample batch
        sample_loader = list(data_loaders.values())[0]
        sample_batch = next(iter(sample_loader))
        sample_images, sample_labels, _, _ = sample_batch
        
        print(f"Multiclass batch shape: {sample_images.shape}")
        print(f"Multiclass labels: {sample_labels}")
        print(f"Unique labels: {torch.unique(sample_labels)}")
        
        # Test with 5 classes (SIPaKMeD)
        model = create_model(
            model_name="efficientnet_b0",
            num_classes=5,
            pretrained=False
        )
        
        model.eval()
        with torch.no_grad():
            outputs = model(sample_images)
        
        print(f"Multiclass output shape: {outputs.shape}")
        print(f"✅ Multiclass model working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiclass test failed: {e}")
        return False


def visualize_sample_batch():
    """Visualize a sample batch of images."""
    print("\n" + "="*80)
    print("VISUALIZING SAMPLE BATCH")
    print("="*80)
    
    try:
        # Create dataset manager
        manager = DatasetManager(
            root_dir="datasets",
            classification_mode="binary",
            random_seed=42
        )
        
        # Load and split
        manager.load_datasets()
        manager.create_splits()
        
        # Get transforms (use validation transforms for better visualization)
        val_transform = get_val_transforms()
        data_loaders = manager.create_data_loaders(
            train_transform=val_transform,  # Use val transforms for better visualization
            val_transform=val_transform,
            batch_size=8,
            num_workers=0
        )
        
        # Get sample batch
        sample_loader = list(data_loaders.values())[0]
        sample_batch = next(iter(sample_loader))
        images, labels, original_labels, paths = sample_batch
        
        print(f"Visualizing {len(images)} samples")
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        class_names = ["NORMAL", "ABNORMAL"]  # Binary classification
        
        for i in range(min(len(images), 8)):
            # Convert image for display
            img = images[i]
            
            # Handle different tensor formats
            if img.dim() == 3:
                if img.shape[0] == 3:  # CHW format
                    img = img.permute(1, 2, 0)  # Convert to HWC
            
            # Denormalize if needed
            if img.max() <= 1.0:
                img = img * 255
            img = img.cpu().numpy().astype(np.uint8)
            
            # Display
            axes[i].imshow(img)
            axes[i].set_title(f"{class_names[labels[i].item()]}\n{os.path.basename(paths[i])}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), 8):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("dataset_sample_batch.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Sample batch visualization saved as 'dataset_sample_batch.png'")
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("COMPREHENSIVE DATASET PIPELINE TEST")
    print("="*80)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Data Loaders", test_data_loaders),
        ("Model Compatibility", test_model_compatibility),
        ("Multiclass Models", test_multiclass_models),
        ("Visualization", visualize_sample_batch)
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
        print("🎉 ALL TESTS PASSED! Dataset pipeline is working correctly.")
        print("\nYou can now proceed with training:")
        print("  python main.py --binary")
        print("  python main.py --multiclass")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comprehensive test
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
