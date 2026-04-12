#!/usr/bin/env python3
"""
Test dashboard components individually to ensure they work correctly.
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_dual_model_classifier():
    """Test dual model classifier."""
    print("Testing Dual Model Classifier...")
    
    try:
        from dual_model_classifier import DualModelClassifier
        
        # Create dummy models
        from models.model_factory import CervicalCancerModel
        import os
        
        os.makedirs("outputs/models", exist_ok=True)
        os.makedirs("outputs/cross_dataset_training/models", exist_ok=True)
        
        cnn_model = CervicalCancerModel(
            model_name="efficientnet_b0.ra_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        
        swin_model = CervicalCancerModel(
            model_name="swin_tiny_patch4_window7_224.ms_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        
        torch.save(cnn_model.state_dict(), "outputs/models/best_model.pth")
        torch.save(swin_model.state_dict(), "outputs/cross_dataset_training/models/best_swin_model.pth")
        
        # Initialize classifier
        device = torch.device("cpu")
        classifier = DualModelClassifier("outputs/models/best_model.pth", 
                                   "outputs/cross_dataset_training/models/best_swin_model.pth", 
                                   device)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        result = classifier.predict(dummy_input)
        
        print(f"✅ Prediction: {result['prediction']}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        print(f"✅ CNN features shape: {result['cnn_features_shape']}")
        print(f"✅ Swin features shape: {result['swin_features_shape']}")
        print(f"✅ Fused features shape: {result['fused_features_shape']}")
        print(f"✅ Grad-CAM heatmap shape: {result['gradcam_heatmap'].shape}")
        print(f"✅ Attention map shape: {result['attention_map'].shape}")
        print(f"✅ CNN weight: {result['cnn_weight']:.3f}")
        print(f"✅ Swin weight: {result['swin_weight']:.3f}")
        
        # Test visualizations
        print("\nTesting visualizations...")
        
        # Test overlay creation
        from dual_model_classifier import create_overlay
        original_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay = create_overlay(original_image, result['gradcam_heatmap'])
        print(f"✅ Overlay created with shape: {overlay.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test image preprocessing."""
    print("\nTesting Image Preprocessing...")
    
    try:
        from preprocessing.transforms import get_val_transforms
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Apply transforms
        transform = get_val_transforms()
        transformed = transform(image=test_image)
        image_tensor = transformed['image']
        
        print(f"✅ Original image shape: {test_image.shape}")
        print(f"✅ Transformed tensor shape: {image_tensor.shape}")
        print(f"✅ Tensor dtype: {image_tensor.dtype}")
        print(f"✅ Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_creation():
    """Test visualization creation."""
    print("\nTesting Visualization Creation...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Create test data
        gradcam_heatmap = np.random.rand(224, 224)
        attention_map = np.random.rand(224, 224)
        original_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Create test visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gradcam_heatmap, cmap='jet')
        axes[0, 1].set_title('CNN Grad-CAM')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(attention_map, cmap='viridis')
        axes[1, 0].set_title('Transformer Attention')
        axes[1, 0].axis('off')
        
        # Test overlay
        from dual_model_classifier import create_overlay
        overlay = create_overlay(original_image, gradcam_heatmap)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Grad-CAM Overlay')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save test visualization
        os.makedirs("test_outputs", exist_ok=True)
        plt.savefig("test_outputs/test_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Test visualization saved to test_outputs/test_visualization.png")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all component tests."""
    print("DASHBOARD COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        ("Dual Model Classifier", test_dual_model_classifier),
        ("Image Preprocessing", test_preprocessing),
        ("Visualization Creation", test_visualization_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All dashboard components are working correctly!")
        print("\nDashboard is ready for live demo.")
        print("Run: streamlit run dual_model_dashboard_robust.py")
    else:
        print("⚠️ Some components need fixes.")


if __name__ == "__main__":
    main()
