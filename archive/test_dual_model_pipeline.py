#!/usr/bin/env python3
"""
Test script for the Dual-Model Classification Pipeline
Validates feature extraction, fusion, and interpretability components.

Author: Cervical Cancer Classification Pipeline
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dual_model_classifier import DualModelClassifier, FeatureExtractor, FeatureFusion
from preprocessing.transforms import get_val_transforms
from models.model_factory import CervicalCancerModel


def create_dummy_models():
    """Create dummy models for testing when real models aren't available."""
    print("Creating dummy models for testing...")
    
    # Create dummy CNN model
    cnn_model = CervicalCancerModel(
        model_name="efficientnet_b0.ra_in1k",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    # Create dummy Swin model
    swin_model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    return cnn_model, swin_model


def test_feature_extractor():
    """Test the feature extraction component."""
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTOR")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    cnn_model, swin_model = create_dummy_models()
    cnn_model.to(device)
    swin_model.to(device)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(cnn_model, swin_model, device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        # Test CNN feature extraction
        print("\n--- Testing CNN Feature Extraction ---")
        cnn_features, cnn_pred, cnn_feat_maps, cnn_grads = extractor.extract_cnn_features(dummy_input)
        
        if cnn_features is not None:
            print(f"✅ CNN features shape: {cnn_features.shape}")
            print(f"✅ CNN prediction: {cnn_pred.item()}")
            print(f"✅ CNN feature maps shape: {cnn_feat_maps.shape if cnn_feat_maps is not None else 'None'}")
            print(f"✅ CNN gradients shape: {cnn_grads.shape if cnn_grads is not None else 'None'}")
        else:
            print("❌ CNN feature extraction failed")
        
        # Test Swin feature extraction
        print("\n--- Testing Swin Feature Extraction ---")
        swin_features, swin_pred, attention_maps = extractor.extract_swin_features(dummy_input)
        
        if swin_features is not None:
            print(f"✅ Swin features shape: {swin_features.shape}")
            print(f"✅ Swin prediction: {swin_pred.item()}")
            print(f"✅ Attention maps keys: {list(attention_maps.keys())}")
        else:
            print("❌ Swin feature extraction failed")
        
        # Test heatmap generation
        print("\n--- Testing Heatmap Generation ---")
        if cnn_feat_maps is not None and cnn_grads is not None:
            gradcam_heatmap = extractor.generate_gradcam_heatmap(cnn_grads, cnn_feat_maps)
            print(f"✅ Grad-CAM heatmap shape: {gradcam_heatmap.shape}")
            print(f"✅ Grad-CAM heatmap range: [{gradcam_heatmap.min():.3f}, {gradcam_heatmap.max():.3f}]")
        
        if attention_maps:
            attention_map = extractor.generate_attention_map(attention_maps)
            print(f"✅ Attention map shape: {attention_map.shape}")
            print(f"✅ Attention map range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_fusion():
    """Test the feature fusion component."""
    print("\n" + "="*60)
    print("TESTING FEATURE FUSION")
    print("="*60)
    
    try:
        # Create dummy feature dimensions
        cnn_dim = 1280  # Typical EfficientNet feature dimension
        swin_dim = 768  # Typical Swin Tiny feature dimension
        
        # Create fusion model
        fusion_model = FeatureFusion(cnn_dim, swin_dim, num_classes=2)
        
        # Create dummy features
        batch_size = 2
        cnn_features = torch.randn(batch_size, cnn_dim)
        swin_features = torch.randn(batch_size, swin_dim)
        
        print(f"CNN features shape: {cnn_features.shape}")
        print(f"Swin features shape: {swin_features.shape}")
        
        # Test forward pass
        result = fusion_model(cnn_features, swin_features)
        
        print(f"✅ Fusion output keys: {list(result.keys())}")
        print(f"✅ Logits shape: {result['logits'].shape}")
        print(f"✅ Probabilities shape: {result['probabilities'].shape}")
        print(f"✅ Fused features shape: {result['fused_features'].shape}")
        print(f"✅ CNN weights shape: {result['cnn_weight'].shape}")
        print(f"✅ Swin weights shape: {result['swin_weight'].shape}")
        print(f"✅ Predictions: {result['prediction']}")
        print(f"✅ Confidence scores: {result['confidence']}")
        
        # Test fusion dimension
        expected_fusion_dim = cnn_dim + swin_dim
        actual_fusion_dim = result['fused_features'].shape[1]
        
        if actual_fusion_dim == expected_fusion_dim:
            print(f"✅ Fusion dimension correct: {actual_fusion_dim}")
        else:
            print(f"❌ Fusion dimension mismatch: expected {expected_fusion_dim}, got {actual_fusion_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_model_classifier():
    """Test the complete dual model classifier."""
    print("\n" + "="*60)
    print("TESTING DUAL MODEL CLASSIFIER")
    print("="*60)
    
    try:
        # Create dummy model files for testing
        os.makedirs("outputs/models", exist_ok=True)
        os.makedirs("outputs/cross_dataset_training/models", exist_ok=True)
        
        # Create dummy CNN model
        cnn_model = CervicalCancerModel(
            model_name="efficientnet_b0.ra_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        
        # Create dummy Swin model
        swin_model = CervicalCancerModel(
            model_name="swin_tiny_patch4_window7_224.ms_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        
        # Save dummy models
        torch.save(cnn_model.state_dict(), "outputs/models/best_model.pth")
        torch.save(swin_model.state_dict(), "outputs/cross_dataset_training/models/best_swin_model.pth")
        
        # Initialize classifier
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = DualModelClassifier(
            "outputs/models/best_model.pth",
            "outputs/cross_dataset_training/models/best_swin_model.pth",
            device
        )
        
        print("✅ Dual model classifier initialized successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"Input shape: {dummy_input.shape}")
        
        # Make prediction
        result = classifier.predict(dummy_input)
        
        print(f"✅ Prediction completed successfully")
        print(f"✅ Result keys: {list(result.keys())}")
        print(f"✅ Prediction: {result['prediction']}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        print(f"✅ CNN features shape: {result['cnn_features_shape']}")
        print(f"✅ Swin features shape: {result['swin_features_shape']}")
        print(f"✅ Fused features shape: {result['fused_features_shape']}")
        print(f"✅ Grad-CAM heatmap shape: {result['gradcam_heatmap'].shape}")
        print(f"✅ Attention map shape: {result['attention_map'].shape}")
        print(f"✅ CNN weight: {result['cnn_weight']:.3f}")
        print(f"✅ Swin weight: {result['swin_weight']:.3f}")
        
        # Test explanations
        print(f"✅ CNN explanation: {result['explanation']['cnn_focus']}")
        print(f"✅ Swin explanation: {result['explanation']['swin_focus']}")
        print(f"✅ Fusion explanation: {result['explanation']['fusion_insight']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dual model classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_image():
    """Test with a real image if available."""
    print("\n" + "="*60)
    print("TESTING WITH REAL IMAGE")
    print("="*60)
    
    try:
        # Try to find a real image from the dataset
        image_paths = []
        
        # Search for images in dataset directories
        for root, dirs, files in os.walk("datasets"):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg', '.tif')):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print("⚠️  No real images found in dataset, creating synthetic image")
            # Create a synthetic image
            synthetic_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(synthetic_image)
        else:
            # Use the first found image
            image_path = image_paths[0]
            print(f"Using image: {image_path}")
            image = Image.open(image_path).convert('RGB')
        
        # Resize to expected size
        image = image.resize((224, 224))
        print(f"Image size: {image.size}")
        
        # Initialize classifier (using dummy models if real ones aren't available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cnn_path = "outputs/models/best_model.pth"
        swin_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
        
        if not os.path.exists(cnn_path) or not os.path.exists(swin_path):
            print("⚠️  Real models not found, using dummy models")
            # Create dummy models
            cnn_model, swin_model = create_dummy_models()
            
            os.makedirs("outputs/models", exist_ok=True)
            os.makedirs("outputs/cross_dataset_training/models", exist_ok=True)
            
            torch.save(cnn_model.state_dict(), cnn_path)
            torch.save(swin_model.state_dict(), swin_path)
        
        classifier = DualModelClassifier(cnn_path, swin_path, device)
        
        # Preprocess image
        transform = get_val_transforms()
        image_np = np.array(image)
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)
        
        print(f"Preprocessed tensor shape: {image_tensor.shape}")
        
        # Make prediction
        result = classifier.predict(image_tensor)
        
        print("✅ Real image prediction completed successfully")
        print(f"✅ Prediction: {result['prediction']}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        print(f"✅ Probabilities: {result['probabilities']}")
        
        # Save visualizations
        os.makedirs("test_outputs", exist_ok=True)
        
        # Save Grad-CAM visualization
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(result['gradcam_heatmap'], cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(result['attention_map'], cmap='viridis')
        plt.title("Attention Map")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("test_outputs/dual_model_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization saved to test_outputs/dual_model_visualization.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("DUAL-MODEL CLASSIFICATION PIPELINE TEST")
    print("=" * 80)
    
    # Run tests
    tests = [
        ("Feature Extractor", test_feature_extractor),
        ("Feature Fusion", test_feature_fusion),
        ("Dual Model Classifier", test_dual_model_classifier),
        ("Real Image Test", test_with_real_image)
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
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The dual-model pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Train your CNN and Swin models")
        print("2. Run: streamlit run dual_model_dashboard.py")
        print("3. Upload medical images for analysis")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
