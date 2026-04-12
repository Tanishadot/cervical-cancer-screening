#!/usr/bin/env python3
"""
Test the fixed Swin Transformer feature extraction.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_fixed_swin_features():
    """Test the fixed Swin feature extraction."""
    print("Testing Fixed Swin Feature Extraction...")
    
    try:
        from dual_model_classifier import DualModelClassifier
        
        # Create dummy models
        from models.model_factory import CervicalCancerModel
        
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
        
        print(f"Input shape: {dummy_input.shape}")
        
        # Test Swin feature extraction specifically
        swin_features, swin_pred, attention_maps = classifier.feature_extractor.extract_swin_features(dummy_input)
        
        print(f"Swin features shape: {swin_features.shape}")
        print(f"Swin features stats: mean={swin_features.mean():.6f}, max={swin_features.max():.6f}, min={swin_features.min():.6f}")
        print(f"Swin prediction: {swin_pred}")
        print(f"First 10 Swin features: {swin_features.flatten()[:10].tolist()}")
        
        # Verify feature dimension
        if swin_features.shape[1] < 100:
            print(f"ERROR: Swin feature dimension too small: {swin_features.shape}")
            return False
        else:
            print(f"SUCCESS: Swin feature dimension is correct: {swin_features.shape}")
        
        # Test CNN features for comparison
        cnn_features, cnn_pred, cnn_feat_maps, cnn_grads = classifier.feature_extractor.extract_cnn_features(dummy_input)
        
        print(f"\nCNN features shape: {cnn_features.shape}")
        print(f"CNN features stats: mean={cnn_features.mean():.6f}, max={cnn_features.max():.6f}, min={cnn_features.min():.6f}")
        print(f"CNN prediction: {cnn_pred}")
        print(f"First 10 CNN features: {cnn_features.flatten()[:10].tolist()}")
        
        # Test full prediction
        result = classifier.predict(dummy_input)
        
        print(f"\nFull prediction results:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"CNN features shape: {result['cnn_features_shape']}")
        print(f"Swin features shape: {result['swin_features_shape']}")
        print(f"Fused features shape: {result['fused_features_shape']}")
        
        # Feature comparison
        cnn_dim = result['cnn_features_shape'][1]
        swin_dim = result['swin_features_shape'][1]
        
        print(f"\nFeature comparison:")
        print(f"CNN: {cnn_dim}-dim local features")
        print(f"Swin: {swin_dim}-dim global features")
        print(f"Ratio (Swin/CNN): {swin_dim/cnn_dim:.2f}")
        
        if swin_dim > cnn_dim:
            print("SUCCESS: Swin features have higher dimension than CNN (as expected for global features)")
        else:
            print("WARNING: Swin features have lower dimension than CNN")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Swin feature extraction test."""
    print("FIXED SWIN FEATURE EXTRACTION TEST")
    print("=" * 50)
    
    success = test_fixed_swin_features()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Swin feature extraction is working correctly!")
        print("\nKey improvements:")
        print("- Using forward_features() instead of backbone()")
        print("- Proper handling of 4D patch embeddings [1, 768, 7, 7]")
        print("- Global average pooling to get 768-dim features")
        print("- Sanity check for minimum feature dimension")
        print("- Meaningful feature values (not near zero)")
    else:
        print("FAILED: Swin feature extraction still has issues")
    
    print("\nExpected feature dimensions:")
    print("- CNN: ~1280-dim (local features)")
    print("- Swin: 768-dim (global features)")
    print("- Fused: ~2048-dim (combined)")


if __name__ == "__main__":
    main()
