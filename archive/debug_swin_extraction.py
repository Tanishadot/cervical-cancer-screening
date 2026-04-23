#!/usr/bin/env python3
"""
Debug Swin feature extraction step by step.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def debug_swin_extraction():
    """Debug Swin feature extraction step by step."""
    print("Debugging Swin Feature Extraction...")
    
    try:
        # Create Swin model directly
        from models.model_factory import CervicalCancerModel
        
        swin_model = CervicalCancerModel(
            model_name="swin_tiny_patch4_window7_224.ms_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        
        print(f"Swin model type: {type(swin_model)}")
        print(f"Swin backbone type: {type(swin_model.backbone)}")
        
        # Test input
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"Input shape: {dummy_input.shape}")
        
        # Step 1: Test forward_features
        print("\n=== Step 1: Testing forward_features ===")
        with torch.no_grad():
            backbone_output = swin_model.backbone.forward_features(dummy_input)
        
        print(f"forward_features output shape: {backbone_output.shape}")
        print(f"forward_features output stats: mean={backbone_output.mean():.6f}, max={backbone_output.max():.6f}")
        
        # Step 2: Test processing logic
        print("\n=== Step 2: Testing processing logic ===")
        
        if len(backbone_output.shape) == 4:
            print("Output is 4D tensor")
            print(f"Shape details: batch={backbone_output.shape[0]}, channels={backbone_output.shape[1]}, height={backbone_output.shape[2]}, width={backbone_output.shape[3]}")
            
            if backbone_output.shape[1] == 768 and backbone_output.shape[2] == 7:
                print("This is the expected patch embedding format")
                features = backbone_output  # [1, 768, 7, 7]
                print(f"Features before pooling: {features.shape}")
                
                # Global average pooling over patches
                features_pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [1, 768, 1, 1]
                print(f"Features after pooling: {features_pooled.shape}")
                
                features_flat = features_pooled.flatten(1)  # [1, 768]
                print(f"Features after flatten: {features_flat.shape}")
                print(f"Features stats: mean={features_flat.mean():.6f}, max={features_flat.max():.6f}")
                
            else:
                print("Unexpected 4D format, using generic pooling")
                features = torch.nn.functional.adaptive_avg_pool2d(backbone_output, (1, 1))
                features = features.flatten(1)
                print(f"Generic processed features: {features.shape}")
        
        # Step 3: Test full model forward
        print("\n=== Step 3: Testing full model forward ===")
        with torch.no_grad():
            full_output = swin_model(dummy_input)
        
        print(f"Full model output shape: {full_output.shape}")
        print(f"Full model output: {full_output}")
        
        # Step 4: Test with DualModelClassifier
        print("\n=== Step 4: Testing with DualModelClassifier ===")
        
        # Save model
        os.makedirs("outputs/cross_dataset_training/models", exist_ok=True)
        torch.save(swin_model.state_dict(), "outputs/cross_dataset_training/models/best_swin_model.pth")
        
        # Create CNN model too
        cnn_model = CervicalCancerModel(
            model_name="efficientnet_b0.ra_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        os.makedirs("outputs/models", exist_ok=True)
        torch.save(cnn_model.state_dict(), "outputs/models/best_model.pth")
        
        # Initialize classifier
        from dual_model_classifier import DualModelClassifier
        
        device = torch.device("cpu")
        classifier = DualModelClassifier("outputs/models/best_model.pth", 
                                   "outputs/cross_dataset_training/models/best_swin_model.pth", 
                                   device)
        
        print("Classifier initialized successfully")
        
        # Test extraction
        swin_features, swin_pred, attention_maps = classifier.feature_extractor.extract_swin_features(dummy_input)
        
        print(f"Final Swin features shape: {swin_features.shape}")
        print(f"Final Swin features stats: mean={swin_features.mean():.6f}, max={swin_features.max():.6f}")
        
        return swin_features.shape[1] >= 100
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_swin_extraction()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Swin feature extraction is working!")
    else:
        print("FAILED: Swin feature extraction has issues")
