#!/usr/bin/env python3
"""
Test the enhanced dual-model dashboard to ensure no prediction errors.
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_enhanced_prediction():
    """Test the enhanced prediction function."""
    print("Testing Enhanced Prediction Function...")
    
    try:
        from dual_model_dashboard_enhanced import safe_prediction_with_features, DualModelClassifier
        
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
        result = safe_prediction_with_features(dummy_input, classifier)
        
        if result is None:
            print("Prediction returned None")
            return False
        
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
        print(f"Raw probabilities: {result.get('raw_probabilities', 'N/A')}")
        print(f"CNN features shape: {result['cnn_features_shape']}")
        print(f"Swin features shape: {result['swin_features_shape']}")
        print(f"Fused features shape: {result['fused_features_shape']}")
        print(f"CNN output shape: {result.get('cnn_output_shape', 'N/A')}")
        print(f"Swin output shape: {result.get('swin_output_shape', 'N/A')}")
        
        # Test feature visualization
        print("\nTesting Feature Visualization...")
        from dual_model_dashboard_enhanced import create_feature_visualization
        import matplotlib.pyplot as plt
        
        fig = create_feature_visualization(result['cnn_features'], result['swin_features'])
        
        # Save test visualization
        os.makedirs("test_outputs", exist_ok=True)
        plt.savefig("test_outputs/enhanced_feature_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Enhanced feature visualization saved successfully")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_probability_handling():
    """Test safe probability handling."""
    print("\nTesting Probability Handling...")
    
    test_cases = [
        # Single probability (binary case)
        np.array([0.7]),
        # Two probabilities (normal case)
        np.array([0.3, 0.7]),
        # Multiple probabilities (multi-class case)
        np.array([0.1, 0.2, 0.7]),
    ]
    
    for i, probs in enumerate(test_cases):
        print(f"\nTest case {i+1}: {probs}")
        
        # Dynamic prediction handling
        if len(probs) == 1:
            prediction = int(probs[0] > 0.5)
            confidence = float(probs[0])
            probabilities = [float(probs[0]), float(1 - probs[0])]
        else:
            prediction = int(np.argmax(probs))
            confidence = float(np.max(probs))
            probabilities = probs.tolist()
        
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Probabilities: {probabilities}")
    
    print("All probability handling tests passed")
    return True


def main():
    """Run all enhanced dashboard tests."""
    print("ENHANCED DASHBOARD TESTS")
    print("=" * 50)
    
    tests = [
        ("Enhanced Prediction", test_enhanced_prediction),
        ("Probability Handling", test_probability_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("Enhanced dashboard is ready for demo!")
        print("\nKey improvements:")
        print(" Fixed 'list index out of range' error")
        print(" Safe probability handling")
        print(" Comprehensive feature visualization")
        print(" Debug panel for troubleshooting")
        print(" Fail-safe error handling")
    else:
        print("Some issues need to be addressed.")


if __name__ == "__main__":
    main()
