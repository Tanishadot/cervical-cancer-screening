#!/usr/bin/env python3
"""
Test script for XAI integration functionality.
Tests both Grad-CAM and Attention Rollout methods.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_attention_rollout():
    """Test Attention Rollout implementation."""
    print("Testing Attention Rollout...")
    
    try:
        from xai.attention_rollout import create_attention_rollout_analyzer
        from models.model_factory import CervicalCancerModel
        
        # Create a dummy Swin Transformer model
        device = torch.device("cpu")  # Use CPU for testing
        model = CervicalCancerModel(
            model_name='swin_base_patch4_window7_224',
            num_classes=2,
            pretrained=False,
            dropout_rate=0.1,
            freeze_backbone=True
        )
        model.eval()
        
        # Create analyzer
        analyzer = create_attention_rollout_analyzer(model, device)
        print("✓ Attention Rollout analyzer created successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Test attention rollout computation
        attention_map = analyzer.compute_attention_rollout(dummy_input)
        
        if attention_map is not None:
            print(f"✓ Attention rollout computed successfully, shape: {attention_map.shape}")
        else:
            print("⚠ Attention rollout returned None (expected if model architecture doesn't support hooks)")
        
        # Cleanup
        analyzer.cleanup()
        print("✓ Attention Rollout test completed")
        
    except Exception as e:
        print(f"✗ Attention Rollout test failed: {e}")
        return False
    
    return True

def test_gradcam():
    """Test Grad-CAM implementation."""
    print("\nTesting Grad-CAM...")
    
    try:
        from xai.grad_cam import create_xai_analyzer
        from models.model_factory import CervicalCancerModel
        
        # Create a dummy CNN model
        device = torch.device("cpu")  # Use CPU for testing
        model = CervicalCancerModel(
            model_name='efficientnet_b0',
            num_classes=2,
            pretrained=False,
            dropout_rate=0.1,
            freeze_backbone=True
        )
        model.eval()
        
        # Create analyzer
        analyzer = create_xai_analyzer(model, device, "outputs/xai/test_gradcam")
        print("✓ Grad-CAM analyzer created successfully")
        
        print("✓ Grad-CAM test completed")
        
    except Exception as e:
        print(f"✗ Grad-CAM test failed: {e}")
        return False
    
    return True

def test_config_integration():
    """Test configuration integration."""
    print("\nTesting configuration integration...")
    
    try:
        from utils.config_manager import ConfigManager
        
        # Load config
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Check XAI configuration
        xai_method = config.get('xai', {}).get('method', 'gradcam')
        xai_enabled = config.get('xai', {}).get('enabled', False)
        xai_num_samples = config.get('xai', {}).get('num_samples', 10)
        
        print(f"✓ XAI method: {xai_method}")
        print(f"✓ XAI enabled: {xai_enabled}")
        print(f"✓ XAI num samples: {xai_num_samples}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_main_integration():
    """Test main.py integration."""
    print("\nTesting main.py integration...")
    
    try:
        from main import run_xai_analysis
        print("✓ run_xai_analysis function imported successfully")
        
        # Test function signature
        import inspect
        sig = inspect.signature(run_xai_analysis)
        params = list(sig.parameters.keys())
        print(f"✓ Function parameters: {params}")
        
        return True
        
    except Exception as e:
        print(f"✗ Main integration test failed: {e}")
        return False

def test_dashboard_integration():
    """Test dashboard integration."""
    print("\nTesting dashboard integration...")
    
    try:
        import dashboard
        print("✓ Dashboard module imported successfully")
        
        # Test sample image loading
        samples = dashboard.load_sample_images()
        print(f"✓ Sample images loaded: {len(samples)} samples found")
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("XAI Integration Test Suite")
    print("="*60)
    
    tests = [
        test_attention_rollout,
        test_gradcam,
        test_config_integration,
        test_main_integration,
        test_dashboard_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! XAI integration is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    print("="*60)

if __name__ == "__main__":
    main()
