#!/usr/bin/env python3
"""
Script to find the correct layer names for Grad-CAM in CNN model.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.model_factory import CervicalCancerModel

def find_cnn_layers():
    """Find layers in CNN model."""
    print("CNN Model Layers:")
    print("=" * 50)
    
    model = CervicalCancerModel(
        model_name="efficientnet_b0.ra_in1k",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    print("Backbone conv layers:")
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"  Conv2D: {name}")
    
    print("\nFull model conv layers:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"  Conv2D: {name}")
    
    # Test forward pass to see output shapes
    print("\nTesting forward pass:")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model.backbone(dummy_input)
        print(f"Backbone output shape: {output.shape}")
        
        # Try to get intermediate outputs
        x = dummy_input
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                x = module(x)
                print(f"After {name}: {x.shape}")

if __name__ == "__main__":
    find_cnn_layers()
