#!/usr/bin/env python3
"""
Script to find correct layer names for EfficientNet-B0
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.model_factory import CervicalCancerModel

def find_efficientnet_layers():
    """Find all layer names in EfficientNet-B0 model."""
    
    # Create model
    model = CervicalCancerModel(
        model_name="efficientnet_b0",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    print("EfficientNet-B0 Layer Names:")
    print("="*50)
    
    # Print all layer names
    for name, module in model.named_modules():
        if 'conv' in name.lower() or 'features' in name.lower():
            print(f"{name:50} {type(module).__name__}")
    
    print("\n" + "="*50)
    print("Recommended target layers for Grad-CAM:")
    print("- backbone.features.8.0.block.2")
    print("- backbone.features.7.0.block.2") 
    print("- backbone.features.6.0.block.2")

if __name__ == "__main__":
    find_efficientnet_layers()
