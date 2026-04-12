#!/usr/bin/env python3
"""
Script to find the correct layer names for Grad-CAM in different models.
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
    
    print("Backbone structure:")
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"  Conv2D: {name} -> {module}")
        elif isinstance(module, torch.nn.BatchNorm2d):
            print(f"  BatchNorm2D: {name} -> {module}")
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            print(f"  AdaptiveAvgPool2D: {name} -> {module}")
    
    print("\nFull model structure:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"  Conv2D: {name} -> {module}")

def find_swin_layers():
    """Find layers in Swin model."""
    print("\n\nSwin Model Layers:")
    print("=" * 50)
    
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    print("Backbone structure:")
    for name, module in model.backbone.named_modules():
        if 'attn' in name.lower():
            print(f"  Attention: {name} -> {type(module).__name__}")
        elif 'norm' in name.lower():
            print(f"  Norm: {name} -> {type(module).__name__}")
        elif 'patch' in name.lower():
            print(f"  Patch: {name} -> {type(module).__name__}")

if __name__ == "__main__":
    find_cnn_layers()
    find_swin_layers()
