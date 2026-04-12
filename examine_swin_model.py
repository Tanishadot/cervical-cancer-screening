#!/usr/bin/env python3
"""
Examine Swin Transformer structure to fix feature extraction.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.model_factory import CervicalCancerModel

def examine_swin_model():
    """Examine Swin Transformer model structure."""
    print("Examining Swin Transformer Model Structure...")
    
    # Create Swin model
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    print(f"Model type: {type(model)}")
    print(f"Backbone type: {type(model.backbone)}")
    
    # Check if backbone has forward_features
    if hasattr(model.backbone, 'forward_features'):
        print("Backbone has forward_features method")
    
    # Check model structure
    print("\nModel components:")
    for name, module in model.named_children():
        print(f"  {name}: {type(module)}")
    
    print("\nBackbone components:")
    for name, module in model.backbone.named_children():
        print(f"  {name}: {type(module)}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test regular forward
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Regular forward output shape: {output.shape}")
        print(f"Regular forward output: {output}")
    
    # Test backbone forward
    with torch.no_grad():
        backbone_output = model.backbone(dummy_input)
        print(f"Backbone output shape: {backbone_output.shape}")
        print(f"Backbone output type: {type(backbone_output)}")
        
        if hasattr(backbone_output, 'shape'):
            print(f"Backbone output stats: mean={backbone_output.mean():.6f}, max={backbone_output.max():.6f}")
    
    # Test forward_features if available
    if hasattr(model.backbone, 'forward_features'):
        with torch.no_grad():
            features = model.backbone.forward_features(dummy_input)
            print(f"Forward features shape: {features.shape}")
            print(f"Forward features stats: mean={features.mean():.6f}, max={features.max():.6f}")
    
    # Check for head
    if hasattr(model.backbone, 'head'):
        print(f"Head type: {type(model.backbone.head)}")
        print(f"Head: {model.backbone.head}")
    
    if hasattr(model, 'head'):
        print(f"Model head type: {type(model.head)}")
        print(f"Model head: {model.head}")

if __name__ == "__main__":
    examine_swin_model()
