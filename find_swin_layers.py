#!/usr/bin/env python3
"""
Script to find correct layer names for Swin Transformer models
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.model_factory import CervicalCancerModel

def find_swin_transformer_layers():
    """Find all layer names in Swin Transformer model."""
    
    # Create Swin Transformer model
    print("Creating Swin Transformer model...")
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    print("Swin Transformer Layer Names:")
    print("="*80)
    
    # Print all layer names with their types
    swin_layers = []
    for name, module in model.named_modules():
        if 'layers' in name or 'blocks' in name or 'norm' in name:
            swin_layers.append((name, type(module).__name__))
    
    # Sort and print
    swin_layers.sort()
    for name, module_type in swin_layers:
        print(f"{name:80} {module_type}")
    
    print("\n" + "="*80)
    print("Recommended target layers for Grad-CAM:")
    
    # Find the deepest layers
    deepest_layers = []
    for name, module_type in swin_layers:
        if 'blocks[-1]' in name and ('norm1' in name or 'norm2' in name):
            deepest_layers.append(name)
    
    print("Swin Transformer final layers:")
    for layer in deepest_layers[-5:]:  # Last 5 layers
        print(f"  - {layer}")
    
    print("\nRecommended target layer:")
    if deepest_layers:
        print(f"  - {deepest_layers[-1]}")
    
    # Test forward pass to see tensor shapes
    print("\n" + "="*80)
    print("Testing forward pass...")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    def hook_fn(module, input, output):
        print(f"Layer output shape: {output.shape}")
        return output
    
    # Register hook on the recommended layer
    target_layer = deepest_layers[-1] if deepest_layers else None
    if target_layer:
        print(f"Hooking into: {target_layer}")
        
        # Find the module
        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module
                break
        
        if target_module:
            target_module.register_forward_hook(hook_fn)
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"Final output shape: {output.shape}")

if __name__ == "__main__":
    find_swin_transformer_layers()
