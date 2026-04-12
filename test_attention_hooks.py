#!/usr/bin/env python3
"""
Test attention hooks for Swin Transformer
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.model_factory import CervicalCancerModel

def test_attention_hooks():
    """Test attention hooks on Swin model."""
    
    # Create model
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    attention_maps = []
    
    def attention_hook(module, input, output):
        print(f"Hook triggered on {type(module).__name__}")
        print(f"Input shape: {input[0].shape if input else 'None'}")
        
        if isinstance(output, tuple):
            print(f"Output tuple length: {len(output)}")
            for i, out in enumerate(output):
                print(f"  Output[{i}] shape: {out.shape if hasattr(out, 'shape') else type(out)}")
        else:
            print(f"Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        # Try to extract attention
        if hasattr(module, 'attention_probs'):
            attention = module.attention_probs
            print(f"Found attention_probs: {attention.shape}")
            attention_maps.append(attention.detach().cpu())
        elif isinstance(output, tuple) and len(output) > 1:
            attention = output[1]
            print(f"Using output[1] as attention: {attention.shape}")
            attention_maps.append(attention.detach().cpu())
    
    # Find and hook attention layers
    attention_layers = []
    for name, module in model.named_modules():
        if 'WindowAttention' in type(module).__name__:
            attention_layers.append((name, module))
            module.register_forward_hook(attention_hook)
    
    print(f"Found {len(attention_layers)} WindowAttention layers")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nCaptured {len(attention_maps)} attention maps")
    for i, attn in enumerate(attention_maps):
        print(f"  Map {i}: {attn.shape}")

if __name__ == "__main__":
    test_attention_hooks()
