#!/usr/bin/env python3
"""
Dual Model Integration for Cytology Classification
Integrates CNN and Swin Transformer with explainability.

Author: Cervical Cancer Classification Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """CNN model for local feature extraction."""
    
    def __init__(self, num_classes: int = 2):
        super(CNNModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 4 (for Grad-CAM)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension
        self.feature_dim = 256
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with feature extraction."""
        # Extract features
        features = self.features(x)
        
        # Global average pooling
        pooled = self.avgpool(features)
        pooled_flat = torch.flatten(pooled, 1)
        
        # Classification
        logits = self.classifier(pooled_flat)
        
        return {
            'features': pooled_flat,
            'feature_maps': features,
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1)
        }


class SwinTransformerModel(nn.Module):
    """Simplified Swin Transformer for global context."""
    
    def __init__(self, num_classes: int = 2, embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2]):
        super(SwinTransformerModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Simplified transformer blocks
        self.layers = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            layer = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=8,
                    dim_feedforward=dim * 4,
                    dropout=0.1,
                    batch_first=True
                ) for _ in range(depth)
            ])
            self.layers.append(layer)
            
            # Downsample after first two layers
            if i < 2:
                dim *= 2
        
        # Classification head
        self.head = nn.Linear(dim, num_classes)
        
        # Feature dimension
        self.feature_dim = dim
        
        # Store attention maps for explainability
        self.attention_maps = []
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with attention extraction."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/4, W/4]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        
        self.attention_maps = []
        
        # Pass through transformer layers
        dim = self.embed_dim
        for i, layer_group in enumerate(self.layers):
            for layer in layer_group:
                # Store attention weights (simplified)
                attn_output = layer(x)
                x = attn_output
            
            # Store attention map for this layer group
            self.attention_maps.append(x.clone())
            
            # Downsample after first two layers
            if i < 1:
                dim *= 2
                x = x.reshape(B, -1, dim // 2, dim // 2)
                x = x.permute(0, 2, 3, 1).reshape(B, -1, dim // 2)
        
        # Global average pooling
        features = torch.mean(x, dim=1)  # [B, dim]
        
        # Classification
        logits = self.head(features)
        
        return {
            'features': features,
            'attention_maps': self.attention_maps,
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1)
        }


class FeatureFusion(nn.Module):
    """Fusion layer for CNN and Swin features."""
    
    def __init__(self, cnn_dim: int, swin_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super(FeatureFusion, self).__init__()
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + swin_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Feature importance weights
        self.importance_weights = nn.Parameter(torch.ones(2))  # [cnn_weight, swin_weight]
        
    def forward(self, cnn_features: torch.Tensor, swin_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with feature fusion."""
        # Concatenate features
        concatenated = torch.cat([cnn_features, swin_features], dim=1)
        
        # Apply fusion layers
        logits = self.fusion(concatenated)
        probabilities = F.softmax(logits, dim=1)
        
        # Calculate feature importance
        normalized_weights = F.softmax(self.importance_weights, dim=0)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'fused_features': concatenated,
            'cnn_weight': normalized_weights[0],
            'swin_weight': normalized_weights[1],
            'prediction': torch.argmax(logits, dim=1),
            'confidence': torch.max(probabilities, dim=1)[0]
        }


class DualModelClassifier(nn.Module):
    """Dual model classifier combining CNN and Swin Transformer."""
    
    def __init__(self, num_classes: int = 2):
        super(DualModelClassifier, self).__init__()
        
        # Initialize models
        self.cnn_model = CNNModel(num_classes)
        self.swin_model = SwinTransformerModel(num_classes)
        
        # Get feature dimensions
        cnn_dim = self.cnn_model.feature_dim
        swin_dim = self.swin_model.feature_dim
        
        # Initialize fusion layer
        self.fusion_layer = FeatureFusion(cnn_dim, swin_dim, num_classes=num_classes)
        
        # Store for explainability
        self.last_cnn_features = None
        self.last_swin_features = None
        self.last_fusion_result = None
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through both models."""
        # CNN forward pass
        cnn_result = self.cnn_model(x)
        
        # Swin forward pass
        swin_result = self.swin_model(x)
        
        # Feature fusion
        fusion_result = self.fusion_layer(cnn_result['features'], swin_result['features'])
        
        # Store for explainability
        self.last_cnn_features = cnn_result
        self.last_swin_features = swin_result
        self.last_fusion_result = fusion_result
        
        return {
            'cnn': cnn_result,
            'swin': swin_result,
            'fusion': fusion_result,
            'final_prediction': fusion_result['prediction'],
            'final_confidence': fusion_result['confidence'],
            'final_probabilities': fusion_result['probabilities']
        }


def create_dual_model(num_classes: int = 2, device: str = 'cpu') -> DualModelClassifier:
    """Create and return a dual model classifier."""
    model = DualModelClassifier(num_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test dual model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dual_model(device=device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        result = model(dummy_input)
    
    print("Dual model test completed:")
    print(f"CNN features shape: {result['cnn']['features'].shape}")
    print(f"Swin features shape: {result['swin']['features'].shape}")
    print(f"Fused features shape: {result['fusion']['fused_features'].shape}")
    print(f"Final prediction: {result['final_prediction']}")
    print(f"Final confidence: {result['final_confidence']}")
