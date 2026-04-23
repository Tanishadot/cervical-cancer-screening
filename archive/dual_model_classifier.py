#!/usr/bin/env python3
"""
Dual-Model Medical Image Classification System
Combines CNN (EfficientNet) and Swin Transformer with feature fusion and interpretability.

Author: Cervical Cancer Classification Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.model_factory import CervicalCancerModel
from test_gradcam import GradCAM


class FeatureExtractor(nn.Module):
    """Extract features from CNN and Swin Transformer models."""
    
    def __init__(self, cnn_model: nn.Module, swin_model: nn.Module, device: str = 'cpu'):
        super().__init__()
        self.cnn_model = cnn_model
        self.swin_model = swin_model
        self.device = device
        
        # Move models to device
        self.cnn_model.to(device)
        self.swin_model.to(device)
        
        # Set to evaluation mode
        self.cnn_model.eval()
        self.swin_model.eval()
        
        # Hook storage
        self.cnn_features = None
        self.swin_features = None
        self.cnn_gradients = None
        self.swin_attention = {}
        
        # Register hooks
        self._register_cnn_hooks()
        self._register_swin_hooks()
    
    def _register_cnn_hooks(self):
        """Register hooks for CNN feature extraction."""
        def cnn_forward_hook(module, input, output):
            # Store features from last convolutional layer
            self.cnn_features = output.detach().clone()
        
        def cnn_backward_hook(module, grad_input, grad_output):
            # Store gradients for Grad-CAM
            self.cnn_gradients = grad_output[0].detach().clone()
        
        # Find the last convolutional layer in CNN backbone
        target_layer = None
        for name, module in self.cnn_model.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer:
            target_layer.register_forward_hook(cnn_forward_hook)
            target_layer.register_backward_hook(cnn_backward_hook)
    
    def _register_swin_hooks(self):
        """Register hooks for Swin Transformer attention extraction."""
        def swin_forward_hook(module, input, output):
            # Store final features before classification
            if hasattr(output, 'shape') and len(output.shape) == 3:
                self.swin_features = output.detach().clone()
        
        def attention_hook(name):
            def hook(module, input, output):
                # For Swin Transformer, attention might be stored differently
                if hasattr(output, 'attention'):
                    self.swin_attention[name] = output.attention.detach().clone()
                elif hasattr(output, 'attn'):
                    self.swin_attention[name] = output.attn.detach().clone()
            return hook
        
        # Register hooks on attention layers
        for name, module in self.swin_model.backbone.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                module.register_forward_hook(attention_hook(name))
        
        # Register hook on final norm layer for features
        for name, module in self.swin_model.backbone.named_modules():
            if name == 'norm' or name.endswith('.norm'):
                module.register_forward_hook(swin_forward_hook)
                break
    
    def extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features and generate Grad-CAM."""
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Clear previous features/gradients
        self.cnn_features = None
        self.cnn_gradients = None
        
        # Forward pass
        output = self.cnn_model(x)
        
        # Get predicted class
        pred_class = torch.argmax(output, dim=1)
        
        # Backward pass for Grad-CAM
        self.cnn_model.zero_grad()
        class_score = output[0, pred_class[0]]
        class_score.backward(retain_graph=True)
        
        # Extract features
        if self.cnn_features is not None:
            # Global average pooling
            features = F.adaptive_avg_pool2d(self.cnn_features, (1, 1))
            features = features.flatten(1)
            return features, pred_class, self.cnn_features, self.cnn_gradients
        
        return None, pred_class, None, None
    
    def extract_swin_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Swin Transformer features and attention maps."""
        x = x.to(self.device)
        
        # Clear previous features/attention
        self.swin_features = None
        self.swin_attention = {}
        
        # Forward pass through backbone to get features
        with torch.no_grad():
            # Use forward_features to get patch embeddings
            backbone_output = self.swin_model.backbone.forward_features(x)
        
        # Get predicted class from full model
        with torch.no_grad():
            full_output = self.swin_model(x)
        pred_class = torch.argmax(full_output, dim=1)
        
        # Process backbone output to get features
        if len(backbone_output.shape) == 4:
            # Feature maps format [batch, height, width, embed_dim] - expected [1, 7, 7, 768]
            if backbone_output.shape[3] == 768 and backbone_output.shape[1] == 7:
                # This is patch embeddings: [batch, height, width, embed_dim]
                features = backbone_output  # [1, 7, 7, 768]
                # Need to transpose to [batch, embed_dim, height, width] for pooling
                features = features.permute(0, 3, 1, 2)  # [1, 768, 7, 7]
                # Global average pooling over patches
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [1, 768, 1, 1]
                features = features.flatten(1)  # [1, 768]
            else:
                # Generic case: adaptive pooling and flatten
                features = torch.nn.functional.adaptive_avg_pool2d(backbone_output, (1, 1))
                features = features.flatten(1)
        elif len(backbone_output.shape) == 3:
            # Token embeddings format [batch, seq_len, embed_dim]
            features = backbone_output
            if features.shape[1] > 1:  # Has class token
                features = features[:, 1:, :]  # Skip class token
            # Global average pooling
            features = torch.mean(features, dim=1)
        else:
            # Already flattened
            features = backbone_output
        
        # Sanity check for feature dimension
        if features.shape[1] < 100:
            raise ValueError(f"Incorrect Swin feature extraction: got {features.shape}, expected at least 100 dimensions")
        
        return features, pred_class, self.swin_attention
        
    def generate_gradcam_heatmap(self, gradients: torch.Tensor, features: torch.Tensor, 
                               size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Generate Grad-CAM heatmap from gradients and features."""
        if gradients is None or features is None:
            return np.zeros(size)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        # Convert to numpy and resize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, size, interpolation=cv2.INTER_LINEAR)
        
        return cam
    
    def generate_attention_map(self, attention_maps: Dict, image_size: int = 224) -> np.ndarray:
        """Generate attention map from Swin Transformer attention."""
        if not attention_maps:
            return np.zeros((image_size, image_size))
        
        # Use the last attention layer
        last_attention = list(attention_maps.values())[-1]
        
        if last_attention.dim() == 3:  # [batch, heads, seq_len] or [batch, seq_len, seq_len]
            # Take mean across heads if needed
            if last_attention.shape[1] > last_attention.shape[2]:
                attention = last_attention[0]  # First sample
                attention = torch.mean(attention, dim=0)  # Mean across heads
            else:
                attention = last_attention[0]  # First sample
            
            # Use CLS token attention (first row/column)
            if attention.shape[0] > 1:
                cls_attention = attention[0, 1:]  # Skip CLS token itself
            else:
                cls_attention = attention[0]
            
            # Convert to spatial map
            num_patches = len(cls_attention)
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size != num_patches:
                # Pad or truncate to make perfect square
                target_size = grid_size * grid_size
                if num_patches > target_size:
                    cls_attention = cls_attention[:target_size]
                else:
                    padding = target_size - num_patches
                    cls_attention = F.pad(cls_attention, (0, padding), mode='constant')
            
            # Reshape to 2D
            attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
            
            # Normalize and resize
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            attention_resized = cv2.resize(attention_map, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            
            return attention_resized
        
        return np.zeros((image_size, image_size))


class FeatureFusion(nn.Module):
    """Feature fusion mechanism with MLP classifier."""
    
    def __init__(self, cnn_feature_dim: int, swin_feature_dim: int, 
                 num_classes: int = 2, hidden_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        
        # Feature fusion
        self.fusion_dim = cnn_feature_dim + swin_feature_dim
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Attention weights for feature importance (optional)
        self.feature_attention = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # CNN and Swin weights
            nn.Softmax(dim=1)
        )
    
    def forward(self, cnn_features: torch.Tensor, swin_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with feature fusion."""
        # Concatenate features
        fused_features = torch.cat([cnn_features, swin_features], dim=1)
        
        # Get feature importance weights
        importance_weights = self.feature_attention(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'fused_features': fused_features,
            'cnn_weight': importance_weights[:, 0],
            'swin_weight': importance_weights[:, 1],
            'prediction': torch.argmax(logits, dim=1),
            'confidence': torch.max(probabilities, dim=1)[0]
        }


class DualModelClassifier:
    """Main dual-model classification system."""
    
    def __init__(self, cnn_model_path: str, swin_model_path: str, device: str = 'cpu'):
        self.device = device
        
        # Load models
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        self.swin_model = self._load_swin_model(swin_model_path)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.cnn_model, self.swin_model, device)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            # Get CNN feature dimension directly from model
            cnn_output = self.cnn_model.backbone(dummy_input)
            if len(cnn_output.shape) == 4:
                cnn_feat = F.adaptive_avg_pool2d(cnn_output, (1, 1)).flatten(1)
            else:
                cnn_feat = cnn_output
            cnn_dim = cnn_feat.shape[1]
            
            # Get Swin feature dimension directly from model
            swin_output = self.swin_model.backbone.forward_features(dummy_input)
            if len(swin_output.shape) == 4:
                # Expected format: [batch, height, width, embed_dim] = [1, 7, 7, 768]
                if swin_output.shape[3] == 768 and swin_output.shape[1] == 7:
                    # Transpose to [batch, embed_dim, height, width] for pooling
                    swin_output = swin_output.permute(0, 3, 1, 2)
                swin_feat = F.adaptive_avg_pool2d(swin_output, (1, 1)).flatten(1)
            elif len(swin_output.shape) == 3:
                # Token format: [batch, seq_len, embed_dim]
                swin_feat = swin_output
                if swin_feat.shape[1] > 1:  # Has class token
                    swin_feat = swin_feat[:, 1:, :]  # Skip class token
                swin_feat = torch.mean(swin_feat, dim=1)
            else:
                # Already flattened
                swin_feat = swin_output
            swin_dim = swin_feat.shape[1]
        
        # Initialize fusion model
        self.fusion_model = FeatureFusion(cnn_dim, swin_dim).to(device)
        
        # Initialize Grad-CAM for CNN
        self.grad_cam = GradCAM(self.cnn_model, "backbone.conv_head")  # Last conv layer in EfficientNet
    
    def _load_cnn_model(self, model_path: str) -> nn.Module:
        """Load CNN model (EfficientNet)."""
        model = CervicalCancerModel(
            model_name="efficientnet_b0.ra_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _load_swin_model(self, model_path: str) -> nn.Module:
        """Load Swin Transformer model."""
        model = CervicalCancerModel(
            model_name="swin_tiny_patch4_window7_224.ms_in1k",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """Make prediction with dual-model system."""
        image_tensor = image_tensor.to(self.device)
        
        # Extract features from both models
        cnn_features, cnn_pred, cnn_feat_maps, cnn_grads = self.feature_extractor.extract_cnn_features(image_tensor)
        swin_features, swin_pred, attention_maps = self.feature_extractor.extract_swin_features(image_tensor)
        
        # Feature fusion
        fusion_result = self.fusion_model(cnn_features, swin_features)
        
        # Generate interpretability maps
        gradcam_heatmap = self.feature_extractor.generate_gradcam_heatmap(
            cnn_grads, cnn_feat_maps, (224, 224)
        )
        
        attention_map = self.feature_extractor.generate_attention_map(attention_maps, 224)
        
        # Compile results
        result = {
            'prediction': fusion_result['prediction'].item(),
            'confidence': fusion_result['confidence'].item(),
            'probabilities': fusion_result['probabilities'].detach().cpu().numpy().tolist(),
            'cnn_features_shape': tuple(cnn_features.shape),
            'swin_features_shape': tuple(swin_features.shape),
            'fused_features_shape': tuple(fusion_result['fused_features'].shape),
            'cnn_prediction': cnn_pred.item(),
            'swin_prediction': swin_pred.item(),
            'cnn_weight': fusion_result['cnn_weight'].item(),
            'swin_weight': fusion_result['swin_weight'].item(),
            'gradcam_heatmap': gradcam_heatmap,
            'attention_map': attention_map,
            'cnn_features': cnn_features.detach().cpu().numpy(),
            'swin_features': swin_features.detach().cpu().numpy(),
            'fused_features': fusion_result['fused_features'].detach().cpu().numpy(),
            'explanation': {
                'cnn_focus': self._explain_cnn_focus(gradcam_heatmap),
                'swin_focus': self._explain_swin_focus(attention_map),
                'fusion_insight': self._explain_fusion(fusion_result['cnn_weight'].item(), 
                                                      fusion_result['swin_weight'].item())
            }
        }
        
        return result
    
    def _explain_cnn_focus(self, heatmap: np.ndarray) -> str:
        """Generate explanation for CNN focus."""
        if np.max(heatmap) == 0:
            return "CNN model did not identify significant regions"
        
        # Find regions of high activation
        threshold = np.percentile(heatmap, 80)
        high_activation = heatmap > threshold
        
        if np.sum(high_activation) > 0:
            y_coords, x_coords = np.where(high_activation)
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            
            if center_y < 112:
                vertical_pos = "upper"
            elif center_y > 112:
                vertical_pos = "lower"
            else:
                vertical_pos = "center"
            
            if center_x < 112:
                horizontal_pos = "left"
            elif center_x > 112:
                horizontal_pos = "right"
            else:
                horizontal_pos = "center"
            
            return f"CNN focuses on {vertical_pos}-{horizontal_pos} regions with high local patterns"
        
        return "CNN shows distributed attention across the image"
    
    def _explain_swin_focus(self, attention_map: np.ndarray) -> str:
        """Generate explanation for Swin Transformer focus."""
        if np.max(attention_map) == 0:
            return "Transformer model did not identify significant regions"
        
        # Analyze attention distribution
        threshold = np.percentile(attention_map, 80)
        high_attention = attention_map > threshold
        
        if np.sum(high_attention) > 0:
            # Check if attention is focused or distributed
            focus_ratio = np.sum(high_attention) / attention_map.size
            
            if focus_ratio < 0.1:
                return "Transformer shows focused attention on specific patches"
            elif focus_ratio < 0.3:
                return "Transformer shows moderate attention on multiple regions"
            else:
                return "Transformer shows distributed attention across many patches"
        
        return "Transformer shows uniform attention pattern"
    
    def _explain_fusion(self, cnn_weight: float, swin_weight: float) -> str:
        """Generate explanation for fusion decision."""
        if cnn_weight > swin_weight * 1.5:
            return f"Fusion relies more on CNN features ({cnn_weight:.2f} vs {swin_weight:.2f})"
        elif swin_weight > cnn_weight * 1.5:
            return f"Fusion relies more on Transformer features ({swin_weight:.2f} vs {cnn_weight:.2f})"
        else:
            return f"Fusion balances both models (CNN: {cnn_weight:.2f}, Transformer: {swin_weight:.2f})"


def create_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create overlay of heatmap on original image."""
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Normalize heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Resize heatmap to match image
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


if __name__ == "__main__":
    # Test the dual model system
    print("Testing Dual Model Classifier...")
    
    # Initialize (you'll need to provide actual model paths)
    # classifier = DualModelClassifier(
    #     cnn_model_path="outputs/models/best_model.pth",
    #     swin_model_path="outputs/cross_dataset_training/models/best_swin_model.pth",
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    
    # Test with dummy input
    # dummy_input = torch.randn(1, 3, 224, 224)
    # result = classifier.predict(dummy_input)
    # print("Prediction result:", result)
    
    print("Dual Model Classifier module loaded successfully!")
