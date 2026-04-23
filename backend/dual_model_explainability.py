#!/usr/bin/env python3
"""
Dual Model Explainability for Cytology Classification
Integrates CNN Grad-CAM and Swin attention with feature alignment.

Author: Cervical Cancer Classification Pipeline
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM implementation for CNN model."""
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module:
            target_module.register_forward_hook(forward_hook)
            target_module.register_backward_hook(backward_hook)
        else:
            logger.warning(f"Target layer '{self.target_layer}' not found")
    
    def generate_cam(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        if self.activations is None or self.gradients is None:
            return np.zeros((224, 224))
        
        try:
            # Get gradients and activations
            gradients = self.gradients[0]  # [C, H, W]
            activations = self.activations[0]  # [C, H, W]
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(1, 2))  # [C]
            
            # Weighted combination of activation maps
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return np.zeros((224, 224))


class AttentionRollout:
    """Attention rollout for Swin Transformer."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_maps = []
        
    def extract_attention(self, image_tensor: torch.Tensor) -> List[np.ndarray]:
        """Extract attention maps from Swin layers."""
        try:
            # Forward pass to get attention maps
            with torch.no_grad():
                result = self.model(image_tensor)
            
            attention_maps = []
            
            # Process attention from each layer group
            for layer_idx, layer_attention in enumerate(result['attention_maps']):
                # Convert to numpy and average over heads
                attention = layer_attention.detach().cpu().numpy()
                
                # Average over sequence dimension (excluding class token if present)
                if attention.shape[1] > 1:
                    attention = attention[:, 1:, :]  # Remove class token
                
                # Average over sequence length to get importance
                attention_avg = np.mean(attention, axis=1)  # [batch, seq_len]
                
                # Reshape to spatial grid (assuming square patches)
                batch_size, seq_len = attention_avg.shape
                grid_size = int(np.sqrt(seq_len))
                
                if grid_size * grid_size == seq_len:
                    attention_2d = attention_avg.reshape(batch_size, grid_size, grid_size)
                    attention_maps.append(attention_2d[0])  # Take first batch
                else:
                    # Fallback: use 1D attention
                    attention_maps.append(attention_avg[0])
            
            return attention_maps
            
        except Exception as e:
            logger.error(f"Error extracting attention: {e}")
            return []
    
    def compute_rollout(self, attention_maps: List[np.ndarray]) -> np.ndarray:
        """Compute cumulative attention rollout."""
        if not attention_maps:
            return np.zeros((224, 224))
        
        try:
            # Start with identity matrix
            rollout = np.eye(attention_maps[0].shape[-1])
            
            # Multiply attention matrices
            for attention in attention_maps:
                if len(attention.shape) == 2:
                    rollout = rollout @ attention
                elif len(attention.shape) == 3:
                    # Average over spatial dimension
                    attention_2d = np.mean(attention, axis=0)
                    rollout = rollout @ attention_2d
            
            # Get class token attention (first token)
            if rollout.shape[0] > 1:
                class_attention = rollout[0, 1:]  # Exclude class token itself
            else:
                class_attention = rollout[0]
            
            # Reshape to spatial grid
            seq_len = len(class_attention)
            grid_size = int(np.sqrt(seq_len))
            
            if grid_size * grid_size == seq_len:
                attention_map = class_attention.reshape(grid_size, grid_size)
            else:
                # Fallback: create a 1D representation
                attention_map = np.zeros((224, 224))
                for i, val in enumerate(class_attention):
                    if i < 224 * 224:
                        attention_map.flat[i] = val
            
            # Normalize
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            
            return attention_map
            
        except Exception as e:
            logger.error(f"Error computing rollout: {e}")
            return np.zeros((224, 224))


class DualModelExplainer:
    """Unified explainability for dual model system."""
    
    def __init__(self, cnn_model: torch.nn.Module, swin_model: torch.nn.Module, device: str = 'cpu'):
        self.device = device
        
        # Initialize explainers
        self.grad_cam = GradCAM(cnn_model, "features.6")  # Last conv block
        self.attention_rollout = AttentionRollout(swin_model)
        
    def explain(self, image_tensor: torch.Tensor, segmentation_masks: Dict[str, np.ndarray], 
                class_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate explanations for both models.
        
        Args:
            image_tensor: Input image tensor
            segmentation_masks: Dictionary with nucleus, cytoplasm, background masks
            class_idx: Target class index
            
        Returns:
            Dictionary with explanations and importance scores
        """
        explanations = {}
        
        # CNN Grad-CAM
        try:
            # Forward pass for gradients
            image_tensor.requires_grad_(True)
            
            # Get model output
            with torch.no_grad():
                cnn_result = self.grad_cam.model(image_tensor)
            
            if class_idx is None:
                class_idx = torch.argmax(cnn_result['logits'], dim=1).item()
            
            # Backward pass
            cnn_result['logits'][0, class_idx].backward(retain_graph=True)
            
            # Generate Grad-CAM
            grad_cam_heatmap = self.grad_cam.generate_cam(image_tensor, class_idx)
            explanations['grad_cam'] = grad_cam_heatmap
            
        except Exception as e:
            logger.error(f"Error in Grad-CAM: {e}")
            explanations['grad_cam'] = np.zeros((224, 224))
        
        # Swin Attention
        try:
            attention_maps = self.attention_rollout.extract_attention(image_tensor)
            attention_heatmap = self.attention_rollout.compute_rollout(attention_maps)
            explanations['attention'] = attention_heatmap
            
        except Exception as e:
            logger.error(f"Error in attention rollout: {e}")
            explanations['attention'] = np.zeros((224, 224))
        
        # Calculate region importance
        explanations.update(self._calculate_region_importance(
            explanations['grad_cam'], 
            explanations['attention'], 
            segmentation_masks
        ))
        
        return explanations
    
    def _calculate_region_importance(self, grad_cam: np.ndarray, attention: np.ndarray, 
                                   masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate importance for each region."""
        importance = {}
        
        try:
            # Grad-CAM importance
            nucleus_grad = np.mean(grad_cam[masks['nucleus'] > 0]) if np.any(masks['nucleus'] > 0) else 0
            cytoplasm_grad = np.mean(grad_cam[masks['cytoplasm'] > 0]) if np.any(masks['cytoplasm'] > 0) else 0
            background_grad = np.mean(grad_cam[masks['background'] > 0]) if np.any(masks['background'] > 0) else 0
            
            # Attention importance
            nucleus_att = np.mean(attention[masks['nucleus'] > 0]) if np.any(masks['nucleus'] > 0) else 0
            cytoplasm_att = np.mean(attention[masks['cytoplasm'] > 0]) if np.any(masks['cytoplasm'] > 0) else 0
            background_att = np.mean(attention[masks['background'] > 0]) if np.any(masks['background'] > 0) else 0
            
            # Combined importance (50-50 weighted)
            importance['nucleus_importance'] = 0.5 * nucleus_grad + 0.5 * nucleus_att
            importance['cytoplasm_importance'] = 0.5 * cytoplasm_grad + 0.5 * cytoplasm_att
            importance['background_importance'] = 0.5 * background_grad + 0.5 * background_att
            
            # Determine dominant region
            region_scores = {
                'nucleus': importance['nucleus_importance'],
                'cytoplasm': importance['cytoplasm_importance'],
                'background': importance['background_importance']
            }
            importance['dominant_region'] = max(region_scores, key=region_scores.get)
            
            # Store individual scores for transparency
            importance['nucleus_grad_cam'] = nucleus_grad
            importance['cytoplasm_grad_cam'] = cytoplasm_grad
            importance['background_grad_cam'] = background_grad
            importance['nucleus_attention'] = nucleus_att
            importance['cytoplasm_attention'] = cytoplasm_att
            importance['background_attention'] = background_att
            
        except Exception as e:
            logger.error(f"Error calculating region importance: {e}")
            # Default values
            importance = {
                'nucleus_importance': 0.33,
                'cytoplasm_importance': 0.33,
                'background_importance': 0.34,
                'dominant_region': 'background',
                'nucleus_grad_cam': 0,
                'cytoplasm_grad_cam': 0,
                'background_grad_cam': 0,
                'nucleus_attention': 0,
                'cytoplasm_attention': 0,
                'background_attention': 0
            }
        
        return importance
    
    def create_overlays(self, image: np.ndarray, explanations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create overlay visualizations."""
        overlays = {}
        
        try:
            # Grad-CAM overlay
            if 'grad_cam' in explanations:
                grad_cam_colored = plt.cm.jet(explanations['grad_cam'])[:, :, :3]
                grad_cam_colored = (grad_cam_colored * 255).astype(np.uint8)
                
                # Resize to match image if needed
                if grad_cam_colored.shape[:2] != image.shape[:2]:
                    grad_cam_colored = cv2.resize(grad_cam_colored, (image.shape[1], image.shape[0]))
                
                # Blend with original
                alpha = 0.6
                grad_cam_overlay = alpha * grad_cam_colored / 255.0 + (1 - alpha) * image / 255.0
                overlays['grad_cam_overlay'] = (grad_cam_overlay * 255).astype(np.uint8)
            
            # Attention overlay
            if 'attention' in explanations:
                attention_colored = plt.cm.viridis(explanations['attention'])[:, :, :3]
                attention_colored = (attention_colored * 255).astype(np.uint8)
                
                # Resize to match image if needed
                if attention_colored.shape[:2] != image.shape[:2]:
                    attention_colored = cv2.resize(attention_colored, (image.shape[1], image.shape[0]))
                
                # Blend with original
                alpha = 0.6
                attention_overlay = alpha * attention_colored / 255.0 + (1 - alpha) * image / 255.0
                overlays['attention_overlay'] = (attention_overlay * 255).astype(np.uint8)
            
            # Combined overlay
            if 'grad_cam' in explanations and 'attention' in explanations:
                combined = 0.5 * explanations['grad_cam'] + 0.5 * explanations['attention']
                combined_colored = plt.cm.plasma(combined)[:, :, :3]
                combined_colored = (combined_colored * 255).astype(np.uint8)
                
                # Resize to match image if needed
                if combined_colored.shape[:2] != image.shape[:2]:
                    combined_colored = cv2.resize(combined_colored, (image.shape[1], image.shape[0]))
                
                # Blend with original
                alpha = 0.6
                combined_overlay = alpha * combined_colored / 255.0 + (1 - alpha) * image / 255.0
                overlays['combined_overlay'] = (combined_overlay * 255).astype(np.uint8)
                
        except Exception as e:
            logger.error(f"Error creating overlays: {e}")
            overlays = {
                'grad_cam_overlay': image,
                'attention_overlay': image,
                'combined_overlay': image
            }
        
        return overlays


def create_dual_explainer(cnn_model: torch.nn.Module, swin_model: torch.nn.Module, 
                        device: str = 'cpu') -> DualModelExplainer:
    """Create and return a dual model explainer."""
    return DualModelExplainer(cnn_model, swin_model, device)


if __name__ == "__main__":
    # Test dual model explainer
    from .dual_model_integration import create_dual_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dual_model(device=device)
    
    # Create explainer
    explainer = create_dual_explainer(model.cnn_model, model.swin_model, device)
    
    # Test with dummy data
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_masks = {
        'nucleus': np.random.randint(0, 255, (224, 224)),
        'cytoplasm': np.random.randint(0, 255, (224, 224)),
        'background': np.random.randint(0, 255, (224, 224))
    }
    
    # Test explanation
    with torch.no_grad():
        result = model(dummy_input)
    
    explanations = explainer.explain(dummy_input, dummy_masks)
    
    print("Dual model explainer test completed:")
    print(f"Grad-CAM shape: {explanations['grad_cam'].shape}")
    print(f"Attention shape: {explanations['attention'].shape}")
    print(f"Dominant region: {explanations['dominant_region']}")
    print(f"Nucleus importance: {explanations['nucleus_importance']:.3f}")
    print(f"Cytoplasm importance: {explanations['cytoplasm_importance']:.3f}")
