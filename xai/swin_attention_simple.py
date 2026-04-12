#!/usr/bin/env python3
"""
Simplified Attention Rollout for Swin Transformer
Uses feature maps instead of attention weights
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class SwinFeatureExtractor:
    """Extract feature maps from Swin Transformer for visualization."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize feature extractor.
        
        Args:
            model: Swin Transformer model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.feature_maps = []
        self.hooks = []
        
        # Register hooks on final layers
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on feature extraction layers."""
        self.feature_maps = []
        
        # Get Swin backbone
        backbone = self.model.backbone
        
        # Hook into the final stage's norm layer
        def feature_hook(module, input, output):
            # Store feature maps
            if isinstance(output, torch.Tensor):
                self.feature_maps.append(output.detach().cpu())
        
        # Find the final norm layer
        final_norm = None
        for name, module in backbone.named_modules():
            if 'norm' in name and type(module).__name__ == 'LayerNorm':
                if 'layers.3' in name or 'layers.2' in name:  # Use deeper layers
                    final_norm = module
                    break
        
        if final_norm is None:
            # Fallback to any norm layer
            for name, module in backbone.named_modules():
                if type(module).__name__ == 'LayerNorm':
                    final_norm = module
                    break
        
        if final_norm:
            hook = final_norm.register_forward_hook(feature_hook)
            self.hooks.append(hook)
            logger.info(f"Registered hook on final norm layer")
        else:
            logger.warning("No suitable layer found for feature extraction")
    
    def generate_feature_map(self, input_tensor: torch.Tensor, image_size: int = 224) -> np.ndarray:
        """
        Generate feature map visualization.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            image_size: Target image size
            
        Returns:
            Feature map heatmap
        """
        # Clear previous feature maps
        self.feature_maps = []
        
        # Set model to eval mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        if not self.feature_maps:
            raise ValueError("No feature maps captured during forward pass")
        
        # Use the last feature map
        features = self.feature_maps[-1]  # [1, seq_len, embed_dim]
        
        # Handle different feature map shapes
        if len(features.shape) == 3:
            batch_size, seq_len, embed_dim = features.shape
            
            # For Swin, seq_len should be 49 (7x7 patches)
            if seq_len == 49:
                # Reshape to 7x7 grid
                features_2d = features[0].mean(dim=0)  # Average across embed_dim
                
                # Create 2D representation
                grid_size = int(np.sqrt(seq_len))
                feature_map = features_2d.view(grid_size, grid_size).numpy()
            else:
                # Fallback: use first dimension
                feature_map = features[0, :, 0].numpy()
                grid_size = int(np.sqrt(feature_map.shape[0]))
                feature_map = feature_map.reshape(grid_size, grid_size)
        else:
            # Fallback for unexpected shapes
            feature_map = features[0, 0].numpy()
        
        # Resize to image size
        feature_resized = cv2.resize(
            feature_map, 
            (image_size, image_size), 
            interpolation=cv2.INTER_LINEAR
        )
        
        return feature_resized
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.feature_maps = []

def generate_attention_overlay(image: np.ndarray, 
                               feature_map: np.ndarray,
                               alpha: float = 0.6,
                               colormap: str = 'jet') -> np.ndarray:
    """
    Generate feature map overlay on original image.
    
    Args:
        image: Original image (H, W, C) in RGB
        feature_map: Feature heatmap (H, W)
        alpha: Transparency for overlay
        colormap: Colormap for heatmap
        
    Returns:
        Overlay image
    """
    # Normalize feature map
    feature_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    feature_colored = cmap(feature_norm)[:, :, :3]  # Remove alpha channel
    
    # Convert to 0-255 range
    feature_colored = (feature_colored * 255).astype(np.uint8)
    
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize feature map to match image if needed
    if feature_colored.shape[:2] != image.shape[:2]:
        feature_colored = cv2.resize(feature_colored, (image.shape[1], image.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, feature_colored, alpha, 0)
    
    return overlay

def save_feature_visualizations(image: np.ndarray,
                               feature_map: np.ndarray,
                               overlay: np.ndarray,
                               save_path: str,
                               prefix: str = ""):
    """
    Save feature visualizations.
    
    Args:
        image: Original image
        feature_map: Feature heatmap
        overlay: Overlay image
        save_path: Directory to save visualizations
        prefix: Filename prefix
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.savefig(save_dir / f"{prefix}_original.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save feature heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(feature_map, cmap='jet')
    plt.title("Feature Activation Map")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir / f"{prefix}_heatmap.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title("Feature Overlay")
    plt.axis('off')
    plt.savefig(save_dir / f"{prefix}_overlay.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"Saved feature visualizations to {save_dir}")

def visualize_multiple_samples(model: nn.Module,
                              dataset,
                              num_samples: int = 6,
                              device: str = 'cpu',
                              save_dir: str = 'outputs/xai/swin_features/',
                              image_size: int = 224):
    """
    Generate feature visualizations for multiple samples.
    
    Args:
        model: Swin Transformer model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to run on
        save_dir: Directory to save visualizations
        image_size: Target image size
    """
    from preprocessing.transforms import get_val_transforms
    from datasets.dataset import TransformDataset, create_dataloader
    
    # Create feature extractor
    feature_extractor = SwinFeatureExtractor(model, device)
    
    # Get validation transforms
    val_transform = get_val_transforms()
    
    # Create transformed dataset
    class AlbumentationsTransform:
        def __init__(self, albumentations_transform):
            self.transform = albumentations_transform
        
        def __call__(self, image):
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            return self.transform(image=image)['image']
    
    albumentations_transform = AlbumentationsTransform(val_transform)
    transformed_dataset = TransformDataset(dataset, albumentations_transform)
    
    # Create data loader
    data_loader = create_dataloader(
        dataset=transformed_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Collect samples
    normal_samples = []
    abnormal_samples = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 4:
                images, labels, original_images, _ = batch
            elif len(batch) == 3:
                images, labels, original_images = batch
            else:
                images, labels = batch[:2]
                original_images = images
            
            # Get prediction
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Store samples
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                
                sample_data = {
                    'image': images[i:i+1],
                    'original': original_images[i],
                    'label': label,
                    'prediction': pred,
                    'correct': label == pred
                }
                
                if label == 0 and len(normal_samples) < num_samples // 2:
                    normal_samples.append(sample_data)
                elif label == 1 and len(abnormal_samples) < num_samples // 2:
                    abnormal_samples.append(sample_data)
                
                # Stop if we have enough samples
                if len(normal_samples) >= num_samples // 2 and len(abnormal_samples) >= num_samples // 2:
                    break
            
            if len(normal_samples) >= num_samples // 2 and len(abnormal_samples) >= num_samples // 2:
                break
    
    # Generate visualizations
    all_samples = normal_samples + abnormal_samples
    
    for idx, sample in enumerate(all_samples):
        try:
            # Generate feature map
            feature_map = feature_extractor.generate_feature_map(
                sample['image'], image_size
            )
            
            # Convert original image to numpy
            original_img = sample['original'].cpu().numpy()
            if original_img.shape[0] == 3:  # CHW format
                original_img = np.transpose(original_img, (1, 2, 0))
            
            # Normalize to 0-1
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
            
            # Generate overlay
            overlay = generate_attention_overlay(original_img, feature_map)
            
            # Determine label name
            label_name = "normal" if sample['label'] == 0 else "abnormal"
            correct_str = "correct" if sample['correct'] else "incorrect"
            
            # Save visualizations
            prefix = f"{label_name}_{idx+1}_{correct_str}"
            save_feature_visualizations(
                original_img, feature_map, overlay, 
                save_dir, prefix
            )
            
            logger.info(f"Generated feature visualization for {prefix}")
            
        except Exception as e:
            logger.error(f"Error generating visualization for sample {idx}: {e}")
            continue
    
    # Cleanup hooks
    feature_extractor.cleanup()
    
    logger.info(f"Generated feature visualizations for {len(all_samples)} samples in {save_dir}")
