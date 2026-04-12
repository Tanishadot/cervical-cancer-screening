#!/usr/bin/env python3
"""
Attention Rollout Visualization for Swin Transformer
Extracts and visualizes attention maps from Swin Transformer blocks
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

class SwinAttentionRollout:
    """Attention Rollout implementation for Swin Transformer."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize attention rollout.
        
        Args:
            model: Swin Transformer model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.attention_maps = []
        self.hooks = []
        
        # Register hooks on attention layers
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on attention layers."""
        self.attention_maps = []
        
        # Get Swin backbone
        backbone = self.model.backbone
        
        # Find all attention layers in Swin blocks
        def get_attention_layers(module):
            attention_layers = []
            
            # Recursively find attention layers
            for name, child in module.named_children():
                if 'attn' in name and hasattr(child, 'attn'):
                    attention_layers.append(child.attn)
                else:
                    attention_layers.extend(get_attention_layers(child))
            
            return attention_layers
        
        attention_layers = get_attention_layers(backbone)
        
        if not attention_layers:
            logger.warning("No attention layers found in model")
            return
        
        # Register forward hooks
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # For timm Swin models, attention is stored in attn.attn.attention_probs
                if hasattr(module, 'attention_probs'):
                    attention = module.attention_probs
                else:
                    # Try to get from output
                    if isinstance(output, tuple) and len(output) > 1:
                        attention = output[1]
                    elif hasattr(output, 'attention_probs'):
                        attention = output.attention_probs
                    else:
                        # For WindowAttention, get from forward pass result
                        # We'll capture this differently
                        return
                
                # Move to CPU and detach
                if attention is not None:
                    attention = attention.detach().cpu()
                    self.attention_maps.append(attention)
            
            return hook
        
        # Register hooks for all attention layers
        for i, layer in enumerate(attention_layers):
            hook = layer.register_forward_hook(get_attention_hook(i))
            self.hooks.append(hook)
        
        logger.info(f"Registered hooks on {len(attention_layers)} attention layers")
    
    def _compute_rollout(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout from attention maps.
        
        Args:
            attention_maps: List of attention tensors from different layers
            
        Returns:
            Rollout attention map
        """
        if not attention_maps:
            raise ValueError("No attention maps available")
        
        # Process each attention map
        rollout_maps = []
        
        for attn_map in attention_maps:
            # attn_map shape: [batch_size, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attn_map.shape
            
            # Average across heads
            attn_avg = attn_map.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Add identity matrix (residual connection)
            identity = torch.eye(seq_len, device=attn_avg.device)
            attn_avg = attn_avg + identity
            
            # Normalize rows
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            
            rollout_maps.append(attn_avg)
        
        # Multiply attention maps across layers (rollout)
        rollout = rollout_maps[0]
        for i in range(1, len(rollout_maps)):
            rollout = torch.matmul(rollout, rollout_maps[i])
        
        return rollout
    
    def _extract_cls_attention(self, rollout: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token attention from rollout.
        
        Args:
            rollout: Rollout attention map
            
        Returns:
            CLS token attention
        """
        # CLS token is usually the first token
        cls_attention = rollout[:, 0, 1:]  # Skip CLS token itself
        return cls_attention
    
    def _reshape_attention_to_image(self, 
                                  cls_attention: torch.Tensor, 
                                  patch_size: int = 16,
                                  image_size: int = 224) -> np.ndarray:
        """
        Reshape CLS attention to image dimensions.
        
        Args:
            cls_attention: CLS token attention
            patch_size: Size of patches in original image
            image_size: Target image size
            
        Returns:
            Reshaped attention map
        """
        batch_size = cls_attention.shape[0]
        
        # Calculate grid size
        grid_size = int(np.sqrt(cls_attention.shape[-1]))
        
        # Reshape to grid
        attention_grid = cls_attention.view(batch_size, grid_size, grid_size)
        
        # Take first sample (batch_size = 1)
        attention_grid = attention_grid[0].cpu().numpy()
        
        # Resize to image size
        attention_resized = cv2.resize(
            attention_grid, 
            (image_size, image_size), 
            interpolation=cv2.INTER_LINEAR
        )
        
        return attention_resized
    
    def generate_attention_map(self, 
                             input_tensor: torch.Tensor, 
                             patch_size: int = 16,
                             image_size: int = 224) -> np.ndarray:
        """
        Generate attention rollout map for input.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            patch_size: Size of patches
            image_size: Target image size
            
        Returns:
            Attention heatmap
        """
        # Clear previous attention maps
        self.attention_maps = []
        
        # Set model to eval mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        if not self.attention_maps:
            raise ValueError("No attention maps captured during forward pass")
        
        # Compute rollout
        rollout = self._compute_rollout(self.attention_maps)
        
        # Extract CLS attention
        cls_attention = self._extract_cls_attention(rollout)
        
        # Reshape to image dimensions
        attention_map = self._reshape_attention_to_image(
            cls_attention, patch_size, image_size
        )
        
        return attention_map
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = []

def generate_attention_overlay(image: np.ndarray, 
                               attention_map: np.ndarray,
                               alpha: float = 0.6,
                               colormap: str = 'jet') -> np.ndarray:
    """
    Generate attention overlay on original image.
    
    Args:
        image: Original image (H, W, C) in RGB
        attention_map: Attention heatmap (H, W)
        alpha: Transparency for overlay
        colormap: Colormap for heatmap
        
    Returns:
        Overlay image
    """
    # Normalize attention map
    attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    attention_colored = cmap(attention_norm)[:, :, :3]  # Remove alpha channel
    
    # Convert to 0-255 range
    attention_colored = (attention_colored * 255).astype(np.uint8)
    
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize attention to match image if needed
    if attention_colored.shape[:2] != image.shape[:2]:
        attention_colored = cv2.resize(attention_colored, (image.shape[1], image.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, attention_colored, alpha, 0)
    
    return overlay

def save_attention_visualizations(image: np.ndarray,
                                 attention_map: np.ndarray,
                                 overlay: np.ndarray,
                                 save_path: str,
                                 prefix: str = ""):
    """
    Save attention visualizations.
    
    Args:
        image: Original image
        attention_map: Attention heatmap
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
    
    # Save attention heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map, cmap='jet')
    plt.title("Attention Heatmap")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir / f"{prefix}_heatmap.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title("Attention Overlay")
    plt.axis('off')
    plt.savefig(save_dir / f"{prefix}_overlay.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"Saved attention visualizations to {save_dir}")

def visualize_multiple_samples(model: nn.Module,
                              dataset,
                              num_samples: int = 6,
                              device: str = 'cpu',
                              save_dir: str = 'outputs/xai/swin_attention/',
                              patch_size: int = 16,
                              image_size: int = 224):
    """
    Generate attention visualizations for multiple samples.
    
    Args:
        model: Swin Transformer model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to run on
        save_dir: Directory to save visualizations
        patch_size: Patch size for attention reshaping
        image_size: Target image size
    """
    from preprocessing.transforms import get_val_transforms
    from datasets.dataset import TransformDataset, create_dataloader
    
    # Create attention rollout
    attention_rollout = SwinAttentionRollout(model, device)
    
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
            # Generate attention map
            attention_map = attention_rollout.generate_attention_map(
                sample['image'], patch_size, image_size
            )
            
            # Convert original image to numpy
            original_img = sample['original'].cpu().numpy()
            if original_img.shape[0] == 3:  # CHW format
                original_img = np.transpose(original_img, (1, 2, 0))
            
            # Normalize to 0-1
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
            
            # Generate overlay
            overlay = generate_attention_overlay(original_img, attention_map)
            
            # Determine label name
            label_name = "normal" if sample['label'] == 0 else "abnormal"
            correct_str = "correct" if sample['correct'] else "incorrect"
            
            # Save visualizations
            prefix = f"{label_name}_{idx+1}_{correct_str}"
            save_attention_visualizations(
                original_img, attention_map, overlay, 
                save_dir, prefix
            )
            
            logger.info(f"Generated attention visualization for {prefix}")
            
        except Exception as e:
            logger.error(f"Error generating visualization for sample {idx}: {e}")
            continue
    
    # Cleanup hooks
    attention_rollout.cleanup()
    
    logger.info(f"Generated attention visualizations for {len(all_samples)} samples in {save_dir}")
