"""
Attention Rollout Implementation for Swin Transformer Models

Based on "Quantifying Attention Flow in Transformers" by Abnar & Zuidema
https://arxiv.org/abs/2005.00928

This implementation extracts and visualizes attention flow through Swin Transformer layers.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionRolloutAnalyzer:
    """
    Analyzes attention flow in Swin Transformer models using attention rollout.
    
    Attention rollout captures the cumulative attention across all layers,
    providing insights into which parts of the image the model focuses on.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the attention rollout analyzer.
        
        Args:
            model: Swin Transformer model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.attention_maps = []
        self.hooks = []
        
        # Register hooks to capture attention maps
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps from all Swin blocks."""
        try:
            # Find all Swin Transformer blocks
            for name, module in self.model.named_modules():
                if 'swin' in name.lower() and 'attn' in name.lower():
                    # Register hook for attention output
                    hook = module.register_forward_hook(self._attention_hook)
                    self.hooks.append(hook)
                    logger.info(f"Registered attention hook for: {name}")
        except Exception as e:
            logger.error(f"Error registering hooks: {e}")
    
    def _attention_hook(self, module, input, output):
        """Hook function to capture attention maps."""
        # Swin Transformer attention output shape: [B, H*W, H*W, num_heads]
        if isinstance(output, tuple) and len(output) > 0:
            attention = output[0]  # Get attention weights
        else:
            attention = output
        
        # Average across heads and move to CPU
        if attention.dim() == 4:  # [B, N, N, num_heads]
            attention = attention.mean(dim=-1)  # Average across heads
        
        self.attention_maps.append(attention.detach().cpu())
    
    def _reset_attention_maps(self):
        """Reset stored attention maps for new forward pass."""
        self.attention_maps = []
    
    def compute_attention_rollout(self, input_tensor):
        """
        Compute attention rollout for a given input.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            attention_rollout: Cumulative attention map [B, H, W]
        """
        self._reset_attention_maps()
        
        # Forward pass to collect attention maps
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if not self.attention_maps:
            logger.warning("No attention maps captured. Check model architecture.")
            return None
        
        # Get batch size and sequence length
        batch_size = self.attention_maps[0].shape[0]
        seq_len = self.attention_maps[0].shape[1]
        
        # Initialize rollout with identity matrix
        rollout = torch.eye(seq_len, device=self.device)
        
        # Process each layer's attention
        for layer_idx, attention in enumerate(self.attention_maps):
            # Move attention to device
            attention = attention.to(self.device)
            
            # Average across batch (use first sample for visualization)
            attention_avg = attention[0]  # [seq_len, seq_len]
            
            # Normalize attention
            attention_norm = F.normalize(attention_avg, p=1, dim=1)
            
            # Add residual connection (identity matrix)
            attention_residual = 0.5 * attention_norm + 0.5 * torch.eye(seq_len, device=self.device)
            
            # Update rollout: multiply with previous rollout
            rollout = torch.matmul(attention_residual, rollout)
            
            logger.debug(f"Layer {layer_idx}: attention shape {attention_avg.shape}")
        
        # Get class token attention (CLS token is usually first token)
        class_token_attention = rollout[0, 1:]  # Exclude CLS token itself
        
        # Reshape to spatial dimensions for Swin Transformer
        # Swin uses patch-based attention, so we need to reconstruct spatial layout
        spatial_attention = self._reshape_tokens_to_spatial(class_token_attention)
        
        return spatial_attention
    
    def _reshape_tokens_to_spatial(self, token_attention):
        """
        Reshape token attention to spatial map for Swin Transformer.
        
        Args:
            token_attention: Attention weights for patches [num_patches]
            
        Returns:
            spatial_attention: Spatial attention map [H, W]
        """
        # For Swin Transformer, we need to determine patch grid size
        # Assuming square patches, calculate grid dimensions
        num_patches = token_attention.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size != num_patches:
            # If not square, try to find best fit
            grid_size = int(np.ceil(np.sqrt(num_patches)))
            # Pad with zeros if needed
            pad_size = grid_size * grid_size - num_patches
            if pad_size > 0:
                token_attention = F.pad(token_attention, (0, pad_size), 'constant', 0)
        
        # Reshape to spatial grid
        spatial_attention = token_attention[:grid_size*grid_size].reshape(grid_size, grid_size)
        
        return spatial_attention
    
    def create_visualization(self, original_image, attention_map, save_dir, sample_idx):
        """
        Create and save attention rollout visualizations.
        
        Args:
            original_image: Original PIL Image
            attention_map: Attention rollout tensor [H, W]
            save_dir: Directory to save visualizations
            sample_idx: Sample index for naming
        """
        try:
            # Convert attention to numpy and normalize
            if isinstance(attention_map, torch.Tensor):
                attention_np = attention_map.detach().cpu().numpy()
            else:
                attention_np = attention_map
            
            # Normalize attention to [0, 1]
            attention_norm = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
            
            # Resize attention to match original image
            original_size = original_image.size
            attention_resized = cv2.resize(
                attention_norm, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Create heatmap using matplotlib
            plt.figure(figsize=(12, 4))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title('Original Image', fontsize=12, fontweight='bold')
            plt.axis('off')
            
            # Attention heatmap
            plt.subplot(1, 3, 2)
            heatmap = plt.imshow(attention_resized, cmap='jet', vmin=0, vmax=1)
            plt.colorbar(heatmap, fraction=0.046, pad=0.04)
            plt.title('Attention Rollout Heatmap', fontsize=12, fontweight='bold')
            plt.axis('off')
            
            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(original_image)
            plt.imshow(attention_resized, cmap='jet', alpha=0.4, vmin=0, vmax=1)
            plt.title('Attention Overlay', fontsize=12, fontweight='bold')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save combined visualization
            combined_path = os.path.join(save_dir, f'rollout_{sample_idx}.png')
            plt.savefig(combined_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save individual images
            # Original
            orig_path = os.path.join(save_dir, f'rollout_{sample_idx}_original.png')
            original_image.save(orig_path)
            
            # Heatmap
            heatmap_path = os.path.join(save_dir, f'rollout_{sample_idx}_heatmap.png')
            plt.figure(figsize=(6, 6))
            plt.imshow(attention_resized, cmap='jet', vmin=0, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('Attention Rollout Heatmap', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Overlay
            overlay_path = os.path.join(save_dir, f'rollout_{sample_idx}_overlay.png')
            plt.figure(figsize=(6, 6))
            plt.imshow(original_image)
            plt.imshow(attention_resized, cmap='jet', alpha=0.4, vmin=0, vmax=1)
            plt.title('Attention Overlay', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved attention rollout visualizations for sample {sample_idx}")
            
            return {
                'original': orig_path,
                'heatmap': heatmap_path,
                'overlay': overlay_path,
                'combined': combined_path,
                'attention_map': attention_np
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def cleanup(self):
        """Clean up registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def create_attention_rollout_analyzer(model, device='cpu'):
    """
    Factory function to create attention rollout analyzer.
    
    Args:
        model: Swin Transformer model
        device: Device for computation
        
    Returns:
        AttentionRolloutAnalyzer instance
    """
    return AttentionRolloutAnalyzer(model, device)


def analyze_dataset(data_loader, model, device, save_dir, num_samples=10):
    """
    Analyze dataset using attention rollout.
    
    Args:
        data_loader: PyTorch data loader
        model: Swin Transformer model
        device: Device for computation
        save_dir: Directory to save visualizations
        num_samples: Number of samples to analyze
    """
    try:
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create analyzer
        analyzer = create_attention_rollout_analyzer(model, device)
        
        logger.info(f"Analyzing {num_samples} samples with attention rollout...")
        
        sample_metadata = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, *_) in enumerate(data_loader):
                for sample_idx, image in enumerate(images):
                    if len(sample_metadata) >= num_samples:
                        break
                    
                    # Move image to device
                    image = image.unsqueeze(0).to(device)  # Add batch dimension
                    
                    # Compute attention rollout
                    attention_map = analyzer.compute_attention_rollout(image)
                    
                    if attention_map is not None:
                        # Convert tensor back to PIL Image for visualization
                        image_np = image.squeeze(0).cpu().numpy()
                        # Denormalize if needed (assuming ImageNet normalization)
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        image_np = image_np * std[:, None, None] + mean[:, None, None]
                        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                        image_np = np.transpose(image_np, (1, 2, 0))
                        original_image = Image.fromarray(image_np)
                        
                        # Create visualization
                        viz_result = analyzer.create_visualization(
                            original_image, attention_map, save_dir, 
                            len(sample_metadata) + 1
                        )
                        
                        if viz_result:
                            sample_metadata.append({
                                'sample_idx': len(sample_metadata) + 1,
                                'original_path': viz_result['original'],
                                'heatmap_path': viz_result['heatmap'],
                                'overlay_path': viz_result['overlay'],
                                'true_label': 'UNKNOWN',  # Would need to be provided
                                'attention_map': viz_result['attention_map']
                            })
                
                if len(sample_metadata) >= num_samples:
                    break
        
        # Cleanup
        analyzer.cleanup()
        
        # Save metadata
        metadata_path = os.path.join(save_dir, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        
        logger.info(f"Attention rollout analysis completed. Results saved to {save_dir}")
        return sample_metadata
        
    except Exception as e:
        logger.error(f"Error in dataset analysis: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    from models.model_factory import CervicalCancerModel
    from utils.config_manager import ConfigManager
    
    # Load model
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    model = CervicalCancerModel(
        model_name='swin_transformer',
        num_classes=2,
        pretrained=False,
        dropout_rate=0.1,
        freeze_backbone=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create analyzer
    analyzer = create_attention_rollout_analyzer(model, device)
    print("Attention Rollout Analyzer created successfully!")
