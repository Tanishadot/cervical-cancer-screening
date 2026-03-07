"""
Explainable AI (XAI) methods for cervical cancer classification.
Implements Grad-CAM, Grad-CAM++, and Attention Rollout for model interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import logging
from tqdm import tqdm

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    logging.warning("pytorch-grad-cam not available. Using custom implementation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomGradCAM:
    """
    Custom Grad-CAM implementation for models that don't work with pytorch-grad-cam.
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize custom Grad-CAM.
        
        Args:
            model: Model to analyze
            target_layer: Target layer name for feature extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate class activation map.
        
        Args:
            input_tensor: Input tensor
            class_idx: Target class index
            
        Returns:
            Class activation map
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


class AttentionRollout:
    """
    Attention Rollout for transformer models (Swin Transformer).
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention rollout.
        
        Args:
            model: Transformer model
        """
        self.model = model
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention maps."""
        def get_attention(name):
            def hook(module, input, output):
                if hasattr(output, 'attention'):
                    self.attention_maps.append(output.attention.detach())
            return hook
        
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'attention'):
                module.register_forward_hook(get_attention(name))
    
    def generate_attention_rollout(self, input_tensor: torch.Tensor, head_fusion: str = 'mean') -> np.ndarray:
        """
        Generate attention rollout visualization.
        
        Args:
            input_tensor: Input tensor
            head_fusion: Method to fuse multi-head attention ('mean', 'max', 'min')
            
        Returns:
            Attention rollout map
        """
        self.attention_maps = []
        
        # Forward pass to collect attention maps
        with torch.no_grad():
            self.model(input_tensor)
        
        if not self.attention_maps:
            logger.warning("No attention maps captured")
            return np.zeros((224, 224))
        
        # Process attention maps
        rollout_maps = []
        
        for attention_map in self.attention_maps:
            # attention_map shape: [batch_size, num_heads, seq_len, seq_len]
            attention_map = attention_map[0]  # Remove batch dimension
            
            # Fuse multi-head attention
            if head_fusion == 'mean':
                fused_attention = torch.mean(attention_map, dim=0)
            elif head_fusion == 'max':
                fused_attention = torch.max(attention_map, dim=0)[0]
            elif head_fusion == 'min':
                fused_attention = torch.min(attention_map, dim=0)[0]
            else:
                fused_attention = attention_map[0]  # Use first head
            
            rollout_maps.append(fused_attention)
        
        # Compute attention rollout (matrix multiplication of attention matrices)
        if len(rollout_maps) > 1:
            rollout = rollout_maps[0]
            for i in range(1, len(rollout_maps)):
                rollout = torch.matmul(rollout_maps[i], rollout)
        else:
            rollout = rollout_maps[0]
        
        # Get CLS token attention (first token)
        cls_attention = rollout[0, 1:]  # Remove CLS token itself
        
        # Reshape to spatial dimensions (assuming square grid)
        seq_len = cls_attention.shape[0]
        grid_size = int(np.sqrt(seq_len))
        
        if grid_size * grid_size == seq_len:
            attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        else:
            # Fallback: reshape as best as possible
            attention_map = cls_attention[:grid_size*grid_size].reshape(grid_size, grid_size).cpu().numpy()
        
        # Normalize
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        return attention_map


class XAIAnalyzer:
    """
    Main XAI analyzer that combines multiple methods.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = "outputs/xai"
    ):
        """
        Initialize XAI analyzer.
        
        Args:
            model: Model to analyze
            device: Device for computation
            output_dir: Output directory for visualizations
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize XAI methods
        self.grad_cam = None
        self.grad_cam_plus = None
        self.attention_rollout = None
        
        self._setup_xai_methods()
    
    def _setup_xai_methods(self):
        """Setup XAI methods based on model type."""
        model_name = type(self.model).__name__.lower()
        
        # Setup Grad-CAM
        target_layers = self._find_target_layers()
        
        if GRAD_CAM_AVAILABLE and target_layers:
            try:
                self.grad_cam = GradCAM(model=self.model, target_layers=target_layers)
                self.grad_cam_plus = GradCAMPlusPlus(model=self.model, target_layers=target_layers)
                logger.info("Grad-CAM and Grad-CAM++ initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Grad-CAM: {e}")
                self.grad_cam = None
                self.grad_cam_plus = None
        
        # Setup Attention Rollout for transformers
        if 'swin' in model_name or 'transformer' in model_name:
            self.attention_rollout = AttentionRollout(self.model)
            logger.info("Attention Rollout initialized")
    
    def _find_target_layers(self) -> List:
        """Find suitable target layers for Grad-CAM."""
        target_layers = []
        
        # Common target layer names
        layer_names = [
            'backbone.features',
            'backbone.layer4',
            'backbone.stages',
            'features',
            'layer4'
        ]
        
        for name, module in self.model.named_modules():
            for layer_name in layer_names:
                if layer_name in name and isinstance(module, (nn.Conv2d, nn.Sequential)):
                    target_layers.append(module)
                    break
        
        return target_layers
    
    def generate_visualizations(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        predicted_class: int,
        true_class: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        save_name: str = "xai_analysis"
    ) -> Dict[str, np.ndarray]:
        """
        Generate XAI visualizations for an image.
        
        Args:
            image: Original image (H, W, 3)
            input_tensor: Preprocessed input tensor
            predicted_class: Predicted class index
            true_class: True class index (optional)
            class_names: Class names
            save_name: Base name for saving visualizations
            
        Returns:
            Dictionary of visualization maps
        """
        visualizations = {}
        
        # Prepare target for Grad-CAM
        targets = [ClassifierOutputTarget(predicted_class)]
        
        # Generate Grad-CAM
        if self.grad_cam is not None:
            try:
                grayscale_cam = self.grad_cam(input_tensor=input_tensor, targets=targets)
                visualizations['grad_cam'] = grayscale_cam[0]
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")
        
        # Generate Grad-CAM++
        if self.grad_cam_plus is not None:
            try:
                grayscale_cam_plus = self.grad_cam_plus(input_tensor=input_tensor, targets=targets)
                visualizations['grad_cam_plus'] = grayscale_cam_plus[0]
            except Exception as e:
                logger.warning(f"Grad-CAM++ failed: {e}")
        
        # Generate Attention Rollout
        if self.attention_rollout is not None:
            try:
                attention_map = self.attention_rollout.generate_attention_rollout(input_tensor)
                visualizations['attention_rollout'] = attention_map
            except Exception as e:
                logger.warning(f"Attention Rollout failed: {e}")
        
        # Save visualizations
        self._save_visualizations(
            image, visualizations, predicted_class, true_class, class_names, save_name
        )
        
        return visualizations
    
    def _save_visualizations(
        self,
        image: np.ndarray,
        visualizations: Dict[str, np.ndarray],
        predicted_class: int,
        true_class: Optional[int],
        class_names: Optional[List[str]],
        save_name: str
    ):
        """Save XAI visualizations."""
        # Create colormap
        colormap = LinearSegmentedColormap.from_list('custom', ['blue', 'green', 'yellow', 'red'])
        
        # Determine number of subplots
        num_methods = len(visualizations)
        if num_methods == 0:
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, num_methods + 1, figsize=(4 * (num_methods + 1), 4))
        if num_methods == 1:
            axes = [axes[0], axes[1]]
        else:
            axes = [axes[0]] + list(axes[1:])
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # XAI visualizations
        for i, (method_name, cam) in enumerate(visualizations.items()):
            # Resize CAM to match image size
            if cam.shape != image.shape[:2]:
                cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
            
            # Create visualization
            visualization = show_cam_on_image(image / 255.0, cam, use_rgb=True)
            
            axes[i + 1].imshow(visualization)
            
            # Title with class information
            if class_names:
                pred_name = class_names[predicted_class]
                title = f'{method_name.replace("_", " ").title()}\nPred: {pred_name}'
                if true_class is not None:
                    true_name = class_names[true_class]
                    title += f'\nTrue: {true_name}'
            else:
                title = f'{method_name.replace("_", " ").title()}\nClass: {predicted_class}'
            
            axes[i + 1].set_title(title)
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save individual heatmaps
        for method_name, cam in visualizations.items():
            if cam.shape != image.shape[:2]:
                cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cam, cmap=colormap)
            plt.colorbar(label='Activation Intensity')
            plt.title(f'{method_name.replace("_", " ").title()} Heatmap')
            plt.axis('off')
            
            heatmap_path = os.path.join(self.output_dir, f'{save_name}_{method_name}_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_dataset(
        self,
        data_loader,
        class_names: Optional[List[str]] = None,
        num_samples: int = 20,
        save_prefix: str = "dataset_xai"
    ) -> Dict:
        """
        Generate XAI visualizations for dataset samples.
        
        Args:
            data_loader: Data loader
            class_names: Class names
            num_samples: Number of samples to analyze
            save_prefix: Prefix for saving files
            
        Returns:
            Analysis results
        """
        self.model.eval()
        results = {
            'samples_analyzed': 0,
            'class_distribution': {},
            'visualization_files': []
        }
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels, _, image_paths) in enumerate(tqdm(data_loader, desc="Generating XAI visualizations")):
                if sample_count >= num_samples:
                    break
                
                images = images.to(self.device)
                
                for i in range(min(images.shape[0], num_samples - sample_count)):
                    if sample_count >= num_samples:
                        break
                    
                    # Get single image and label
                    single_image = images[i:i+1]
                    single_label = labels[i].item()
                    single_path = image_paths[i]
                    
                    # Get prediction
                    output = self.model(single_image)
                    predicted_class = torch.argmax(output, dim=1).item()
                    
                    # Convert tensor back to numpy for visualization
                    image_np = single_image[0].cpu().numpy()
                    image_np = np.transpose(image_np, (1, 2, 0))  # CHW to HWC
                    image_np = (image_np * 255).astype(np.uint8)  # Denormalize
                    
                    # Generate visualizations
                    save_name = f"{save_prefix}_sample_{sample_count:04d}"
                    visualizations = self.generate_visualizations(
                        image=image_np,
                        input_tensor=single_image,
                        predicted_class=predicted_class,
                        true_class=single_label,
                        class_names=class_names,
                        save_name=save_name
                    )
                    
                    # Update results
                    results['samples_analyzed'] += 1
                    results['visualization_files'].append({
                        'sample_idx': sample_count,
                        'image_path': single_path,
                        'true_class': single_label,
                        'predicted_class': predicted_class,
                        'save_name': save_name,
                        'methods_used': list(visualizations.keys())
                    })
                    
                    # Update class distribution
                    class_name = class_names[single_label] if class_names else str(single_label)
                    results['class_distribution'][class_name] = results['class_distribution'].get(class_name, 0) + 1
                    
                    sample_count += 1
        
        # Save results
        results_path = os.path.join(self.output_dir, f'{save_prefix}_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"XAI analysis completed for {results['samples_analyzed']} samples")
        logger.info(f"Results saved to {results_path}")
        
        return results


def create_xai_analyzer(
    model: nn.Module,
    device: torch.device,
    output_dir: str = "outputs/xai"
) -> XAIAnalyzer:
    """
    Create XAI analyzer instance.
    
    Args:
        model: Model to analyze
        device: Device for computation
        output_dir: Output directory
        
    Returns:
        XAI analyzer instance
    """
    return XAIAnalyzer(model, device, output_dir)


if __name__ == "__main__":
    # Example usage
    from models.model_factory import create_model
    
    # Create model
    model = create_model('efficientnet_b0', num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create XAI analyzer
    analyzer = create_xai_analyzer(model, device)
    
    print("XAI analyzer ready for use")
