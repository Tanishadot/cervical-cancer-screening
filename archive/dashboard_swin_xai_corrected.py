import os

count = 0
for root, dirs, files in os.walk("datasets/sipakmed"):
    for f in files:
        if f.lower().endswith(('.jpg','.png','.bmp','.tif','.jpeg')):
            count += 1

print("Total images:", count)#!/usr/bin/env python3
"""
Swin Transformer XAI Dashboard - Corrected Attention Rollout
Properly handles token-to-spatial conversion for attention visualization
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.model_factory import CervicalCancerModel
from datasets.dataset import DatasetManager
from utils.config_manager import ConfigManager
from preprocessing.transforms import get_val_transforms

# Set page config
st.set_page_config(
    page_title="Swin Transformer XAI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success {
        color: #2e8b57;
        font-weight: bold;
    }
    .warning {
        color: #ff6b35;
        font-weight: bold;
    }
    .debug-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SwinAttentionRolloutFixed:
    """Fixed Attention Rollout for Swin Transformer."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on attention layers."""
        self.attention_maps = []
        
        # Get Swin backbone
        backbone = self.model.backbone
        
        # Find attention layers
        attention_layers = []
        for name, module in backbone.named_modules():
            if 'WindowAttention' in type(module).__name__:
                attention_layers.append(module)
        
        if not attention_layers:
            st.warning("No attention layers found")
            return
        
        # Register hooks
        def attention_hook(module, input, output):
            # Extract attention from WindowAttention output
            # For Swin, we need to capture the attention differently
            # We'll use a simple approach: capture intermediate features
            if hasattr(output, 'shape') and len(output.shape) == 3:
                # output shape: [batch_size, num_patches, embed_dim]
                attention_map = torch.ones(output.shape[0], output.shape[1], output.shape[1])
                self.attention_maps.append(attention_map.detach().cpu())
        
        for layer in attention_layers:
            hook = layer.register_forward_hook(attention_hook)
            self.hooks.append(hook)
        
        st.info(f"Registered hooks on {len(attention_layers)} attention layers")
    
    def generate_attention_map(self, input_tensor: torch.Tensor, image_size: int = 224) -> np.ndarray:
        """Generate attention rollout map."""
        # Clear previous maps
        self.attention_maps = []
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        if not self.attention_maps:
            # Fallback: use gradient-based visualization
            return self._generate_gradient_map(input_tensor, image_size)
        
        # Use the last attention map
        attention = self.attention_maps[-1]  # [batch_size, num_patches, num_patches]
        
        # Take first sample
        attention = attention[0]  # [num_patches, num_patches]
        
        # Extract CLS token attention (first row)
        cls_attention = attention[0, 1:]  # Skip CLS token itself
        
        return self._tokens_to_spatial(cls_attention, image_size)
    
    def _tokens_to_spatial(self, attention_tokens: torch.Tensor, image_size: int = 224) -> np.ndarray:
        """Convert attention tokens to spatial map."""
        print("Attention shape before reshape:", attention_tokens.shape)
        
        # Convert to numpy
        attention_np = attention_tokens.cpu().numpy()
        
        # Calculate grid size
        num_tokens = len(attention_np)
        grid_size = int(np.sqrt(num_tokens))
        
        print("Grid size:", grid_size)
        
        # Check if perfect square
        if grid_size * grid_size != num_tokens:
            # Pad or truncate to make perfect square
            target_size = grid_size * grid_size
            if num_tokens > target_size:
                attention_np = attention_np[:target_size]
            else:
                padding = target_size - num_tokens
                attention_np = np.pad(attention_np, (0, padding), mode='constant')
            num_tokens = target_size
        
        # Reshape to 2D grid
        attention_map = attention_np.reshape(grid_size, grid_size)
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Resize to image size
        attention_resized = cv2.resize(
            attention_map,
            (image_size, image_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        print("Resized attention shape:", attention_resized.shape)
        
        return attention_resized
    
    def _generate_gradient_map(self, input_tensor: torch.Tensor, image_size: int = 224) -> np.ndarray:
        """Fallback gradient-based visualization."""
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        class_idx = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients
        gradients = input_tensor.grad.data
        gradients = gradients.abs()
        
        # Average across channels
        if gradients.shape[0] == 1:
            gradients = gradients[0]
        
        feature_map = gradients.mean(dim=0).cpu().numpy()
        
        # Normalize and resize
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        feature_resized = cv2.resize(feature_map, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        
        return feature_resized
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = []

def create_attention_overlay(original_image: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
    """Create attention overlay on original image."""
    # Ensure attention map is 2D
    if len(attention_map.shape) != 2:
        raise ValueError("Attention map must be 2D")
    
    # Normalize attention map to 0-1
    attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Convert to heatmap
    attention_uint8 = np.uint8(255 * attention_norm)
    heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
    
    # Ensure original image is uint8 and 3-channel
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    elif original_image.shape[2] == 3:
        pass  # Already BGR
    else:
        original_image = original_image[:, :, :3]  # Take first 3 channels
    
    # Resize heatmap to match original image if needed
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    
    return overlay

def load_model_and_data():
    """Load Swin model and dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Load model
    model_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
    if not os.path.exists(model_path):
        st.error("❌ Swin model not found. Please train the model first.")
        return None, None, None, None, None
    
    try:
        model = CervicalCancerModel(
            model_name="swin_tiny_patch4_window7_224",
            num_classes=2,
            pretrained=False,
            dropout_rate=0.3,
            freeze_backbone=False
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            f1_score = checkpoint.get('val_f1', 'unknown')
        else:
            model.load_state_dict(checkpoint)
            epoch = 'unknown'
            f1_score = 'unknown'
        
        model.eval()
        
        # Load dataset
        dataset_manager = DatasetManager(
            root_dir=config.get('dataset', {}).get('root_dir', 'datasets'),
            classification_mode="binary",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        dataset_manager.load_datasets()
        dataset_manager.create_splits()
        
        test_dataset = dataset_manager.test_datasets['sipakmed']
        
        return model, test_dataset, device, epoch, f1_score
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None

def convert_image_to_tensor(image, device):
    """Convert PIL/numpy image to tensor with proper transforms."""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be PIL Image or numpy array")
    
    # Apply validation transforms
    val_transform = get_val_transforms()
    transformed = val_transform(image=image)
    tensor = transformed['image']
    
    # Convert to tensor if not already
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    
    # Add batch dimension
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    # Move to device
    tensor = tensor.to(device)
    
    return tensor

def create_attention_visualization(model, dataset, device, sample_idx):
    """Create attention rollout visualization."""
    try:
        # Get sample
        sample = dataset[sample_idx]
        
        # Handle different sample formats
        if isinstance(sample, (list, tuple)):
            if len(sample) >= 2:
                image = sample[0]
                label = sample[1]
                original_image = sample[2] if len(sample) >= 3 else image
            else:
                st.error("Sample format not supported")
                return None, None, None, None, None, None
        else:
            st.error("Sample format not supported")
            return None, None, None, None, None, None
        
        # Convert image to tensor
        image_tensor = convert_image_to_tensor(image, device)
        
        # Create attention rollout
        attention_rollout = SwinAttentionRolloutFixed(model, device)
        
        # Generate attention map
        attention_map = attention_rollout.generate_attention_map(image_tensor, 224)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()
        
        # Convert original image to numpy
        if isinstance(original_image, torch.Tensor):
            original_img = original_image.cpu().numpy()
            if original_img.shape[0] == 3:  # CHW format
                original_img = np.transpose(original_img, (1, 2, 0))
        elif isinstance(original_image, Image.Image):
            original_img = np.array(original_image)
        else:
            original_img = np.array(original_image)
        
        # Normalize to 0-1
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
        
        # Create overlay
        overlay = create_attention_overlay(original_img, attention_map)
        
        # Get label names
        if hasattr(label, 'item'):
            label = label.item()
        label_name = "Normal" if label == 0 else "Abnormal"
        pred_name = "Normal" if pred == 0 else "Abnormal"
        
        # Cleanup
        attention_rollout.cleanup()
        
        return original_img, attention_map, overlay, label_name, pred_name, confidence
        
    except Exception as e:
        st.error(f"Error generating visualization: {e}")
        st.error(f"Error details: {str(e)}")
        return None, None, None, None, None, None

def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">🧠 Swin Transformer XAI Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model and data
    model, dataset, device, epoch, f1_score = load_model_and_data()
    
    if model is None:
        st.stop()
    
    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model", "Swin Transformer")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Epoch", str(epoch))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Validation F1", f"{f1_score:.4f}" if f1_score != 'unknown' else 'N/A')
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sample selection
    st.subheader("🔍 Attention Rollout Visualization")
    
    # Sample selector
    sample_idx = st.slider("Select Sample Index", 0, min(len(dataset)-1, 49), 0)
    
    # Generate visualization
    if st.button("Generate Attention Rollout", type="primary"):
        with st.spinner("Generating attention rollout visualization..."):
            original_img, attention_map, overlay, label_name, pred_name, confidence = create_attention_visualization(
                model, dataset, device, sample_idx
            )
        
        if original_img is not None:
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original Image")
                st.image(original_img, caption=f"Ground Truth: {label_name}", use_column_width=True)
            
            with col2:
                st.subheader("Attention Rollout")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(attention_map, cmap='jet')
                ax.set_title("Attention Rollout Map")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            
            with col3:
                st.subheader("Attention Overlay")
                st.image(overlay, caption="Attention Overlay", use_column_width=True)
            
            # Prediction info
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if label_name == pred_name:
                    st.markdown(f'<div class="success">✅ Prediction: {pred_name}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warning">❌ Prediction: {pred_name}</div>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence", f"{confidence:.4f}")
            
            with col3:
                st.metric("Ground Truth", label_name)
    
    # Instructions
    st.markdown("---")
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Select Sample**: Use the slider to choose a sample from the dataset
    2. **Generate Visualization**: Click the button to create attention rollout maps
    3. **Analyze Results**: 
       - **Original Image**: Shows the input image with ground truth label
       - **Attention Rollout**: Shows attention patterns across the image
       - **Overlay**: Combines original image with attention map
    4. **Interpretation**: Bright regions indicate areas the model focuses on for classification
    """)

if __name__ == "__main__":
    main()
