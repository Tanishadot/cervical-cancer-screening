#!/usr/bin/env python3
"""
Swin Transformer XAI Dashboard - Final Fixed Version
Properly handles PIL to tensor conversion for attention rollout
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
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

def safe_get_label(dataset, idx):
    """Safely get label from dataset."""
    try:
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)):
            if len(sample) >= 2:
                label = sample[1]
            else:
                return None
        else:
            return None
        
        # Handle both tensor and int cases
        if hasattr(label, 'item'):
            label = label.item()
        return label
    except Exception as e:
        return None

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
    # Debug info
    st.markdown(f'<div class="debug-info">Debug: Input image type: {type(image)}</div>', unsafe_allow_html=True)
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        st.markdown('<div class="debug-info">Debug: Converted PIL to numpy</div>', unsafe_allow_html=True)
    
    # Ensure numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be PIL Image or numpy array")
    
    st.markdown(f'<div class="debug-info">Debug: Numpy array shape: {image.shape}</div>', unsafe_allow_html=True)
    
    # Apply validation transforms
    val_transform = get_val_transforms()
    transformed = val_transform(image=image)
    tensor = transformed['image']
    
    st.markdown(f'<div class="debug-info">Debug: After transforms - tensor type: {type(tensor)}, shape: {tensor.shape}</div>', unsafe_allow_html=True)
    
    # Convert to tensor if not already
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    
    # Add batch dimension
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    # Move to device
    tensor = tensor.to(device)
    
    st.markdown(f'<div class="debug-info">Debug: Final tensor - type: {type(tensor)}, shape: {tensor.shape}, device: {tensor.device}</div>', unsafe_allow_html=True)
    
    return tensor

def create_feature_visualization(model, dataset, device, sample_idx):
    """Create feature visualization with proper tensor handling."""
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
        
        st.markdown(f'<div class="debug-info">Debug: Sample format - type: {type(sample)}, length: {len(sample) if isinstance(sample, (list, tuple)) else "N/A"}</div>', unsafe_allow_html=True)
        
        # Convert image to tensor
        image_tensor = convert_image_to_tensor(image, device)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()
        
        # Create gradient-based feature map
        image_tensor.requires_grad = True
        output = model(image_tensor)
        class_idx = torch.argmax(output, dim=1)
        
        # Backward pass to get gradients
        model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Get gradients
        gradients = image_tensor.grad.data
        gradients = gradients.abs()
        
        # Convert to numpy
        if gradients.shape[0] == 1:
            gradients = gradients[0]
        
        # Average across channels
        feature_map = gradients.mean(dim=0).cpu().numpy()
        
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
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # Create overlay
        overlay = original_img.copy()
        if overlay.shape[:2] != feature_map.shape[:2]:
            # Resize feature map to match image
            from skimage.transform import resize
            feature_map_resized = resize(feature_map, overlay.shape[:2], preserve_range=True)
        else:
            feature_map_resized = feature_map
        
        # Apply heatmap
        cmap = plt.get_cmap('jet')
        heatmap = cmap(feature_map_resized)[:, :, :3]
        
        # Blend with original
        alpha = 0.6
        overlay = (1 - alpha) * original_img + alpha * heatmap
        
        # Get label names
        if hasattr(label, 'item'):
            label = label.item()
        label_name = "Normal" if label == 0 else "Abnormal"
        pred_name = "Normal" if pred == 0 else "Abnormal"
        
        return original_img, feature_map, overlay, label_name, pred_name, confidence
        
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
    st.subheader("🔍 Feature Visualization Explorer")
    
    # Get dataset info
    normal_count = 0
    abnormal_count = 0
    
    try:
        for i in range(min(len(dataset), 50)):  # Check first 50 samples
            label = safe_get_label(dataset, i)
            if label is not None:
                if label == 0:
                    normal_count += 1
                else:
                    abnormal_count += 1
        
        st.info(f"Dataset contains {normal_count} Normal and {abnormal_count} Abnormal samples in first 50 samples")
    except Exception as e:
        st.warning(f"Could not analyze dataset: {e}")
        normal_count = abnormal_count = "N/A"
    
    # Sample selector
    sample_idx = st.slider("Select Sample Index", 0, min(len(dataset)-1, 49), 0)
    
    # Generate visualization
    if st.button("Generate Feature Visualization", type="primary"):
        with st.spinner("Generating feature visualization..."):
            original_img, feature_map, overlay, label_name, pred_name, confidence = create_feature_visualization(
                model, dataset, device, sample_idx
            )
        
        if original_img is not None:
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original Image")
                st.image(original_img, caption=f"Ground Truth: {label_name}", use_column_width=True)
            
            with col2:
                st.subheader("Feature Activation")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(feature_map, cmap='jet')
                ax.set_title("Feature Activation Map")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            
            with col3:
                st.subheader("Overlay Visualization")
                st.image(overlay, caption="Feature Overlay", use_container_width=True)
            
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
    2. **Generate Visualization**: Click the button to create feature activation maps
    3. **Analyze Results**: 
       - **Original Image**: Shows the input image with ground truth label
       - **Feature Activation**: Shows which regions the model focuses on
       - **Overlay**: Combines original image with feature activation
    4. **Interpretation**: Bright regions indicate areas the model considers important for classification
    """)
    
    # Model architecture info
    st.markdown("---")
    st.subheader("🏗️ Model Architecture")
    st.markdown("""
    - **Model**: Swin Transformer (Tiny)
    - **Patch Size**: 4x4
    - **Window Size**: 7x7
    - **Layers**: [2, 2, 6, 2] transformer blocks
    - **Embedding Dim**: [96, 192, 384, 768]
    - **Classification**: Binary (Normal vs Abnormal)
    """)

if __name__ == "__main__":
    main()
