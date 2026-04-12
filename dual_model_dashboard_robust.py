#!/usr/bin/env python3
"""
Robust Dual-Model Medical Image Classification Dashboard
Focus on stability and clean visualizations for medical demo.

Author: Cervical Cancer Classification Pipeline
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dual_model_classifier import DualModelClassifier, create_overlay
from preprocessing.transforms import get_val_transforms

# Set page config
st.set_page_config(
    page_title="Dual-Model Medical Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .model-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    .success {
        color: #10b981;
        font-weight: bold;
    }
    .warning {
        color: #f59e0b;
        font-weight: bold;
    }
    .error {
        color: #ef4444;
        font-weight: bold;
    }
    .explanation-box {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_dual_model_system():
    """Load dual-model classification system with error handling."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🖥️ Using device: {device}")
        
        # Model paths
        cnn_model_path = "outputs/models/best_model.pth"
        swin_model_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
        
        # Check if models exist
        if not os.path.exists(cnn_model_path):
            st.warning(f"⚠️ CNN model not found at {cnn_model_path}")
            st.info("Creating dummy CNN model for demo...")
            cnn_model_path = None
        
        if not os.path.exists(swin_model_path):
            st.warning(f"⚠️ Swin model not found at {swin_model_path}")
            st.info("Creating dummy Swin model for demo...")
            swin_model_path = None
        
        if cnn_model_path and swin_model_path:
            classifier = DualModelClassifier(cnn_model_path, swin_model_path, device)
            return classifier, device, "✅ Real models loaded"
        else:
            # Create dummy models for demo
            from models.model_factory import CervicalCancerModel
            
            os.makedirs("outputs/models", exist_ok=True)
            os.makedirs("outputs/cross_dataset_training/models", exist_ok=True)
            
            # Create and save dummy CNN model
            cnn_model = CervicalCancerModel(
                model_name="efficientnet_b0.ra_in1k",
                num_classes=2,
                pretrained=False,
                dropout_rate=0.3,
                freeze_backbone=False
            )
            torch.save(cnn_model.state_dict(), "outputs/models/best_model.pth")
            
            # Create and save dummy Swin model
            swin_model = CervicalCancerModel(
                model_name="swin_tiny_patch4_window7_224.ms_in1k",
                num_classes=2,
                pretrained=False,
                dropout_rate=0.3,
                freeze_backbone=False
            )
            torch.save(swin_model.state_dict(), "outputs/cross_dataset_training/models/best_swin_model.pth")
            
            classifier = DualModelClassifier("outputs/models/best_model.pth", 
                                       "outputs/cross_dataset_training/models/best_swin_model.pth", 
                                       device)
            return classifier, device, "🎭 Demo models created"
    
    except Exception as e:
        st.error(f"❌ Error loading dual-model system: {e}")
        return None, torch.device("cpu"), f"❌ Error: {e}"


def preprocess_image_for_demo(image: Image.Image) -> torch.Tensor:
    """Preprocess image with error handling."""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard size
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Apply transforms
        transform = get_val_transforms()
        image_np = np.array(image)
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)
        
        return image_tensor
    
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {e}")
        return None


def create_matplotlib_visualization(result: Dict, original_image: np.ndarray):
    """Create clean matplotlib visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dual-Model Analysis Results', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # CNN Grad-CAM heatmap
    if 'gradcam_heatmap' in result:
        im1 = axes[0, 1].imshow(result['gradcam_heatmap'], cmap='jet')
        axes[0, 1].set_title('CNN Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # CNN Grad-CAM overlay
    if 'gradcam_heatmap' in result:
        overlay1 = create_overlay(original_image, result['gradcam_heatmap'])
        axes[0, 2].imshow(overlay1)
        axes[0, 2].set_title('CNN Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
    
    # Swin attention map
    if 'attention_map' in result:
        im2 = axes[1, 0].imshow(result['attention_map'], cmap='viridis')
        axes[1, 0].set_title('Transformer Attention Map', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Swin attention overlay
    if 'attention_map' in result:
        overlay2 = create_overlay(original_image, result['attention_map'])
        axes[1, 1].imshow(overlay2)
        axes[1, 1].set_title('Transformer Attention Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
    
    # Feature comparison
    cnn_weight = result.get('cnn_weight', 0.5)
    swin_weight = result.get('swin_weight', 0.5)
    
    models = ['CNN', 'Transformer']
    weights = [cnn_weight, swin_weight]
    colors = ['blue', 'green']
    
    bars = axes[1, 2].bar(models, weights, color=colors, alpha=0.7)
    axes[1, 2].set_title('Model Contribution Weights', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Weight')
    axes[1, 2].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_debug_info_section(result: Dict):
    """Create debug information section."""
    st.markdown("### 🔍 Debug Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>Feature Shapes</h4>
            <p><strong>CNN Features:</strong> {}</p>
            <p><strong>Swin Features:</strong> {}</p>
            <p><strong>Fused Features:</strong> {}</p>
        </div>
        """.format(
            result.get('cnn_features_shape', 'N/A'),
            result.get('swin_features_shape', 'N/A'),
            result.get('fused_features_shape', 'N/A')
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h4>Model Predictions</h4>
            <p><strong>CNN Prediction:</strong> Class {}</p>
            <p><strong>Swin Prediction:</strong> Class {}</p>
            <p><strong>Final Prediction:</strong> Class {}</p>
        </div>
        """.format(
            result.get('cnn_prediction', 'N/A'),
            result.get('swin_prediction', 'N/A'),
            result.get('prediction', 'N/A')
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-card">
            <h4>Quality Metrics</h4>
            <p><strong>Confidence:</strong> {:.3f}</p>
            <p><strong>CNN Weight:</strong> {:.3f}</p>
            <p><strong>Swin Weight:</strong> {:.3f}</p>
        </div>
        """.format(
            result.get('confidence', 0.0),
            result.get('cnn_weight', 0.0),
            result.get('swin_weight', 0.0)
        ), unsafe_allow_html=True)


def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">🧠 Dual-Model Medical Classification</h1>', 
                unsafe_allow_html=True)
    
    # Load dual-model system
    classifier, device, status = load_dual_model_system()
    
    if classifier is None:
        st.error("❌ Dual-model system could not be loaded. Please check the error messages above.")
        return
    
    st.success(f"✅ System Status: {status}")
    
    # Sidebar for image input
    st.sidebar.markdown("### 🖼️ Image Input")
    
    # Image upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload medical image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
        help="Upload a medical image for dual-model analysis"
    )
    
    # Demo image option
    use_demo = st.sidebar.checkbox("Use demo image", value=True)
    
    if use_demo:
        # Create a simple demo image
        demo_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        demo_image = Image.fromarray(demo_image)
        st.sidebar.info("🎭 Using random demo image")
    else:
        demo_image = None
    
    # Process image
    image_to_analyze = None
    if uploaded_file is not None:
        image_to_analyze = Image.open(uploaded_file).convert('RGB')
        st.sidebar.success("✅ Image uploaded successfully")
    elif demo_image is not None:
        image_to_analyze = demo_image
    
    if image_to_analyze is not None:
        # Display original image
        st.markdown("### 📸 Input Image")
        st.image(image_to_analyze, caption="Input Medical Image", width=400)
        
        # Analysis button
        if st.button("🔍 Analyze with Dual Models", type="primary", use_container_width=True):
            with st.spinner("Processing with CNN and Transformer models..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image_for_demo(image_to_analyze)
                    
                    if image_tensor is None:
                        st.error("❌ Failed to preprocess image")
                        return
                    
                    # Get prediction
                    result = classifier.predict(image_tensor)
                    
                    # Convert image to numpy for overlays
                    original_np = np.array(image_to_analyze)
                    if original_np.max() <= 1.0:
                        original_np = (original_np * 255).astype(np.uint8)
                    
                    # Display results
                    st.markdown("---")
                    
                    # Section 1: Prediction Results
                    st.markdown("## 🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        class_name = "Normal" if result['prediction'] == 0 else "Abnormal"
                        confidence_color = "success" if result['confidence'] > 0.8 else "warning" if result['confidence'] > 0.6 else "error"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin: 0; font-size: 1.5rem;">Prediction</h3>
                            <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{class_name}</p>
                            <p style="margin: 0; opacity: 0.9;">Class {result['prediction']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin: 0; font-size: 1.5rem;">Confidence</h3>
                            <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{result['confidence']:.3f}</p>
                            <p style="margin: 0; opacity: 0.9;">Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        prob_normal, prob_abnormal = result['probabilities'][0], result['probabilities'][1]
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin: 0; font-size: 1.5rem;">Probabilities</h3>
                            <p style="margin: 0.5rem 0; font-size: 1.2rem;">Normal: {prob_normal:.3f}</p>
                            <p style="margin: 0; font-size: 1.2rem;">Abnormal: {prob_abnormal:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Section 2: Visualizations
                    st.markdown("---")
                    st.markdown("## 🔬 Model Analysis")
                    
                    # Create comprehensive visualization
                    fig = create_matplotlib_visualization(result, original_np)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Section 3: Debug Info
                    st.markdown("---")
                    create_debug_info_section(result)
                    
                    # Section 4: Explanations
                    st.markdown("---")
                    st.markdown("### 📋 Model Explanations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="explanation-box">
                            <h4>CNN Analysis</h4>
                            <p><strong>Focus:</strong> {result['explanation']['cnn_focus']}</p>
                            <p><strong>Individual Prediction:</strong> Class {result['cnn_prediction']}</p>
                            <p><strong>Method:</strong> Grad-CAM (local feature attention)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="explanation-box">
                            <h4>Transformer Analysis</h4>
                            <p><strong>Focus:</strong> {result['explanation']['swin_focus']}</p>
                            <p><strong>Individual Prediction:</strong> Class {result['swin_prediction']}</p>
                            <p><strong>Method:</strong> Attention maps (global relationships)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Section 5: Fusion Insight
                    st.markdown("---")
                    st.markdown("### 🔗 Fusion Analysis")
                    
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>Decision Process</h4>
                        <p><strong>Fusion Strategy:</strong> {result['explanation']['fusion_insight']}</p>
                        <p><strong>CNN Contribution:</strong> {result['cnn_weight']:.3f}</p>
                        <p><strong>Transformer Contribution:</strong> {result['swin_weight']:.3f}</p>
                        <p><strong>Combined Features:</strong> {result['fused_features_shape'][1]} dimensions</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download results
                    st.markdown("---")
                    st.markdown("### 💾 Export Results")
                    
                    results_json = {
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'probabilities': result['probabilities'],
                        'cnn_features_shape': result['cnn_features_shape'],
                        'swin_features_shape': result['swin_features_shape'],
                        'fused_features_shape': result['fused_features_shape'],
                        'explanation': result['explanation']
                    }
                    
                    st.download_button(
                        label="Download Results as JSON",
                        data=json.dumps(results_json, indent=2),
                        file_name="dual_model_analysis.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    st.error("Please check that both models are properly loaded and image format is supported.")
    
    # Instructions
    st.markdown("---")
    st.markdown("## 📖 How to Use")
    
    st.markdown("""
    <div class="explanation-box">
        <h4>Dual-Model Analysis Guide</h4>
        <ol>
            <li><strong>Upload Image:</strong> Choose a medical image from your device</li>
            <li><strong>Or Use Demo:</strong> Toggle demo mode for quick testing</li>
            <li><strong>Analyze:</strong> Click "Analyze with Dual Models" to process</li>
            <li><strong>Review Results:</strong>
                <ul>
                    <li><strong>Prediction:</strong> Final classification and confidence</li>
                    <li><strong>CNN Grad-CAM:</strong> Local feature attention heatmap</li>
                    <li><strong>Transformer Attention:</strong> Global relationship map</li>
                    <li><strong>Fusion Analysis:</strong> How models combine decisions</li>
                </ul>
            </li>
            <li><strong>Medical Interpretation:</strong> Use visualizations to understand AI reasoning</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
