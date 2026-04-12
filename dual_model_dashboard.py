#!/usr/bin/env python3
"""
Dual-Model Medical Image Classification Dashboard
Combines CNN and Swin Transformer with comprehensive interpretability visualizations.

Author: Cervical Cancer Classification Pipeline
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
import json
from typing import Dict, List, Tuple, Optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Using matplotlib instead.")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dual_model_classifier import DualModelClassifier, create_overlay
from preprocessing.transforms import get_val_transforms
from datasets.dataset import DatasetManager
from utils.config_manager import ConfigManager
from utils.label_mapping import LabelMapper

# Set page config
st.set_page_config(
    page_title="Dual-Model Medical Image Classification",
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
    .feature-comparison {
        background: linear-gradient(45deg, #f0f9ff, #e0f2fe);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_dual_model_system():
    """Load the dual-model classification system."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model paths (adjust as needed)
        cnn_model_path = "outputs/models/best_model.pth"
        swin_model_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
        
        # Check if models exist
        if not os.path.exists(cnn_model_path):
            st.warning(f"CNN model not found at {cnn_model_path}")
            cnn_model_path = None
        
        if not os.path.exists(swin_model_path):
            st.warning(f"Swin model not found at {swin_model_path}")
            swin_model_path = None
        
        if cnn_model_path and swin_model_path:
            classifier = DualModelClassifier(cnn_model_path, swin_model_path, device)
            return classifier, device
        else:
            return None, device
    
    except Exception as e:
        st.error(f"Error loading dual-model system: {e}")
        return None, torch.device("cpu")


@st.cache_data
def load_sample_images():
    """Load sample images for testing."""
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Create dataset manager
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
        
        # Get test samples
        samples = []
        if 'sipakmed' in dataset_manager.test_datasets:
            test_dataset = dataset_manager.test_datasets['sipakmed']
            for i in range(min(10, len(test_dataset))):
                sample = test_dataset[i]
                if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    image = sample[0]
                    label = sample[1]
                    
                    # Convert to PIL Image if needed
                    if isinstance(image, torch.Tensor):
                        image_np = image.cpu().numpy()
                        if image_np.shape[0] == 3:  # CHW format
                            image_np = np.transpose(image_np, (1, 2, 0))
                        image = Image.fromarray((image_np * 255).astype(np.uint8))
                    elif isinstance(image, np.ndarray):
                        image = Image.fromarray((image * 255).astype(np.uint8))
                    
                    samples.append({
                        'image': image,
                        'label': label.item() if hasattr(label, 'item') else label,
                        'index': i
                    })
        
        return samples
    
    except Exception as e:
        st.error(f"Error loading sample images: {e}")
        return []


def preprocess_image(image: Image.Image, transform) -> torch.Tensor:
    """Preprocess image for model input."""
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)
    return image_tensor


def create_prediction_section(result: Dict):
    """Create prediction results section."""
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
        prob_normal, prob_abnormal = result['probabilities'][0]
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 1.5rem;">Probabilities</h3>
            <p style="margin: 0.5rem 0; font-size: 1.2rem;">Normal: {prob_normal:.3f}</p>
            <p style="margin: 0; font-size: 1.2rem;">Abnormal: {prob_abnormal:.3f}</p>
        </div>
        """, unsafe_allow_html=True)


def create_cnn_insights_section(result: Dict, original_image: np.ndarray):
    """Create CNN insights section with Grad-CAM."""
    st.markdown("### 🔬 CNN Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Grad-CAM Heatmap")
        
        # Create heatmap visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(result['gradcam_heatmap'], cmap='jet')
        ax.set_title("CNN Grad-CAM Heatmap", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("#### Grad-CAM Overlay")
        
        # Create overlay
        overlay = create_overlay(original_image, result['gradcam_heatmap'])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title("CNN Grad-CAM Overlay", fontsize=14, fontweight='bold')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    # CNN explanation
    st.markdown("""
    <div class="explanation-box">
        <h4>CNN Model Analysis</h4>
        <p><strong>What CNN looks for:</strong> Local features, textures, and patterns</p>
        <p><strong>Interpretation:</strong> {}</p>
        <p><strong>Individual CNN prediction:</strong> Class {}</p>
    </div>
    """.format(
        result['explanation']['cnn_focus'],
        "Normal" if result['cnn_prediction'] == 0 else "Abnormal"
    ), unsafe_allow_html=True)


def create_transformer_insights_section(result: Dict, original_image: np.ndarray):
    """Create Transformer insights section with attention maps."""
    st.markdown("### 🧠 Transformer Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Attention Map")
        
        # Create attention map visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(result['attention_map'], cmap='viridis')
        ax.set_title("Transformer Attention Map", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("#### Attention Overlay")
        
        # Create overlay
        overlay = create_overlay(original_image, result['attention_map'])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title("Transformer Attention Overlay", fontsize=14, fontweight='bold')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    # Transformer explanation
    st.markdown("""
    <div class="explanation-box">
        <h4>Transformer Model Analysis</h4>
        <p><strong>What Transformer looks for:</strong> Global relationships and spatial context</p>
        <p><strong>Interpretation:</strong> {}</p>
        <p><strong>Individual Transformer prediction:</strong> Class {}</p>
    </div>
    """.format(
        result['explanation']['swin_focus'],
        "Normal" if result['swin_prediction'] == 0 else "Abnormal"
    ), unsafe_allow_html=True)


def create_feature_comparison_section(result: Dict):
    """Create feature comparison section."""
    st.markdown("### 📊 Feature Comparison")
    
    # Feature weights comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Contribution Weights")
        
        # Create bar chart for weights
        fig = go.Figure(data=[
            go.Bar(name='CNN Weight', x=['CNN'], y=[result['cnn_weight']], marker_color='blue'),
            go.Bar(name='Transformer Weight', x=['Transformer'], y=[result['swin_weight']], marker_color='green')
        ])
        
        fig.update_layout(
            title="Feature Fusion Weights",
            xaxis_title="Model",
            yaxis_title="Weight",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Feature Dimensions")
        
        # Create feature dimension comparison
        dimensions = {
            'CNN Features': result['cnn_features_shape'][1],
            'Swin Features': result['swin_features_shape'][1],
            'Fused Features': result['fused_features_shape'][1]
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(dimensions.keys()), y=list(dimensions.values()))
        ])
        
        fig.update_layout(
            title="Feature Vector Dimensions",
            xaxis_title="Feature Type",
            yaxis_title="Dimension",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature similarity visualization
    st.markdown("#### Feature Vector Analysis")
    
    # Create heatmap of feature correlations
    cnn_feat = result['cnn_features'].flatten()
    swin_feat = result['swin_features'].flatten()
    
    # Take first 100 dimensions for visualization
    viz_size = min(100, len(cnn_feat), len(swin_feat))
    cnn_viz = cnn_feat[:viz_size]
    swin_viz = swin_feat[:viz_size]
    
    correlation_matrix = np.corrcoef(cnn_viz, swin_viz)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                xticklabels=['CNN', 'Transformer'],
                yticklabels=['CNN', 'Transformer'],
                ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()


def create_fusion_insight_section(result: Dict):
    """Create fusion insight section."""
    st.markdown("### 🔗 Fusion Insight")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Fusion Decision Process")
        
        st.markdown(f"""
        <div class="feature-comparison">
            <h4>Model Contributions</h4>
            <p><strong>CNN Weight:</strong> {result['cnn_weight']:.3f}</p>
            <p><strong>Transformer Weight:</strong> {result['swin_weight']:.3f}</p>
            <p><strong>Fused Feature Dimension:</strong> {result['fused_features_shape'][1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Decision Explanation")
        
        st.markdown(f"""
        <div class="explanation-box">
            <h4>Fusion Analysis</h4>
            <p>{result['explanation']['fusion_insight']}</p>
            <p><strong>Final Decision:</strong> Combines both models' strengths</p>
            <p><strong>Confidence Source:</strong> Weighted ensemble confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature fusion visualization
    st.markdown("#### Fused Feature Vector")
    
    # Visualize first 50 dimensions of fused features
    fused_feat = result['fused_features'].flatten()
    viz_dims = min(50, len(fused_feat))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(viz_dims)),
        y=fused_feat[:viz_dims],
        mode='lines+markers',
        name='Fused Features',
        line=dict(color='purple', width=2)
    ))
    
    fig.update_layout(
        title=f"Fused Feature Vector (First {viz_dims} Dimensions)",
        xaxis_title="Feature Dimension",
        yaxis_title="Feature Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">🧠 Dual-Model Medical Image Classification</h1>', 
                unsafe_allow_html=True)
    
    # Load dual-model system
    classifier, device = load_dual_model_system()
    
    if classifier is None:
        st.error("❌ Dual-model system could not be loaded. Please ensure both CNN and Swin models are trained and available.")
        st.info("Required models:")
        st.info("- CNN: outputs/models/best_model.pth")
        st.info("- Swin: outputs/cross_dataset_training/models/best_swin_model.pth")
        return
    
    # Load sample images
    sample_images = load_sample_images()
    
    if not sample_images:
        st.error("❌ No sample images found. Please ensure dataset is properly set up.")
        return
    
    # Sidebar for image selection
    st.sidebar.markdown("### 🖼️ Image Selection")
    
    # Image upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif']
    )
    
    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.sidebar.success("✅ Image uploaded successfully")
        use_uploaded = True
    else:
        # Use sample images
        st.sidebar.markdown("#### Or select from dataset:")
        sample_idx = st.sidebar.slider(
            "Sample Index",
            0,
            len(sample_images) - 1,
            0
        )
        
        image = sample_images[sample_idx]['image']
        true_label = sample_images[sample_idx]['label']
        st.sidebar.info(f"True Label: {'Normal' if true_label == 0 else 'Abnormal'}")
        use_uploaded = False
    
    # Display original image
    st.markdown("### 📸 Original Image")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(image, caption="Input Image", use_container_width=True)
    
    # Prediction button
    if st.button("🔍 Analyze with Dual Models", type="primary", use_container_width=True):
        with st.spinner("Processing with CNN and Transformer models..."):
            try:
                # Preprocess image
                transform = get_val_transforms()
                image_tensor = preprocess_image(image, transform)
                
                # Get prediction
                result = classifier.predict(image_tensor)
                
                # Convert image to numpy for overlays
                original_np = np.array(image)
                if original_np.max() <= 1.0:
                    original_np = (original_np * 255).astype(np.uint8)
                
                # Display results
                st.markdown("---")
                
                # Section 1: Prediction
                st.markdown("## 🎯 Prediction Results")
                create_prediction_section(result)
                
                # Section 2: CNN Insights
                st.markdown("---")
                create_cnn_insights_section(result, original_np)
                
                # Section 3: Transformer Insights
                st.markdown("---")
                create_transformer_insights_section(result, original_np)
                
                # Section 4: Feature Comparison
                st.markdown("---")
                create_feature_comparison_section(result)
                
                # Section 5: Fusion Insight
                st.markdown("---")
                create_fusion_insight_section(result)
                
                # Section 6: Summary
                st.markdown("---")
                st.markdown("## 📋 Analysis Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>Model Performance Summary</h4>
                        <p><strong>CNN Prediction:</strong> Class {result['cnn_prediction']}</p>
                        <p><strong>Transformer Prediction:</strong> Class {result['swin_prediction']}</p>
                        <p><strong>Fused Prediction:</strong> Class {result['prediction']}</p>
                        <p><strong>Final Confidence:</strong> {result['confidence']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>Interpretability Insights</h4>
                        <p><strong>CNN Focus:</strong> {result['explanation']['cnn_focus']}</p>
                        <p><strong>Transformer Focus:</strong> {result['explanation']['swin_focus']}</p>
                        <p><strong>Fusion Strategy:</strong> {result['explanation']['fusion_insight']}</p>
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
                    file_name="dual_model_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.error("Please check that both models are properly loaded and the image format is supported.")
    
    # Instructions
    st.markdown("---")
    st.markdown("## 📖 How to Use")
    
    st.markdown("""
    <div class="explanation-box">
        <h4>Dual-Model Analysis Guide</h4>
        <ol>
            <li><strong>Upload or Select Image:</strong> Choose an image from the dataset or upload your own</li>
            <li><strong>Run Analysis:</strong> Click "Analyze with Dual Models" to process with both CNN and Transformer</li>
            <li><strong>Review Results:</strong>
                <ul>
                    <li><strong>Prediction:</strong> Final classification and confidence score</li>
                    <li><strong>CNN Insights:</strong> Grad-CAM heatmap showing local feature attention</li>
                    <li><strong>Transformer Insights:</strong> Attention map showing global relationships</li>
                    <li><strong>Feature Comparison:</strong> How each model contributes to the decision</li>
                    <li><strong>Fusion Insight:</strong> How both models are combined</li>
                </ul>
            </li>
            <li><strong>Interpret for Medical Context:</strong> Use the visualizations to understand what features influenced the decision</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
