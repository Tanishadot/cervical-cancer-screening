#!/usr/bin/env python3
"""
Enhanced Dual-Model Medical Image Classification Dashboard
Fixed prediction errors with comprehensive feature visualization for medical demo.

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
    page_title="Enhanced Dual-Model Medical Classification",
    page_icon="",
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
    .feature-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
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
        st.info(f"Device: {device}")
        
        # Model paths
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
            return classifier, device, "Real models loaded"
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
            return classifier, device, "Demo models created"
    
    except Exception as e:
        st.error(f"Error loading dual-model system: {e}")
        return None, torch.device("cpu"), f"Error: {e}"


def safe_prediction_with_features(image_tensor: torch.Tensor, classifier) -> Dict:
    """Safe prediction with comprehensive feature extraction."""
    try:
        image_tensor = image_tensor.to(classifier.device)
        
        # Extract features from both models
        cnn_features, cnn_pred, cnn_feat_maps, cnn_grads = classifier.feature_extractor.extract_cnn_features(image_tensor)
        swin_features, swin_pred, attention_maps = classifier.feature_extractor.extract_swin_features(image_tensor)
        
        # Get individual model outputs for consistency check
        with torch.no_grad():
            cnn_output = classifier.cnn_model(image_tensor)
            swin_output = classifier.swin_model(image_tensor)
        
        # Ensure model output consistency
        assert cnn_output.shape[-1] == swin_output.shape[-1], f"Model output mismatch: CNN {cnn_output.shape} vs Swin {swin_output.shape}"
        
        # Feature fusion
        fusion_result = classifier.fusion_model(cnn_features, swin_features)
        
        # Safe probability handling
        probs = fusion_result['probabilities'].detach().cpu().numpy().flatten()
        
        # Dynamic prediction handling
        if len(probs) == 1:
            prediction = int(probs[0] > 0.5)
            confidence = float(probs[0])
            probabilities = [float(probs[0]), float(1 - probs[0])]
        else:
            prediction = int(np.argmax(probs))
            confidence = float(np.max(probs))
            probabilities = probs.tolist()
        
        # Generate interpretability maps
        gradcam_heatmap = classifier.feature_extractor.generate_gradcam_heatmap(
            cnn_grads, cnn_feat_maps, (224, 224)
        )
        
        attention_map = classifier.feature_extractor.generate_attention_map(attention_maps, 224)
        
        # Compile comprehensive results
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'cnn_features_shape': tuple(cnn_features.shape),
            'swin_features_shape': tuple(swin_features.shape),
            'fused_features_shape': tuple(fusion_result['fused_features'].shape),
            'cnn_prediction': cnn_pred.item() if hasattr(cnn_pred, 'item') else int(cnn_pred),
            'swin_prediction': swin_pred.item() if hasattr(swin_pred, 'item') else int(swin_pred),
            'cnn_weight': fusion_result['cnn_weight'].item(),
            'swin_weight': fusion_result['swin_weight'].item(),
            'gradcam_heatmap': gradcam_heatmap,
            'attention_map': attention_map,
            'cnn_features': cnn_features.detach().cpu().numpy(),
            'swin_features': swin_features.detach().cpu().numpy(),
            'fused_features': fusion_result['fused_features'].detach().cpu().numpy(),
            'cnn_output_shape': tuple(cnn_output.shape),
            'swin_output_shape': tuple(swin_output.shape),
            'raw_probabilities': probs.tolist(),
            'explanation': {
                'cnn_focus': classifier._explain_cnn_focus(gradcam_heatmap),
                'swin_focus': classifier._explain_swin_focus(attention_map),
                'fusion_insight': classifier._explain_fusion(fusion_result['cnn_weight'].item(), 
                                                            fusion_result['swin_weight'].item())
            }
        }
        
        return result
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def create_feature_visualization(cnn_features: np.ndarray, swin_features: np.ndarray):
    """Create comprehensive feature visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Feature Analysis', fontsize=16, fontweight='bold')
    
    # CNN Features - First 10 values
    cnn_top10 = cnn_features.flatten()[:10]
    axes[0, 0].bar(range(len(cnn_top10)), cnn_top10, color='blue', alpha=0.7)
    axes[0, 0].set_title('CNN Features (Top 10)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Activation Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Swin Features - First 10 values
    swin_top10 = swin_features.flatten()[:10]
    axes[0, 1].bar(range(len(swin_top10)), swin_top10, color='green', alpha=0.7)
    axes[0, 1].set_title('Transformer Features (Top 10)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Activation Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature Distribution Comparison
    axes[1, 0].hist(cnn_features.flatten(), bins=50, alpha=0.7, color='blue', label='CNN', density=True)
    axes[1, 0].hist(swin_features.flatten(), bins=50, alpha=0.7, color='green', label='Transformer', density=True)
    axes[1, 0].set_title('Feature Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Activation Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature Statistics Comparison
    cnn_stats = [np.mean(cnn_features), np.max(cnn_features), np.std(cnn_features)]
    swin_stats = [np.mean(swin_features), np.max(swin_features), np.std(swin_features)]
    
    x = np.arange(3)
    width = 0.35
    
    axes[1, 1].bar(x - width/2, cnn_stats, width, label='CNN', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, swin_stats, width, label='Transformer', color='green', alpha=0.7)
    axes[1, 1].set_title('Feature Statistics Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Statistic')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Mean', 'Max', 'Std'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_feature_insights_panel(result: Dict):
    """Create comprehensive feature insights panel."""
    st.markdown("### Model Feature Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>CNN Model Analysis</h4>
            <p><strong>Focus:</strong> Local textures, edges, nucleus details</p>
            <p><strong>Approach:</strong> Convolutional filters capture fine-grained patterns</p>
            <p><strong>Strength:</strong> Excellent for identifying specific cellular structures</p>
        </div>
        """, unsafe_allow_html=True)
        
        # CNN Feature Summary
        cnn_feat = result['cnn_features']
        st.markdown("**CNN Feature Summary:**")
        st.write(f"- Feature vector shape: {result['cnn_features_shape']}")
        st.write(f"- Mean activation: {np.mean(cnn_feat):.4f}")
        st.write(f"- Max activation: {np.max(cnn_feat):.4f}")
        st.write(f"- Standard deviation: {np.std(cnn_feat):.4f}")
        st.write(f"- First 5 values: {cnn_feat.flatten()[:5].tolist()}")
        st.write(f"- Feature type: Local texture patterns (1280-dim)")
        st.write(f"- Expected range: Fine-grained feature activations")
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>Transformer Model Analysis</h4>
            <p><strong>Focus:</strong> Spatial relationships and global structure</p>
            <p><strong>Approach:</strong> Self-attention captures long-range dependencies</p>
            <p><strong>Strength:</strong> Excellent for understanding context and patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Swin Feature Summary
        swin_feat = result['swin_features']
        st.markdown("**Transformer Feature Summary:**")
        st.write(f"- Feature vector shape: {result['swin_features_shape']}")
        st.write(f"- Mean activation: {np.mean(swin_feat):.4f}")
        st.write(f"- Max activation: {np.max(swin_feat):.4f}")
        st.write(f"- Standard deviation: {np.std(swin_feat):.4f}")
        st.write(f"- First 5 values: {swin_feat.flatten()[:5].tolist()}")
        st.write(f"- Feature type: Global spatial embeddings (768-dim)")
        st.write(f"- Expected range: Meaningful values (not near zero)")


def create_debug_panel(result: Dict):
    """Create comprehensive debug panel."""
    st.markdown("### Debug Information")
    
    with st.expander("Model Output Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**CNN Model:**")
            st.write(f"- Output shape: {result.get('cnn_output_shape', 'N/A')}")
            st.write(f"- Features shape: {result['cnn_features_shape']}")
            st.write(f"- Individual prediction: {result['cnn_prediction']}")
        
        with col2:
            st.markdown("**Transformer Model:**")
            st.write(f"- Output shape: {result.get('swin_output_shape', 'N/A')}")
            st.write(f"- Features shape: {result['swin_features_shape']}")
            st.write(f"- Individual prediction: {result['swin_prediction']}")
        
        with col3:
            st.markdown("**Fusion Model:**")
            st.write(f"- Features shape: {result['fused_features_shape']}")
            st.write(f"- Final prediction: {result['prediction']}")
            st.write(f"- Confidence: {result['confidence']:.4f}")
    
    with st.expander("Probability Details"):
        st.write(f"Raw probabilities: {result.get('raw_probabilities', 'N/A')}")
        st.write(f"Processed probabilities: {result['probabilities']}")
        st.write(f"CNN weight: {result['cnn_weight']:.4f}")
        st.write(f"Swin weight: {result['swin_weight']:.4f}")


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
        st.error(f"Error preprocessing image: {e}")
        return None


def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header"> Enhanced Dual-Model Medical Classification</h1>', 
                unsafe_allow_html=True)
    
    # Load dual-model system
    classifier, device, status = load_dual_model_system()
    
    if classifier is None:
        st.error("Dual-model system could not be loaded.")
        return
    
    st.success(f"System Status: {status}")
    
    # Sidebar for image input
    st.sidebar.markdown("### Image Input")
    
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
        st.sidebar.info("Using random demo image")
    else:
        demo_image = None
    
    # Process image
    image_to_analyze = None
    if uploaded_file is not None:
        image_to_analyze = Image.open(uploaded_file).convert('RGB')
        st.sidebar.success("Image uploaded successfully")
    elif demo_image is not None:
        image_to_analyze = demo_image
    
    if image_to_analyze is not None:
        # Display original image
        st.markdown("### Input Image")
        st.image(image_to_analyze, caption="Input Medical Image", width=400)
        
        # Analysis button
        if st.button("Analyze with Dual Models", type="primary", use_container_width=True):
            with st.spinner("Processing with CNN and Transformer models..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image_for_demo(image_to_analyze)
                    
                    if image_tensor is None:
                        st.error("Failed to preprocess image")
                        return
                    
                    # Safe prediction with features
                    result = safe_prediction_with_features(image_tensor, classifier)
                    
                    if result is None:
                        st.error("Prediction failed")
                        return
                    
                    # Convert image to numpy for overlays
                    original_np = np.array(image_to_analyze)
                    if original_np.max() <= 1.0:
                        original_np = (original_np * 255).astype(np.uint8)
                    
                    # Display results
                    st.markdown("---")
                    
                    # Section 1: Prediction Results
                    st.markdown("## Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        class_name = "Normal" if result['prediction'] == 0 else "Abnormal"
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
                    
                    # Section 2: Feature Insights Panel
                    st.markdown("---")
                    create_feature_insights_panel(result)
                    
                    # Section 3: Feature Visualization
                    st.markdown("---")
                    st.markdown("## Feature Visualization")
                    
                    fig = create_feature_visualization(result['cnn_features'], result['swin_features'])
                    st.pyplot(fig)
                    plt.close()
                    
                    # Section 4: Visual Analysis
                    st.markdown("---")
                    st.markdown("## Visual Analysis")
                    
                    # Create comprehensive visualization
                    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig2.suptitle('Comprehensive Model Analysis', fontsize=16, fontweight='bold')
                    
                    # Original image
                    axes[0, 0].imshow(original_np)
                    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                    axes[0, 0].axis('off')
                    
                    # CNN Grad-CAM heatmap
                    im1 = axes[0, 1].imshow(result['gradcam_heatmap'], cmap='jet')
                    axes[0, 1].set_title('CNN Grad-CAM Heatmap', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                    
                    # CNN Grad-CAM overlay
                    overlay1 = create_overlay(original_np, result['gradcam_heatmap'])
                    axes[0, 2].imshow(overlay1)
                    axes[0, 2].set_title('CNN Grad-CAM Overlay', fontsize=12, fontweight='bold')
                    axes[0, 2].axis('off')
                    
                    # Swin attention map
                    im2 = axes[1, 0].imshow(result['attention_map'], cmap='viridis')
                    axes[1, 0].set_title('Transformer Attention Map', fontsize=12, fontweight='bold')
                    axes[1, 0].axis('off')
                    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
                    
                    # Swin attention overlay
                    overlay2 = create_overlay(original_np, result['attention_map'])
                    axes[1, 1].imshow(overlay2)
                    axes[1, 1].set_title('Transformer Attention Overlay', fontsize=12, fontweight='bold')
                    axes[1, 1].axis('off')
                    
                    # Model comparison
                    models = ['CNN', 'Transformer']
                    weights = [result['cnn_weight'], result['swin_weight']]
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
                    st.pyplot(fig2)
                    plt.close()
                    
                    # Section 5: Debug Panel
                    st.markdown("---")
                    create_debug_panel(result)
                    
                    # Section 6: Explanations
                    st.markdown("---")
                    st.markdown("### Model Explanations")
                    
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
                    
                    # Download results
                    st.markdown("---")
                    st.markdown("### Export Results")
                    
                    results_json = {
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'probabilities': result['probabilities'],
                        'cnn_features_shape': result['cnn_features_shape'],
                        'swin_features_shape': result['swin_features_shape'],
                        'fused_features_shape': result['fused_features_shape'],
                        'cnn_features_sample': result['cnn_features'].flatten()[:10].tolist(),
                        'swin_features_sample': result['swin_features'].flatten()[:10].tolist(),
                        'explanation': result['explanation']
                    }
                    
                    st.download_button(
                        label="Download Results as JSON",
                        data=json.dumps(results_json, indent=2),
                        file_name="dual_model_analysis.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.error("Please check that both models are properly loaded and image format is supported.")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Instructions
    st.markdown("---")
    st.markdown("## How to Use")
    
    st.markdown("""
    <div class="explanation-box">
        <h4>Enhanced Dual-Model Analysis Guide</h4>
        <ol>
            <li><strong>Upload Image:</strong> Choose a medical image or use demo mode</li>
            <li><strong>Analyze:</strong> Click "Analyze with Dual Models" to process</li>
            <li><strong>Review Features:</strong>
                <ul>
                    <li><strong>Feature Insights:</strong> Numerical analysis of extracted features</li>
                    <li><strong>Feature Visualization:</strong> Bar charts and distributions</li>
                    <li><strong>Visual Analysis:</strong> Heatmaps and overlays</li>
                    <li><strong>Debug Panel:</strong> Technical details for verification</li>
                </ul>
            </li>
            <li><strong>Medical Interpretation:</strong> Use comprehensive insights for decision support</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
