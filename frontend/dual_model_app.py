#!/usr/bin/env python3
"""
Dual Model Frontend for Cervical Cytology Classification
Integrates CNN and Swin Transformer with comprehensive explainability.

Author: Cervical Cancer Classification Pipeline
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.dual_model_inference import create_dual_inference_engine
from backend.feature_extraction import CytologyFeatureExtractor

# Configure Streamlit - MUST be the first Streamlit command
st.set_page_config(
    page_title="Dual Model Clinical Cytology Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .bethesda-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .risk-low { border-left: 4px solid #10b981; }
    .risk-moderate { border-left: 4px solid #f59e0b; }
    .risk-high { border-left: 4px solid #ef4444; }
    .risk-urgent { border-left: 4px solid #991b1b; }
    .clinical-insight {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .dual-model-badge {
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_dual_inference_engine():
    """Load and cache the dual model inference engine."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🖥️ Device: {device}")
        
        # Try to load real models
        cnn_path = "outputs/models/best_model.pth"
        swin_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
        engine = create_dual_inference_engine(cnn_path, swin_path, device)
        
        return engine, device
    except Exception as e:
        st.error(f"❌ Error loading dual model inference engine: {e}")
        return None, torch.device("cpu")


def create_dual_model_visual_analysis(result: Dict):
    """Create comprehensive visual analysis for dual models."""
    st.markdown("## 📊 Dual Model Visual Analysis")
    
    # Get visualizations and overlays
    grad_cam = result['visualizations']['grad_cam']
    attention = result['visualizations']['attention']
    overlays = result['overlays']
    masks = result['visualizations']['segmentation_masks']
    
    # Create 4-column layout for dual model visualization
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### CNN Grad-CAM")
        if grad_cam is not None and grad_cam.max() > 0:
            st.image(grad_cam, caption="CNN Attention Heatmap", use_container_width=True)
            st.markdown("**Local Feature Focus**")
        else:
            st.warning("⚠️ Grad-CAM not available")
    
    with col2:
        st.markdown("### Swin Attention")
        if attention is not None and attention.max() > 0:
            st.image(attention, caption="Swin Attention Map", use_container_width=True)
            st.markdown("**Global Context**")
        else:
            st.warning("⚠️ Attention not available")
    
    with col3:
        st.markdown("### Combined Overlay")
        if 'combined_overlay' in overlays:
            st.image(overlays['combined_overlay'], caption="Combined Attention", use_container_width=True)
            st.markdown("**Unified Explanation**")
        else:
            st.warning("⚠️ Combined overlay not available")
    
    with col4:
        st.markdown("### Region Importance")
        explanations = result.get('explanations', {})
        
        st.markdown("**Importance Scores:**")
        st.write(f"🔬 Nucleus: {explanations.get('nucleus_importance', 0):.3f}")
        st.write(f"🧫 Cytoplasm: {explanations.get('cytoplasmic_importance', 0):.3f}")
        st.write(f"🔍 Background: {explanations.get('background_importance', 0):.3f}")
        
        dominant_region = explanations.get('dominant_region', 'background')
        st.markdown(f"**Dominant: {dominant_region.title()}**")
    
    # Model weights visualization
    st.markdown("### Model Contribution Analysis")
    
    model_weights = result['summary']['model_weights']
    cnn_weight = model_weights['cnn_weight']
    swin_weight = model_weights['swin_weight']
    
    fig = go.Figure(data=[
        go.Bar(
            x=['CNN (Local)', 'Swin (Global)'],
            y=[cnn_weight, swin_weight],
            marker_color=['#3b82f6', '#8b5cf6']
        )
    ])
    
    fig.update_layout(
        title="Model Weight Distribution",
        xaxis_title="Model Type",
        yaxis_title="Weight",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Region importance breakdown
    st.markdown("### Region Importance Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nucleus_grad = explanations.get('nucleus_grad_cam', 0)
        nucleus_att = explanations.get('nucleus_attention', 0)
        st.metric("CNN Nucleus", f"{nucleus_grad:.3f}")
        st.metric("Swin Nucleus", f"{nucleus_att:.3f}")
    
    with col2:
        cytoplasm_grad = explanations.get('cytoplasm_grad_cam', 0)
        cytoplasm_att = explanations.get('cytoplasm_attention', 0)
        st.metric("CNN Cytoplasm", f"{cytoplasm_grad:.3f}")
        st.metric("Swin Cytoplasm", f"{cytoplasm_att:.3f}")
    
    with col3:
        background_grad = explanations.get('background_grad_cam', 0)
        background_att = explanations.get('background_attention', 0)
        st.metric("CNN Background", f"{background_grad:.3f}")
        st.metric("Swin Background", f"{background_att:.3f}")


def create_dual_model_clinical_reasoning(result: Dict):
    """Create enhanced clinical reasoning panel for dual models."""
    st.markdown("## 🩺 Dual Model Clinical Reasoning")
    
    bethesda = result['bethesda_classification']
    dual_insights = bethesda.get('dual_model_insights', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="clinical-insight">
            <h4>🤖 Dual Model Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Model weights
        cnn_weight = dual_insights.get('cnn_weight', 0.5)
        swin_weight = dual_insights.get('swin_weight', 0.5)
        
        st.write(f"**CNN Weight:** {cnn_weight:.3f}")
        st.write(f"**Swin Weight:** {swin_weight:.3f}")
        
        if cnn_weight > 0.6:
            st.write("🔬 **CNN dominates** (local features)")
        elif swin_weight > 0.6:
            st.write("🌐 **Swin dominates** (global context)")
        else:
            st.write("⚖️ **Balanced contribution**")
        
        st.write(f"**Clinical Score:** {bethesda['clinical_score']:.3f}")
        st.write(f"**Confidence Level:** {bethesda['confidence']:.3f}")
        
        # Decision type
        decision_type = bethesda.get('decision_type', 'dual-model-based')
        if decision_type == 'rule-based':
            st.write(f"**Decision Type:** 📋 Rule-based (Parabasal Override)")
            if bethesda.get('parabasal_override', False):
                st.info("✅ Parabasal cell override applied - classified as NILM")
        elif decision_type == 'dual-model-based':
            st.write(f"**Decision Type:** 🤖 Dual Model-based")
            st.success("✅ Both models contributed to decision")
        else:
            st.write(f"**Decision Type:** 🤖 Model-based")
        
        # Risk indicator
        risk_color = {
            'Low Risk': '🟢',
            'Mild Risk': '🟡',
            'Moderate Risk': '🟡',
            'High Risk': '🔴',
            'Very High Risk': '🔴'
        }
        
        risk_emoji = risk_color.get(bethesda['risk_level'].split(' - ')[0], '⚪')
        st.write(f"**Risk Level:** {risk_emoji} {bethesda['risk_level']}")
    
    with col2:
        st.markdown("""
        <div class="clinical-insight">
            <h4>🧠 Enhanced Reasoning</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(bethesda['clinical_reasoning'])
        
        # Feature breakdown
        st.markdown("**Feature Breakdown:**")
        features = bethesda['feature_scores']
        for feature, score in features.items():
            st.write(f"- {feature.title()}: {score:.3f}")
        
        # Dual model insights
        st.markdown("**Model Insights:**")
        dominant_region = dual_insights.get('dominant_region', 'background')
        st.write(f"- **Dominant Region:** {dominant_region.title()}")
        
        nucleus_imp = dual_insights.get('nucleus_importance', 0.33)
        cytoplasm_imp = dual_insights.get('cytoplasmic_importance', 0.33)
        background_imp = dual_insights.get('background_importance', 0.34)
        
        st.write(f"- **Nucleus Importance:** {nucleus_imp:.3f}")
        st.write(f"- **Cytoplasm Importance:** {cytoplasm_imp:.3f}")
        st.write(f"- **Background Importance:** {background_imp:.3f}")


def create_dual_model_summary_dashboard(result: Dict):
    """Create summary dashboard for dual model system."""
    st.markdown("## 📈 Dual Model Summary Dashboard")
    
    # Key metrics with dual model badges
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Classification",
            result['summary']['class'],
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        confidence = result['summary']['confidence']
        if confidence < 0.6:
            st.warning(f"⚠️ Low Confidence: {confidence:.1%}")
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
            delta=f"{(confidence - 0.5) * 100:.1f}%",
            delta_color="normal" if confidence > 0.7 else "inverse"
        )
    
    with col3:
        dominant_region = result['summary']['dominant_region']
        region_display = dominant_region.title()
        st.metric(
            "Dominant Region",
            region_display,
            delta=None,
            delta_color="normal"
        )
    
    with col4:
        clinical_score = result['bethesda_classification']['clinical_score']
        st.metric(
            "Clinical Score",
            f"{clinical_score:.3f}",
            delta=None,
            delta_color="normal"
        )
    
    # Dual model status
    st.markdown("### 🤖 Dual Model Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cnn_weight = result['summary']['model_weights']['cnn_weight']
        swin_weight = result['summary']['model_weights']['swin_weight']
        
        st.markdown('<span class="dual-model-badge">CNN Model</span>', unsafe_allow_html=True)
        st.progress(cnn_weight, text=f"Local Features: {cnn_weight:.1%}")
    
    with col2:
        st.markdown('<span class="dual-model-badge">Swin Transformer</span>', unsafe_allow_html=True)
        st.progress(swin_weight, text=f"Global Context: {swin_weight:.1%}")
    
    with col3:
        decision_type = result['bethesda_classification'].get('decision_type', 'dual-model-based')
        if decision_type == 'dual-model-based':
            st.markdown('<span class="dual-model-badge">Dual Model</span>', unsafe_allow_html=True)
            st.info("✅ Both models active")
        elif decision_type == 'rule-based':
            st.markdown('<span class="dual-model-badge">Rule Override</span>', unsafe_allow_html=True)
            st.warning("⚠️ Clinical rule applied")
        else:
            st.markdown('<span class="dual-model-badge">Single Model</span>', unsafe_allow_html=True)
            st.info("📊 Model-based decision")
    
    # Feature scores with dual model alignment
    st.markdown("### 🎯 Feature Alignment")
    
    nuclear_score = result['extracted_features']['nuclear']['overall_nuclear_score']
    cytoplasmic_score = result['extracted_features']['cytoplasmic']['overall_cytoplasmic_score']
    background_score = result['extracted_features']['background']['overall_background_score']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.progress(nuclear_score, text=f"Nuclear: {nuclear_score:.1%}")
    
    with col2:
        st.progress(cytoplasmic_score, text=f"Cytoplasmic: {cytoplasmic_score:.1%}")
    
    with col3:
        st.progress(background_score, text=f"Background: {background_score:.1%}")
    
    # Add cell maturity ratio if available
    if 'cell_maturity_ratio' in result['extracted_features']['cytoplasmic']:
        maturity_ratio = result['extracted_features']['cytoplasmic']['cell_maturity_ratio']
        st.markdown("### Cell Maturity Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Cell Maturity Ratio",
                f"{maturity_ratio:.2f}",
                delta=None,
                delta_color="normal"
            )
        
        with col2:
            if maturity_ratio < 2.0:
                st.info("🟡 Immature cells detected")
            elif maturity_ratio > 4.0:
                st.info("🟢 Mature cells detected")
            else:
                st.info("🔵 Normal maturity")


def main():
    """Main dual model application function."""
    st.markdown('<h1 class="main-header">🤖 Dual Model Clinical Cytology Classification</h1>', 
                unsafe_allow_html=True)
    
    # Load dual model inference engine
    engine, device = load_dual_inference_engine()
    
    if engine is None:
        st.error("❌ Could not load dual model inference engine. Please check the configuration.")
        return
    
    st.success("✅ Dual Model System Ready for Analysis")
    
    # Sidebar
    st.sidebar.markdown("### 📁 Patient Sample")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload cytology slide",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
        help="Upload a cervical cytology image for dual model analysis"
    )
    
    # Demo option
    use_demo = st.sidebar.checkbox("Use demo image", value=True)
    
    # Process image
    image = None
    image_id = None
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_id = uploaded_file.name
        st.sidebar.success("✅ Image uploaded successfully")
    elif use_demo:
        # Create demo image with cell-like structures
        demo_image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add cell-like structures
        cv2.circle(demo_image, (112, 112), 25, (150, 100, 80), -1)  # Nucleus
        cv2.circle(demo_image, (80, 80), 20, (140, 90, 70), -1)  # Another nucleus
        cv2.circle(demo_image, (150, 140), 22, (160, 110, 90), -1)  # Third nucleus
        
        # Add cytoplasm-like regions
        cv2.circle(demo_image, (112, 112), 40, (200, 150, 120), 2)  # Cytoplasm boundary
        cv2.circle(demo_image, (80, 80), 35, (190, 140, 110), 2)
        cv2.circle(demo_image, (150, 140), 38, (210, 160, 130), 2)
        
        image = Image.fromarray(demo_image)
        image_id = "demo_image"
        st.sidebar.info("🎭 Using demo image with simulated cell structures")
    
    if image is not None:
        # Store current image in session state
        st.session_state.current_image = np.array(image)
        
        # Display original image
        st.markdown("### 🖼️ Input Cytology Slide")
        st.image(image, caption="Original Cytology Image", width=400)
        
        # Analysis button
        if st.button("🤖 Analyze with Dual AI Models", type="primary", use_container_width=True):
            # Check if we already have results for this image
            if st.session_state.get('last_image_id') == image_id and st.session_state.get('last_dual_result') is not None:
                st.info("📋 Using cached dual model analysis results")
                result = st.session_state.last_dual_result
            else:
                with st.spinner("🔄 Performing comprehensive dual model analysis..."):
                    try:
                        # Run dual model inference
                        result = engine.predict_with_dual_models(image)
                        
                        # Cache the result
                        st.session_state.last_dual_result = result
                        st.session_state.last_image_id = image_id
                        
                    except Exception as e:
                        st.error(f"❌ Error during dual model analysis: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                        return
            
            # Display results
            st.markdown("---")
            
            # Section 1: Bethesda Classification
            from frontend.app import create_bethesda_display
            create_bethesda_display(result['bethesda_classification'])
            
            # Section 2: Dual Model Summary Dashboard
            create_dual_model_summary_dashboard(result)
            
            # Section 3: Feature Analysis
            from frontend.app import create_feature_analysis
            create_feature_analysis(result)
            
            # Section 4: Dual Model Visual Analysis
            create_dual_model_visual_analysis(result)
            
            # Section 5: Dual Model Clinical Reasoning
            create_dual_model_clinical_reasoning(result)
            
            # Success message
            st.success("✅ Dual Model Analysis completed successfully!")
            
            # Add clear cache button
            if st.button("🗑️ Clear Cache"):
                st.session_state.last_dual_result = None
                st.session_state.last_image_id = None
                st.rerun()
    
    # Information section
    st.markdown("---")
    st.markdown("## ℹ️ Dual Model System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Dual Model Architecture
        - **CNN Model**: Local feature extraction
        - **Swin Transformer**: Global context understanding
        - **Feature Fusion**: Combined representation
        - **Explainability**: Grad-CAM + Attention
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Clinical Advantages
        - **Multi-scale analysis**: Local + Global
        - **Real explainability**: Feature-aligned
        - **Clinical reasoning**: Rule-based overrides
        - **Transparent decisions**: Model weights shown
        """)


if __name__ == "__main__":
    main()
