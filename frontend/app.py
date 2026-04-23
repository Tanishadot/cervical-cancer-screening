#!/usr/bin/env python3
"""
Clean Frontend for Cervical Cytology Classification
Production-ready Streamlit app with real feature extraction.

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

from backend.inference import create_inference_engine
from backend.feature_extraction import CytologyFeatureExtractor

# Note: st.set_page_config() is called in the main entry file to avoid duplicate calls

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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inference_engine():
    """Load and cache the inference engine."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🖥️ Device: {device}")
        
        # Try to load real model
        model_path = "outputs/models/best_model.pth"
        engine = create_inference_engine(model_path, device)
        
        return engine, device
    except Exception as e:
        st.error(f"❌ Error loading inference engine: {e}")
        return None, torch.device("cpu")


@st.cache_data
def extract_features_cached(image_array: np.ndarray):
    """Cache feature extraction results."""
    extractor = CytologyFeatureExtractor()
    return extractor.extract_features(image_array)


def create_bethesda_display(bethesda_result: Dict):
    """Display Bethesda classification with risk assessment."""
    risk_colors = {
        'NILM': '#10b981',
        'ASC-US': '#f59e0b', 
        'LSIL': '#f59e0b',
        'HSIL': '#ef4444',
        'SCC': '#991b1b'
    }
    
    color = risk_colors.get(bethesda_result['bethesda_class'], '#6b7280')
    
    st.markdown(f"""
    <div class="bethesda-card">
        <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">{bethesda_result['bethesda_class']}</h2>
        <p style="margin: 0 0 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">{bethesda_result['bethesda_full']}</p>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="margin: 0; font-size: 1rem;">Clinical Score: <strong>{bethesda_result['clinical_score']:.3f}</strong></p>
                <p style="margin: 0; font-size: 1rem;">Confidence: <strong>{bethesda_result['confidence']:.3f}</strong></p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 1.1rem; font-weight: bold;">{bethesda_result['risk_level']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_feature_analysis(result: Dict):
    """Create comprehensive feature analysis display."""
    st.markdown("## 🔬 Feature Analysis")
    
    # Get feature scores
    nuclear = result['extracted_features']['nuclear']
    cytoplasmic = result['extracted_features']['cytoplasmic']
    background = result['extracted_features']['background']
    
    # Create metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card risk-low">
            <h4>🔬 Nuclear Features</h4>
            <p><strong>Primary but not sufficient alone</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Overall Score", f"{nuclear['overall_nuclear_score']:.3f}")
        st.metric("Chromatin Density", f"{nuclear['chromatin_density']:.3f}")
        st.metric("N:C Ratio", f"{nuclear['nc_ratio']:.3f}")
        st.metric("Hyperchromasia", f"{nuclear['hyperchromasia']:.3f}")
    
    with col2:
        st.markdown("""
        <div class="feature-card risk-moderate">
            <h4>🧫 Cytoplasmic Features</h4>
            <p><strong>CRITICAL for classification</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Overall Score", f"{cytoplasmic['overall_cytoplasmic_score']:.3f}")
        st.metric("Perinuclear Halo", f"{cytoplasmic['perinuclear_halo']:.3f}")
        st.metric("Keratinization", f"{cytoplasmic['keratinization']:.3f}")
        st.metric("Koilocytosis", f"{cytoplasmic['koilocytosis_score']:.3f}")
    
    with col3:
        st.markdown("""
        <div class="feature-card risk-high">
            <h4>🔍 Background Features</h4>
            <p><strong>Contextual diagnosis</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Overall Score", f"{background['overall_background_score']:.3f}")
        st.metric("Debris Level", f"{background['background_debris']:.3f}")
        st.metric("Tumor Diathesis", f"{background['tumor_diathesis']:.3f}")
        st.metric("Cleanliness", f"{background['background_cleanliness']:.3f}")
    
    # Create feature contribution chart
    st.markdown("### Feature Contribution Analysis")
    
    feature_names = ['Model', 'Nuclear', 'Cytoplasmic', 'Background']
    feature_values = [
        result['feature_fusion']['feature_contributions']['model'],
        result['feature_fusion']['feature_contributions']['nuclear'],
        result['feature_fusion']['feature_contributions']['cytoplasmic'],
        result['feature_fusion']['feature_contributions']['background']
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=feature_values,
            marker_color=['#3b82f6', '#ef4444', '#f59e0b', '#10b981']
        )
    ])
    
    fig.update_layout(
        title="Feature Contribution to Final Classification",
        xaxis_title="Feature Type",
        yaxis_title="Contribution Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_visual_analysis(result: Dict):
    """Create visual analysis with segmentation and Grad-CAM."""
    st.markdown("## 📊 Visual Analysis")
    
    # Get visualizations
    grad_cam = result['visualizations']['grad_cam']
    masks = result['visualizations']['segmentation_masks']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Region Segmentation")
        
        try:
            # Get original image from session state
            original = st.session_state.get('current_image')
            if original is None:
                st.warning("No image available for segmentation overlay")
                return
            
            # Ensure original is numpy array and proper shape
            if isinstance(original, Image.Image):
                original = np.array(original)
            
            # Resize to match masks if needed
            target_size = (masks['nucleus'].shape[1], masks['nucleus'].shape[0])
            if original.shape[:2] != target_size:
                original_resized = cv2.resize(original, target_size)
            else:
                original_resized = original
            
            # Create colored segmentation overlay
            segmentation = np.zeros((*target_size, 3), dtype=np.uint8)
            segmentation[masks['nucleus'] > 0] = [255, 0, 0]  # Red for nucleus
            segmentation[masks['cytoplasm'] > 0] = [0, 255, 0]  # Green for cytoplasm
            segmentation[masks['background'] > 0] = [0, 0, 255]  # Blue for background
            
            # Normalize original image to [0, 1]
            if original_resized.max() > 1.0:
                original_norm = original_resized / 255.0
            else:
                original_norm = original_resized
            
            # Blend with original image
            alpha = 0.6
            overlay = alpha * segmentation / 255.0 + (1 - alpha) * original_norm
            overlay = (overlay * 255).astype(np.uint8)
            
            st.image(overlay, caption="Region Segmentation (Red: Nucleus, Green: Cytoplasm, Blue: Background)")
            
            # Region statistics
            st.markdown("**Region Statistics:**")
            nucleus_area = np.sum(masks['nucleus'] > 0)
            cytoplasm_area = np.sum(masks['cytoplasm'] > 0)
            background_area = np.sum(masks['background'] > 0)
            total_area = nucleus_area + cytoplasm_area + background_area
            
            if total_area > 0:
                st.write(f"- Nucleus: {nucleus_area/total_area:.1%} of image")
                st.write(f"- Cytoplasm: {cytoplasm_area/total_area:.1%} of image")
                st.write(f"- Background: {background_area/total_area:.1%} of image")
            else:
                st.warning("No regions detected")
                
        except Exception as e:
            st.error(f"Error creating segmentation overlay: {e}")
            st.info("Displaying segmentation masks separately")
            
            # Display masks separately if overlay fails
            st.image(masks['nucleus'], caption="Nucleus Mask", width=200)
            st.image(masks['cytoplasm'], caption="Cytoplasm Mask", width=200)
            st.image(masks['background'], caption="Background Mask", width=200)
    
    with col2:
        st.markdown("### Model Attention")
        
        # Check if Grad-CAM has meaningful values
        if grad_cam is not None and grad_cam.max() > 0:
            try:
                # Get original image for blending
                original = st.session_state.get('current_image')
                if original is None:
                    st.warning("No image available for Grad-CAM overlay")
                    return
                
                if isinstance(original, Image.Image):
                    original = np.array(original)
                
                # Resize to match Grad-CAM if needed
                target_size = (grad_cam.shape[1], grad_cam.shape[0])
                if original.shape[:2] != target_size:
                    original_resized = cv2.resize(original, target_size)
                else:
                    original_resized = original
                
                # Normalize Grad-CAM
                grad_cam_norm = grad_cam / grad_cam.max()
                
                # Apply colormap
                grad_cam_colored = plt.cm.jet(grad_cam_norm)[:, :, :3]
                grad_cam_colored = (grad_cam_colored * 255).astype(np.uint8)
                
                # Normalize original image
                if original_resized.max() > 1.0:
                    original_norm = original_resized / 255.0
                else:
                    original_norm = original_resized
                
                # Blend with original
                grad_cam_overlay = 0.6 * grad_cam_colored + 0.4 * original_norm
                grad_cam_overlay = (grad_cam_overlay * 255).astype(np.uint8)
                
                st.image(grad_cam_overlay, caption="Grad-CAM Attention Heatmap")
                
                # Attention statistics
                st.markdown("**Attention Analysis:**")
                attention_mean = np.mean(grad_cam_norm)
                attention_max = np.max(grad_cam_norm)
                attention_coverage = np.sum(grad_cam_norm > 0.1) / grad_cam_norm.size if grad_cam_norm.size > 0 else 0
                
                st.write(f"- Mean Attention: {attention_mean:.3f}")
                st.write(f"- Peak Attention: {attention_max:.3f}")
                st.write(f"- Attention Coverage: {attention_coverage:.1%}")
                
            except Exception as e:
                st.error(f"Error creating Grad-CAM overlay: {e}")
                st.info("Displaying Grad-CAM separately")
                st.image(grad_cam, caption="Grad-CAM Heatmap", width=400)
        else:
            # Grad-CAM failed or has no meaningful values
            st.warning("⚠️ Grad-CAM heatmap not available")
            st.info("Model attention could not be computed. Using segmentation-based feature importance instead.")
            
            # Show segmentation-based importance
            st.markdown("**Segmentation-Based Importance:**")
            nucleus_area = np.sum(masks['nucleus'] > 0)
            cytoplasm_area = np.sum(masks['cytoplasm'] > 0)
            background_area = np.sum(masks['background'] > 0)
            total_area = nucleus_area + cytoplasm_area + background_area
            
            if total_area > 0:
                st.write(f"- Nucleus region: {nucleus_area/total_area:.1%}")
                st.write(f"- Cytoplasm region: {cytoplasm_area/total_area:.1%}")
                st.write(f"- Background region: {background_area/total_area:.1%}")
                
                # Show which region is dominant
                areas = {'Nucleus': nucleus_area, 'Cytoplasm': cytoplasm_area, 'Background': background_area}
                dominant_region = max(areas, key=areas.get)
                st.write(f"- **Dominant region: {dominant_region}**")
            else:
                st.warning("No regions detected for importance analysis")


def create_clinical_reasoning(result: Dict):
    """Create clinical reasoning panel."""
    st.markdown("## 🩺 Clinical Reasoning")
    
    bethesda = result['bethesda_classification']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="clinical-insight">
            <h4>Medical Interpretation</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Get dominant feature from fusion result
        dominant = result.get('feature_fusion', {}).get('dominant_feature', 'model')
        if dominant == 'model':
            dominant_display = "CNN Model"
        elif dominant == 'nuclear':
            dominant_display = "Nuclear Features"
        elif dominant == 'cytoplasmic':
            dominant_display = "Cytoplasmic Features"
        elif dominant == 'background':
            dominant_display = "Background Features"
        else:
            dominant_display = dominant.title()
        
        st.write(f"**Dominant Feature:** {dominant_display}")
        st.write(f"**Clinical Score:** {bethesda['clinical_score']:.3f}")
        st.write(f"**Confidence Level:** {bethesda['confidence']:.3f}")
        
        # Decision type
        decision_type = bethesda.get('decision_type', 'model-based')
        if decision_type == 'rule-based':
            st.write(f"**Decision Type:** 📋 Rule-based (Parabasal Override)")
            if bethesda.get('parabasal_override', False):
                st.info("✅ Parabasal cell override applied - classified as NILM")
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
            <h4>Diagnostic Reasoning</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(bethesda['clinical_reasoning'])
        
        # Feature breakdown
        st.markdown("**Feature Breakdown:**")
        features = bethesda['feature_scores']
        for feature, score in features.items():
            st.write(f"- {feature.title()}: {score:.3f}")
        
        # Add feature scores comparison if available
        if 'feature_fusion' in result and 'feature_scores' in result['feature_fusion']:
            st.markdown("**Raw Feature Scores:**")
            raw_scores = result['feature_fusion']['feature_scores']
            for feature, score in raw_scores.items():
                st.write(f"- {feature.title()}: {score:.3f}")


def create_summary_dashboard(result: Dict):
    """Create summary dashboard with key metrics."""
    st.markdown("## 📈 Summary Dashboard")
    
    # Key metrics
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
        # Add low confidence warning
        if confidence < 0.6:
            st.warning(f"⚠️ Low Confidence: {confidence:.1%}")
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
            delta=f"{(confidence - 0.5) * 100:.1f}%",
            delta_color="normal" if confidence > 0.7 else "inverse"
        )
    
    with col3:
        # Get actual dominant feature from fusion result
        dominant = result.get('feature_fusion', {}).get('dominant_feature', 'model')
        # Replace "model" with more descriptive name
        if dominant == 'model':
            dominant_display = "CNN Model"
        elif dominant == 'nuclear':
            dominant_display = "Nuclear Features"
        elif dominant == 'cytoplasmic':
            dominant_display = "Cytoplasmic Features"
        elif dominant == 'background':
            dominant_display = "Background Features"
        else:
            dominant_display = dominant.title()
        
        # Add decision type indicator
        decision_type = result.get('bethesda_classification', {}).get('decision_type', 'model-based')
        if decision_type == 'rule-based':
            dominant_display += " 📋"
        
        st.metric(
            "Key Feature",
            dominant_display,
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
    
    # Progress bars for feature scores
    st.markdown("### Feature Score Distribution")
    
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
    """Main application function."""
    st.markdown('<h1 class="main-header">🔬 Clinical Cytology Classification System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'engine' not in st.session_state:
        st.session_state.engine = None
        st.session_state.device = None
        st.session_state.last_result = None
        st.session_state.last_image_id = None
    
    # Load inference engine (cached)
    if st.session_state.engine is None:
        with st.spinner("🔄 Loading AI models..."):
            engine, device = load_inference_engine()
            st.session_state.engine = engine
            st.session_state.device = device
    
    if st.session_state.engine is None:
        st.error("❌ Could not load inference engine. Please check the configuration.")
        return
    
    st.success("✅ System ready for analysis")
    
    # Sidebar
    st.sidebar.markdown("### 📁 Patient Sample")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload cytology slide",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
        help="Upload a cervical cytology image for analysis"
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
        if st.button("🔬 Analyze with AI", type="primary", use_container_width=True):
            # Check if we already have results for this image
            if st.session_state.last_image_id == image_id and st.session_state.last_result is not None:
                st.info("📋 Using cached analysis results")
                result = st.session_state.last_result
            else:
                with st.spinner("🔄 Performing comprehensive analysis..."):
                    try:
                        # Run inference
                        result = st.session_state.engine.predict(image)
                        
                        # Cache the result
                        st.session_state.last_result = result
                        st.session_state.last_image_id = image_id
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                        return
            
            # Display results
            st.markdown("---")
            
            # Section 1: Bethesda Classification
            create_bethesda_display(result['bethesda_classification'])
            
            # Section 2: Summary Dashboard
            create_summary_dashboard(result)
            
            # Section 3: Feature Analysis
            create_feature_analysis(result)
            
            # Section 4: Visual Analysis
            create_visual_analysis(result)
            
            # Section 5: Clinical Reasoning
            create_clinical_reasoning(result)
            
            # Success message
            st.success("✅ Analysis completed successfully!")
            
            # Add clear cache button
            if st.button("🗑️ Clear Cache"):
                st.session_state.last_result = None
                st.session_state.last_image_id = None
                st.rerun()
    
    # Information section
    st.markdown("---")
    st.markdown("## ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Bethesda Classification System
        - **NILM**: Negative for Intraepithelial Lesion
        - **ASC-US**: Atypical Squamous Cells (Undetermined)
        - **LSIL**: Low-grade Squamous Intraepithelial Lesion
        - **HSIL**: High-grade Squamous Intraepithelial Lesion
        - **SCC**: Squamous Cell Carcinoma
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Features
        - **Real Feature Extraction** using OpenCV
        - **Multi-factor Analysis** (Nuclear + Cytoplasmic + Background)
        - **Clinical Reasoning** with explainable AI
        - **Visual Explainability** with Grad-CAM
        - **Bethesda Alignment** for medical standards
        """)


if __name__ == "__main__":
    main()
