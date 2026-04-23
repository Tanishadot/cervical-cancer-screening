#!/usr/bin/env python3
"""
Clinical Cytology Dashboard with Bethesda-Aligned Classification
Enhanced medical interpretation with clinical reasoning.

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
import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from clinical_dual_model_classifier import ClinicalDualModelClassifier, create_clinical_overlay
from preprocessing.transforms import get_val_transforms

# Set page config
st.set_page_config(
    page_title="Clinical Cytology Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .clinical-header {
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_clinical_system():
    """Load clinical dual-model system."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Device: {device}")
        
        # Model paths
        cnn_model_path = "outputs/models/best_model.pth"
        swin_model_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
        
        # Check if models exist
        if not os.path.exists(cnn_model_path) or not os.path.exists(swin_model_path):
            st.warning("Models not found. Creating demo models...")
            return create_demo_models(device)
        
        classifier = ClinicalDualModelClassifier(cnn_model_path, swin_model_path, device)
        return classifier, device, "Real models loaded"
    
    except Exception as e:
        st.error(f"Error loading clinical system: {e}")
        return None, torch.device("cpu"), f"Error: {e}"


def create_demo_models(device):
    """Create demo models for testing."""
    from models.model_factory import CervicalCancerModel
    
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/cross_dataset_training/models", exist_ok=True)
    
    # Create and save dummy models
    cnn_model = CervicalCancerModel(
        model_name="efficientnet_b0.ra_in1k",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    torch.save(cnn_model.state_dict(), "outputs/models/best_model.pth")
    
    swin_model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    torch.save(swin_model.state_dict(), "outputs/cross_dataset_training/models/best_swin_model.pth")
    
    classifier = ClinicalDualModelClassifier("outputs/models/best_model.pth", 
                                   "outputs/cross_dataset_training/models/best_swin_model.pth", 
                                   device)
    return classifier, device, "Demo models created"


def preprocess_image_clinical(image):
    """Preprocess image with clinical validation."""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_np = np.array(image)
        
        # Clinical validation
        if image_np.max() < 10:
            st.warning("Image appears very dark - may affect analysis")
        if image_np.std() < 20:
            st.warning("Low contrast image - may affect feature extraction")
        
        transform = get_val_transforms()
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)
        
        return image_tensor
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None


def create_bethesda_display(bethesda_result):
    """Display Bethesda classification with risk assessment."""
    risk_class = {
        'NILM': 'risk-low',
        'ASC-US': 'risk-moderate', 
        'LSIL': 'risk-moderate',
        'HSIL': 'risk-high',
        'SCC': 'risk-urgent'
    }
    
    risk_css = risk_class.get(bethesda_result['bethesda_class'], 'risk-low')
    
    st.markdown(f"""
    <div class="bethesda-card">
        <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">{bethesda_result['bethesda_class']}</h2>
        <p style="margin: 0 0 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">{bethesda_result['bethesda_full']}</p>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="margin: 0; font-size: 1rem;">Clinical Score: <strong>{bethesda_result['clinical_score']:.3f}</strong></p>
                <p style="margin: 0; font-size: 1rem;">Model Confidence: <strong>{bethesda_result['model_confidence']:.3f}</strong></p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 1.1rem; font-weight: bold;">{bethesda_result['risk_level']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_feature_analysis_panels(result):
    """Create detailed feature analysis panels."""
    st.markdown("## Feature Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card risk-low">
            <h4>🔬 Nuclear Features</h4>
            <p><strong>Primary but not sufficient alone</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        nuclear = result['nuclear_analysis']
        st.write("**Nuclear Analysis:**")
        st.write(f"- Enlargement: {nuclear['nuclear_enlargement']:.3f}")
        st.write(f"- Chromatin Density: {nuclear['chromatin_density']:.3f}")
        st.write(f"- Contour Irregularity: {nuclear['nuclear_contours']:.3f}")
        st.write(f"- N:C Ratio: {nuclear['nc_ratio']:.3f}")
        st.write(f"- Hyperchromasia: {nuclear['hyperchromasia']:.3f}")
        st.write(f"- Overall Score: {nuclear['overall_nuclear_score']:.3f}")
    
    with col2:
        st.markdown("""
        <div class="feature-card risk-moderate">
            <h4>🧫 Cytoplasmic Features</h4>
            <p><strong>CRITICAL for classification</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        cytoplasmic = result['cytoplasmic_analysis']
        st.write("**Cytoplasmic Analysis:**")
        st.write(f"- Perinuclear Halo: {cytoplasmic['perinuclear_halo']:.3f}")
        st.write(f"- Keratinization: {cytoplasmic['keratinization']:.3f}")
        st.write(f"- Texture Changes: {cytoplasmic['cytoplasmic_texture']:.3f}")
        st.write(f"- Cell Maturity: {cytoplasmic['cell_maturity']:.3f}")
        st.write(f"- Koilocytosis: {cytoplasmic['koilocytosis_score']:.3f}")
        st.write(f"- Overall Score: {cytoplasmic['overall_cytoplasmic_score']:.3f}")
    
    with col3:
        st.markdown("""
        <div class="feature-card risk-high">
            <h4>🔍 Background Features</h4>
            <p><strong>Contextual diagnosis</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        background = result['background_analysis']
        st.write("**Background Analysis:**")
        st.write(f"- Debris Level: {background['background_debris']:.3f}")
        st.write(f"- Inflammatory Cells: {background['inflammatory_cells']:.3f}")
        st.write(f"- Tumor Diathesis: {background['tumor_diathesis']:.3f}")
        st.write(f"- Cleanliness: {background['background_cleanliness']:.3f}")
        st.write(f"- Overall Score: {background['overall_background_score']:.3f}")


def create_clinical_reasoning_panel(result):
    """Create clinical reasoning panel."""
    st.markdown("## Clinical Reasoning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="clinical-insight">
            <h4>🩺 Medical Interpretation</h4>
            <p><strong>Dominant Feature:</strong> {result['dominant_feature_type'].title()}</p>
            <p><strong>Model-Clinical Alignment:</strong> {result['model_clinical_alignment']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="clinical-insight">
            <h4>📋 Diagnostic Reasoning</h4>
            <p>{result['clinical_reasoning']}</p>
        </div>
        """, unsafe_allow_html=True)


def create_visual_analysis_section(result, original_image):
    """Create comprehensive visual analysis."""
    st.markdown("## Visual Analysis")
    
    # Create clinical visualization
    fig = result['classifier'].create_clinical_visualization(result, original_image)
    st.pyplot(fig)
    plt.close()
    
    # Create combined overlay
    st.markdown("### Combined Feature Overlay")
    combined_overlay = create_clinical_overlay(
        original_image, 
        result['gradcam_heatmap'], 
        result['attention_map']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(combined_overlay, caption="Combined Clinical Overlay (Red: Nuclear, Green: Spatial)", use_container_width=True)
    
    with col2:
        # Feature contribution chart
        feature_scores = [
            result['nuclear_analysis']['overall_nuclear_score'],
            result['cytoplasmic_analysis']['overall_cytoplasmic_score'],
            result['background_analysis']['overall_background_score']
        ]
        feature_labels = ['Nuclear', 'Cytoplasmic', 'Background']
        feature_colors = ['#ef4444', '#f59e0b', '#3b82f6']
        
        fig2, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(feature_labels, feature_scores, color=feature_colors)
        ax.set_title('Feature Contribution Analysis', fontweight='bold', fontsize=14)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, max(feature_scores) * 1.2)
        
        # Add value labels on bars
        for bar, score in zip(bars, feature_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


def create_export_section(result):
    """Create export section for clinical reports."""
    st.markdown("## Export Clinical Report")
    
    clinical_report = {
        'patient_id': f"PAT_{np.random.randint(10000, 99999)}",
        'analysis_date': str(datetime.datetime.now()),
        'bethesda_classification': result['bethesda_classification'],
        'clinical_features': result['clinical_features'],
        'feature_analysis': {
            'nuclear': result['nuclear_analysis'],
            'cytoplasmic': result['cytoplasmic_analysis'],
            'background': result['background_analysis']
        },
        'clinical_reasoning': result['clinical_reasoning'],
        'risk_assessment': result['risk_assessment'],
        'model_confidence': result['confidence'],
        'alignment': result['model_clinical_alignment']
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download Clinical Report (JSON)",
            data=json.dumps(clinical_report, indent=2),
            file_name=f"clinical_report_{clinical_report['patient_id']}.json",
            mime="application/json"
        )
    
    with col2:
        st.download_button(
            label="📊 Download Feature Analysis (CSV)",
            data="Feature,Score\n" + "\n".join([
                f"Nuclear,{result['nuclear_analysis']['overall_nuclear_score']:.3f}",
                f"Cytoplasmic,{result['cytoplasmic_analysis']['overall_cytoplasmic_score']:.3f}",
                f"Background,{result['background_analysis']['overall_background_score']:.3f}"
            ]),
            file_name=f"feature_analysis_{clinical_report['patient_id']}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Generate summary text
        summary = f"""Clinical Cytology Analysis Report
================================
Patient ID: {clinical_report['patient_id']}
Date: {clinical_report['analysis_date']}

Bethesda Classification: {result['bethesda_classification']['bethesda_class']}
Full Classification: {result['bethesda_classification']['bethesda_full']}

Risk Assessment: {result['risk_assessment']}
Model Confidence: {result['confidence']:.3f}

Clinical Reasoning:
{result['clinical_reasoning']}

Feature Analysis:
- Nuclear Score: {result['nuclear_analysis']['overall_nuclear_score']:.3f}
- Cytoplasmic Score: {result['cytoplasmic_analysis']['overall_cytoplasmic_score']:.3f}
- Background Score: {result['background_analysis']['overall_background_score']:.3f}

Model-Clinical Alignment: {result['model_clinical_alignment']}
        """
        
        st.download_button(
            label="📝 Download Summary (TXT)",
            data=summary,
            file_name=f"clinical_summary_{clinical_report['patient_id']}.txt",
            mime="text/plain"
        )


def main():
    """Main clinical dashboard function."""
    st.markdown('<h1 class="clinical-header"> Clinical Cytology Classification System</h1>', 
                unsafe_allow_html=True)
    
    # Load clinical system
    classifier, device, status = load_clinical_system()
    
    if classifier is None:
        st.error("Clinical system could not be loaded.")
        return
    
    st.success(f"System Status: {status}")
    
    # Sidebar for image input
    st.sidebar.markdown("### Patient Sample Input")
    
    # Image upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload cytology slide image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
        help="Upload a cervical cytology image for clinical analysis"
    )
    
    # Demo image option
    use_demo = st.sidebar.checkbox("Use demo image", value=True)
    
    if use_demo:
        # Create clinical demo image
        demo_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        # Add some "cell-like" structures
        demo_image = cv2.circle(demo_image, (112, 112), 30, (200, 150, 100), -1)
        demo_image = cv2.circle(demo_image, (80, 80), 20, (150, 100, 80), -1)
        demo_image = cv2.circle(demo_image, (150, 140), 25, (180, 120, 90), -1)
        demo_image = Image.fromarray(demo_image)
        st.sidebar.info("Using synthetic demo image with cell-like structures")
    else:
        demo_image = None
    
    # Process image
    image_to_analyze = None
    if uploaded_file is not None:
        image_to_analyze = Image.open(uploaded_file).convert('RGB')
        st.sidebar.success("Cytology slide uploaded successfully")
    elif demo_image is not None:
        image_to_analyze = demo_image
    
    if image_to_analyze is not None:
        # Display original image
        st.markdown("### Cytology Slide")
        st.image(image_to_analyze, caption="Input Cytology Slide", width=400)
        
        # Analysis button
        if st.button("🔬 Analyze with Clinical AI", type="primary", use_container_width=True):
            with st.spinner("Performing comprehensive clinical analysis..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image_clinical(image_to_analyze)
                    
                    if image_tensor is None:
                        st.error("Failed to preprocess image")
                        return
                    
                    # Clinical prediction
                    result = classifier.predict_with_clinical_reasoning(image_tensor)
                    
                    # Store classifier for visualization
                    result['classifier'] = classifier
                    
                    # Convert image to numpy for overlays
                    original_np = np.array(image_to_analyze)
                    if original_np.max() <= 1.0:
                        original_np = (original_np * 255).astype(np.uint8)
                    
                    # Display results
                    st.markdown("---")
                    
                    # Section 1: Bethesda Classification
                    create_bethesda_display(result['bethesda_classification'])
                    
                    # Section 2: Feature Analysis
                    create_feature_analysis_panels(result)
                    
                    # Section 3: Clinical Reasoning
                    create_clinical_reasoning_panel(result)
                    
                    # Section 4: Visual Analysis
                    create_visual_analysis_section(result, original_np)
                    
                    # Section 5: Export
                    create_export_section(result)
                    
                except Exception as e:
                    st.error(f"Error during clinical analysis: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Clinical guidelines
    st.markdown("---")
    st.markdown("## Clinical Guidelines")
    
    st.markdown("""
    <div class="clinical-insight">
        <h4>🩺 Bethesda System Classification</h4>
        <p><strong>NILM:</strong> Negative for Intraepithelial Lesion or Malignancy</p>
        <p><strong>ASC-US:</strong> Atypical Squamous Cells of Undetermined Significance</p>
        <p><strong>LSIL:</strong> Low-grade Squamous Intraepithelial Lesion</p>
        <p><strong>HSIL:</strong> High-grade Squamous Intraepithelial Lesion</p>
        <p><strong>SCC:</strong> Squamous Cell Carcinoma</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="clinical-insight">
        <h4>🔬 Key Diagnostic Principles</h4>
        <ol>
            <li><strong>Multi-factor Classification:</strong> Nuclear + Cytoplasmic + Background</li>
            <li><strong>Cytoplasmic Features are Critical:</strong> Essential for LSIL detection</li>
            <li><strong>Context Matters:</strong> Same features → different classes based on combination</li>
            <li><strong>Background as Diagnostic Signal:</strong> Especially for HSIL and SCC</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
