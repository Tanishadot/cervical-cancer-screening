#!/usr/bin/env python3
"""
Clinical Cytology Dashboard with Bethesda-Aligned Classification
Simplified version for demonstration.

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


def create_demo_clinical_analysis():
    """Create demo clinical analysis for demonstration."""
    # Simulate clinical features
    nuclear_features = {
        'nuclear_enlargement': np.random.uniform(0.2, 0.8),
        'chromatin_density': np.random.uniform(0.1, 0.7),
        'nuclear_contours': np.random.uniform(0.3, 0.9),
        'nc_ratio': np.random.uniform(0.4, 0.8),
        'hyperchromasia': np.random.uniform(0.2, 0.6),
        'overall_nuclear_score': np.random.uniform(0.3, 0.7)
    }
    
    cytoplasmic_features = {
        'perinuclear_halo': np.random.uniform(0.1, 0.9),
        'keratinization': np.random.uniform(0.2, 0.8),
        'cytoplasmic_texture': np.random.uniform(0.3, 0.7),
        'cell_maturity': np.random.uniform(0.4, 0.8),
        'koilocytosis_score': np.random.uniform(0.2, 0.7),
        'overall_cytoplasmic_score': np.random.uniform(0.4, 0.8)
    }
    
    background_features = {
        'background_debris': np.random.uniform(0.1, 0.6),
        'inflammatory_cells': np.random.uniform(0.2, 0.7),
        'tumor_diathesis': np.random.uniform(0.0, 0.4),
        'background_cleanliness': np.random.uniform(0.3, 0.9),
        'overall_background_score': np.random.uniform(0.2, 0.6)
    }
    
    # Determine Bethesda class
    clinical_score = (nuclear_features['overall_nuclear_score'] + 
                    cytoplasmic_features['overall_cytoplasmic_score'] + 
                    background_features['overall_background_score']) / 3
    
    if clinical_score < 0.3:
        bethesda_class = 'NILM'
        bethesda_full = 'Negative for Intraepithelial Lesion or Malignancy'
        risk_level = 'Low Risk - Normal findings'
    elif clinical_score < 0.5:
        bethesda_class = 'ASC-US'
        bethesda_full = 'Atypical Squamous Cells of Undetermined Significance'
        risk_level = 'Mild Risk - Monitor/Repeat'
    elif clinical_score < 0.7:
        bethesda_class = 'LSIL'
        bethesda_full = 'Low-grade Squamous Intraepithelial Lesion'
        risk_level = 'Moderate Risk - Treatment considered'
    elif clinical_score < 0.9:
        bethesda_class = 'HSIL'
        bethesda_full = 'High-grade Squamous Intraepithelial Lesion'
        risk_level = 'High Risk - Immediate treatment'
    else:
        bethesda_class = 'SCC'
        bethesda_full = 'Squamous Cell Carcinoma'
        risk_level = 'Very High Risk - Urgent intervention'
    
    # Generate reasoning
    reasoning_parts = []
    if nuclear_features['hyperchromasia'] > 0.4:
        reasoning_parts.append("Hyperchromasia (dense chromatin) present")
    if cytoplasmic_features['perinuclear_halo'] > 0.5:
        reasoning_parts.append("Perinuclear halo (koilocytosis) - HPV indicator")
    if background_features['background_debris'] > 0.4:
        reasoning_parts.append("Background debris present")
    
    if bethesda_class == 'LSIL':
        reasoning_parts.append("Consistent with low-grade HPV-related changes")
    elif bethesda_class == 'HSIL':
        reasoning_parts.append("High-grade changes with marked atypia")
    
    clinical_reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Multiple subtle features detected"
    
    return {
        'nuclear_analysis': nuclear_features,
        'cytoplasmic_analysis': cytoplasmic_features,
        'background_analysis': background_features,
        'bethesda_classification': {
            'bethesda_class': bethesda_class,
            'bethesda_full': bethesda_full,
            'clinical_score': clinical_score,
            'model_confidence': np.random.uniform(0.7, 0.95),
            'risk_level': risk_level
        },
        'clinical_reasoning': clinical_reasoning,
        'dominant_feature_type': 'cytoplasmic' if cytoplasmic_features['overall_cytoplasmic_score'] > nuclear_features['overall_nuclear_score'] else 'nuclear',
        'confidence': np.random.uniform(0.7, 0.95)
    }


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
            <p><strong>Model-Clinical Alignment:</strong> High alignment between model and clinical reasoning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="clinical-insight">
            <h4>📋 Diagnostic Reasoning</h4>
            <p>{result['clinical_reasoning']}</p>
        </div>
        """, unsafe_allow_html=True)


def create_visual_analysis_section(result):
    """Create comprehensive visual analysis."""
    st.markdown("## Visual Analysis")
    
    # Feature contribution chart
    feature_scores = [
        result['nuclear_analysis']['overall_nuclear_score'],
        result['cytoplasmic_analysis']['overall_cytoplasmic_score'],
        result['background_analysis']['overall_background_score']
    ]
    feature_labels = ['Nuclear', 'Cytoplasmic', 'Background']
    feature_colors = ['#ef4444', '#f59e0b', '#3b82f6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
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
    st.pyplot(fig)
    plt.close()
    
    # Create demo visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Demo heatmap
        demo_heatmap = np.random.rand(224, 224)
        st.image(demo_heatmap, caption="CNN Grad-CAM (Nuclear Features)", use_container_width=True)
    
    with col2:
        # Demo attention map
        demo_attention = np.random.rand(224, 224)
        st.image(demo_attention, caption="Transformer Attention (Cytoplasmic Features)", use_container_width=True)


def create_export_section(result):
    """Create export section for clinical reports."""
    st.markdown("## Export Clinical Report")
    
    clinical_report = {
        'patient_id': f"PAT_{np.random.randint(10000, 99999)}",
        'analysis_date': str(datetime.datetime.now()),
        'bethesda_classification': result['bethesda_classification'],
        'feature_analysis': {
            'nuclear': result['nuclear_analysis'],
            'cytoplasmic': result['cytoplasmic_analysis'],
            'background': result['background_analysis']
        },
        'clinical_reasoning': result['clinical_reasoning'],
        'risk_assessment': result['bethesda_classification']['risk_level'],
        'model_confidence': result['confidence'],
        'dominant_feature': result['dominant_feature_type']
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

Risk Assessment: {result['bethesda_classification']['risk_level']}
Model Confidence: {result['confidence']:.3f}

Clinical Reasoning:
{result['clinical_reasoning']}

Feature Analysis:
- Nuclear Score: {result['nuclear_analysis']['overall_nuclear_score']:.3f}
- Cytoplasmic Score: {result['cytoplasmic_analysis']['overall_cytoplasmic_score']:.3f}
- Background Score: {result['background_analysis']['overall_background_score']:.3f}

Dominant Feature Type: {result['dominant_feature_type'].title()}
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
    
    st.info("🩺 Bethesda-Aligned Clinical Analysis System")
    st.info("Demonstrating multi-factor classification with clinical reasoning")
    
    # Sidebar for image input
    st.sidebar.markdown("### Patient Sample Input")
    
    # Demo image option
    use_demo = st.sidebar.checkbox("Use demo analysis", value=True)
    
    if use_demo:
        # Create clinical demo image
        demo_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        # Add some "cell-like" structures
        demo_image = cv2.circle(demo_image, (112, 112), 30, (200, 150, 100), -1)
        demo_image = cv2.circle(demo_image, (80, 80), 20, (150, 100, 80), -1)
        demo_image = cv2.circle(demo_image, (150, 140), 25, (180, 120, 90), -1)
        demo_image = Image.fromarray(demo_image)
        st.sidebar.info("Using synthetic demo with clinical features")
        
        # Display original image
        st.markdown("### Cytology Slide")
        st.image(demo_image, caption="Input Cytology Slide", width=400)
        
        # Analysis button
        if st.button("🔬 Analyze with Clinical AI", type="primary", use_container_width=True):
            with st.spinner("Performing comprehensive clinical analysis..."):
                try:
                    # Generate demo clinical analysis
                    result = create_demo_clinical_analysis()
                    
                    # Display results
                    st.markdown("---")
                    
                    # Section 1: Bethesda Classification
                    create_bethesda_display(result['bethesda_classification'])
                    
                    # Section 2: Feature Analysis
                    create_feature_analysis_panels(result)
                    
                    # Section 3: Clinical Reasoning
                    create_clinical_reasoning_panel(result)
                    
                    # Section 4: Visual Analysis
                    create_visual_analysis_section(result)
                    
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
