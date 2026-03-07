#!/usr/bin/env python3
"""
Cervical Cancer Screening Dashboard
A production-quality Streamlit dashboard for model visualization and XAI explanations.

Author: Cervical Cancer Classification Pipeline
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.cm as cm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.model_factory import CervicalCancerModel
from preprocessing.transforms import get_val_transforms
from datasets.dataset import DatasetManager, DatasetType, ClassificationMode
from utils.label_mapping import LabelMapper
from utils.config_manager import ConfigManager
from test_gradcam import GradCAM

# Set page configuration
st.set_page_config(
    page_title="Cervical Cancer Screening Dashboard",
    page_icon=":material/dashboard:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2a9d8f;
        margin: 20px 0 10px 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header .material-icons {
        margin-right: 8px;
        font-size: 1.5rem;
    }
    .info-text {
        color: #34495e;
        font-size: 1rem;
        line-height: 1.6;
    }
    .sidebar-section {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    h1 a, h2 a, h3 a {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

def section_header(title, icon):
    """Create a section header with Material icon."""
    st.markdown(
        f''' <h2 class="section-header">
            <span class="material-icons">{icon}</span>
            {title}
        </h2>
        ''',
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model_and_config():
    """Load trained model and configuration with proper caching."""
    try:
        # Device detection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Create label mapper
        label_mapper = LabelMapper(ClassificationMode.BINARY)
        
        # Load model
        checkpoint_path = "outputs/models/best_model.pth"
        if not os.path.exists(checkpoint_path):
            st.error("Model checkpoint not found. Please train model first.")
            return None, None, None, device
        
        model = CervicalCancerModel(
            model_name=config['model']['architecture'],
            num_classes=config['model']['num_classes'],
            pretrained=False,
            dropout_rate=config['model']['dropout_rate'],
            freeze_backbone=config['model']['freeze_backbone']
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model, config, label_mapper, device
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, torch.device("cpu")

@st.cache_data
def load_evaluation_results():
    """Load evaluation metrics from pipeline output files."""
    try:
        metrics_path = "outputs/metrics/evaluation_results.json"
        
        if not os.path.exists(metrics_path):
            st.warning(f"Evaluation results file not found: {metrics_path}")
            st.info("Please run model evaluation first using: python main.py --binary --eval-only")
            return None
        
        with open(metrics_path, 'r') as f:
            results = json.load(f)
        
        # Convert confusion matrix to numpy array
        if 'confusion_matrix' in results:
            results['confusion_matrix'] = np.array(results['confusion_matrix'])
        
        # Validate required metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
        missing_metrics = [metric for metric in required_metrics if metric not in results]
        
        if missing_metrics:
            st.error(f"Missing required metrics in evaluation results: {missing_metrics}")
            return None
        
        return results
    
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")
        return None

@st.cache_data
def load_sample_images():
    """Load sample images for visualization with metadata from JSON file."""
    try:
        # Try to load metadata from Grad-CAM output
        metadata_path = "outputs/xai/gradcam/metadata.json"
        sample_dir = "outputs/xai/gradcam/"
        samples = []
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                sample_metadata = json.load(f)
        else:
            # Fallback to hardcoded metadata with image paths from actual dataset
            sample_metadata = [
                {"image_path": "datasets/sipakmed/archive/im_Metaplastic/im_Metaplastic/146.bmp", "true_label": "ABNORMAL"},
                {"image_path": "datasets/sipakmed/archive/im_Koilocytotic/im_Koilocytotic/025.bmp", "true_label": "ABNORMAL"},
                {"image_path": "datasets/sipakmed/archive/im_Koilocytotic/im_Koilocytotic/003.bmp", "true_label": "ABNORMAL"},
                {"image_path": "datasets/sipakmed/archive/im_Superficial-Intermediate/im_Superficial-Intermediate/121.bmp", "true_label": "NORMAL"},
                {"image_path": "datasets/sipakmed/archive/im_Metaplastic/im_Metaplastic/037.bmp", "true_label": "ABNORMAL"}
            ]
        
        if os.path.exists(sample_dir):
            for i, metadata in enumerate(sample_metadata, 1):
                base_path = os.path.join(sample_dir, f"gradcam_{i}")
                if os.path.exists(f"{base_path}_original.png"):
                    samples.append({
                        'original': f"{base_path}_original.png",
                        'heatmap': f"{base_path}_heatmap.png",
                        'overlay': f"{base_path}_overlay.png",
                        'combined': f"{base_path}.png",
                        'index': i,
                        'image_path': metadata['image_path'],
                        'true_label': metadata['true_label']
                    })
        
        return samples
    
    except Exception as e:
        st.error(f"Error loading sample images: {e}")
        return []

def create_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """Create confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_class_distribution_chart(class_dist: Dict[str, int]):
    """Create class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    colors = ['#3498db', '#e74c3c']  # Blue for NORMAL, Red for ABNORMAL
    
    bars = ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def display_sample_prediction(sample: Dict, model, label_mapper, transform, device):
    """Display sample prediction with XAI explanations using real model inference."""
    try:
        # Load images
        original_img = Image.open(sample['original'])
        heatmap_img = Image.open(sample['heatmap'])
        overlay_img = Image.open(sample['overlay'])
        
        # Get image path for inference
        image_path = sample.get('image_path')
        if not image_path or not os.path.exists(image_path):
            st.warning(f"Image not found: {image_path}")
            return
        
        # Preprocess image for model inference
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Run model inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_class_idx = predicted_class.item()
            confidence_score = confidence.item()
        
        # Get class names
        class_names = label_mapper.get_class_names()
        predicted_label = class_names[predicted_class_idx]
        
        # Get true label (this would need to be stored with the sample)
        true_label = sample.get('true_label', 'UNKNOWN')
        
        # Create columns for side-by-side display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(original_img, caption="Original Image", use_container_width=True)
        
        with col2:
            st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
        
        with col3:
            st.image(overlay_img, caption="Grad-CAM Overlay", use_container_width=True)
        
        # Display real prediction details
        st.markdown(f"""
        <div class="metric-card">
            <h4>Real Prediction Results (Sample {sample['index']})</h4>
            <p><strong>Predicted Class:</strong> {predicted_label}</p>
            <p><strong>True Class:</strong> {true_label}</p>
            <p><strong>Confidence Score:</strong> {confidence_score:.3f}</p>
            <p><strong>Model:</strong> {getattr(model, 'model_name', 'Unknown')}</p>
            <p><strong>Class Probabilities:</strong> 
            Normal: {probabilities[0][0].item():.3f} | 
            Abnormal: {probabilities[0][1].item():.3f}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing sample {sample.get('index', 'unknown')}: {e}")
        st.info("Make sure the model and image paths are correctly configured.")

def main():
    """Main dashboard function with sidebar navigation."""
    # Load data
    model, config, label_mapper, device = load_model_and_config()
    evaluation_results = load_evaluation_results()
    sample_images = load_sample_images()
    
    if model is None:
        st.error("Cannot load model. Please check the model checkpoint.")
        st.info("Make sure you have trained the model using: python main.py --binary")
        return
    
    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Overview", "Model Performance", "Confusion Matrix", "Dataset Overview", "Explainable AI Explorer"],
        index=0
    )
    
    # Header
    st.markdown('<h1 class="main-header">Cervical Cancer Screening Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Overview page
    if page == "Overview":
        section_header("Overview", "dashboard")
        
        if evaluation_results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{evaluation_results['accuracy']:.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{evaluation_results['precision']:.1%}</div>
                    <div class="metric-label">Precision</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{evaluation_results['recall']:.1%}</div>
                    <div class="metric-label">Recall</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{evaluation_results['f1_score']:.1%}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Model and dataset info
        if config:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Model Information</h4>
                    <p><strong>Architecture:</strong> {config['model']['architecture']}</p>
                    <p><strong>Pretrained:</strong> {'Yes' if config['model']['pretrained'] else 'No'}</p>
                    <p><strong>Backbone Frozen:</strong> {'Yes' if config['model']['freeze_backbone'] else 'No'}</p>
                    <p><strong>Dropout Rate:</strong> {config['model']['dropout_rate']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if evaluation_results and 'class_distribution' in evaluation_results:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Dataset Statistics</h4>
                        <p><strong>Total Samples:</strong> {evaluation_results.get('total_samples', 'N/A')}</p>
                        <p><strong>Classes:</strong> {len(label_mapper.get_class_names())} (Binary Classification)</p>
                        <p><strong>Dataset:</strong> {config.get('data', {}).get('dataset', config.get('train_dataset', 'Unknown'))}</p>
                        <p><strong>Image Size:</strong> {config.get('augmentation', {}).get('image_size', 224)}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Model Performance page
    elif page == "Model Performance":
        section_header("Model Performance", "bar_chart")
        
        if evaluation_results is None:
            st.warning("No evaluation results available. Please run model evaluation first.")
            st.info("Use: python main.py --binary --eval-only to generate evaluation results.")
        else:
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{evaluation_results['accuracy']:.2%}", 
                         help="Overall accuracy of the model")
            
            with col2:
                st.metric("Precision", f"{evaluation_results['precision']:.2%}",
                         help="Precision for positive predictions")
            
            with col3:
                st.metric("Recall", f"{evaluation_results['recall']:.2%}",
                         help="Recall for positive class")
            
            with col4:
                st.metric("F1 Score", f"{evaluation_results['f1_score']:.2%}",
                         help="Harmonic mean of precision and recall")
    
    # Confusion Matrix page
    elif page == "Confusion Matrix":
        section_header("Confusion Matrix", "grid_view")
        
        if evaluation_results is not None:
            class_names = label_mapper.get_class_names()
            cm_fig = create_confusion_matrix(evaluation_results['confusion_matrix'], class_names)
            st.pyplot(cm_fig)
        else:
            st.warning("No evaluation results available for confusion matrix.")
    
    # Dataset Overview page
    elif page == "Dataset Overview":
        section_header("Dataset Overview", "dataset")
        
        if evaluation_results is not None and 'class_distribution' in evaluation_results:
            # Class distribution chart
            dist_fig = create_class_distribution_chart(evaluation_results['class_distribution'])
            st.pyplot(dist_fig)
        else:
            st.warning("Dataset overview not available. Please run evaluation first.")
    
    # Explainable AI Explorer page
    elif page == "Explainable AI Explorer":
        section_header("Explainable AI Explorer", "psychology")
        
        if not sample_images:
            st.warning("No sample images found. Please run Grad-CAM visualization first.")
            st.info("Use: python test_gradcam.py to generate XAI visualizations.")
            return
        
        # Sample selector
        selected_sample = st.selectbox(
            "Select a sample to explore:",
            options=range(len(sample_images)),
            format_func=lambda i: f"Sample {i+1}",
            index=0
        )
        
        # Display selected sample
        sample = sample_images[selected_sample]
        display_sample_prediction(sample, model, label_mapper, get_val_transforms(), device)
        
        # XAI explanation
        st.markdown("""
        <div class="info-text">
        <h4>Grad-CAM Explanations</h4>
        <p>Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the spatial regions 
        in the image that were most influential for the model's prediction. The red areas indicate 
        regions where the model focused its attention when making the classification decision.</p>
        
        <h4>How to Interpret</h4>
        <ul>
            <li><strong>Original Image:</strong> The input Pap smear cell image</li>
            <li><strong>Heatmap:</strong> Shows attention intensity (red = high attention)</li>
            <li><strong>Overlay:</strong> Combines original image with heatmap for interpretation</li>
        </ul>
        
        <p>This helps clinicians understand which cellular features (nuclei, cytoplasm, etc.) 
        influenced the model's decision, increasing trust and transparency in AI-assisted diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
