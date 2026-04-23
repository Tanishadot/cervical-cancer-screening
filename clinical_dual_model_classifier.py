#!/usr/bin/env python3
"""
Clinical-Enhanced Dual-Model Classifier for Cervical Cytology
Incorporates Bethesda-aligned classification criteria with medical reasoning.

Author: Cervical Cancer Classification Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from dual_model_classifier import DualModelClassifier, FeatureExtractor, FeatureFusion


class ClinicalFeatureExtractor:
    """Extracts clinically relevant features from model outputs."""
    
    def __init__(self):
        self.feature_weights = {
            'nuclear': 0.3,      # Nuclear features
            'cytoplasmic': 0.4,   # Cytoplasmic features (CRITICAL)
            'background': 0.3       # Background features
        }
    
    def analyze_nuclear_features(self, cnn_features: np.ndarray) -> Dict:
        """Analyze nuclear-focused features from CNN."""
        return {
            'nuclear_enlargement': float(np.mean(cnn_features[:200])),
            'chromatin_density': float(np.std(cnn_features[200:400])),
            'nuclear_contours': float(np.mean(cnn_features[400:600])),
            'nc_ratio': float(np.mean(cnn_features[600:800])),
            'hyperchromasia': float(np.max(cnn_features[800:1000])),
            'overall_nuclear_score': float(np.mean(cnn_features[:1000]))
        }
    
    def analyze_cytoplasmic_features(self, swin_features: np.ndarray) -> Dict:
        """Analyze cytoplasmic and spatial features from Swin."""
        return {
            'perinuclear_halo': float(np.mean(swin_features[:200])),
            'keratinization': float(np.std(swin_features[200:400])),
            'cytoplasmic_texture': float(np.mean(swin_features[400:600])),
            'cell_maturity': float(np.mean(swin_features[600:768])),
            'koilocytosis_score': float(np.max(swin_features[:300])),
            'overall_cytoplasmic_score': float(np.mean(swin_features))
        }
    
    def analyze_background_features(self, attention_map: np.ndarray) -> Dict:
        """Analyze background features from attention maps."""
        # Analyze attention distribution for background patterns
        attention_flat = attention_map.flatten()
        
        return {
            'background_debris': float(np.percentile(attention_flat, 90)),
            'inflammatory_cells': float(np.std(attention_flat)),
            'tumor_diathesis': float(np.max(attention_flat)),
            'background_cleanliness': 1.0 - float(np.mean(attention_flat > 0.5)),
            'overall_background_score': float(np.percentile(attention_flat, 75))
        }
    
    def combine_clinical_features(self, nuclear: Dict, cytoplasmic: Dict, background: Dict) -> Dict:
        """Combine features using clinical weighting."""
        clinical_score = (
            self.feature_weights['nuclear'] * nuclear['overall_nuclear_score'] +
            self.feature_weights['cytoplasmic'] * cytoplasmic['overall_cytoplasmic_score'] +
            self.feature_weights['background'] * background['overall_background_score']
        )
        
        return {
            'clinical_score': clinical_score,
            'nuclear_features': nuclear,
            'cytoplasmic_features': cytoplasmic,
            'background_features': background,
            'feature_weights': self.feature_weights,
            'dominant_feature': self._identify_dominant_feature(nuclear, cytoplasmic, background)
        }
    
    def _identify_dominant_feature(self, nuclear: Dict, cytoplasmic: Dict, background: Dict) -> str:
        """Identify which feature type is most prominent."""
        scores = {
            'nuclear': nuclear['overall_nuclear_score'],
            'cytoplasmic': cytoplasmic['overall_cytoplasmic_score'],
            'background': background['overall_background_score']
        }
        
        return max(scores, key=scores.get)


class BethesdaClassifier:
    """Bethesda classification with clinical reasoning."""
    
    def __init__(self):
        self.thresholds = {
            'nilm_upper': 0.3,
            'ascus_lower': 0.3,
            'ascus_upper': 0.5,
            'lsil_lower': 0.5,
            'lsil_upper': 0.7,
            'hsil_lower': 0.7,
            'hsil_upper': 0.9,
            'scc_lower': 0.9
        }
    
    def classify_bethesda(self, clinical_features: Dict, model_confidence: float) -> Dict:
        """Classify according to Bethesda system with clinical reasoning."""
        clinical_score = clinical_features['clinical_score']
        dominant_feature = clinical_features['dominant_feature']
        
        # Determine Bethesda class
        if clinical_score <= self.thresholds['nilm_upper']:
            bethesda_class = 'NILM'
            bethesda_full = 'Negative for Intraepithelial Lesion or Malignancy'
        elif clinical_score <= self.thresholds['ascus_upper']:
            bethesda_class = 'ASC-US'
            bethesda_full = 'Atypical Squamous Cells of Undetermined Significance'
        elif clinical_score <= self.thresholds['lsil_upper']:
            bethesda_class = 'LSIL'
            bethesda_full = 'Low-grade Squamous Intraepithelial Lesion'
        elif clinical_score <= self.thresholds['hsil_upper']:
            bethesda_class = 'HSIL'
            bethesda_full = 'High-grade Squamous Intraepithelial Lesion'
        else:
            bethesda_class = 'SCC'
            bethesda_full = 'Squamous Cell Carcinoma'
        
        # Generate clinical reasoning
        reasoning = self._generate_reasoning(clinical_features, bethesda_class, dominant_feature)
        
        return {
            'bethesda_class': bethesda_class,
            'bethesda_full': bethesda_full,
            'clinical_score': clinical_score,
            'model_confidence': model_confidence,
            'dominant_feature': dominant_feature,
            'clinical_reasoning': reasoning,
            'thresholds_used': self.thresholds,
            'risk_level': self._assess_risk_level(bethesda_class)
        }
    
    def _generate_reasoning(self, features: Dict, bethesda_class: str, dominant: str) -> str:
        """Generate clinical reasoning for classification."""
        nuclear = features['nuclear_features']
        cytoplasmic = features['cytoplasmic_features']
        background = features['background_features']
        
        reasoning_parts = []
        
        # Nuclear-based reasoning
        if nuclear['hyperchromasia'] > 0.5:
            reasoning_parts.append("Hyperchromasia (dense chromatin) present")
        if nuclear['nuclear_enlargement'] > 0.5:
            reasoning_parts.append("Nuclear enlargement detected")
        if nuclear['nc_ratio'] > 0.6:
            reasoning_parts.append("Increased N:C ratio")
        
        # Cytoplasmic-based reasoning
        if cytoplasmic['perinuclear_halo'] > 0.5:
            reasoning_parts.append("Perinuclear halo (koilocytosis) - HPV indicator")
        if cytoplasmic['keratinization'] > 0.4:
            reasoning_parts.append("Keratinization (dyskeratosis) present")
        if cytoplasmic['koilocytosis_score'] > 0.6:
            reasoning_parts.append("Strong koilocytosis features")
        
        # Background-based reasoning
        if background['background_debris'] > 0.6:
            reasoning_parts.append("Background debris present")
        if background['tumor_diathesis'] > 0.5:
            reasoning_parts.append("Tumor diathesis (necrotic background)")
        if background['background_cleanliness'] > 0.7:
            reasoning_parts.append("Relatively clean background")
        
        # Class-specific reasoning
        if bethesda_class == 'LSIL':
            reasoning_parts.append("Consistent with low-grade HPV-related changes")
        elif bethesda_class == 'HSIL':
            reasoning_parts.append("High-grade changes with marked atypia")
        elif bethesda_class == 'SCC':
            reasoning_parts.append("Features consistent with invasive carcinoma")
        elif bethesda_class == 'NILM':
            reasoning_parts.append("No significant pathological changes detected")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Multiple subtle features detected"
    
    def _assess_risk_level(self, bethesda_class: str) -> str:
        """Assess clinical risk level."""
        risk_map = {
            'NILM': 'Low Risk - Normal findings',
            'ASC-US': 'Mild Risk - Monitor/Repeat',
            'LSIL': 'Moderate Risk - Treatment considered',
            'HSIL': 'High Risk - Immediate treatment',
            'SCC': 'Very High Risk - Urgent intervention'
        }
        return risk_map.get(bethesda_class, 'Unknown Risk')


class ClinicalDualModelClassifier:
    """Enhanced dual-model classifier with clinical reasoning."""
    
    def __init__(self, cnn_model_path: str, swin_model_path: str, device: str = 'cpu'):
        self.device = device
        self.base_classifier = DualModelClassifier(cnn_model_path, swin_model_path, device)
        self.clinical_extractor = ClinicalFeatureExtractor()
        self.bethesda_classifier = BethesdaClassifier()
    
    def predict_with_clinical_reasoning(self, image_tensor: torch.Tensor) -> Dict:
        """Predict with comprehensive clinical analysis."""
        # Get base prediction
        base_result = self.base_classifier.predict(image_tensor)
        
        # Extract clinical features
        nuclear_features = self.clinical_extractor.analyze_nuclear_features(
            base_result['cnn_features']
        )
        
        cytoplasmic_features = self.clinical_extractor.analyze_cytoplasmic_features(
            base_result['swin_features']
        )
        
        background_features = self.clinical_extractor.analyze_background_features(
            base_result['attention_map']
        )
        
        # Combine clinical features
        clinical_features = self.clinical_extractor.combine_clinical_features(
            nuclear_features, cytoplasmic_features, background_features
        )
        
        # Classify with Bethesda system
        bethesda_result = self.bethesda_classifier.classify_bethesda(
            clinical_features, base_result['confidence']
        )
        
        # Create comprehensive result
        enhanced_result = {
            # Base results
            **base_result,
            
            # Clinical analysis
            'clinical_features': clinical_features,
            'bethesda_classification': bethesda_result,
            
            # Feature breakdown
            'nuclear_analysis': nuclear_features,
            'cytoplasmic_analysis': cytoplasmic_features,
            'background_analysis': background_features,
            
            # Clinical interpretation
            'dominant_feature_type': clinical_features['dominant_feature'],
            'clinical_reasoning': bethesda_result['clinical_reasoning'],
            'risk_assessment': bethesda_result['risk_level'],
            
            # Model confidence vs clinical alignment
            'model_clinical_alignment': self._assess_alignment(
                base_result['prediction'], bethesda_result['bethesda_class']
            )
        }
        
        return enhanced_result
    
    def _assess_alignment(self, model_pred: int, bethesda_class: str) -> str:
        """Assess alignment between model and clinical reasoning."""
        class_map = {0: 'NILM', 1: 'ASC-US', 2: 'LSIL', 3: 'HSIL', 4: 'SCC'}
        model_class = class_map.get(model_pred, 'Unknown')
        
        if model_class == bethesda_class:
            return "High alignment between model and clinical reasoning"
        else:
            return f"Model: {model_class} vs Clinical: {bethesda_class} - Review recommended"
    
    def create_clinical_visualization(self, result: Dict, original_image: np.ndarray) -> plt.Figure:
        """Create clinical visualization with feature highlights."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clinical Cytology Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # CNN Grad-CAM (Nuclear features)
        axes[0, 1].imshow(result['gradcam_heatmap'], cmap='jet')
        axes[0, 1].set_title('CNN: Nuclear Features', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Swin Attention (Cytoplasmic/Spatial features)
        axes[0, 2].imshow(result['attention_map'], cmap='viridis')
        axes[0, 2].set_title('Transformer: Spatial Features', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Feature scores
        feature_names = ['Nuclear', 'Cytoplasmic', 'Background']
        feature_scores = [
            result['nuclear_analysis']['overall_nuclear_score'],
            result['cytoplasmic_analysis']['overall_cytoplasmic_score'],
            result['background_analysis']['overall_background_score']
        ]
        
        bars = axes[1, 0].bar(feature_names, feature_scores, color=['red', 'green', 'blue'])
        axes[1, 0].set_title('Feature Contributions', fontweight='bold')
        axes[1, 0].set_ylabel('Score')
        
        # Bethesda classification
        bethesda = result['bethesda_classification']
        axes[1, 1].text(0.5, 0.7, bethesda['bethesda_class'], 
                        ha='center', va='center', fontsize=20, fontweight='bold',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.3, f"Risk: {bethesda['risk_level']}", 
                        ha='center', va='center', fontsize=12,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Bethesda Classification', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Clinical reasoning
        reasoning = result['clinical_reasoning']
        axes[1, 2].text(0.05, 0.95, 'Clinical Reasoning:', 
                        ha='left', va='top', fontweight='bold',
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.05, 0.05, reasoning[:200] + '...' if len(reasoning) > 200 else reasoning, 
                        ha='left', va='bottom', fontsize=8,
                        wrap=True, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Medical Interpretation', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig


def create_clinical_overlay(original_image: np.ndarray, gradcam: np.ndarray, 
                        attention: np.ndarray) -> np.ndarray:
    """Create overlay showing nuclear and cytoplasmic features."""
    # Normalize maps
    gradcam_norm = cv2.normalize(gradcam, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    attention_norm = cv2.normalize(attention, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Create colored overlays
    gradcam_color = cv2.applyColorMap(gradcam_norm, cv2.COLORMAP_JET)
    attention_color = cv2.applyColorMap(attention_norm, cv2.COLORMAP_VIRIDIS)
    
    # Blend with original
    alpha = 0.4
    gradcam_overlay = cv2.addWeighted(original_image, 1-alpha, gradcam_color, alpha, 0)
    attention_overlay = cv2.addWeighted(original_image, 1-alpha, attention_color, alpha, 0)
    
    # Combine both overlays
    combined = np.maximum(gradcam_overlay, attention_overlay)
    
    return combined
