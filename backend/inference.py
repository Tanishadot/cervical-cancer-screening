#!/usr/bin/env python3
"""
Inference Engine for Cervical Cytology Classification
Combines model prediction, feature extraction, and clinical reasoning.

Author: Cervical Cancer Classification Pipeline
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from PIL import Image
import cv2

from .feature_extraction import CytologyFeatureExtractor
from .model import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalReasoningEngine:
    """Clinical reasoning engine for Bethesda classification."""
    
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
    
    def classify_bethesda(self, fused_result: Dict) -> Dict:
        """
        Classify according to Bethesda system with clinical reasoning.
        
        Args:
            fused_result: Result from feature fusion
            
        Returns:
            Bethesda classification with reasoning
        """
        try:
            # Get feature scores
            nuclear_score = fused_result['extracted_features']['nuclear']['overall_nuclear_score']
            cytoplasmic_score = fused_result['extracted_features']['cytoplasmic']['overall_cytoplasmic_score']
            background_score = fused_result['extracted_features']['background']['overall_background_score']
            final_score = fused_result['final_score']
            
            # Get specific features for reasoning
            nuclear_features = fused_result['extracted_features']['nuclear']
            cytoplasmic_features = fused_result['extracted_features']['cytoplasmic']
            background_features = fused_result['extracted_features']['background']
            
            # Determine Bethesda class
            bethesda_class, bethesda_full, risk_level = self._determine_class(final_score)
            
            # Generate clinical reasoning
            reasoning = self._generate_reasoning(
                nuclear_features, cytoplasmic_features, background_features, 
                bethesda_class, final_score
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(final_score, bethesda_class)
            
            return {
                'bethesda_class': bethesda_class,
                'bethesda_full': bethesda_full,
                'clinical_score': final_score,
                'model_confidence': fused_result['final_confidence'],
                'risk_level': risk_level,
                'clinical_reasoning': reasoning,
                'feature_scores': {
                    'nuclear': nuclear_score,
                    'cytoplasmic': cytoplasmic_score,
                    'background': background_score
                },
                'confidence': confidence,
                'dominant_feature': fused_result['dominant_feature']
            }
            
        except Exception as e:
            logger.error(f"Error in clinical reasoning: {e}")
            return self._get_default_bethesda_result()
    
    def _determine_class(self, score: float) -> Tuple[str, str, str]:
        """Determine Bethesda class from score."""
        if score <= self.thresholds['nilm_upper']:
            return 'NILM', 'Negative for Intraepithelial Lesion or Malignancy', 'Low Risk - Normal findings'
        elif score <= self.thresholds['ascus_upper']:
            return 'ASC-US', 'Atypical Squamous Cells of Undetermined Significance', 'Mild Risk - Monitor/Repeat'
        elif score <= self.thresholds['lsil_upper']:
            return 'LSIL', 'Low-grade Squamous Intraepithelial Lesion', 'Moderate Risk - Treatment considered'
        elif score <= self.thresholds['hsil_upper']:
            return 'HSIL', 'High-grade Squamous Intraepithelial Lesion', 'High Risk - Immediate treatment'
        else:
            return 'SCC', 'Squamous Cell Carcinoma', 'Very High Risk - Urgent intervention'
    
    def _generate_reasoning(self, nuclear: Dict, cytoplasmic: Dict, background: Dict, 
                          bethesda_class: str, final_score: float) -> str:
        """Generate clinical reasoning based on features."""
        reasoning_parts = []
        
        # Nuclear-based reasoning
        if nuclear['hyperchromasia'] > 0.5:
            reasoning_parts.append("Hyperchromasia (dense chromatin) present")
        if nuclear['nuclear_enlargement'] > 0.5:
            reasoning_parts.append("Nuclear enlargement detected")
        if nuclear['nc_ratio'] > 0.6:
            reasoning_parts.append("Increased N:C ratio")
        if nuclear['nuclear_contours'] > 0.5:
            reasoning_parts.append("Irregular nuclear contours")
        
        # Cytoplasmic-based reasoning
        if cytoplasmic['perinuclear_halo'] > 0.5:
            reasoning_parts.append("Perinuclear halo (koilocytosis) - HPV indicator")
        if cytoplasmic['keratinization'] > 0.4:
            reasoning_parts.append("Keratinization (dyskeratosis) present")
        if cytoplasmic['koilocytosis_score'] > 0.6:
            reasoning_parts.append("Strong koilocytosis features")
        
        # Background-based reasoning
        if background['background_debris'] > 0.4:
            reasoning_parts.append("Background debris present")
        if background['tumor_diathesis'] > 0.3:
            reasoning_parts.append("Tumor diathesis (necrotic background)")
        if background['background_cleanliness'] > 0.7:
            reasoning_parts.append("Relatively clean background")
        
        # Class-specific reasoning
        if bethesda_class == 'LSIL':
            if cytoplasmic['perinuclear_halo'] > 0.5:
                reasoning_parts.append("Consistent with low-grade HPV-related changes")
            else:
                reasoning_parts.append("Low-grade changes detected")
        elif bethesda_class == 'HSIL':
            if nuclear['nc_ratio'] > 0.6:
                reasoning_parts.append("High-grade changes with marked atypia")
            else:
                reasoning_parts.append("High-grade epithelial changes")
        elif bethesda_class == 'SCC':
            reasoning_parts.append("Features consistent with invasive carcinoma")
        elif bethesda_class == 'NILM':
            reasoning_parts.append("No significant pathological changes detected")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Multiple subtle features detected"
    
    def _calculate_confidence(self, score: float, bethesda_class: str) -> float:
        """Calculate confidence based on score and class."""
        base_confidence = abs(score - 0.5) * 2  # Distance from 0.5
        
        # Adjust confidence based on class
        if bethesda_class == 'NILM':
            confidence = base_confidence * 0.8 + 0.2
        elif bethesda_class == 'SCC':
            confidence = base_confidence * 0.9 + 0.1
        else:
            confidence = base_confidence * 0.7 + 0.3
        
        return min(max(confidence, 0.5), 0.95)
    
    def _get_default_bethesda_result(self) -> Dict:
        """Return default Bethesda result when error occurs."""
        return {
            'bethesda_class': 'ASC-US',
            'bethesda_full': 'Atypical Squamous Cells of Undetermined Significance',
            'clinical_score': 0.4,
            'model_confidence': 0.6,
            'risk_level': 'Mild Risk - Monitor/Repeat',
            'clinical_reasoning': 'Analysis incomplete - manual review recommended',
            'feature_scores': {'nuclear': 0.3, 'cytoplasmic': 0.3, 'background': 0.3},
            'confidence': 0.5,
            'dominant_feature': 'model'
        }


class InferenceEngine:
    """Main inference engine combining all components."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.feature_extractor = CytologyFeatureExtractor()
        self.model_manager = ModelManager(model_path, device)
        self.reasoning_engine = ClinicalReasoningEngine()
    
    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (tensor, numpy_array)
        """
        try:
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to standard size
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Normalize to [0, 1]
            image_tensor = torch.from_numpy(image_np).float() / 255.0
            
            # Permute dimensions: [H, W, C] -> [C, H, W]
            image_tensor = image_tensor.permute(2, 0, 1)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor, image_np
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return dummy tensor
            return torch.randn(1, 3, 224, 224), np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Complete inference pipeline.
        
        Args:
            image: PIL Image
            
        Returns:
            Comprehensive prediction results
        """
        try:
            # Preprocess image
            image_tensor, image_np = self.preprocess_image(image)
            
            # Extract features
            extracted_features = self.feature_extractor.extract_features(image_np)
            
            # Model prediction with feature fusion
            fused_result = self.model_manager.predict_with_features(image_tensor, extracted_features)
            
            # Clinical reasoning
            bethesda_result = self.reasoning_engine.classify_bethesda(fused_result)
            
            # Combine all results
            final_result = {
                'image_info': {
                    'shape': image_np.shape,
                    'dtype': str(image_np.dtype)
                },
                'extracted_features': extracted_features,
                'model_prediction': fused_result['model_prediction'],
                'feature_fusion': fused_result,
                'bethesda_classification': bethesda_result,
                'visualizations': {
                    'grad_cam': fused_result['grad_cam'],
                    'segmentation_masks': extracted_features['segmentation_masks']
                },
                'summary': {
                    'class': bethesda_result['bethesda_class'],
                    'confidence': bethesda_result['confidence'],
                    'reasoning': bethesda_result['clinical_reasoning'],
                    'dominant_feature': bethesda_result['dominant_feature']
                }
            }
            
            logger.info(f"Inference completed: {bethesda_result['bethesda_class']} with {bethesda_result['confidence']:.3f} confidence")
            return final_result
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict:
        """Return default result when inference fails."""
        return {
            'image_info': {'shape': (224, 224, 3), 'dtype': 'uint8'},
            'extracted_features': self.feature_extractor._get_default_features(),
            'model_prediction': {'prediction': 0, 'confidence': 0.5},
            'feature_fusion': {'final_class': 0, 'final_score': 0.5},
            'bethesda_classification': self.reasoning_engine._get_default_bethesda_result(),
            'visualizations': {
                'grad_cam': np.zeros((224, 224)),
                'segmentation_masks': {
                    'nucleus': np.zeros((224, 224), dtype=np.uint8),
                    'cytoplasm': np.zeros((224, 224), dtype=np.uint8),
                    'background': np.ones((224, 224), dtype=np.uint8) * 255
                }
            },
            'summary': {
                'class': 'ASC-US',
                'confidence': 0.5,
                'reasoning': 'Analysis incomplete - manual review recommended',
                'dominant_feature': 'model'
            }
        }


def create_inference_engine(model_path: Optional[str] = None, device: str = 'cpu') -> InferenceEngine:
    """
    Create and return an inference engine instance.
    
    Args:
        model_path: Path to trained model
        device: Device to run on
        
    Returns:
        InferenceEngine instance
    """
    return InferenceEngine(model_path, device)


if __name__ == "__main__":
    # Test inference engine
    engine = create_inference_engine()
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Test inference
    result = engine.predict(test_image)
    print("Inference engine test completed")
    print(f"Class: {result['summary']['class']}")
    print(f"Confidence: {result['summary']['confidence']:.3f}")
    print(f"Reasoning: {result['summary']['reasoning']}")
