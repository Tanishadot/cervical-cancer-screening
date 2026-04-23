#!/usr/bin/env python3
"""
Dual Model Inference Engine for Cytology Classification
Integrates CNN and Swin Transformer with clinical reasoning.

Author: Cervical Cancer Classification Pipeline
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from PIL import Image
import cv2

from .dual_model_integration import create_dual_model
from .dual_model_explainability import create_dual_explainer
from .feature_extraction import CytologyFeatureExtractor
from .inference import ClinicalReasoningEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualModelInferenceEngine:
    """Inference engine with dual model integration."""
    
    def __init__(self, cnn_model_path: Optional[str] = None, swin_model_path: Optional[str] = None, 
                 device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Initialize dual model
        self.dual_model = create_dual_model(device=self.device)
        
        # Load models if paths provided
        if cnn_model_path and swin_model_path:
            self._load_models(cnn_model_path, swin_model_path)
        else:
            logger.info("Using randomly initialized models for demo")
        
        # Initialize explainers
        self.explainer = create_dual_explainer(
            self.dual_model.cnn_model, 
            self.dual_model.swin_model, 
            device
        )
        
        # Initialize feature extractor and clinical reasoning
        self.feature_extractor = CytologyFeatureExtractor()
        self.clinical_reasoning = ClinicalReasoningEngine()
        
        # Set models to eval mode
        self.dual_model.eval()
    
    def _load_models(self, cnn_path: str, swin_path: str) -> bool:
        """Load pretrained models."""
        try:
            # Load CNN model
            if Path(cnn_path).exists():
                cnn_checkpoint = torch.load(cnn_path, map_location=self.device)
                # Handle different checkpoint formats
                if 'state_dict' in cnn_checkpoint:
                    self.dual_model.cnn_model.load_state_dict(cnn_checkpoint['state_dict'])
                elif 'model_state_dict' in cnn_checkpoint:
                    self.dual_model.cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
                else:
                    self.dual_model.cnn_model.load_state_dict(cnn_checkpoint)
                logger.info(f"CNN model loaded from {cnn_path}")
            else:
                logger.warning(f"CNN model path not found: {cnn_path}")
            
            # Load Swin model
            if Path(swin_path).exists():
                swin_checkpoint = torch.load(swin_path, map_location=self.device)
                # Handle different checkpoint formats
                if 'state_dict' in swin_checkpoint:
                    self.dual_model.swin_model.load_state_dict(swin_checkpoint['state_dict'])
                elif 'model_state_dict' in swin_checkpoint:
                    self.dual_model.swin_model.load_state_dict(swin_checkpoint['model_state_dict'])
                else:
                    self.dual_model.swin_model.load_state_dict(swin_checkpoint)
                logger.info(f"Swin model loaded from {swin_path}")
            else:
                logger.warning(f"Swin model path not found: {swin_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess image for dual model inference."""
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
    
    def predict_with_dual_models(self, image: Image.Image) -> Dict:
        """
        Complete dual model inference pipeline.
        
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
            
            # Dual model prediction
            with torch.no_grad():
                dual_result = self.dual_model(image_tensor)
            
            # Generate explanations
            explanations = self.explainer.explain(
                image_tensor, 
                extracted_features['segmentation_masks'],
                class_idx=dual_result['final_prediction'].item()
            )
            
            # Create overlays
            overlays = self.explainer.create_overlays(image_np, explanations)
            
            # Align explanations with clinical features
            aligned_features = self._align_explanations_with_features(
                extracted_features, explanations
            )
            
            # Clinical reasoning with dual model insights
            bethesda_result = self.clinical_reasoning.classify_bethesda_with_dual_models(
                dual_result, aligned_features, explanations
            )
            
            # Combine all results
            final_result = {
                'image_info': {
                    'shape': image_np.shape,
                    'dtype': str(image_np.dtype)
                },
                'extracted_features': extracted_features,
                'dual_model_prediction': dual_result,
                'explanations': explanations,
                'overlays': overlays,
                'aligned_features': aligned_features,
                'bethesda_classification': bethesda_result,
                'visualizations': {
                    'grad_cam': explanations.get('grad_cam', np.zeros((224, 224))),
                    'attention': explanations.get('attention', np.zeros((224, 224))),
                    'segmentation_masks': extracted_features['segmentation_masks']
                },
                'summary': {
                    'class': bethesda_result['bethesda_class'],
                    'confidence': bethesda_result['confidence'],
                    'reasoning': bethesda_result['clinical_reasoning'],
                    'dominant_region': explanations.get('dominant_region', 'background'),
                    'model_weights': {
                        'cnn_weight': dual_result['fusion']['cnn_weight'].item(),
                        'swin_weight': dual_result['fusion']['swin_weight'].item()
                    }
                }
            }
            
            # Ensure feature_fusion is always present
            final_result = self.ensure_feature_fusion(final_result)
            
            logger.info(f"Dual model inference completed: {bethesda_result['bethesda_class']} with {bethesda_result['confidence']:.3f} confidence")
            return final_result
            
        except Exception as e:
            logger.error(f"Error during dual model inference: {e}")
            return self._get_default_result()
    
    def _align_explanations_with_features(self, extracted_features: Dict, explanations: Dict) -> Dict:
        """Align model explanations with clinical features."""
        try:
            aligned = extracted_features.copy()
            
            # Get region importance
            nucleus_importance = explanations.get('nucleus_importance', 0.33)
            cytoplasmic_importance = explanations.get('cytoplasmic_importance', 0.33)
            background_importance = explanations.get('background_importance', 0.34)
            
            # Adjust feature scores based on model importance
            # High nucleus importance → influences nuclear features
            if nucleus_importance > 0.4:
                aligned['nuclear']['chromatin_density'] *= (1 + 0.2 * nucleus_importance)
                aligned['nuclear']['nc_ratio'] *= (1 + 0.2 * nucleus_importance)
                aligned['nuclear']['nuclear_enlargement'] *= (1 + 0.2 * nucleus_importance)
            
            # High cytoplasmic importance → influences cytoplasmic features
            if cytoplasmic_importance > 0.4:
                aligned['cytoplasmic']['perinuclear_halo'] *= (1 + 0.2 * cytoplasmic_importance)
                aligned['cytoplasmic']['keratinization'] *= (1 + 0.2 * cytoplasmic_importance)
                aligned['cytoplasmic']['koilocytosis_score'] *= (1 + 0.2 * cytoplasmic_importance)
            
            # High background importance → influences background features
            if background_importance > 0.4:
                aligned['background']['background_debris'] *= (1 + 0.2 * background_importance)
                aligned['background']['tumor_diathesis'] *= (1 + 0.2 * background_importance)
                aligned['background']['background_cleanliness'] *= (1 - 0.2 * background_importance)
            
            # Clamp values to [0, 1]
            for feature_type in ['nuclear', 'cytoplasmic', 'background']:
                for feature_name, feature_value in aligned[feature_type].items():
                    if isinstance(feature_value, (int, float)):
                        aligned[feature_type][feature_name] = np.clip(feature_value, 0.0, 1.0)
            
            # Recalculate overall scores
            for feature_type in ['nuclear', 'cytoplasmic', 'background']:
                features = aligned[feature_type]
                if 'overall_nuclear_score' in features:
                    score_features = [
                        features.get('nuclear_enlargement', 0),
                        features.get('chromatin_density', 0),
                        features.get('nuclear_contours', 0),
                        features.get('nc_ratio', 0),
                        features.get('hyperchromasia', 0)
                    ]
                    aligned[feature_type]['overall_nuclear_score'] = np.mean(score_features)
                elif 'overall_cytoplasmic_score' in features:
                    score_features = [
                        features.get('perinuclear_halo', 0),
                        features.get('keratinization', 0),
                        features.get('cytoplasmic_texture', 0),
                        features.get('cell_maturity', 0),
                        features.get('koilocytosis_score', 0)
                    ]
                    aligned[feature_type]['overall_cytoplasmic_score'] = np.mean(score_features)
                elif 'overall_background_score' in features:
                    score_features = [
                        features.get('background_debris', 0),
                        features.get('inflammatory_cells', 0),
                        features.get('tumor_diathesis', 0),
                        features.get('background_cleanliness', 1)
                    ]
                    aligned[feature_type]['overall_background_score'] = np.mean(score_features)
            
            # Add explanation alignment info
            aligned['explanation_alignment'] = {
                'nucleus_importance': nucleus_importance,
                'cytoplasmic_importance': cytoplasmic_importance,
                'background_importance': background_importance,
                'dominant_region': explanations.get('dominant_region', 'background')
            }
            
            return aligned
            
        except Exception as e:
            logger.error(f"Error aligning explanations with features: {e}")
            return extracted_features
    
    def ensure_feature_fusion(self, result: Dict) -> Dict:
        """Ensure feature_fusion is always present in result."""
        if "feature_fusion" not in result:
            # Extract available scores from result
            dual_pred = result.get("dual_model_prediction", {})
            fusion_info = dual_pred.get("fusion", {})
            
            # Get feature scores from aligned features
            aligned_features = result.get("aligned_features", {})
            nuclear_score = aligned_features.get("nuclear", {}).get("overall_nuclear_score", 0.0)
            cytoplasmic_score = aligned_features.get("cytoplasmic", {}).get("overall_cytoplasmic_score", 0.0)
            background_score = aligned_features.get("background", {}).get("overall_background_score", 0.0)
            
            # Get model scores
            cnn_weight = fusion_info.get("cnn_weight", 0.5)
            swin_weight = fusion_info.get("swin_weight", 0.5)
            model_confidence = dual_pred.get("final_confidence", 0.5)
            
            result["feature_fusion"] = {
                "feature_contributions": {
                    "cnn": float(cnn_weight),
                    "swin": float(swin_weight),
                    "nuclear": float(nuclear_score),
                    "cytoplasmic": float(cytoplasmic_score),
                    "background": float(background_score),
                    "model": float(model_confidence)
                },
                "fusion_method": "fallback",
                "dominant_feature": result.get("summary", {}).get("dominant_region", "background"),
                "feature_scores": {
                    "nuclear": float(nuclear_score),
                    "cytoplasmic": float(cytoplasmic_score),
                    "background": float(background_score),
                    "model": float(model_confidence)
                }
            }
        return result
    
    def _get_default_result(self) -> Dict:
        """Return default result when inference fails."""
        default_result = {
            'image_info': {'shape': (224, 224, 3), 'dtype': 'uint8'},
            'extracted_features': self.feature_extractor._get_default_features(),
            'dual_model_prediction': {
                'final_prediction': 0,
                'final_confidence': 0.5,
                'fusion': {'cnn_weight': 0.5, 'swin_weight': 0.5}
            },
            'explanations': {
                'grad_cam': np.zeros((224, 224)),
                'attention': np.zeros((224, 224)),
                'dominant_region': 'background'
            },
            'overlays': {
                'grad_cam_overlay': np.zeros((224, 224, 3), dtype=np.uint8),
                'attention_overlay': np.zeros((224, 224, 3), dtype=np.uint8),
                'combined_overlay': np.zeros((224, 224, 3), dtype=np.uint8)
            },
            'aligned_features': self.feature_extractor._get_default_features(),
            'bethesda_classification': self.clinical_reasoning._get_default_bethesda_result(),
            'visualizations': {
                'grad_cam': np.zeros((224, 224)),
                'attention': np.zeros((224, 224)),
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
                'dominant_region': 'background',
                'model_weights': {'cnn_weight': 0.5, 'swin_weight': 0.5}
            }
        }
        
        # Ensure feature_fusion is present
        return self.ensure_feature_fusion(default_result)


def create_dual_inference_engine(cnn_model_path: Optional[str] = None, 
                               swin_model_path: Optional[str] = None, 
                               device: str = 'cpu') -> DualModelInferenceEngine:
    """
    Create and return a dual model inference engine.
    
    Args:
        cnn_model_path: Path to CNN model
        swin_model_path: Path to Swin model
        device: Device to run on
        
    Returns:
        DualModelInferenceEngine instance
    """
    return DualModelInferenceEngine(cnn_model_path, swin_model_path, device)


if __name__ == "__main__":
    # Test dual model inference engine
    engine = create_dual_inference_engine()
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Test inference
    result = engine.predict_with_dual_models(test_image)
    print("Dual model inference engine test completed")
    print(f"Class: {result['summary']['class']}")
    print(f"Confidence: {result['summary']['confidence']:.3f}")
    print(f"Dominant region: {result['summary']['dominant_region']}")
    print(f"CNN weight: {result['summary']['model_weights']['cnn_weight']:.3f}")
    print(f"Swin weight: {result['summary']['model_weights']['swin_weight']:.3f}")
