#!/usr/bin/env python3
"""
Model Management for Cervical Cytology Classification
Handles model loading, inference, and feature fusion.

Author: Cervical Cancer Classification Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CytologyModel:
    """Main model wrapper for cytology classification."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.model_loaded = False
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No model path provided or model not found. Using demo mode.")
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained model."""
        try:
            # Create a simple CNN model for demo
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)  # Binary classification
            ).to(self.device)
            
            # Try to load weights if available
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}. Using random weights.")
            
            self.model.eval()
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """
        Make prediction on input image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor [1, 3, H, W]
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            # Return demo prediction
            return self._get_demo_prediction(image_tensor)
        
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
                
                return {
                    'prediction': prediction.item(),
                    'probabilities': probabilities.cpu().numpy().tolist(),
                    'confidence': confidence.item(),
                    'logits': outputs.cpu().numpy().tolist()
                }
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self._get_demo_prediction(image_tensor)
    
    def _get_demo_prediction(self, image_tensor: torch.Tensor) -> Dict:
        """Generate demo prediction when model is not loaded."""
        # Random but consistent prediction based on image content
        image_mean = torch.mean(image_tensor).item()
        
        if image_mean > 0.5:
            prediction = 1
            confidence = 0.7 + (image_mean * 0.1)
        else:
            prediction = 0
            confidence = 0.6 + ((1 - image_mean) * 0.1)
        
        confidence = min(confidence, 0.95)
        
        return {
            'prediction': prediction,
            'probabilities': [[1 - confidence, confidence]],
            'confidence': confidence,
            'logits': [[-1.0, 1.0]]
        }


class FeatureFusion:
    """Combine model predictions with extracted features."""
    
    def __init__(self, feature_weights: Optional[Dict] = None):
        self.feature_weights = feature_weights or {
            'model': 0.4,
            'nuclear': 0.2,
            'cytoplasmic': 0.3,
            'background': 0.1
        }
        
        # Normalize weights
        total_weight = sum(self.feature_weights.values())
        self.feature_weights = {k: v/total_weight for k, v in self.feature_weights.items()}
    
    def fuse_features(self, model_prediction: Dict, extracted_features: Dict) -> Dict:
        """
        Fuse model prediction with extracted features.
        
        Args:
            model_prediction: Model output dictionary
            extracted_features: Extracted cytological features
            
        Returns:
            Fused prediction with feature contributions
        """
        try:
            # Get model confidence
            model_confidence = model_prediction['confidence']
            
            # Get feature scores
            nuclear_score = extracted_features['nuclear']['overall_nuclear_score']
            cytoplasmic_score = extracted_features['cytoplasmic']['overall_cytoplasmic_score']
            background_score = extracted_features['background']['overall_background_score']
            
            # Calculate weighted fusion
            final_score = (
                self.feature_weights['model'] * model_confidence +
                self.feature_weights['nuclear'] * nuclear_score +
                self.feature_weights['cytoplasmic'] * cytoplasmic_score +
                self.feature_weights['background'] * background_score
            )
            
            # Determine final class
            final_class = 1 if final_score > 0.5 else 0
            
            # Calculate feature contributions
            contributions = {
                'model': self.feature_weights['model'] * model_confidence,
                'nuclear': self.feature_weights['nuclear'] * nuclear_score,
                'cytoplasmic': self.feature_weights['cytoplasmic'] * cytoplasmic_score,
                'background': self.feature_weights['background'] * background_score
            }
            
            # Find dominant feature based on actual feature scores (not weighted contributions)
            feature_scores = {
                'nuclear': nuclear_score,
                'cytoplasmic': cytoplasmic_score,
                'background': background_score,
                'model': model_confidence
            }
            dominant_feature = max(feature_scores, key=feature_scores.get)
            
            return {
                'final_class': final_class,
                'final_score': final_score,
                'final_confidence': max(final_score, 1 - final_score),
                'feature_contributions': contributions,
                'dominant_feature': dominant_feature,
                'model_prediction': model_prediction,
                'extracted_features': extracted_features,
                'feature_scores': feature_scores
            }
            
        except Exception as e:
            logger.error(f"Error in feature fusion: {e}")
            return self._get_default_fusion(model_prediction, extracted_features)
    
    def _get_default_fusion(self, model_prediction: Dict, extracted_features: Dict) -> Dict:
        """Return default fusion when error occurs."""
        return {
            'final_class': model_prediction['prediction'],
            'final_score': model_prediction['confidence'],
            'final_confidence': model_prediction['confidence'],
            'feature_contributions': {'model': 1.0, 'nuclear': 0.0, 'cytoplasmic': 0.0, 'background': 0.0},
            'dominant_feature': 'model',
            'model_prediction': model_prediction,
            'extracted_features': extracted_features
        }


class GradCAM:
    """Generate Grad-CAM visualizations for model explainability."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module:
            target_module.register_forward_hook(forward_hook)
            target_module.register_backward_hook(backward_hook)
        else:
            logger.warning(f"Target layer '{self.target_layer}' not found")
    
    def generate_cam(self, image_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image_tensor: Input image tensor
            class_idx: Target class index
            
        Returns:
            Grad-CAM heatmap
        """
        if self.activations is None or self.gradients is None:
            return np.zeros((224, 224))
        
        try:
            # Get gradients and activations
            gradients = self.gradients[0]  # [C, H, W]
            activations = self.activations[0]  # [C, H, W]
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(1, 2))  # [C]
            
            # Weighted combination of activation maps
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return np.zeros((224, 224))


class ModelManager:
    """Main model management class."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.model = CytologyModel(model_path, device)
        self.feature_fusion = FeatureFusion()
        self.grad_cam = None
        
        # Initialize Grad-CAM if model is loaded
        if self.model.model_loaded:
            self.grad_cam = GradCAM(self.model.model, "6")  # Last conv layer
    
    def predict_with_features(self, image_tensor: torch.Tensor, extracted_features: Dict) -> Dict:
        """
        Make prediction with feature fusion.
        
        Args:
            image_tensor: Preprocessed image tensor
            extracted_features: Extracted cytological features
            
        Returns:
            Comprehensive prediction results
        """
        # Get model prediction
        model_prediction = self.model.predict(image_tensor)
        
        # Fuse with extracted features
        fused_result = self.feature_fusion.fuse_features(model_prediction, extracted_features)
        
        # Add Grad-CAM if available
        if self.grad_cam:
            try:
                cam = self.grad_cam.generate_cam(image_tensor, fused_result['final_class'])
                fused_result['grad_cam'] = cam
            except Exception as e:
                logger.error(f"Error generating Grad-CAM: {e}")
                fused_result['grad_cam'] = np.zeros((224, 224))
        else:
            fused_result['grad_cam'] = np.zeros((224, 224))
        
        return fused_result


def create_model_manager(model_path: Optional[str] = None, device: str = 'cpu') -> ModelManager:
    """
    Create and return a model manager instance.
    
    Args:
        model_path: Path to trained model
        device: Device to run on
        
    Returns:
        ModelManager instance
    """
    return ModelManager(model_path, device)


if __name__ == "__main__":
    # Test model management
    manager = create_model_manager()
    
    # Create dummy input
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    # Create dummy features
    dummy_features = {
        'nuclear': {'overall_nuclear_score': 0.5},
        'cytoplasmic': {'overall_cytoplasmic_score': 0.6},
        'background': {'overall_background_score': 0.3}
    }
    
    # Test prediction
    result = manager.predict_with_features(dummy_tensor, dummy_features)
    print("Model management test completed")
    print(f"Final class: {result['final_class']}")
    print(f"Final score: {result['final_score']:.3f}")
    print(f"Dominant feature: {result['dominant_feature']}")
