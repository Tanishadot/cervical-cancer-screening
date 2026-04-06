"""
Test-time augmentation and ensemble methods for improved robustness.
Includes TTA with multiple augmentations and lightweight model ensembling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from enhanced_transforms import get_test_time_augmentation_transforms
from models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class TestTimeAugmentation:
    """
    Test-time augmentation for improved prediction robustness.
    """
    
    def __init__(self, model: nn.Module, tta_steps: int = 5, 
                 device: torch.device = None):
        """
        Initialize TTA.
        
        Args:
            model: Trained model
            tta_steps: Number of augmentation steps
            device: Device to run inference on
        """
        self.model = model
        self.tta_steps = tta_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Get TTA transforms
        self.tta_transforms = get_test_time_augmentation_transforms(
            tta_steps=tta_steps
        )
    
    def predict_single_with_tta(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on single image with TTA.
        
        Args:
            image: Input image tensor [C, H, W]
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            # Convert tensor to numpy for augmentation
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.shape[0] == 3:  # CHW format
                    image_np = np.transpose(image_np, (1, 2, 0))  # Convert to HWC
                image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8
            else:
                image_np = image
            
            # Apply TTA transforms
            for i, transform in enumerate(self.tta_transforms):
                if i >= self.tta_steps:
                    break
                
                # Apply augmentation
                if hasattr(transform, '__call__') and hasattr(transform, 'transforms'):
                    # Albumentations transform
                    transformed = transform(image=image_np)
                    aug_image = transformed['image']
                else:
                    # Regular transform
                    aug_image = transform(image_np)
                
                # Add batch dimension and move to device
                aug_image = aug_image.unsqueeze(0).to(self.device)
                
                # Model prediction
                outputs = self.model(aug_image)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
                
                all_predictions.append(prediction.cpu().numpy()[0])
                all_probabilities.append(probabilities.cpu().numpy()[0])
        
        # Average predictions
        avg_probabilities = np.mean(all_probabilities, axis=0)
        avg_prediction = np.argmax(avg_probabilities)
        
        return avg_prediction, avg_probabilities
    
    def predict_batch_with_tta(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on batch with TTA.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        batch_predictions = []
        batch_probabilities = []
        
        for i in range(images.shape[0]):
            pred, prob = self.predict_single_with_tta(images[i])
            batch_predictions.append(pred)
            batch_probabilities.append(prob)
        
        return np.array(batch_predictions), np.array(batch_probabilities)


class ModelEnsemble:
    """
    Lightweight model ensemble for improved performance.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None,
                 device: torch.device = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for averaging (default: equal weights)
            device: Device to run inference on
        """
        self.models = models
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Move models to device and set to eval
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        logger.info(f"Ensemble created with {len(models)} models")
        logger.info(f"Model weights: {self.weights}")
    
    def predict_single(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on single image with ensemble.
        
        Args:
            image: Input image tensor [C, H, W]
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_probabilities = []
        
        with torch.no_grad():
            # Add batch dimension
            image_batch = image.unsqueeze(0).to(self.device)
            
            # Get predictions from each model
            for i, model in enumerate(self.models):
                outputs = model(image_batch)
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu().numpy()[0])
        
        # Weighted average of probabilities
        avg_probabilities = np.average(all_probabilities, axis=0, weights=self.weights)
        avg_prediction = np.argmax(avg_probabilities)
        
        return avg_prediction, avg_probabilities
    
    def predict_batch(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on batch with ensemble.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        batch_predictions = []
        batch_probabilities = []
        
        for i in range(images.shape[0]):
            pred, prob = self.predict_single(images[i])
            batch_predictions.append(pred)
            batch_probabilities.append(prob)
        
        return np.array(batch_predictions), np.array(batch_probabilities)


class EnsembleTTA:
    """
    Combined ensemble and test-time augmentation for maximum robustness.
    """
    
    def __init__(self, models: List[nn.Module], tta_steps: int = 3,
                 ensemble_weights: Optional[List[float]] = None,
                 device: torch.device = None):
        """
        Initialize Ensemble+TTA.
        
        Args:
            models: List of trained models
            tta_steps: Number of TTA steps
            ensemble_weights: Optional ensemble weights
            device: Device to run inference on
        """
        self.ensemble = ModelEnsemble(models, ensemble_weights, device)
        self.tta_steps = tta_steps
        self.device = device
        
        # Get TTA transforms
        self.tta_transforms = get_test_time_augmentation_transforms(
            tta_steps=tta_steps
        )
    
    def predict_single(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on single image with ensemble+TTA.
        
        Args:
            image: Input image tensor [C, H, W]
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_probabilities = []
        
        with torch.no_grad():
            # Convert tensor to numpy for augmentation
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.shape[0] == 3:  # CHW format
                    image_np = np.transpose(image_np, (1, 2, 0))  # Convert to HWC
                image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8
            else:
                image_np = image
            
            # Apply TTA transforms
            for i, transform in enumerate(self.tta_transforms):
                if i >= self.tta_steps:
                    break
                
                # Apply augmentation
                if hasattr(transform, '__call__') and hasattr(transform, 'transforms'):
                    # Albumentations transform
                    transformed = transform(image=image_np)
                    aug_image = transformed['image']
                else:
                    # Regular transform
                    aug_image = transform(image_np)
                
                # Add batch dimension
                aug_image = aug_image.unsqueeze(0)
                
                # Ensemble prediction on augmented image
                pred, prob = self.ensemble.predict_single(aug_image[0])
                all_probabilities.append(prob)
        
        # Average across TTA
        avg_probabilities = np.mean(all_probabilities, axis=0)
        avg_prediction = np.argmax(avg_probabilities)
        
        return avg_prediction, avg_probabilities


def create_ensemble_models(model_names: List[str], model_paths: List[str],
                          num_classes: int = 2, device: torch.device = None) -> List[nn.Module]:
    """
    Create ensemble models from checkpoints.
    
    Args:
        model_names: List of model architecture names
        model_paths: List of model checkpoint paths
        num_classes: Number of output classes
        device: Device to load models on
        
    Returns:
        List of loaded models
    """
    models = []
    
    for name, path in zip(model_names, model_paths):
        try:
            # Create model
            model = ModelFactory.create_model(
                model_name=name,
                num_classes=num_classes,
                pretrained=False  # Loading from checkpoint
            )
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            models.append(model)
            logger.info(f"Loaded model: {name} from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model {name} from {path}: {e}")
            continue
    
    return models


def create_lightweight_ensemble(base_model: nn.Module, 
                               checkpoint_paths: List[str],
                               device: torch.device = None) -> ModelEnsemble:
    """
    Create ensemble from multiple checkpoints of same architecture.
    
    Args:
        base_model: Base model architecture
        checkpoint_paths: List of checkpoint paths
        device: Device to load models on
        
    Returns:
        Model ensemble
    """
    models = []
    
    for path in checkpoint_paths:
        try:
            # Create copy of base model
            import copy
            model = copy.deepcopy(base_model)
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            models.append(model)
            logger.info(f"Loaded checkpoint: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")
            continue
    
    return ModelEnsemble(models, device=device)


if __name__ == "__main__":
    # Test TTA
    from models.model_factory import create_model
    
    # Create dummy model
    model = create_model('efficientnet_b0', num_classes=2)
    
    # Test TTA
    tta = TestTimeAugmentation(model, tta_steps=3)
    
    # Create dummy image
    dummy_image = torch.randn(3, 224, 224)
    pred, prob = tta.predict_single_with_tta(dummy_image)
    
    print(f"TTA Prediction: {pred}")
    print(f"TTA Probabilities: {prob}")
    
    # Test ensemble
    models = [create_model('efficientnet_b0', num_classes=2) for _ in range(3)]
    ensemble = ModelEnsemble(models)
    
    pred, prob = ensemble.predict_single(dummy_image)
    print(f"Ensemble Prediction: {pred}")
    print(f"Ensemble Probabilities: {prob}")
    
    print("TTA and Ensemble functionality working!")
