"""
Confidence-based prediction system with uncertainty handling for clinical safety.
Provides prediction confidence scores and uncertainty flags for robust deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class UncertaintyLevel(Enum):
    """Uncertainty level enumeration."""
    CERTAIN = "certain"
    UNCERTAIN = "uncertain"
    VERY_UNCERTAIN = "very_uncertain"


@dataclass
class PredictionResult:
    """Prediction result with confidence and uncertainty information."""
    prediction: int
    confidence: float
    uncertainty_level: UncertaintyLevel
    probabilities: Dict[str, float]
    raw_logits: Optional[np.ndarray] = None
    entropy: Optional[float] = None
    margin: Optional[float] = None


class ConfidenceBasedPredictor:
    """
    Confidence-based predictor with uncertainty handling for clinical safety.
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.5,
        device: torch.device = None
    ):
        """
        Initialize confidence-based predictor.
        
        Args:
            model: Trained model
            class_names: List of class names
            confidence_threshold: Threshold for certain predictions
            uncertainty_threshold: Threshold for uncertain predictions
            device: Device to run inference on
        """
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Confidence-based predictor initialized with threshold: {confidence_threshold}")
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate prediction entropy as uncertainty measure.
        
        Args:
            probabilities: Class probabilities
            
        Returns:
            Entropy value
        """
        # Avoid log(0)
        probabilities = np.clip(probabilities, 1e-8, 1.0)
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy
    
    def _calculate_margin(self, probabilities: np.ndarray) -> float:
        """
        Calculate prediction margin (difference between top two probabilities).
        
        Args:
            probabilities: Class probabilities
            
        Returns:
            Margin value
        """
        if len(probabilities) < 2:
            return 0.0
        
        # Sort probabilities in descending order
        sorted_probs = np.sort(probabilities)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]
        return margin
    
    def _determine_uncertainty_level(
        self,
        confidence: float,
        entropy: float,
        margin: float
    ) -> UncertaintyLevel:
        """
        Determine uncertainty level based on confidence, entropy, and margin.
        
        Args:
            confidence: Prediction confidence
            entropy: Prediction entropy
            margin: Prediction margin
            
        Returns:
            Uncertainty level
        """
        if confidence >= self.confidence_threshold and margin >= 0.3:
            return UncertaintyLevel.CERTAIN
        elif confidence >= self.uncertainty_threshold and margin >= 0.1:
            return UncertaintyLevel.UNCERTAIN
        else:
            return UncertaintyLevel.VERY_UNCERTAIN
    
    def predict_single(
        self,
        image: torch.Tensor,
        return_raw: bool = False
    ) -> PredictionResult:
        """
        Make prediction with confidence estimation for a single image.
        
        Args:
            image: Input image tensor
            return_raw: Whether to return raw logits
            
        Returns:
            Prediction result with confidence and uncertainty
        """
        with torch.no_grad():
            # Ensure image is in correct format
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            image = image.to(self.device)
            
            # Forward pass
            logits = self.model(image)
            
            # Convert to probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get prediction
            pred_class = torch.argmax(probabilities, dim=1).item()
            max_prob = torch.max(probabilities, dim=1).values.item()
            
            # Convert to numpy for calculations
            probs_np = probabilities.cpu().numpy().squeeze()
            logits_np = logits.cpu().numpy().squeeze()
            
            # Calculate uncertainty measures
            entropy = self._calculate_entropy(probs_np)
            margin = self._calculate_margin(probs_np)
            
            # Determine uncertainty level
            uncertainty_level = self._determine_uncertainty_level(
                max_prob, entropy, margin
            )
            
            # Create probability dictionary
            prob_dict = {
                self.class_names[i]: float(probs_np[i])
                for i in range(len(self.class_names))
            }
            
            # Create prediction result
            result = PredictionResult(
                prediction=pred_class,
                confidence=max_prob,
                uncertainty_level=uncertainty_level,
                probabilities=prob_dict,
                entropy=entropy,
                margin=margin
            )
            
            if return_raw:
                result.raw_logits = logits_np
            
            return result
    
    def predict_batch(
        self,
        images: torch.Tensor,
        return_raw: bool = False
    ) -> List[PredictionResult]:
        """
        Make predictions with confidence estimation for a batch of images.
        
        Args:
            images: Batch of input image tensors
            return_raw: Whether to return raw logits
            
        Returns:
            List of prediction results
        """
        results = []
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            
            # Convert to probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get predictions and confidences
            pred_classes = torch.argmax(probabilities, dim=1)
            max_probs = torch.max(probabilities, dim=1).values
            
            # Process each prediction
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Get individual prediction
                pred_class = pred_classes[i].item()
                max_prob = max_probs[i].item()
                
                # Convert to numpy
                probs_np = probabilities[i].cpu().numpy()
                logits_np = logits[i].cpu().numpy()
                
                # Calculate uncertainty measures
                entropy = self._calculate_entropy(probs_np)
                margin = self._calculate_margin(probs_np)
                
                # Determine uncertainty level
                uncertainty_level = self._determine_uncertainty_level(
                    max_prob, entropy, margin
                )
                
                # Create probability dictionary
                prob_dict = {
                    self.class_names[j]: float(probs_np[j])
                    for j in range(len(self.class_names))
                }
                
                # Create prediction result
                result = PredictionResult(
                    prediction=pred_class,
                    confidence=max_prob,
                    uncertainty_level=uncertainty_level,
                    probabilities=prob_dict,
                    entropy=entropy,
                    margin=margin
                )
                
                if return_raw:
                    result.raw_logits = logits_np
                
                results.append(result)
        
        return results
    
    def predict_with_uncertainty_threshold(
        self,
        image: torch.Tensor,
        min_confidence: Optional[float] = None
    ) -> Dict:
        """
        Make prediction with uncertainty-based decision making.
        
        Args:
            image: Input image tensor
            min_confidence: Minimum confidence threshold (overrides default)
            
        Returns:
            Dictionary with prediction, confidence, and uncertainty decision
        """
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        
        result = self.predict_single(image)
        
        # Make uncertainty-based decision
        if result.confidence >= threshold:
            decision = "ACCEPT"
            recommendation = "Prediction is reliable"
        elif result.confidence >= self.uncertainty_threshold:
            decision = "REVIEW"
            recommendation = "Prediction requires expert review"
        else:
            decision = "REJECT"
            recommendation = "Prediction is unreliable, requires re-examination"
        
        return {
            "prediction": self.class_names[result.prediction],
            "confidence": result.confidence,
            "uncertainty_level": result.uncertainty_level.value,
            "decision": decision,
            "recommendation": recommendation,
            "probabilities": result.probabilities,
            "entropy": result.entropy,
            "margin": result.margin
        }
    
    def evaluate_uncertainty_calibration(
        self,
        dataloader,
        num_batches: Optional[int] = None
    ) -> Dict:
        """
        Evaluate uncertainty calibration on validation data.
        
        Args:
            dataloader: Validation data loader
            num_batches: Number of batches to evaluate
            
        Returns:
            Calibration metrics
        """
        self.model.eval()
        
        all_confidences = []
        all_correct = []
        all_uncertainty_levels = []
        
        batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if num_batches is not None and batch_count >= num_batches:
                    break
                
                # Extract images and labels
                if len(batch) >= 4:
                    images, labels, _, _ = batch[:4]
                else:
                    images, labels = batch[:2]
                
                # Get predictions
                results = self.predict_batch(images)
                
                # Collect results
                for result, label in zip(results, labels):
                    all_confidences.append(result.confidence)
                    all_correct.append(result.prediction == label.item())
                    all_uncertainty_levels.append(result.uncertainty_level.value)
                
                batch_count += 1
        
        # Calculate calibration metrics
        all_confidences = np.array(all_confidences)
        all_correct = np.array(all_correct)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = all_correct[in_bin].mean()
                avg_confidence_in_bin = all_confidences[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Uncertainty distribution
        uncertainty_dist = {
            level: all_uncertainty_levels.count(level.value)
            for level in UncertaintyLevel
        }
        
        return {
            "expected_calibration_error": ece,
            "average_confidence": np.mean(all_confidences),
            "average_accuracy": np.mean(all_correct),
            "uncertainty_distribution": uncertainty_dist,
            "total_samples": len(all_confidences)
        }
    
    def get_uncertainty_statistics(self, results: List[PredictionResult]) -> Dict:
        """
        Get statistics about uncertainty in prediction results.
        
        Args:
            results: List of prediction results
            
        Returns:
            Uncertainty statistics
        """
        confidences = [r.confidence for r in results]
        entropies = [r.entropy for r in results if r.entropy is not None]
        margins = [r.margin for r in results if r.margin is not None]
        
        uncertainty_counts = {level.value: 0 for level in UncertaintyLevel}
        for result in results:
            uncertainty_counts[result.uncertainty_level.value] += 1
        
        return {
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "median": np.median(confidences)
            },
            "entropy_stats": {
                "mean": np.mean(entropies) if entropies else 0,
                "std": np.std(entropies) if entropies else 0,
                "min": np.min(entropies) if entropies else 0,
                "max": np.max(entropies) if entropies else 0
            },
            "margin_stats": {
                "mean": np.mean(margins) if margins else 0,
                "std": np.std(margins) if margins else 0,
                "min": np.min(margins) if margins else 0,
                "max": np.max(margins) if margins else 0
            },
            "uncertainty_distribution": uncertainty_counts,
            "total_predictions": len(results)
        }


def create_confidence_predictor(
    model: nn.Module,
    class_names: List[str],
    confidence_threshold: float = 0.7,
    uncertainty_threshold: float = 0.5,
    device: torch.device = None
) -> ConfidenceBasedPredictor:
    """
    Create confidence-based predictor.
    
    Args:
        model: Trained model
        class_names: List of class names
        confidence_threshold: Threshold for certain predictions
        uncertainty_threshold: Threshold for uncertain predictions
        device: Device to run inference on
        
    Returns:
        Confidence-based predictor instance
    """
    return ConfidenceBasedPredictor(
        model=model,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        uncertainty_threshold=uncertainty_threshold,
        device=device
    )


if __name__ == "__main__":
    # Test confidence-based prediction
    from models.model_factory import create_model
    
    # Create dummy model
    model = create_model('efficientnet_b0', num_classes=2)
    class_names = ['NORMAL', 'ABNORMAL']
    
    # Create predictor
    predictor = create_confidence_predictor(
        model=model,
        class_names=class_names,
        confidence_threshold=0.7
    )
    
    # Test single prediction
    dummy_image = torch.randn(1, 3, 224, 224)
    result = predictor.predict_single(dummy_image)
    
    print(f"Prediction: {class_names[result.prediction]}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Uncertainty Level: {result.uncertainty_level.value}")
    print(f"Probabilities: {result.probabilities}")
    
    # Test uncertainty-based decision
    decision_result = predictor.predict_with_uncertainty_threshold(dummy_image)
    print(f"Decision: {decision_result['decision']}")
    print(f"Recommendation: {decision_result['recommendation']}")
    
    print("Confidence-based prediction system ready for use")
