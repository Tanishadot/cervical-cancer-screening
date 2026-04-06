"""
Advanced loss functions and threshold optimization for improved performance.
Includes focal loss and optimal threshold search.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Tuple, Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses on hard-to-classify examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate cross entropy
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        # Apply alpha weighting
        if self.alpha != 1.0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ThresholdOptimizer:
    """
    Optimize decision threshold based on validation set performance.
    """
    
    def __init__(self, threshold_range: Tuple[float, float] = (0.3, 0.7), 
                 num_steps: int = 40, metric: str = 'f1'):
        """
        Initialize threshold optimizer.
        
        Args:
            threshold_range: Range of thresholds to search
            num_steps: Number of threshold values to test
            metric: Metric to optimize ('f1', 'precision', 'recall')
        """
        self.threshold_range = threshold_range
        self.num_steps = num_steps
        self.metric = metric
        self.best_threshold = 0.5
        self.best_score = 0.0
        self.threshold_scores = {}
    
    def optimize(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Find optimal threshold based on validation set.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for positive class
            
        Returns:
            Dictionary with optimization results
        """
        thresholds = np.linspace(self.threshold_range[0], self.threshold_range[1], self.num_steps)
        best_threshold = 0.5
        best_score = 0.0
        threshold_scores = {}
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Select metric to optimize
            if self.metric == 'f1':
                score = f1
            elif self.metric == 'precision':
                score = precision
            elif self.metric == 'recall':
                score = recall
            else:
                score = f1
            
            threshold_scores[float(threshold)] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'score': score
            }
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        self.best_score = best_score
        self.threshold_scores = threshold_scores
        
        return {
            'best_threshold': best_threshold,
            'best_score': best_score,
            'metric': self.metric,
            'all_scores': threshold_scores
        }
    
    def apply_threshold(self, y_prob: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Apply optimal threshold to predictions.
        
        Args:
            y_prob: Predicted probabilities
            threshold: Threshold to use (if None, use best found)
            
        Returns:
            Binary predictions
        """
        if threshold is None:
            threshold = self.best_threshold
        
        return (y_prob >= threshold).astype(int)
    
    def get_summary(self) -> Dict:
        """Get optimization summary."""
        if not self.threshold_scores:
            return {}
        
        # Find best threshold for each metric
        best_f1_threshold = max(self.threshold_scores.keys(), 
                               key=lambda x: self.threshold_scores[x]['f1'])
        best_precision_threshold = max(self.threshold_scores.keys(), 
                                     key=lambda x: self.threshold_scores[x]['precision'])
        best_recall_threshold = max(self.threshold_scores.keys(), 
                                   key=lambda x: self.threshold_scores[x]['recall'])
        
        return {
            'optimized_threshold': self.best_threshold,
            'optimized_score': self.best_score,
            'optimized_metric': self.metric,
            'best_f1': {
                'threshold': best_f1_threshold,
                'score': self.threshold_scores[best_f1_threshold]['f1']
            },
            'best_precision': {
                'threshold': best_precision_threshold,
                'score': self.threshold_scores[best_precision_threshold]['precision']
            },
            'best_recall': {
                'threshold': best_recall_threshold,
                'score': self.threshold_scores[best_recall_threshold]['recall']
            }
        }


def create_loss_function(loss_type: str = "bce", pos_weight: float = 1.8, 
                         focal_alpha: float = 1.0, focal_gamma: float = 2.0,
                         device: torch.device = None) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        loss_type: Type of loss ('bce' or 'focal')
        pos_weight: Positive weight for BCE
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        device: Device to create tensors on
        
    Returns:
        Loss function
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if loss_type.lower() == "bce":
        pos_weight_tensor = torch.tensor(pos_weight).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    elif loss_type.lower() == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'bce' or 'focal'")


def apply_threshold_to_predictions(predictions: torch.Tensor, 
                                  threshold: float = 0.5) -> torch.Tensor:
    """
    Apply threshold to model predictions.
    
    Args:
        predictions: Model logits or probabilities
        threshold: Decision threshold
        
    Returns:
        Binary predictions
    """
    if predictions.dim() == 2:  # Multi-class format
        predictions = predictions[:, 1]  # Take positive class
    
    # Apply sigmoid if needed (check if values are logits)
    if (predictions < 0).any() or (predictions > 1).any():
        predictions = torch.sigmoid(predictions)
    
    return (predictions >= threshold).long()


if __name__ == "__main__":
    # Test threshold optimization
    optimizer = ThresholdOptimizer(threshold_range=(0.3, 0.7), num_steps=20)
    
    # Create dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.random.beta(2, 2, 1000)  # Beta distribution for realistic probabilities
    
    # Optimize threshold
    results = optimizer.optimize(y_true, y_prob)
    
    print("Threshold Optimization Results:")
    print(f"Best threshold: {results['best_threshold']:.3f}")
    print(f"Best {results['metric']}: {results['best_score']:.4f}")
    
    # Get summary
    summary = optimizer.get_summary()
    print(f"Best F1: {summary['best_f1']['score']:.4f} at threshold {summary['best_f1']['threshold']:.3f}")
    print(f"Best Precision: {summary['best_precision']['score']:.4f} at threshold {summary['best_precision']['threshold']:.3f}")
    print(f"Best Recall: {summary['best_recall']['score']:.4f} at threshold {summary['best_recall']['threshold']:.3f}")
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Test with dummy data
    inputs = torch.randn(10, 1)
    targets = torch.randint(0, 2, (10, 1)).float()
    
    focal_result = focal_loss(inputs, targets)
    bce_result = bce_loss(inputs, targets)
    
    print(f"Focal Loss: {focal_result:.4f}")
    print(f"BCE Loss: {bce_result:.4f}")
