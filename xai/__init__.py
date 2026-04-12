"""
XAI (Explainable AI) Module for Cervical Cancer Classification Pipeline

This module provides various explainable AI methods for model interpretation:
- Grad-CAM for CNN models (EfficientNet, ResNet)
- Attention Rollout for Transformer models (Swin Transformer)

Supported XAI Methods:
- gradcam: Gradient-weighted Class Activation Mapping
- attention_rollout: Attention flow visualization for Transformers
"""

from .grad_cam import (
    CustomGradCAM,
    AttentionRollout,
    XAIAnalyzer,
    create_xai_analyzer
)
from .attention_rollout import (
    AttentionRolloutAnalyzer, 
    create_attention_rollout_analyzer, 
    analyze_dataset as rollout_analyze_dataset
)

__all__ = [
    'CustomGradCAM',
    'AttentionRollout',
    'XAIAnalyzer',
    'create_xai_analyzer',
    'AttentionRolloutAnalyzer',
    'create_attention_rollout_analyzer',
    'rollout_analyze_dataset'
]
