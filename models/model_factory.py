"""
Model factory for creating deep learning models for cervical cancer classification.
Supports EfficientNet-B0, ResNet50, and Swin Transformer with transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CervicalCancerModel(nn.Module):
    """
    Base model class for cervical cancer classification.
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize model.
        
        Args:
            model_name: Name of the backbone model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone initially
            dropout_rate: Dropout rate for classification head
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        
        # Create backbone
        self.backbone = self._create_backbone()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Create classification head
        self.classifier = self._create_classifier()
        
        # Initialize weights
        self._initialize_weights()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _create_backbone(self) -> nn.Module:
        """Create backbone model using TIMM."""
        try:
            backbone = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=0,  # Remove classification head
                global_pool=''   # Remove global pooling
            )
            return backbone
        except Exception as e:
            logger.error(f"Failed to create model {self.model_name}: {e}")
            raise
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from backbone."""
        # Create dummy input to determine feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        
        # Handle different output formats
        if isinstance(features, dict):
            # Some models return dict with 'features' key
            features = features['features']
        
        if len(features.shape) == 4:
            # Convolutional feature maps - apply adaptive pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
            feature_dim = features.shape[1]
        else:
            # Already flattened features
            feature_dim = features.shape[1]
        
        return feature_dim
    
    def _create_classifier(self) -> nn.Module:
        """Create classification head."""
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if len(self.backbone(torch.randn(1, 3, 224, 224)).shape) == 4 else nn.Identity(),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
        return classifier
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info(f"Frozen backbone: {self.model_name}")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info(f"Unfrozen backbone: {self.model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.backbone(x)
        
        # Handle different feature formats
        if isinstance(features, dict):
            features = features['features']
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature maps for XAI purposes."""
        return self.backbone(x)


class EfficientNetB0Model(CervicalCancerModel):
    """EfficientNet-B0 model for cervical cancer classification."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__(
            model_name="efficientnet_b0.ra_in1k",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )


class ResNet50Model(CervicalCancerModel):
    """ResNet50 model for cervical cancer classification."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__(
            model_name="resnet50.a1_in1k",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )


class SwinTransformerModel(CervicalCancerModel):
    """Swin Transformer Tiny model for cervical cancer classification."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__(
            model_name="swin_tiny_patch4_window7_224.ms_in1k",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    
    def _create_classifier(self) -> nn.Module:
        """Create classifier for Swin Transformer."""
        # Swin Transformer already has global pooling
        classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
        return classifier
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for XAI (specific to Swin Transformer).
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of attention maps
        """
        attention_maps = {}
        
        # Hook function to capture attention
        def get_attention(name):
            def hook(module, input, output):
                if hasattr(output, 'attention'):
                    attention_maps[name] = output.attention
            return hook
        
        # Register hooks for attention layers
        for name, module in self.backbone.named_modules():
            if 'attn' in name and hasattr(module, 'attention'):
                module.register_forward_hook(get_attention(name))
        
        # Forward pass
        with torch.no_grad():
            self.backbone(x)
        
        return attention_maps


class ModelFactory:
    """Factory class for creating models."""
    
    MODELS = {
        'efficientnet_b0': EfficientNetB0Model,
        'resnet50': ResNet50Model,
        'swin_transformer': SwinTransformerModel
    }
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3,
        **kwargs
    ) -> nn.Module:
        """
        Create model instance.
        
        Args:
            model_name: Name of the model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone
            dropout_rate: Dropout rate
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls.MODELS.keys())}")
        
        model_class = cls.MODELS[model_name]
        model = model_class(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
            **kwargs
        )
        
        logger.info(f"Created model: {model_name} with {num_classes} classes")
        return model
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available models."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """
        Get model information.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create temporary model to get info
        temp_model = cls.create_model(model_name, num_classes=2, pretrained=False, freeze_backbone=False)
        
        # Count parameters
        total_params = sum(p.numel() for p in temp_model.parameters())
        trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        
        info = {
            'model_name': model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': sum(p.numel() for p in temp_model.backbone.parameters()),
            'classifier_parameters': sum(p.numel() for p in temp_model.classifier.parameters())
        }
        
        del temp_model
        return info


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Convenience function to create a model.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone
        dropout_rate: Dropout rate
        
    Returns:
        Model instance
    """
    return ModelFactory.create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> Dict:
    """
    Get model summary information.
    
    Args:
        model: Model instance
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Dictionary with model summary
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get output size
    with torch.no_grad():
        output = model(dummy_input)
    
    summary = {
        'input_size': input_size,
        'output_size': tuple(output.shape),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }
    
    return summary


if __name__ == "__main__":
    # Test model creation
    print("Available models:", ModelFactory.get_available_models())
    
    # Create EfficientNet-B0
    model = create_model('efficientnet_b0', num_classes=2)
    print(f"Created EfficientNet-B0: {type(model).__name__}")
    
    # Get model info
    info = ModelFactory.get_model_info('efficientnet_b0')
    print(f"Model info: {info}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test unfreezing
    model.unfreeze_backbone()
    trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreeze: {trainable_after_unfreeze}")
