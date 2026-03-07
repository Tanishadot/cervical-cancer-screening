"""
Configuration manager for the cervical cancer classification pipeline.
Handles loading, validation, and management of experiment configurations.
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration data class for experiments."""
    
    # Experiment metadata
    name: str = "cervical_cancer_classification"
    description: str = "Deep learning pipeline for cervical cancer screening"
    seed: int = 42
    output_dir: str = "outputs"
    
    # Dataset configuration
    dataset_root: str = "datasets"
    train_dataset: str = "sipakmed"
    test_dataset: str = "herlev"
    classification_mode: str = "binary"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    
    # Model configuration
    model_architecture: str = "efficientnet_b0"
    num_classes: int = 2
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epoch: int = 10
    dropout_rate: float = 0.3
    
    # Training configuration
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 15
    min_delta: float = 0.001
    use_class_weights: bool = True
    use_amp: bool = True
    
    # Augmentation configuration
    image_size: int = 224
    use_color_augmentation: bool = True
    use_geometric_augmentation: bool = True
    
    # XAI configuration
    xai_enabled: bool = True
    xai_methods: list = field(default_factory=lambda: ["grad_cam", "grad_cam_plus", "attention_rollout"])
    xai_num_samples: int = 20
    
    # Hardware configuration
    device: str = "auto"
    mixed_precision: bool = True
    
    # Paths
    paths: Dict[str, str] = field(default_factory=lambda: {
        'datasets': 'datasets',
        'models': 'outputs/models',
        'visualizations': 'outputs/visualizations',
        'xai_outputs': 'outputs/xai',
        'logs': 'outputs/logs'
    })


class ConfigManager:
    """
    Configuration manager for loading and managing experiment configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "configs/config.yaml"
        self.config = None
        self.experiment_config = None
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}. Using default configuration.")
            self.config = self._get_default_config()
        else:
            try:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}. Using default configuration.")
                self.config = self._get_default_config()
        
        # Validate configuration
        self._validate_config()
        
        # Create experiment config
        self.experiment_config = self._create_experiment_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'experiment': {
                'name': 'cervical_cancer_classification',
                'description': 'Deep learning pipeline for cervical cancer screening',
                'seed': 42,
                'output_dir': 'outputs'
            },
            'dataset': {
                'root_dir': 'datasets',
                'train_dataset': 'sipakmed',
                'test_dataset': 'herlev',
                'classification_mode': 'binary',
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'batch_size': 32,
                'num_workers': 4
            },
            'model': {
                'architecture': 'efficientnet_b0',
                'num_classes': 2,
                'pretrained': True,
                'freeze_backbone': True,
                'unfreeze_epoch': 10,
                'dropout_rate': 0.3
            },
            'training': {
                'num_epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'early_stopping': {
                    'patience': 15,
                    'min_delta': 0.001,
                    'restore_best_weights': True
                },
                'use_class_weights': True,
                'use_amp': True
            },
            'augmentation': {
                'image_size': 224,
                'use_color_augmentation': True,
                'use_geometric_augmentation': True
            },
            'xai': {
                'enabled': True,
                'methods': ['grad_cam', 'grad_cam_plus', 'attention_rollout'],
                'num_samples': 20
            },
            'hardware': {
                'device': 'auto',
                'mixed_precision': True
            },
            'paths': {
                'datasets': 'datasets',
                'models': 'outputs/models',
                'visualizations': 'outputs/visualizations',
                'xai_outputs': 'outputs/xai',
                'logs': 'outputs/logs'
            }
        }
    
    def _validate_config(self):
        """Validate configuration values."""
        if not self.config:
            return
        
        # Validate classification mode
        classification_mode = self.config.get('dataset', {}).get('classification_mode', 'binary')
        if classification_mode not in ['binary', 'multiclass']:
            raise ValueError(f"Invalid classification mode: {classification_mode}. Must be 'binary' or 'multiclass'")
        
        # Validate model architecture
        architecture = self.config.get('model', {}).get('architecture', 'efficientnet_b0')
        valid_architectures = ['efficientnet_b0', 'resnet50', 'swin_transformer']
        if architecture not in valid_architectures:
            raise ValueError(f"Invalid model architecture: {architecture}. Must be one of {valid_architectures}")
        
        # Validate dataset splits
        train_split = self.config.get('dataset', {}).get('train_split', 0.7)
        val_split = self.config.get('dataset', {}).get('val_split', 0.15)
        test_split = self.config.get('dataset', {}).get('test_split', 0.15)
        
        if abs(train_split + val_split + test_split - 1.0) > 0.01:
            raise ValueError(f"Dataset splits must sum to 1.0. Got: {train_split + val_split + test_split}")
        
        # Validate paths
        paths = self.config.get('paths', {})
        for path_name, path_value in paths.items():
            if not isinstance(path_value, str):
                raise ValueError(f"Path {path_name} must be a string")
        
        logger.info("Configuration validation passed")
    
    def _create_experiment_config(self) -> ExperimentConfig:
        """Create experiment configuration from loaded config."""
        if not self.config:
            return ExperimentConfig()
        
        # Extract configuration sections
        experiment = self.config.get('experiment', {})
        dataset = self.config.get('dataset', {})
        model = self.config.get('model', {})
        training = self.config.get('training', {})
        augmentation = self.config.get('augmentation', {})
        xai = self.config.get('xai', {})
        hardware = self.config.get('hardware', {})
        paths = self.config.get('paths', {})
        
        # Get early stopping config
        early_stopping = training.get('early_stopping', {})
        
        # Create experiment config
        return ExperimentConfig(
            name=experiment.get('name', 'cervical_cancer_classification'),
            description=experiment.get('description', 'Deep learning pipeline for cervical cancer screening'),
            seed=experiment.get('seed', 42),
            output_dir=experiment.get('output_dir', 'outputs'),
            
            dataset_root=dataset.get('root_dir', 'datasets'),
            train_dataset=dataset.get('train_dataset', 'sipakmed'),
            test_dataset=dataset.get('test_dataset', 'herlev'),
            classification_mode=dataset.get('classification_mode', 'binary'),
            train_split=dataset.get('train_split', 0.7),
            val_split=dataset.get('val_split', 0.15),
            test_split=dataset.get('test_split', 0.15),
            batch_size=dataset.get('batch_size', 32),
            num_workers=dataset.get('num_workers', 4),
            
            model_architecture=model.get('architecture', 'efficientnet_b0'),
            num_classes=model.get('num_classes', 2),
            pretrained=model.get('pretrained', True),
            freeze_backbone=model.get('freeze_backbone', True),
            unfreeze_epoch=model.get('unfreeze_epoch', 10),
            dropout_rate=model.get('dropout_rate', 0.3),
            
            num_epochs=training.get('num_epochs', 100),
            learning_rate=training.get('learning_rate', 0.001),
            weight_decay=training.get('weight_decay', 0.0001),
            patience=early_stopping.get('patience', 15),
            min_delta=early_stopping.get('min_delta', 0.001),
            use_class_weights=training.get('use_class_weights', True),
            use_amp=training.get('use_amp', True),
            
            image_size=augmentation.get('image_size', 224),
            use_color_augmentation=augmentation.get('use_color_augmentation', True),
            use_geometric_augmentation=augmentation.get('use_geometric_augmentation', True),
            
            xai_enabled=xai.get('enabled', True),
            xai_methods=xai.get('methods', ['grad_cam', 'grad_cam_plus', 'attention_rollout']),
            xai_num_samples=xai.get('num_samples', 20),
            
            device=hardware.get('device', 'auto'),
            mixed_precision=hardware.get('mixed_precision', True),
            
            paths=paths
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        return self.config
    
    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration object."""
        return self.experiment_config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        if not self.config:
            self.config = self._get_default_config()
        
        # Deep update configuration
        self._deep_update(self.config, updates)
        
        # Re-validate and recreate experiment config
        self._validate_config()
        self.experiment_config = self._create_experiment_config()
        
        logger.info("Configuration updated")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, save_path: Optional[str] = None):
        """
        Save configuration to YAML file.
        
        Args:
            save_path: Path to save configuration
        """
        save_path = save_path or self.config_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def create_experiment_directories(self):
        """Create necessary directories for the experiment."""
        config = self.experiment_config
        
        if not config:
            return
        
        directories = [
            config.output_dir,
            config.paths.get('models', 'outputs/models'),
            config.paths.get('visualizations', 'outputs/visualizations'),
            config.paths.get('xai_outputs', 'outputs/xai'),
            config.paths.get('logs', 'outputs/logs'),
            config.paths.get('datasets', 'datasets')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Experiment directories created")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        config = self.experiment_config
        
        return {
            'architecture': config.model_architecture,
            'num_classes': config.num_classes,
            'pretrained': config.pretrained,
            'freeze_backbone': config.freeze_backbone,
            'dropout_rate': config.dropout_rate
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        config = self.experiment_config
        
        return {
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'patience': config.patience,
            'min_delta': config.min_delta,
            'use_class_weights': config.use_class_weights,
            'use_amp': config.use_amp,
            'unfreeze_epoch': config.unfreeze_epoch
        }
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset-specific configuration."""
        config = self.experiment_config
        
        return {
            'root_dir': config.dataset_root,
            'train_dataset': config.train_dataset,
            'test_dataset': config.test_dataset,
            'classification_mode': config.classification_mode,
            'batch_size': config.batch_size,
            'num_workers': config.num_workers
        }
    
    def print_config_summary(self):
        """Print configuration summary."""
        config = self.experiment_config
        
        if not config:
            logger.warning("No configuration available")
            return
        
        print("\n" + "="*60)
        print("EXPERIMENT CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"Experiment Name: {config.name}")
        print(f"Description: {config.description}")
        print(f"Seed: {config.seed}")
        print(f"Output Directory: {config.output_dir}")
        
        print(f"\nDataset Configuration:")
        print(f"  Root Directory: {config.dataset_root}")
        print(f"  Train Dataset: {config.train_dataset}")
        print(f"  Test Dataset: {config.test_dataset}")
        print(f"  Classification Mode: {config.classification_mode}")
        print(f"  Batch Size: {config.batch_size}")
        
        print(f"\nModel Configuration:")
        print(f"  Architecture: {config.model_architecture}")
        print(f"  Number of Classes: {config.num_classes}")
        print(f"  Pretrained: {config.pretrained}")
        print(f"  Freeze Backbone: {config.freeze_backbone}")
        print(f"  Dropout Rate: {config.dropout_rate}")
        
        print(f"\nTraining Configuration:")
        print(f"  Number of Epochs: {config.num_epochs}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Weight Decay: {config.weight_decay}")
        print(f"  Early Stopping Patience: {config.patience}")
        print(f"  Use Class Weights: {config.use_class_weights}")
        print(f"  Mixed Precision: {config.use_amp}")
        
        print(f"\nXAI Configuration:")
        print(f"  Enabled: {config.xai_enabled}")
        print(f"  Methods: {config.xai_methods}")
        print(f"  Number of Samples: {config.xai_num_samples}")
        
        print("="*60 + "\n")


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


def create_experiment_config(
    name: str,
    classification_mode: str = "binary",
    model_architecture: str = "efficientnet_b0",
    **kwargs
) -> ExperimentConfig:
    """
    Create experiment configuration programmatically.
    
    Args:
        name: Experiment name
        classification_mode: Classification mode
        model_architecture: Model architecture
        **kwargs: Additional configuration parameters
        
    Returns:
        ExperimentConfig instance
    """
    config = ExperimentConfig(
        name=name,
        classification_mode=classification_mode,
        model_architecture=model_architecture,
        **kwargs
    )
    
    # Adjust number of classes based on classification mode
    if classification_mode == "binary":
        config.num_classes = 2
    else:
        config.num_classes = 5  # SIPaKMeD has 5 classes
    
    return config


if __name__ == "__main__":
    # Test configuration manager
    config_manager = load_config()
    config_manager.print_config_summary()
    
    # Create experiment directories
    config_manager.create_experiment_directories()
    
    print("Configuration manager ready for use")
