"""
Hyperparameter configuration and search space for cervical cancer classification.
Provides configurable options for systematic performance optimization.
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for training optimization."""
    
    # Learning rates
    classifier_lr: float = 1e-4
    backbone_lr: float = 1e-5
    
    # Regularization
    dropout_rate: float = 0.5
    weight_decay: float = 1e-4
    
    # Loss function
    pos_weight: float = 1.8
    loss_type: str = "bce"  # "bce" or "focal"
    
    # Training strategy
    unfreeze_epoch: int = 3
    early_stopping_patience: int = 2
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine" or "plateau"
    plateau_patience: int = 1
    plateau_factor: float = 0.5
    
    # Data handling
    use_weighted_sampler: bool = False
    use_tta: bool = False
    tta_steps: int = 5
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_range: tuple = (0.3, 0.7)
    
    # Ensemble
    use_ensemble: bool = False
    ensemble_models: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ["efficientnet_b0"]


class HyperparameterSearch:
    """Hyperparameter search space for systematic optimization."""
    
    def __init__(self):
        self.search_space = {
            "classifier_lr": [1e-4, 5e-5, 3e-5],
            "backbone_lr": [1e-5, 5e-6],
            "dropout_rate": [0.3, 0.5, 0.6],
            "pos_weight": [1.5, 1.8, 2.0, 2.5],
            "scheduler_type": ["cosine", "plateau"],
            "use_weighted_sampler": [False, True],
            "use_tta": [False, True],
            "optimize_threshold": [True, False]
        }
        
        self.best_configs = []
    
    def get_grid_configs(self, max_configs: int = 10) -> List[HyperparameterConfig]:
        """
        Generate grid search configurations.
        
        Args:
            max_configs: Maximum number of configurations to generate
            
        Returns:
            List of hyperparameter configurations
        """
        configs = []
        
        # Generate some key combinations
        key_combinations = [
            # High recall focus
            {
                "classifier_lr": 5e-5,
                "backbone_lr": 5e-6,
                "dropout_rate": 0.6,
                "pos_weight": 2.0,
                "use_weighted_sampler": True,
                "optimize_threshold": True
            },
            # Balanced approach
            {
                "classifier_lr": 1e-4,
                "backbone_lr": 1e-5,
                "dropout_rate": 0.5,
                "pos_weight": 1.8,
                "use_weighted_sampler": False,
                "optimize_threshold": True
            },
            # Conservative approach
            {
                "classifier_lr": 3e-5,
                "backbone_lr": 5e-6,
                "dropout_rate": 0.3,
                "pos_weight": 1.5,
                "use_weighted_sampler": False,
                "optimize_threshold": False
            },
            # Aggressive approach
            {
                "classifier_lr": 1e-4,
                "backbone_lr": 1e-5,
                "dropout_rate": 0.6,
                "pos_weight": 2.5,
                "use_weighted_sampler": True,
                "optimize_threshold": True
            }
        ]
        
        for i, combo in enumerate(key_combinations[:max_configs]):
            config = HyperparameterConfig(**combo)
            configs.append(config)
        
        return configs
    
    def save_config(self, config: HyperparameterConfig, filepath: str):
        """Save configuration to file."""
        config_dict = asdict(config)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, filepath: str) -> HyperparameterConfig:
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return HyperparameterConfig(**config_dict)
    
    def add_result(self, config: HyperparameterConfig, metrics: Dict[str, float]):
        """Add experiment result for tracking."""
        result = {
            "config": asdict(config),
            "metrics": metrics,
            "f1_score": metrics.get("f1_score", 0.0)
        }
        self.best_configs.append(result)
        
        # Sort by F1 score
        self.best_configs.sort(key=lambda x: x["f1_score"], reverse=True)
        
        # Keep only top 10
        self.best_configs = self.best_configs[:10]
    
    def get_best_config(self) -> HyperparameterConfig:
        """Get the best performing configuration."""
        if not self.best_configs:
            return HyperparameterConfig()
        
        best_dict = self.best_configs[0]["config"]
        return HyperparameterConfig(**best_dict)
    
    def print_comparison(self):
        """Print comparison of all experiments."""
        if not self.best_configs:
            print("No experiments run yet.")
            return
        
        print("\n" + "="*100)
        print("HYPERPARAMETER SEARCH RESULTS")
        print("="*100)
        
        for i, result in enumerate(self.best_configs):
            config = result["config"]
            metrics = result["metrics"]
            
            print(f"\n{i+1}. F1: {metrics['f1_score']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"   Config: LR={config['classifier_lr']:.0e}, Dropout={config['dropout_rate']}, PosWeight={config['pos_weight']}")
            print(f"   Sampler: {config['use_weighted_sampler']}, TTA: {config['use_tta']}, ThresholdOpt: {config['optimize_threshold']}")
        
        print("="*100)


# Predefined configurations for quick testing
PREDEFINED_CONFIGS = {
    "baseline": HyperparameterConfig(),
    "high_recall": HyperparameterConfig(
        classifier_lr=5e-5,
        backbone_lr=5e-6,
        dropout_rate=0.6,
        pos_weight=2.0,
        use_weighted_sampler=True,
        optimize_threshold=True
    ),
    "balanced": HyperparameterConfig(
        classifier_lr=1e-4,
        backbone_lr=1e-5,
        dropout_rate=0.5,
        pos_weight=1.8,
        scheduler_type="plateau",
        use_tta=True,
        optimize_threshold=True
    ),
    "conservative": HyperparameterConfig(
        classifier_lr=3e-5,
        backbone_lr=5e-6,
        dropout_rate=0.3,
        pos_weight=1.5,
        use_weighted_sampler=False,
        optimize_threshold=False
    )
}


def get_config(config_name: str = "baseline") -> HyperparameterConfig:
    """
    Get predefined configuration.
    
    Args:
        config_name: Name of predefined config
        
    Returns:
        Hyperparameter configuration
    """
    return PREDEFINED_CONFIGS.get(config_name, PREDEFINED_CONFIGS["baseline"])


if __name__ == "__main__":
    # Test hyperparameter search
    search = HyperparameterSearch()
    configs = search.get_grid_configs(max_configs=3)
    
    print("Generated configurations:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config}")
    
    # Test saving/loading
    search.save_config(configs[0], "test_config.json")
    loaded_config = search.load_config("test_config.json")
    print(f"Loaded config: {loaded_config}")
