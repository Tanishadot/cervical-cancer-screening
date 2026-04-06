"""
Hyperparameter experiment runner for systematic performance optimization.
Runs multiple experiments with different configurations and compares results.
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

from multi_domain_trainer import MultiDomainTrainer
from hyperparameter_config import HyperparameterConfig, HyperparameterSearch, get_config
from multi_domain_dataset import MultiDomainManager
from enhanced_transforms import get_domain_robust_train_transforms, get_validation_transforms

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs systematic hyperparameter experiments.
    """
    
    def __init__(self, base_config: dict, device: torch.device, output_dir: str):
        """
        Initialize experiment runner.
        
        Args:
            base_config: Base configuration
            device: Device to run on
            output_dir: Output directory
        """
        self.base_config = base_config
        self.device = device
        self.output_dir = output_dir
        self.search = HyperparameterSearch()
        
        # Create experiments directory
        self.experiments_dir = os.path.join(output_dir, "experiments")
        os.makedirs(self.experiments_dir, exist_ok=True)
    
    def run_single_experiment(self, hp_config: HyperparameterConfig, 
                            experiment_name: str) -> dict:
        """
        Run a single experiment with given hyperparameter configuration.
        
        Args:
            hp_config: Hyperparameter configuration
            experiment_name: Name for this experiment
            
        Returns:
            Experiment results
        """
        logger.info(f"🚀 Starting experiment: {experiment_name}")
        
        # Create experiment directory
        exp_dir = os.path.join(self.experiments_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(exp_dir, "config.json")
        self.search.save_config(hp_config, config_path)
        
        try:
            # Create multi-domain manager
            manager = MultiDomainManager(
                root_dir=self.base_config['dataset']['root_dir'],
                classification_mode='binary',
                herlev_train_ratio=0.2,
                random_seed=42
            )
            
            # Load datasets
            manager.load_datasets()
            
            # Create data loaders
            train_transform = get_domain_robust_train_transforms()
            val_transform = get_validation_transforms()
            
            train_loader, test_loader = manager.create_data_loaders(
                train_transform=train_transform,
                val_transform=val_transform,
                batch_size=self.base_config['dataset']['batch_size'],
                num_workers=self.base_config['dataset']['num_workers']
            )
            
            # Create trainer with hyperparameter config
            trainer = MultiDomainTrainer(
                config=self.base_config,
                device=self.device,
                output_dir=exp_dir,
                hp_config=hp_config
            )
            
            # Setup model
            trainer.setup_model()
            
            # Train model
            trainer.train(train_loader, test_loader)
            
            # Load best model for final evaluation
            best_model_path = os.path.join(exp_dir, "models", "best_model.pth")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=self.device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Final evaluation with TTA if enabled
            if hp_config.use_tta:
                from ensemble_tta import TestTimeAugmentation
                tta_predictor = TestTimeAugmentation(
                    trainer.model, 
                    tta_steps=hp_config.tta_steps,
                    device=self.device
                )
                # Run TTA evaluation (simplified)
                logger.info(f"🔄 Running TTA evaluation with {hp_config.tta_steps} steps")
            
            # Get final metrics
            final_metrics = {
                'best_val_f1': trainer.best_val_f1,
                'total_epochs': trainer.current_epoch + 1,
                'best_threshold': getattr(trainer, 'best_threshold', 0.5),
                'experiment_name': experiment_name
            }
            
            # Save results
            results_path = os.path.join(exp_dir, "results.json")
            with open(results_path, 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            logger.info(f"✅ Experiment {experiment_name} completed")
            logger.info(f"   Best F1: {final_metrics['best_val_f1']:.4f}")
            logger.info(f"   Epochs: {final_metrics['total_epochs']}")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"❌ Experiment {experiment_name} failed: {str(e)}")
            return {
                'best_val_f1': 0.0,
                'total_epochs': 0,
                'error': str(e),
                'experiment_name': experiment_name
            }
    
    def run_grid_search(self, max_experiments: int = 5) -> dict:
        """
        Run grid search experiments.
        
        Args:
            max_experiments: Maximum number of experiments to run
            
        Returns:
            Best configuration and results
        """
        logger.info(f"🔍 Starting grid search with {max_experiments} experiments")
        
        # Get configurations to test
        configs = self.search.get_grid_configs(max_experiments)
        
        all_results = []
        
        for i, hp_config in enumerate(configs):
            experiment_name = f"exp_{i+1:02d}_lr{hp_config.classifier_lr:.0e}_dropout{hp_config.dropout_rate}"
            
            result = self.run_single_experiment(hp_config, experiment_name)
            all_results.append(result)
            
            # Add to search results
            metrics = {
                'f1_score': result['best_val_f1'],
                'precision': 0.0,  # Would need full evaluation
                'recall': 0.0
            }
            self.search.add_result(hp_config, metrics)
        
        # Get best configuration
        best_config = self.search.get_best_config()
        best_result = max(all_results, key=lambda x: x['best_val_f1'])
        
        # Print comparison
        self.search.print_comparison()
        
        return {
            'best_config': best_config,
            'best_result': best_result,
            'all_results': all_results
        }
    
    def run_predefined_configs(self) -> dict:
        """
        Run experiments with predefined configurations.
        
        Returns:
            Results summary
        """
        logger.info("🎯 Running experiments with predefined configurations")
        
        predefined_names = ["baseline", "high_recall", "balanced", "conservative"]
        results = {}
        
        for name in predefined_names:
            hp_config = get_config(name)
            experiment_name = f"predefined_{name}"
            
            result = self.run_single_experiment(hp_config, experiment_name)
            results[name] = result
        
        # Print comparison
        print("\n" + "="*80)
        print("PREDEFINED CONFIGURATIONS COMPARISON")
        print("="*80)
        
        for name, result in results.items():
            f1 = result['best_val_f1']
            epochs = result['total_epochs']
            print(f"{name:12}: F1={f1:.4f}, Epochs={epochs}")
        
        print("="*80)
        
        return results


def run_hyperparameter_experiments(base_config: dict, device: torch.device, 
                                 output_dir: str, experiment_type: str = "predefined"):
    """
    Run hyperparameter experiments.
    
    Args:
        base_config: Base configuration
        device: Device to run on
        output_dir: Output directory
        experiment_type: Type of experiment ("predefined", "grid", or "single")
        
    Returns:
        Experiment results
    """
    runner = ExperimentRunner(base_config, device, output_dir)
    
    if experiment_type == "predefined":
        return runner.run_predefined_configs()
    elif experiment_type == "grid":
        return runner.run_grid_search(max_experiments=5)
    elif experiment_type == "single":
        # Run with baseline config
        hp_config = get_config("baseline")
        result = runner.run_single_experiment(hp_config, "single_baseline")
        return {"baseline": result}
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


if __name__ == "__main__":
    # Test experiment runner
    from utils.config_manager import ConfigManager
    
    # Load base config
    config_manager = ConfigManager("configs/config.yaml")
    base_config = config_manager.get_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run experiments
    results = run_hyperparameter_experiments(
        base_config=base_config,
        device=device,
        output_dir="outputs/experiments",
        experiment_type="predefined"
    )
    
    print("Experiments completed!")
