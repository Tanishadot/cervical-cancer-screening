"""
Baseline vs Optimized EfficientNet-B0 Comparison Script
Demonstrates the improvements and expected performance gains.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_guard import run_dataset_guard
from utils.config_manager import load_config


class BaselineOptimizedComparison:
    """Compare baseline vs optimized EfficientNet-B0 configurations."""
    
    def __init__(self):
        self.comparison_results = {}
        
    def load_configurations(self):
        """Load baseline and optimized configurations."""
        print("🔧 LOADING CONFIGURATIONS")
        print("=" * 60)
        
        # Baseline config
        baseline_config_path = "configs/efficientnet_baseline_config.yaml"
        if os.path.exists(baseline_config_path):
            config_manager = load_config(baseline_config_path)
            baseline_config = config_manager.get_config()
            print(f"✅ Baseline config loaded: {baseline_config_path}")
        else:
            baseline_config = self.create_baseline_config()
            print(f"✅ Baseline config created (default)")
        
        # Optimized config
        optimized_config_path = "configs/efficientnet_optimized_config.yaml"
        if os.path.exists(optimized_config_path):
            config_manager = load_config(optimized_config_path)
            optimized_config = config_manager.get_config()
            print(f"✅ Optimized config loaded: {optimized_config_path}")
        else:
            optimized_config = self.create_optimized_config()
            print(f"✅ Optimized config created (default)")
        
        self.comparison_results['baseline_config'] = baseline_config
        self.comparison_results['optimized_config'] = optimized_config
        
        return baseline_config, optimized_config
    
    def create_baseline_config(self):
        """Create baseline configuration."""
        return {
            'experiment': {
                'name': 'efficientnet_b0_baseline',
                'description': 'Standard EfficientNet-B0 baseline'
            },
            'model': {
                'architecture': 'efficientnet_b0',
                'pretrained': True,
                'dropout_rate': 0.3
            },
            'training': {
                'num_epochs': 25,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'weight_decay': 0.00001,
                'early_stopping_patience': 5
            },
            'dataset': {
                'classification_mode': 'binary',
                'use_cropped_only': True
            }
        }
    
    def create_optimized_config(self):
        """Create optimized configuration."""
        return {
            'experiment': {
                'name': 'efficientnet_b0_optimized',
                'description': 'Optimized EfficientNet-B0 with advanced techniques'
            },
            'model': {
                'architecture': 'efficientnet_b0',
                'pretrained': True,
                'dropout_rate': 0.4
            },
            'training': {
                'num_epochs': 30,
                'batch_size': 16,
                'learning_rate': 0.0001,
                'weight_decay': 0.00001,
                'early_stopping_patience': 5,
                'label_smoothing': 0.1,
                'unfreeze_epoch': 5
            },
            'dataset': {
                'classification_mode': 'binary',
                'use_cropped_only': True
            }
        }
    
    def compare_configurations(self, baseline_config, optimized_config):
        """Compare key configuration differences."""
        print("\n📊 CONFIGURATION COMPARISON")
        print("=" * 60)
        
        comparison_table = []
        
        # Model comparison
        comparison_table.append({
            'Aspect': 'Model Architecture',
            'Baseline': baseline_config['model']['architecture'],
            'Optimized': optimized_config['model']['architecture'],
            'Improvement': 'Same'
        })
        
        comparison_table.append({
            'Aspect': 'Dropout Rate',
            'Baseline': baseline_config['model']['dropout_rate'],
            'Optimized': optimized_config['model']['dropout_rate'],
            'Improvement': f"+{optimized_config['model']['dropout_rate'] - baseline_config['model']['dropout_rate']:.1f} (Better regularization)"
        })
        
        # Training comparison
        comparison_table.append({
            'Aspect': 'Epochs',
            'Baseline': baseline_config['training']['num_epochs'],
            'Optimized': optimized_config['training']['num_epochs'],
            'Improvement': f"+{optimized_config['training']['num_epochs'] - baseline_config['training']['num_epochs']} (More training)"
        })
        
        comparison_table.append({
            'Aspect': 'Batch Size',
            'Baseline': baseline_config['training']['batch_size'],
            'Optimized': optimized_config['training']['batch_size'],
            'Improvement': f"-{baseline_config['training']['batch_size'] - optimized_config['training']['batch_size']} (Better generalization)"
        })
        
        comparison_table.append({
            'Aspect': 'Label Smoothing',
            'Baseline': 'None',
            'Optimized': optimized_config['training'].get('label_smoothing', 'N/A'),
            'Improvement': 'Prevents overconfidence'
        })
        
        comparison_table.append({
            'Aspect': 'Progressive Unfreezing',
            'Baseline': 'None',
            'Optimized': f"Epoch {optimized_config['training'].get('unfreeze_epoch', 'N/A')}",
            'Improvement': 'Better feature learning'
        })
        
        # Print comparison table
        print(f"{'Aspect':<25} {'Baseline':<15} {'Optimized':<15} {'Improvement':<30}")
        print("-" * 85)
        
        for row in comparison_table:
            print(f"{row['Aspect']:<25} {row['Baseline']:<15} {row['Optimized']:<15} {row['Improvement']:<30}")
        
        self.comparison_results['config_comparison'] = comparison_table
        return comparison_table
    
    def compare_augmentation_strategies(self):
        """Compare augmentation strategies."""
        print("\n🎨 AUGMENTATION STRATEGY COMPARISON")
        print("=" * 60)
        
        baseline_aug = [
            "Resize → 224x224",
            "HorizontalFlip (p=0.5)",
            "VerticalFlip (p=0.5)",
            "RandomRotation (±15°)",
            "ColorJitter (0.2, 0.2)",
            "ImageNet normalization"
        ]
        
        optimized_aug = [
            "Resize → 256 → CenterCrop(224)",
            "HorizontalFlip (p=0.5)",
            "VerticalFlip (p=0.5)",
            "RandomRotation (±15°)",
            "RandomResizedCrop (scale=0.9-1.0)",
            "ColorJitter (0.2, 0.2, 0.2, 0.1)",
            "GaussianBlur (p=0.2, σ=0.1-1.5)",
            "ImageNet normalization"
        ]
        
        print("BASELINE AUGMENTATION:")
        for i, aug in enumerate(baseline_aug, 1):
            print(f"  {i}. {aug}")
        
        print("\nOPTIMIZED AUGMENTATION:")
        for i, aug in enumerate(optimized_aug, 1):
            print(f"  {i}. {aug}")
        
        improvements = [
            "✅ CenterCrop for consistent focus",
            "✅ RandomResizedCrop for scale invariance",
            "✅ Enhanced ColorJitter (4 parameters)",
            "✅ GaussianBlur for medical noise robustness"
        ]
        
        print("\nOPTIMIZATION IMPROVEMENTS:")
        for improvement in improvements:
            print(f"  {improvement}")
        
        self.comparison_results['augmentation_comparison'] = {
            'baseline': baseline_aug,
            'optimized': optimized_aug,
            'improvements': improvements
        }
    
    def compare_training_strategies(self):
        """Compare training strategies."""
        print("\n🚀 TRAINING STRATEGY COMPARISON")
        print("=" * 60)
        
        strategies = {
            'Loss Function': {
                'Baseline': 'CrossEntropyLoss',
                'Optimized': 'CrossEntropyLoss + Label Smoothing (0.1)',
                'Benefit': 'Prevents overconfidence, improves generalization'
            },
            'Backbone Training': {
                'Baseline': 'Frozen throughout',
                'Optimized': 'Progressive unfreezing (epoch 5)',
                'Benefit': 'Better feature adaptation, domain transfer'
            },
            'Learning Rate': {
                'Baseline': 'Fixed 1e-4',
                'Optimized': 'Adaptive with ReduceLROnPlateau',
                'Benefit': 'Automatic convergence optimization'
            },
            'Batch Size': {
                'Baseline': '32',
                'Optimized': '16',
                'Benefit': 'Better generalization, more gradient updates'
            },
            'Regularization': {
                'Baseline': 'Dropout 0.3, Weight decay 1e-5',
                'Optimized': 'Dropout 0.4, Weight decay 1e-5, Label smoothing',
                'Benefit': 'Stronger regularization, less overfitting'
            }
        }
        
        for aspect, details in strategies.items():
            print(f"\n{aspect}:")
            print(f"  Baseline:   {details['Baseline']}")
            print(f"  Optimized:  {details['Optimized']}")
            print(f"  Benefit:    {details['Benefit']}")
        
        self.comparison_results['training_strategy_comparison'] = strategies
    
    def predict_performance_gains(self):
        """Predict expected performance gains."""
        print("\n🎯 EXPECTED PERFORMANCE GAINS")
        print("=" * 60)
        
        performance_predictions = {
            'baseline': {
                'validation_accuracy': '75-85%',
                'herlev_accuracy': '60-70%',
                'generalization_gap': '15-20%',
                'training_stability': 'Good',
                'overfitting_risk': 'Moderate'
            },
            'optimized': {
                'validation_accuracy': '85-95%',
                'herlev_accuracy': '65-80%',
                'generalization_gap': '10-20%',
                'training_stability': 'Excellent',
                'overfitting_risk': 'Low'
            }
        }
        
        print("BASELINE EXPECTED PERFORMANCE:")
        for metric, value in performance_predictions['baseline'].items():
            print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        print("\nOPTIMIZED EXPECTED PERFORMANCE:")
        for metric, value in performance_predictions['optimized'].items():
            print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        improvements = {
            'validation_accuracy': '+10-15%',
            'herlev_accuracy': '+5-10%',
            'generalization_gap': '-5-10%',
            'training_stability': 'Significantly improved',
            'overfitting_risk': 'Reduced'
        }
        
        print("\nPREDICTED IMPROVEMENTS:")
        for metric, improvement in improvements.items():
            print(f"  {metric.replace('_', ' ').title()}: {improvement}")
        
        self.comparison_results['performance_predictions'] = performance_predictions
        self.comparison_results['predicted_improvements'] = improvements
    
    def generate_training_commands(self):
        """Generate training commands for both versions."""
        print("\n💻 TRAINING COMMANDS")
        print("=" * 60)
        
        print("BASELINE TRAINING:")
        print("python train_efficientnet_baseline.py --mode binary")
        print("python train_efficientnet_baseline.py --mode multiclass")
        print("")
        
        print("OPTIMIZED TRAINING:")
        print("python train_efficientnet_optimized.py --mode binary")
        print("python train_efficientnet_optimized.py --mode multiclass")
        print("")
        
        print("CUSTOM PARAMETERS:")
        print("# Baseline with custom settings")
        print("python train_efficientnet_baseline.py --mode binary --epochs 30 --batch-size 16")
        print("")
        print("# Optimized with custom settings")
        print("python train_efficientnet_optimized.py --mode binary --epochs 35 --dropout 0.5 --label-smoothing 0.15")
        print("")
        
        print("CONFIG FILE BASED:")
        print("# Baseline")
        print("python train_efficientnet_baseline.py --config configs/efficientnet_baseline_config.yaml")
        print("")
        print("# Optimized")
        print("python train_efficientnet_optimized.py --config configs/efficientnet_optimized_config.yaml")
    
    def create_comparison_summary(self):
        """Create comprehensive comparison summary."""
        print("\n📋 COMPARISON SUMMARY")
        print("=" * 60)
        
        summary = {
            'comparison_date': datetime.now().isoformat(),
            'objective': 'Achieve 85-95% validation accuracy with strong generalization',
            'key_improvements': [
                'Higher dropout (0.3 → 0.4) for better regularization',
                'Label smoothing (0.1) to prevent overconfidence',
                'Progressive unfreezing for better feature learning',
                'Enhanced augmentation with GaussianBlur',
                'Smaller batch size (32 → 16) for better generalization',
                'Adaptive learning rate scheduling',
                'More training epochs (25 → 30)'
            ],
            'expected_outcomes': {
                'validation_accuracy_target': '85-95%',
                'generalization_gap_target': '<20%',
                'training_stability': 'Excellent',
                'overfitting_risk': 'Low'
            },
            'success_criteria': [
                'Validation accuracy ≥85%',
                'F1-score ≥0.85',
                'Generalization gap <20%',
                'Stable training curves',
                'No severe overfitting'
            ]
        }
        
        print("OBJECTIVE:")
        print(f"  {summary['objective']}")
        
        print("\nKEY IMPROVEMENTS:")
        for i, improvement in enumerate(summary['key_improvements'], 1):
            print(f"  {i}. {improvement}")
        
        print("\nEXPECTED OUTCOMES:")
        for metric, target in summary['expected_outcomes'].items():
            print(f"  {metric.replace('_', ' ').title()}: {target}")
        
        print("\nSUCCESS CRITERIA:")
        for i, criterion in enumerate(summary['success_criteria'], 1):
            print(f"  {i}. {criterion}")
        
        # Save summary
        output_dir = Path("outputs/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "baseline_optimized_comparison.json", 'w') as f:
            json.dump({
                **summary,
                **self.comparison_results
            }, f, indent=2, default=str)
        
        print(f"\n✅ Comparison summary saved to: {output_dir / 'baseline_optimized_comparison.json'}")
        
        return summary
    
    def run_complete_comparison(self):
        """Run complete baseline vs optimized comparison."""
        print("🚀 BASELINE vs OPTIMIZED EFFICIENTNET-B0 COMPARISON")
        print("=" * 80)
        print("Objective: Achieve 85-95% validation accuracy with strong generalization")
        print("=" * 80)
        
        # Load configurations
        baseline_config, optimized_config = self.load_configurations()
        
        # Compare configurations
        self.compare_configurations(baseline_config, optimized_config)
        
        # Compare augmentation strategies
        self.compare_augmentation_strategies()
        
        # Compare training strategies
        self.compare_training_strategies()
        
        # Predict performance gains
        self.predict_performance_gains()
        
        # Generate training commands
        self.generate_training_commands()
        
        # Create summary
        summary = self.create_comparison_summary()
        
        print("\n" + "=" * 80)
        print("🎉 COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\nRECOMMENDATIONS:")
        print("1. ✅ Use the optimized version for production training")
        print("2. ✅ Target 85-95% validation accuracy")
        print("3. ✅ Monitor generalization gap (<20%)")
        print("4. ✅ Use F1-score as primary metric")
        print("5. ✅ Compare results with baseline for validation")
        
        print("\nNEXT STEPS:")
        print("1. Install PyTorch: pip install torch torchvision")
        print("2. Run optimized training: python train_efficientnet_optimized.py --mode binary")
        print("3. Monitor results in outputs/efficientnet_optimized/")
        print("4. Compare with baseline results")
        print("5. Use for hybrid model comparison")
        
        return True


def main():
    """Main comparison function."""
    comparison = BaselineOptimizedComparison()
    success = comparison.run_complete_comparison()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
