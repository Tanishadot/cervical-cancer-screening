#!/usr/bin/env python3
"""
Cross-Dataset Evaluation Script for Cervical Cancer Classification

This script automates evaluation for remaining dataset combinations:
1. Train on SIPaKMeD → Test on SIPaKMeD
2. Train on Herlev → Test on Herlev

Results are merged with existing cross-dataset evaluation metrics.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import random
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.dataset import DatasetManager, DatasetType, ClassificationMode
from preprocessing.transforms import get_train_transforms, get_val_transforms
from models.model_factory import CervicalCancerModel
from training.trainer import Trainer
from utils.config_manager import ConfigManager
from utils.label_mapping import LabelMapper
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CrossDatasetEvaluator:
    """Handles cross-dataset evaluation experiments."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize evaluator with configuration."""
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Create output directories
        self.metrics_dir = Path("outputs/metrics")
        self.visualizations_dir = Path("outputs/visualizations/cross_dataset")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_name(self, train_dataset: str, test_dataset: str) -> str:
        """Create experiment name from dataset names."""
        return f"{train_dataset}_train_{test_dataset}_test"
    
    def load_datasets(self, train_dataset: str, test_dataset: str) -> Tuple[Any, Any, Any, Any]:
        """Load train and test datasets."""
        logger.info(f"Loading datasets: Train={train_dataset}, Test={test_dataset}")
        
        # Create dataset manager
        dataset_manager = DatasetManager(
            root_dir=self.config.get('dataset', {}).get('root_dir', 'datasets'),
            classification_mode="binary",
            train_ratio=self.config.get('dataset', {}).get('train_split', 0.7),
            val_ratio=self.config.get('dataset', {}).get('val_split', 0.15),
            test_ratio=self.config.get('dataset', {}).get('test_split', 0.15),
            random_seed=42
        )
        
        # Load all datasets
        dataset_manager.load_datasets()
        
        # Create train/val/test splits
        dataset_manager.create_splits()
        
        # Get specific datasets
        train_data = dataset_manager.train_datasets.get(train_dataset)
        test_data = dataset_manager.test_datasets.get(test_dataset)
        
        if train_data is None:
            raise ValueError(f"Training dataset '{train_dataset}' not found")
        if test_data is None:
            raise ValueError(f"Test dataset '{test_dataset}' not found")
        
        return train_data, test_data, dataset_manager
    
    def create_data_loaders(self, train_data, test_data, dataset_manager, train_dataset, test_dataset, batch_size: int = 32):
        """Create data loaders for training and testing."""
        # Get transforms
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        # Create data loaders using the dataset manager
        data_loaders = dataset_manager.create_data_loaders(
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=batch_size,
            num_workers=self.config.get('dataset', {}).get('num_workers', 0)
        )
        
        # Get specific loaders for the datasets we need
        # This depends on the actual structure returned by create_data_loaders
        train_loader = data_loaders.get('train', {}).get(train_dataset.lower())
        test_loader = data_loaders.get('test', {}).get(test_dataset.lower())
        
        if train_loader is None or test_loader is None:
            logger.info("Using fallback data loader creation")
            # Fallback: create individual data loaders
            from datasets.dataset import create_dataloader
            
            # Apply transforms to datasets
            train_dataset = TransformDataset(train_data, train_transform)
            test_dataset = TransformDataset(test_data, val_transform)
            
            train_loader = create_dataloader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config.get('dataset', {}).get('num_workers', 0)
            )
            
            test_loader = create_dataloader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.get('dataset', {}).get('num_workers', 0)
            )
        
        return train_loader, test_loader
    
    def create_model(self) -> CervicalCancerModel:
        """Create model for training."""
        model_config = self.config.get('model', {})
        
        model = CervicalCancerModel(
            model_name=model_config.get('architecture', 'efficientnet_b0'),
            num_classes=model_config.get('num_classes', 2),
            pretrained=model_config.get('pretrained', True),
            dropout_rate=model_config.get('dropout_rate', 0.3),
            freeze_backbone=model_config.get('freeze_backbone', True)
        )
        
        return model.to(self.device)
    
    def train_model(self, model: CervicalCancerModel, train_loader, val_loader, experiment_name: str) -> CervicalCancerModel:
        """Train model on training dataset."""
        logger.info(f"Training model for experiment: {experiment_name}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            output_dir=str(self.metrics_dir),
            experiment_name=experiment_name
        )
        
        # Train model
        history = trainer.train()
        
        # Save model
        model_path = self.metrics_dir / f"{experiment_name}_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
    
    def evaluate_model(self, model: CervicalCancerModel, test_loader) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info("Evaluating model on test dataset")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'num_samples': len(y_true)
        }
        
        logger.info(f"Evaluation completed: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        return metrics
    
    def run_experiment(self, train_dataset: str, test_dataset: str) -> Dict[str, Any]:
        """Run a single cross-dataset experiment."""
        experiment_name = self.create_experiment_name(train_dataset, test_dataset)
        logger.info(f"Running experiment: {experiment_name}")
        
        try:
            # Load datasets
            train_data, test_data, dataset_manager = self.load_datasets(train_dataset, test_dataset)
            
            # Create data loaders
            batch_size = self.config.get('dataset', {}).get('batch_size', 32)
            train_loader, test_loader = self.create_data_loaders(
                train_data, test_data, dataset_manager, train_dataset, test_dataset, batch_size
            )
            
            # Create model
            model = self.create_model()
            
            # Train model
            trained_model = self.train_model(model, train_loader, test_loader, experiment_name)
            
            # Evaluate model
            metrics = self.evaluate_model(trained_model, test_loader)
            
            logger.info(f"Experiment {experiment_name} completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {e}")
            return None
    
    def load_existing_results(self) -> Dict[str, Any]:
        """Load existing cross-dataset results."""
        results_file = self.metrics_dir / "cross_dataset_results.json"
        
        if results_file.exists():
            logger.info(f"Loading existing results from {results_file}")
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            logger.info("No existing results found")
            return {}
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        results_file = self.metrics_dir / "cross_dataset_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def create_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table from results."""
        summary_data = []
        
        for experiment_name, metrics in results.items():
            # Parse experiment name
            parts = experiment_name.split('_')
            train_dataset = parts[0].upper()
            test_dataset = parts[2].upper()
            
            summary_data.append({
                'Experiment': f"{train_dataset} → {test_dataset}",
                'Train Dataset': train_dataset,
                'Test Dataset': test_dataset,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score']
            })
        
        return pd.DataFrame(summary_data)
    
    def save_summary_table(self, summary_df: pd.DataFrame):
        """Save summary table to CSV."""
        summary_file = self.metrics_dir / "cross_dataset_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary table saved to {summary_file}")
    
    def create_visualizations(self, summary_df: pd.DataFrame):
        """Create comparison visualizations."""
        logger.info("Creating visualizations")
        
        # Check if DataFrame is empty
        if summary_df.empty:
            logger.warning("No data available for visualization")
            return
        
        # Set style
        plt.style.use('default')  # Use default style instead of seaborn-v0_8
        sns.set_palette("husl")
        
        # Accuracy comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(summary_df['Experiment'], summary_df['Accuracy'], 
                     color=sns.color_palette("husl", len(summary_df)))
        ax.set_title('Cross-Dataset Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, summary_df['Accuracy']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # F1 score comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(summary_df['Experiment'], summary_df['F1 Score'], 
                     color=sns.color_palette("husl", len(summary_df)))
        ax.set_title('Cross-Dataset F1 Score Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, summary_df['F1 Score']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.visualizations_dir}")
    
    def print_results(self, summary_df: pd.DataFrame):
        """Print results to console."""
        print("\n" + "="*60)
        print("CROSS DATASET EVALUATION")
        print("="*60)
        
        if summary_df.empty:
            print("No results to display.")
        else:
            for _, row in summary_df.iterrows():
                print(f"\nTrain: {row['Train Dataset']} | Test: {row['Test Dataset']}")
                print(f"Accuracy: {row['Accuracy']:.4f}")
                print(f"Precision: {row['Precision']:.4f}")
                print(f"Recall: {row['Recall']:.4f}")
                print(f"F1 Score: {row['F1 Score']:.4f}")
        
        print(f"\nResults saved to {self.metrics_dir}")
        print("="*60)
    
    def run_all_experiments(self):
        """Run all remaining cross-dataset experiments."""
        logger.info("Starting cross-dataset evaluation")
        
        # Define experiments to run
        experiments = [
            ("sipakmed", "sipakmed"),
            ("herlev", "herlev")
        ]
        
        # Load existing results
        self.results = self.load_existing_results()
        
        # Run experiments
        for train_dataset, test_dataset in experiments:
            experiment_name = self.create_experiment_name(train_dataset, test_dataset)
            
            # Skip if already exists
            if experiment_name in self.results:
                logger.info(f"Experiment {experiment_name} already exists, skipping...")
                continue
            
            # Run experiment
            metrics = self.run_experiment(train_dataset, test_dataset)
            
            if metrics:
                self.results[experiment_name] = metrics
        
        # Save results
        self.save_results(self.results)
        
        # Create summary table
        summary_df = self.create_summary_table(self.results)
        self.save_summary_table(summary_df)
        
        # Create visualizations
        self.create_visualizations(summary_df)
        
        # Print results
        self.print_results(summary_df)
        
        logger.info("Cross-dataset evaluation completed")
        return self.results


def main():
    """Main function."""
    print("Cross-Dataset Evaluation Script")
    print("="*50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Create evaluator
    evaluator = CrossDatasetEvaluator(config_manager)
    
    # Run all experiments
    results = evaluator.run_all_experiments()
    
    print("\nEvaluation completed successfully!")
    return results


if __name__ == "__main__":
    main()
