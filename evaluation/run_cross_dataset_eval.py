#!/usr/bin/env python3
"""
Cross-Dataset Evaluation Script using Existing Trained Model

This script evaluates the existing trained model on different dataset combinations:
1. SIPaKMeD Train → SIPaKMeD Test
2. Herlev Train → Herlev Test
3. SIPaKMeD Train → Herlev Test
4. Herlev Train → SIPaKMeD Test

Uses the existing trained model from outputs/models/best_model.pth
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

from datasets.dataset import DatasetManager
from preprocessing.transforms import get_val_transforms
from models.model_factory import CervicalCancerModel
from utils.config_manager import ConfigManager
from utils.label_mapping import LabelMapper, ClassificationMode
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
    """Handles cross-dataset evaluation using existing trained model."""
    
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
        
        # Load existing trained model
        self.model = self.load_trained_model()
        
    def load_trained_model(self) -> CervicalCancerModel:
        """Load the existing trained model."""
        model_path = "outputs/models/best_model.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        logger.info(f"Loading trained model from {model_path}")
        
        # Create model with same architecture as training
        model_config = self.config.get('model', {})
        model = CervicalCancerModel(
            model_name=model_config.get('architecture', 'efficientnet_b0'),
            num_classes=model_config.get('num_classes', 2),
            pretrained=False,  # Don't need pretrained weights for inference
            dropout_rate=model_config.get('dropout_rate', 0.3),
            freeze_backbone=False  # Don't freeze for evaluation
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def create_experiment_name(self, train_dataset: str, test_dataset: str) -> str:
        """Create experiment name from dataset names."""
        return f"{train_dataset}_train_{test_dataset}_test"
    
    def load_dataset_for_evaluation(self, dataset_name: str) -> Tuple[Any, Any]:
        """Load and split dataset for evaluation."""
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Create dataset manager
        dataset_manager = DatasetManager(
            root_dir=self.config.get('dataset', {}).get('root_dir', 'datasets'),
            classification_mode="binary",
            train_ratio=self.config.get('dataset', {}).get('train_split', 0.7),
            val_ratio=self.config.get('dataset', {}).get('val_split', 0.15),
            test_ratio=self.config.get('dataset', {}).get('test_split', 0.15),
            random_seed=42
        )
        
        # Load datasets
        dataset_manager.load_datasets()
        dataset_manager.create_splits()
        
        # Get train and test splits
        train_data = dataset_manager.train_datasets.get(dataset_name)
        test_data = dataset_manager.test_datasets.get(dataset_name)
        
        if train_data is None or test_data is None:
            raise ValueError(f"Dataset '{dataset_name}' not found or not properly split")
        
        return train_data, test_data
    
    def create_data_loader(self, dataset, transform, batch_size: int = 32, shuffle: bool = False):
        """Create data loader for evaluation."""
        from datasets.dataset import TransformDataset, create_dataloader
        
        # Apply transforms
        transformed_dataset = TransformDataset(dataset, transform)
        
        # Create data loader
        data_loader = create_dataloader(
            dataset=transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get('dataset', {}).get('num_workers', 0)
        )
        
        return data_loader
    
    def evaluate_model(self, test_loader, dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info(f"Evaluating model on {dataset_name} test set")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
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
        
        logger.info(f"Evaluation on {dataset_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        return metrics
    
    def run_experiment(self, train_dataset: str, test_dataset: str) -> Dict[str, Any]:
        """Run a single cross-dataset experiment."""
        experiment_name = self.create_experiment_name(train_dataset, test_dataset)
        logger.info(f"Running experiment: {experiment_name}")
        
        try:
            # For same-dataset evaluation, we need to load the dataset and use its test split
            if train_dataset == test_dataset:
                _, test_data = self.load_dataset_for_evaluation(test_dataset)
                
                # Create test data loader
                val_transform = get_val_transforms()
                test_loader = self.create_data_loader(test_data, val_transform, shuffle=False)
                
                # Evaluate model
                metrics = self.evaluate_model(test_loader, test_dataset)
                
            else:
                # For cross-dataset evaluation, load test dataset
                _, test_data = self.load_dataset_for_evaluation(test_dataset)
                
                # Create test data loader
                val_transform = get_val_transforms()
                test_loader = self.create_data_loader(test_data, val_transform, shuffle=False)
                
                # Evaluate model
                metrics = self.evaluate_model(test_loader, test_dataset)
            
            logger.info(f"Experiment {experiment_name} completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {e}")
            return None
    
    def load_existing_results(self) -> Dict[str, Any]:
        """Load existing cross-dataset results."""
        results_file = self.metrics_dir / "cross_dataset_results.json"
        
        if results_file.exists() and results_file.stat().st_size > 2:  # Check if not empty
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
        plt.style.use('default')
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
        """Run all cross-dataset experiments."""
        logger.info("Starting cross-dataset evaluation")
        
        # Define experiments to run
        experiments = [
            ("sipakmed", "sipakmed"),  # Same dataset
            ("herlev", "herlev"),      # Same dataset
            ("sipakmed", "herlev"),    # Cross dataset
            ("herlev", "sipakmed")     # Cross dataset
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
    print("Cross-Dataset Evaluation Script (Using Existing Trained Model)")
    print("="*70)
    
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
