"""
Cross-dataset evaluation for cervical cancer classification.
Evaluates model generalization across SIPaKMeD and Herlev datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from collections import defaultdict

from datasets.dataset import create_dataset, get_dataset_info
from utils.label_mapping import LabelMapper, ClassificationMode, DatasetType
from training.trainer import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossDatasetEvaluator:
    """
    Cross-dataset evaluation for model generalization assessment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        datasets_root: str,
        output_dir: str = "outputs",
        device: Optional[torch.device] = None
    ):
        """
        Initialize cross-dataset evaluator.
        
        Args:
            model: Trained model to evaluate
            datasets_root: Root directory containing datasets
            output_dir: Output directory for results
            device: Device for evaluation
        """
        self.model = model
        self.datasets_root = datasets_root
        self.output_dir = output_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'cross_dataset'), exist_ok=True)
    
    def evaluate_cross_dataset(
        self,
        train_dataset: str,
        test_dataset: str,
        classification_mode: str = "binary",
        batch_size: int = 32,
        transform=None
    ) -> Dict:
        """
        Evaluate model trained on one dataset and tested on another.
        
        Args:
            train_dataset: Dataset model was trained on ("sipakmed" or "herlev")
            test_dataset: Dataset to test on ("sipakmed" or "herlev")
            classification_mode: "binary" or "multiclass"
            batch_size: Batch size for evaluation
            transform: Image transforms
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {train_dataset} -> {test_dataset} ({classification_mode})")
        
        # Create label mapper
        if classification_mode == "binary":
            label_mapper = LabelMapper(ClassificationMode.BINARY)
        else:
            label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        
        # Create test dataset
        test_dataset_obj = create_dataset(
            root_dir=self.datasets_root,
            dataset_name=test_dataset,
            classification_mode=classification_mode,
            transform=transform,
            mode="test"
        )
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset_obj,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Get dataset info
        dataset_info = get_dataset_info(test_dataset_obj)
        num_classes = dataset_info['num_classes']
        class_names = dataset_info['class_names']
        
        # Initialize metrics calculator
        metrics_calculator = MetricsCalculator(num_classes, class_names)
        
        # Evaluate
        results = self._evaluate_model(
            test_loader, 
            metrics_calculator, 
            class_names,
            f"{train_dataset}_to_{test_dataset}"
        )
        
        # Add dataset information
        results['train_dataset'] = train_dataset
        results['test_dataset'] = test_dataset
        results['classification_mode'] = classification_mode
        results['test_dataset_info'] = dataset_info
        
        return results
    
    def _evaluate_model(
        self,
        data_loader: DataLoader,
        metrics_calculator: MetricsCalculator,
        class_names: List[str],
        experiment_name: str
    ) -> Dict:
        """
        Evaluate model on given data loader.
        
        Args:
            data_loader: Data loader for evaluation
            metrics_calculator: Metrics calculator
            class_names: Class names
            experiment_name: Name for saving results
            
        Returns:
            Evaluation results
        """
        all_preds = []
        all_labels = []
        all_probs = []
        all_image_paths = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating")
            for images, labels, original_labels, image_paths in pbar:
                images = images.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_image_paths.extend(image_paths)
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(all_labels, all_preds, all_probs)
        
        # Generate confusion matrix
        cm = metrics_calculator.get_confusion_matrix(all_labels, all_preds)
        
        # Analyze per-class performance
        per_class_analysis = self._analyze_per_class_performance(
            all_labels, all_preds, all_probs, class_names, all_image_paths
        )
        
        # Save results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'per_class_analysis': per_class_analysis,
            'sample_predictions': self._get_sample_predictions(
                all_labels, all_preds, all_probs, all_image_paths, class_names, n=10
            )
        }
        
        # Save to file
        results_path = os.path.join(self.output_dir, 'cross_dataset', f'{experiment_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, class_names, experiment_name)
        
        # Plot per-class metrics
        self._plot_per_class_metrics(per_class_analysis, experiment_name)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        
        return results
    
    def _analyze_per_class_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        image_paths: List[str]
    ) -> Dict:
        """
        Analyze performance per class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            class_names: Class names
            image_paths: Image paths
            
        Returns:
            Per-class analysis
        """
        analysis = {}
        
        for i, class_name in enumerate(class_names):
            # Get indices for this class
            class_indices = np.where(y_true == i)[0]
            
            if len(class_indices) == 0:
                continue
            
            # Get predictions for this class
            class_true = y_true[class_indices]
            class_pred = y_pred[class_indices]
            class_prob = y_prob[class_indices]
            class_paths = [image_paths[idx] for idx in class_indices]
            
            # Calculate metrics for this class
            correct_predictions = class_pred == class_true
            accuracy = correct_predictions.mean()
            
            # Average confidence for correct and incorrect predictions
            correct_probs = class_prob[correct_predictions, i] if correct_predictions.any() else []
            incorrect_probs = class_prob[~correct_predictions, i] if (~correct_predictions).any() else []
            
            avg_confidence_correct = np.mean(correct_probs) if len(correct_probs) > 0 else 0
            avg_confidence_incorrect = np.mean(incorrect_probs) if len(incorrect_probs) > 0 else 0
            
            # Get most confident correct and incorrect predictions
            if len(correct_probs) > 0:
                most_confident_correct_idx = class_indices[np.argmax(correct_probs)]
                most_confident_correct_path = image_paths[most_confident_correct_idx]
                most_confident_correct_prob = np.max(correct_probs)
            else:
                most_confident_correct_path = None
                most_confident_correct_prob = 0
            
            if len(incorrect_probs) > 0:
                most_confident_incorrect_idx = class_indices[np.where(~correct_predictions)[0][np.argmax(incorrect_probs)]]
                most_confident_incorrect_path = image_paths[most_confident_incorrect_idx]
                most_confident_incorrect_prob = np.max(incorrect_probs)
            else:
                most_confident_incorrect_path = None
                most_confident_incorrect_prob = 0
            
            analysis[class_name] = {
                'num_samples': len(class_indices),
                'accuracy': accuracy,
                'avg_confidence_correct': avg_confidence_correct,
                'avg_confidence_incorrect': avg_confidence_incorrect,
                'most_confident_correct_path': most_confident_correct_path,
                'most_confident_correct_prob': most_confident_correct_prob,
                'most_confident_incorrect_path': most_confident_incorrect_path,
                'most_confident_incorrect_prob': most_confident_incorrect_prob
            }
        
        return analysis
    
    def _get_sample_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        image_paths: List[str],
        class_names: List[str],
        n: int = 10
    ) -> List[Dict]:
        """
        Get sample predictions for analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            image_paths: Image paths
            class_names: Class names
            n: Number of samples to return
            
        Returns:
            List of sample predictions
        """
        samples = []
        
        # Get random indices
        indices = np.random.choice(len(y_true), min(n, len(y_true)), replace=False)
        
        for idx in indices:
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = float(y_prob[idx, y_pred[idx]])
            correct = y_true[idx] == y_pred[idx]
            
            samples.append({
                'image_path': image_paths[idx],
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'correct': correct
            })
        
        return samples
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], experiment_name: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Cross-Dataset Confusion Matrix: {experiment_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'cross_dataset', f'{experiment_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, per_class_analysis: Dict, experiment_name: str):
        """Plot per-class performance metrics."""
        class_names = list(per_class_analysis.keys())
        accuracies = [per_class_analysis[name]['accuracy'] for name in class_names]
        confidences_correct = [per_class_analysis[name]['avg_confidence_correct'] for name in class_names]
        confidences_incorrect = [per_class_analysis[name]['avg_confidence_incorrect'] for name in class_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy per class
        ax1.bar(class_names, accuracies, color='skyblue')
        ax1.set_title(f'Per-Class Accuracy: {experiment_name}')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Confidence comparison
        x = np.arange(len(class_names))
        width = 0.35
        
        ax2.bar(x - width/2, confidences_correct, width, label='Correct Predictions', color='lightgreen')
        ax2.bar(x + width/2, confidences_incorrect, width, label='Incorrect Predictions', color='lightcoral')
        
        ax2.set_title(f'Prediction Confidence: {experiment_name}')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Average Confidence')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'cross_dataset', f'{experiment_name}_per_class_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_cross_dataset_evaluation(
        self,
        classification_mode: str = "binary",
        batch_size: int = 32,
        transform=None
    ) -> Dict:
        """
        Run comprehensive cross-dataset evaluation.
        
        Args:
            classification_mode: "binary" or "multiclass"
            batch_size: Batch size
            transform: Image transforms
            
        Returns:
            All evaluation results
        """
        all_results = {}
        
        # Define dataset combinations
        datasets = ["sipakmed", "herlev"]
        
        # Run all cross-dataset evaluations
        for train_dataset in datasets:
            for test_dataset in datasets:
                if train_dataset != test_dataset:  # Only cross-dataset, not same dataset
                    experiment_name = f"{train_dataset}_to_{test_dataset}"
                    
                    results = self.evaluate_cross_dataset(
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        classification_mode=classification_mode,
                        batch_size=batch_size,
                        transform=transform
                    )
                    
                    all_results[experiment_name] = results
        
        # Generate summary report
        self._generate_summary_report(all_results, classification_mode)
        
        # Save all results
        all_results_path = os.path.join(self.output_dir, 'cross_dataset', 'all_cross_dataset_results.json')
        with open(all_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"All cross-dataset evaluations completed. Results saved to {all_results_path}")
        
        return all_results
    
    def _generate_summary_report(self, all_results: Dict, classification_mode: str):
        """Generate summary report of cross-dataset evaluations."""
        report = {
            'classification_mode': classification_mode,
            'summary': {},
            'detailed_results': all_results
        }
        
        # Calculate summary statistics
        accuracies = []
        f1_scores = []
        
        for experiment_name, results in all_results.items():
            metrics = results['metrics']
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_macro'])
        
        report['summary'] = {
            'num_experiments': len(all_results),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1_macro': np.mean(f1_scores),
            'std_f1_macro': np.std(f1_scores),
            'best_accuracy': np.max(accuracies),
            'worst_accuracy': np.min(accuracies)
        }
        
        # Plot summary
        self._plot_summary_results(all_results, classification_mode)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'cross_dataset', 'summary_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {report_path}")
    
    def _plot_summary_results(self, all_results: Dict, classification_mode: str):
        """Plot summary of cross-dataset results."""
        experiment_names = list(all_results.keys())
        accuracies = [results['metrics']['accuracy'] for results in all_results.values()]
        f1_scores = [results['metrics']['f1_macro'] for results in all_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(experiment_names, accuracies, color='skyblue')
        ax1.set_title(f'Cross-Dataset Accuracy Comparison ({classification_mode})')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1 score comparison
        bars2 = ax2.bar(experiment_names, f1_scores, color='lightgreen')
        ax2.set_title(f'Cross-Dataset F1 Score Comparison ({classification_mode})')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('F1 Score (Macro)')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'cross_dataset', f'summary_comparison_{classification_mode}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Example usage
    from models.model_factory import create_model
    from preprocessing.transforms import get_val_transforms
    
    # Create model (would load trained weights in practice)
    model = create_model('efficientnet_b0', num_classes=2)
    
    # Create evaluator
    evaluator = CrossDatasetEvaluator(
        model=model,
        datasets_root="datasets",
        output_dir="outputs"
    )
    
    print("Cross-dataset evaluator ready for use")
