"""
Visualization utilities for cervical cancer classification research.
Includes training curves, confusion matrices, ROC curves, and class activation maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import cv2
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Comprehensive visualization utilities for model analysis.
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Output directory for saving visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(
        self,
        history: Dict,
        experiment_name: str = "training",
        save_fig: bool = True
    ) -> plt.Figure:
        """
        Plot comprehensive training curves.
        
        Args:
            history: Training history dictionary
            experiment_name: Name for saving
            save_fig: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(bottom=0)
        
        # Accuracy curves
        if 'train_accuracy' in history and 'val_accuracy' in history:
            axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
        
        # F1 Score curves
        if 'train_f1' in history and 'val_f1' in history:
            axes[2].plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2)
            axes[2].plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2)
            axes[2].set_title('F1 Score Curves', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('F1 Score')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, 1)
        
        # Learning rate curve
        if 'lr' in history:
            axes[3].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[3].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Learning Rate')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_yscale('log')
        
        # Precision and Recall curves
        if 'train_precision' in history and 'val_precision' in history:
            axes[4].plot(epochs, history['train_precision'], 'b-', label='Training Precision', linewidth=2)
            axes[4].plot(epochs, history['val_precision'], 'r-', label='Validation Precision', linewidth=2)
            axes[4].plot(epochs, history.get('train_recall', []), 'b--', label='Training Recall', linewidth=2)
            axes[4].plot(epochs, history.get('val_recall', []), 'r--', label='Validation Recall', linewidth=2)
            axes[4].set_title('Precision and Recall Curves', fontsize=14, fontweight='bold')
            axes[4].set_xlabel('Epoch')
            axes[4].set_ylabel('Score')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
            axes[4].set_ylim(0, 1)
        
        # Metric comparison
        metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        for metric in metrics_to_plot:
            if f'val_{metric}' in history:
                axes[5].plot(epochs, history[f'val_{metric}'], label=metric.replace('_', ' ').title(), linewidth=2)
        
        axes[5].set_title('Validation Metrics Comparison', fontsize=14, fontweight='bold')
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('Score')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        axes[5].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'{experiment_name}_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        experiment_name: str = "confusion_matrix",
        normalize: bool = False,
        save_fig: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix with detailed annotations.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            experiment_name: Name for saving
            normalize: Whether to normalize the matrix
            save_fig: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'{experiment_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        experiment_name: str = "roc_curves",
        save_fig: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: List of class names
            experiment_name: Name for saving
            save_fig: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        n_classes = len(class_names)
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Multi-class ROC curves
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax1.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Multi-class ROC Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Macro and micro average ROC
        # Micro average
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        # Macro average
        all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)
        
        ax2.plot(fpr_micro, tpr_micro, color='deeppink', linewidth=2,
                label=f'Micro-average ROC (AUC = {roc_auc_micro:.3f})')
        ax2.plot(all_fpr, mean_tpr, color='navy', linewidth=2,
                label=f'Macro-average ROC (AUC = {roc_auc_macro:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Average ROC Curves', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'{experiment_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        experiment_name: str = "pr_curves",
        save_fig: bool = True
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: List of class names
            experiment_name: Name for saving
            save_fig: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        n_classes = len(class_names)
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Per-class PR curves
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            
            ax1.plot(recall, precision, color=color, linewidth=2,
                    label=f'{class_names[i]} (AP = {avg_precision:.3f})')
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax1.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower left", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Micro average PR curve
        precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
        avg_precision_micro = average_precision_score(y_true_bin, y_prob, average="micro")
        
        ax2.plot(recall_micro, precision_micro, color='deeppink', linewidth=2,
                label=f'Micro-average PR (AP = {avg_precision_micro:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax2.set_title('Micro-average Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'{experiment_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curves saved to {save_path}")
        
        return fig
    
    def plot_class_distribution(
        self,
        labels: np.ndarray,
        class_names: List[str],
        experiment_name: str = "class_distribution",
        save_fig: bool = True
    ) -> plt.Figure:
        """
        Plot class distribution.
        
        Args:
            labels: Class labels
            class_names: List of class names
            experiment_name: Name for saving
            save_fig: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count samples per class
        class_counts = pd.Series(labels).value_counts().sort_index()
        
        # Bar plot
        bars = ax1.bar(class_names, class_counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(class_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_names, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax2.axis('equal')
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'{experiment_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution saved to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        experiment_name: str = "metrics_comparison",
        save_fig: bool = True
    ) -> plt.Figure:
        """
        Plot comparison of metrics across different models or experiments.
        
        Args:
            metrics_dict: Dictionary of metrics for each experiment
            experiment_name: Name for saving
            save_fig: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract metric names
        all_metrics = set()
        for exp_metrics in metrics_dict.values():
            all_metrics.update(exp_metrics.keys())
        
        # Filter to common metrics
        common_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_roc']
        metrics_to_plot = [m for m in common_metrics if any(m in exp_metrics for exp_metrics in metrics_dict.values())]
        
        if not metrics_to_plot:
            logger.warning("No common metrics found for comparison")
            return None
        
        # Prepare data
        experiments = list(metrics_dict.keys())
        data = {metric: [] for metric in metrics_to_plot}
        
        for exp_name in experiments:
            exp_metrics = metrics_dict[exp_name]
            for metric in metrics_to_plot:
                data[metric].append(exp_metrics.get(metric, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(experiments))
        width = 0.15
        colors = plt.cm.Set1(np.linspace(0, 1, len(metrics_to_plot)))
        
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            offset = (i - len(metrics_to_plot)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), color=color)
            
            # Add value labels on bars
            for bar, value in zip(bars, data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Metrics Comparison Across Experiments', fontsize=16, fontweight='bold')
        ax.set_xlabel('Experiments')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'{experiment_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        return fig
    
    def plot_activation_map_overlay(
        self,
        image: np.ndarray,
        activation_map: np.ndarray,
        class_name: str,
        confidence: float,
        experiment_name: str = "activation_overlay",
        save_fig: bool = True
    ) -> plt.Figure:
        """
        Plot activation map overlaid on original image.
        
        Args:
            image: Original image (H, W, 3)
            activation_map: Activation map (H, W)
            class_name: Predicted class name
            confidence: Prediction confidence
            experiment_name: Name for saving
            save_fig: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Activation map
        im = axes[1].imshow(activation_map, cmap='jet')
        axes[1].set_title('Activation Map', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        # Resize activation map to match image if needed
        if activation_map.shape != image.shape[:2]:
            activation_map = cv2.resize(activation_map, (image.shape[1], image.shape[0]))
        
        # Create colormap
        heatmap = plt.cm.jet(activation_map)
        heatmap[:, :, 3] = 0.6  # Set alpha
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(heatmap)
        axes[2].set_title(f'Overlay\n{class_name} ({confidence:.3f})', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'{experiment_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Activation overlay saved to {save_path}")
        
        return fig
    
    def create_comprehensive_report(
        self,
        history: Dict,
        metrics: Dict,
        cm: np.ndarray,
        class_names: List[str],
        experiment_name: str = "comprehensive_report"
    ):
        """
        Create a comprehensive visualization report.
        
        Args:
            history: Training history
            metrics: Test metrics
            cm: Confusion matrix
            class_names: Class names
            experiment_name: Name for saving
        """
        logger.info(f"Creating comprehensive report for {experiment_name}")
        
        # Plot training curves
        self.plot_training_curves(history, f"{experiment_name}_training")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, class_names, f"{experiment_name}_confusion")
        
        # Plot metrics summary
        metrics_summary = {experiment_name: metrics}
        self.plot_metrics_comparison(metrics_summary, f"{experiment_name}_metrics")
        
        logger.info(f"Comprehensive report completed for {experiment_name}")


def create_visualizer(output_dir: str = "outputs/visualizations") -> Visualizer:
    """
    Create visualizer instance.
    
    Args:
        output_dir: Output directory
        
    Returns:
        Visualizer instance
    """
    return Visualizer(output_dir)


if __name__ == "__main__":
    # Example usage
    visualizer = create_visualizer()
    
    # Create dummy data for testing
    history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
        'train_accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.65, 0.75, 0.8, 0.82, 0.83]
    }
    
    visualizer.plot_training_curves(history)
    print("Visualization utilities ready for use")
