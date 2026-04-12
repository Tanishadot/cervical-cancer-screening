"""
Cross-dataset evaluation script for cervical cancer classification.
Evaluates trained model on Herlev dataset for cross-dataset generalization performance.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import argparse

from utils.config_manager import ConfigManager
from utils.label_mapping import LabelMapper, ClassificationMode, DatasetType
from datasets.dataset import HerlevDataset, create_dataloader
from models.model_factory import ModelFactory
from preprocessing.transforms import get_val_transforms


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'herlev_evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_model(checkpoint_path: str, model_config: dict, device: torch.device) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    
    # Create model
    model = ModelFactory.create_model(
        model_name=model_config['architecture'],
        num_classes=model_config['num_classes'],
        pretrained=False,  # Don't use pretrained weights when loading checkpoint
        freeze_backbone=False,
        dropout_rate=model_config.get('dropout_rate', 0.3)
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info("✅ Model loaded successfully from checkpoint")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list
) -> dict:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_paths = []
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 4:  # (image, mapped_label, original_label, img_path)
                images, labels, original_labels, paths = batch
            else:  # (image, label)
                images, labels = batch[:2]
                original_labels = labels
                paths = [f"sample_{batch_idx}" for _ in range(len(images))]
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
    metrics['precision'] = precision_score(all_labels, all_predictions, average='binary')
    metrics['recall'] = recall_score(all_labels, all_predictions, average='binary')
    metrics['f1_score'] = f1_score(all_labels, all_predictions, average='binary')
    
    # ROC-AUC (need probabilities for positive class)
    if len(np.unique(all_labels)) == 2:  # Binary classification
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probabilities[:, 1])
    else:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(all_labels, all_predictions, average=None).tolist()
    metrics['recall_per_class'] = recall_score(all_labels, all_predictions, average=None).tolist()
    metrics['f1_per_class'] = f1_score(all_labels, all_predictions, average=None).tolist()
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report
    
    # Raw predictions for further analysis
    metrics['predictions'] = all_predictions.tolist()
    metrics['probabilities'] = all_probabilities.tolist()
    metrics['true_labels'] = all_labels.tolist()
    metrics['image_paths'] = all_paths
    
    logger.info(f"✅ Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Herlev Dataset Evaluation')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(metrics: dict, config: dict, save_path: str):
    """Save evaluation results to JSON file."""
    # Create results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'herlev',
        'model_checkpoint': 'outputs/models/best_model.pth',
        'configuration': config,
        'metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'precision_per_class': metrics['precision_per_class'],
            'recall_per_class': metrics['recall_per_class'],
            'f1_per_class': metrics['f1_per_class'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report']
        },
        'dataset_info': {
            'total_samples': len(metrics['true_labels']),
            'normal_samples': metrics['true_labels'].count(0),
            'abnormal_samples': metrics['true_labels'].count(1)
        }
    }
    
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {save_path}")


def print_results_summary(metrics: dict, class_names: list):
    """Print comprehensive results summary."""
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION RESULTS - HERLEV DATASET")
    print("="*80)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\n📈 PER-CLASS METRICS:")
    for i, class_name in enumerate(class_names):
        if i < len(metrics['precision_per_class']):
            print(f"   {class_name}:")
            print(f"     Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"     Recall: {metrics['recall_per_class'][i]:.4f}")
            print(f"     F1 Score: {metrics['f1_per_class'][i]:.4f}")
    
    print(f"\n🎯 CONFUSION MATRIX:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"   Predicted →   {class_names[0]:<12} {class_names[1]:<12}")
    print(f"   True ↓")
    for i, class_name in enumerate(class_names):
        print(f"   {class_name:<12} {cm[i,0]:<12} {cm[i,1]:<12}")
    
    # Dataset statistics
    total_samples = len(metrics['true_labels'])
    normal_samples = metrics['true_labels'].count(0)
    abnormal_samples = metrics['true_labels'].count(1)
    
    print(f"\n📋 DATASET STATISTICS:")
    print(f"   Total Samples: {total_samples}")
    print(f"   Normal Samples: {normal_samples} ({normal_samples/total_samples*100:.1f}%)")
    print(f"   Abnormal Samples: {abnormal_samples} ({abnormal_samples/total_samples*100:.1f}%)")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    if metrics['recall_per_class'][1] > 0.8:  # ABNORMAL class recall
        print(f"   ✅ High abnormal class detection ({metrics['recall_per_class'][1]:.1%})")
    else:
        print(f"   ⚠️  Low abnormal class detection ({metrics['recall_per_class'][1]:.1%})")
    
    if metrics['accuracy'] > 0.8:
        print(f"   ✅ Good overall accuracy ({metrics['accuracy']:.1%})")
    else:
        print(f"   ⚠️  Moderate overall accuracy ({metrics['accuracy']:.1%})")
    
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Cross-dataset evaluation on Herlev dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='outputs/models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/metrics',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Setup
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # Setup logging
    logger = setup_logging('outputs/logs')
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Load model configuration
        logger.info("Loading model configuration...")
        model_config = {
            'architecture': config['model']['architecture'],
            'num_classes': config['model']['num_classes'],
            'dropout_rate': config['model'].get('dropout_rate', 0.3)
        }
        
        # 2. Load trained model
        logger.info("Loading trained model...")
        model = load_model(args.checkpoint, model_config, device)
        
        # 3. Setup label mapper for binary classification
        logger.info("Setting up label mapper...")
        label_mapper = LabelMapper(ClassificationMode.BINARY)
        class_names = label_mapper.get_class_names()
        
        # 4. Create validation transforms
        logger.info("Creating validation transforms...")
        image_size = config['augmentation']['image_size']
        val_transform = get_val_transforms(image_size=image_size)
        
        # 5. Load Herlev dataset
        logger.info("Loading Herlev dataset...")
        herlev_dataset = HerlevDataset(
            root_dir=config['dataset']['root_dir'],
            label_mapper=label_mapper,
            transform=val_transform,
            mode="test",
            use_cropped_only=False  # Use all images for Herlev (no CROPPED images available)
        )
        
        logger.info(f"Loaded {len(herlev_dataset)} samples from Herlev dataset")
        
        # 6. Create dataloader
        dataloader = create_dataloader(
            dataset=herlev_dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=False,
            num_workers=config['dataset']['num_workers'],
            pin_memory=True
        )
        
        # 7. Run evaluation
        logger.info("Running cross-dataset evaluation...")
        metrics = evaluate_model(model, dataloader, device, class_names)
        
        # 8. Save results
        results_path = os.path.join(args.output_dir, 'herlev_evaluation.json')
        save_results(metrics, config, results_path)
        
        # 9. Save confusion matrix plot
        cm_plot_path = os.path.join('outputs/visualizations', 'herlev_confusion_matrix.png')
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_names, cm_plot_path)
        
        # 10. Print summary
        print_results_summary(metrics, class_names)
        
        logger.info("✅ Cross-dataset evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
