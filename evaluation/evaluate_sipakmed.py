#!/usr/bin/env python3
"""
Evaluate Trained Model on SIPaKMeD Dataset

This script loads the existing trained model (trained on SIPaKMeD, tested on Herlev)
and evaluates it on SIPaKMeD dataset to get the same-dataset performance.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.dataset import DatasetManager
from preprocessing.transforms import get_val_transforms
from models.model_factory import CervicalCancerModel
from utils.config_manager import ConfigManager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trained_model():
    """Load the existing trained model."""
    model_path = "outputs/models/best_model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    logger.info(f"Loading trained model from {model_path}")
    
    # Load configuration to get model architecture
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Create model with same architecture as training
    model_config = config.get('model', {})
    model = CervicalCancerModel(
        model_name=model_config.get('architecture', 'efficientnet_b0'),
        num_classes=model_config.get('num_classes', 2),
        pretrained=False,  # Don't need pretrained weights for inference
        dropout_rate=model_config.get('dropout_rate', 0.3),
        freeze_backbone=False  # Don't freeze for evaluation
    )
    
    # Load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, device

def load_sipakmed_dataset():
    """Load SIPaKMeD dataset for evaluation."""
    logger.info("Loading SIPaKMeD dataset")
    
    # Create dataset manager
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    dataset_manager = DatasetManager(
        root_dir=config.get('dataset', {}).get('root_dir', 'datasets'),
        classification_mode="binary",
        train_ratio=config.get('dataset', {}).get('train_split', 0.7),
        val_ratio=config.get('dataset', {}).get('val_split', 0.15),
        test_ratio=config.get('dataset', {}).get('test_split', 0.15),
        random_seed=42
    )
    
    # Load datasets
    dataset_manager.load_datasets()
    dataset_manager.create_splits()
    
    # Get test split
    test_data = dataset_manager.test_datasets.get('sipakmed')
    
    if test_data is None:
        raise ValueError("SIPaKMeD test dataset not found")
    
    logger.info(f"SIPaKMeD test set loaded: {len(test_data)} samples")
    return test_data

def create_data_loader(dataset, batch_size=32):
    """Create data loader for evaluation."""
    from datasets.dataset import TransformDataset, create_dataloader
    
    # Create a wrapper transform for albumentations
    class AlbumentationsTransform:
        def __init__(self, albumentations_transform):
            self.transform = albumentations_transform
        
        def __call__(self, image):
            # Convert PIL to numpy if needed
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Apply albumentations transform with named argument
            transformed = self.transform(image=image)
            return transformed['image']
    
    # Get validation transforms
    val_transform = get_val_transforms()
    
    # Apply transforms
    albumentations_transform = AlbumentationsTransform(val_transform)
    transformed_dataset = TransformDataset(dataset, albumentations_transform)
    
    # Create data loader
    data_loader = create_dataloader(
        dataset=transformed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return data_loader

def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset."""
    logger.info("Evaluating model on SIPaKMeD test set")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Handle different data loader formats
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 3:
                images, labels, _ = batch
            elif len(batch) == 4:
                images, labels, _, _ = batch
            else:
                logger.warning(f"Unexpected batch format with {len(batch)} items")
                continue
            
            images = images.to(device)
            labels = labels.to(device)
            
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
    
    return metrics

def save_results(metrics):
    """Save evaluation results."""
    # Create results directory
    results_dir = Path("outputs/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual result
    result = {
        "sipakmed_train_sipakmed_test": metrics
    }
    
    results_file = results_dir / "sipakmed_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Update cross-dataset results if they exist
    cross_dataset_file = results_dir / "cross_dataset_results.json"
    
    if cross_dataset_file.exists():
        try:
            with open(cross_dataset_file, 'r') as f:
                cross_results = json.load(f)
        except:
            cross_results = {}
    else:
        cross_results = {}
    
    # Add our result
    cross_results["sipakmed_train_sipakmed_test"] = metrics
    
    # Save updated cross-dataset results
    with open(cross_dataset_file, 'w') as f:
        json.dump(cross_results, f, indent=2)
    
    logger.info(f"Cross-dataset results updated at {cross_dataset_file}")

def print_results(metrics):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("SIPaKMeD EVALUATION RESULTS")
    print("="*60)
    print(f"Model: Trained on SIPaKMeD")
    print(f"Test Dataset: SIPaKMeD")
    print(f"Number of samples: {metrics['num_samples']}")
    print()
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print()
    print("Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"True Negative:  {cm[0,0]}")
    print(f"False Positive: {cm[0,1]}")
    print(f"False Negative: {cm[1,0]}")
    print(f"True Positive:  {cm[1,1]}")
    print("="*60)

def main():
    """Main evaluation function."""
    print("Evaluating Trained Model on SIPaKMeD Dataset")
    print("="*50)
    
    try:
        # Load trained model
        model, device = load_trained_model()
        
        # Load SIPaKMeD test dataset
        test_data = load_sipakmed_dataset()
        
        # Create data loader
        test_loader = create_data_loader(test_data)
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader, device)
        
        # Save results
        save_results(metrics)
        
        # Print results
        print_results(metrics)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
