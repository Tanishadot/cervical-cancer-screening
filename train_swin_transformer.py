#!/usr/bin/env python3
"""
Swin Transformer Training and Evaluation Pipeline
Uses same preprocessing as EfficientNet-B0 for fair comparison
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent))

from datasets.dataset import DatasetManager
from preprocessing.transforms import get_train_transforms, get_val_transforms
from models.model_factory import CervicalCancerModel
from training.trainer import Trainer
from utils.config_manager import ConfigManager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_data_loader_with_transforms(dataset, transform, batch_size=16, shuffle=False):
    """Create data loader with proper transform handling."""
    from datasets.dataset import TransformDataset, create_dataloader
    
    # Create wrapper for albumentations transforms
    class AlbumentationsTransform:
        def __init__(self, albumentations_transform):
            self.transform = albumentations_transform
        
        def __call__(self, image):
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            return self.transform(image=image)['image']
    
    # Apply transforms
    albumentations_transform = AlbumentationsTransform(transform)
    transformed_dataset = TransformDataset(dataset, albumentations_transform)
    
    return create_dataloader(
        dataset=transformed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

def load_training_datasets():
    """Load SIPaKMeD dataset for training and validation only."""
    logger.info("Loading SIPaKMeD dataset for training and validation")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Create dataset manager - only for SIPaKMeD
    dataset_manager = DatasetManager(
        root_dir=config.get('dataset', {}).get('root_dir', 'datasets'),
        classification_mode="binary",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Load only SIPaKMeD dataset
    dataset_manager.load_datasets()
    dataset_manager.create_splits()
    
    # Get SIPaKMeD train/val splits
    train_data = dataset_manager.train_datasets['sipakmed']
    val_data = dataset_manager.val_datasets['sipakmed']
    
    return train_data, val_data, config

def train_swin_model():
    """Train Swin Transformer on SIPaKMeD only."""
    logger.info("Training Swin Transformer on SIPaKMeD")
    
    # Load training datasets (SIPaKMeD only)
    train_data, val_data, config = load_training_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create Swin model
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        pretrained=True,
        dropout_rate=0.3,
        freeze_backbone=False
    ).to(device)
    
    # Load datasets (same preprocessing as EfficientNet)
    dataset_manager = DatasetManager(
        root_dir=config.get('dataset', {}).get('root_dir', 'datasets'),
        classification_mode="binary",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    dataset_manager.load_datasets()
    dataset_manager.create_splits()
    
    # Get SIPaKMeD splits
    train_data = dataset_manager.train_datasets['sipakmed']
    val_data = dataset_manager.val_datasets['sipakmed']
    
    # Create data loaders with SAME transforms as EfficientNet
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_loader = create_data_loader_with_transforms(
        train_data, train_transform, batch_size=16, shuffle=True
    )
    
    val_loader = create_data_loader_with_transforms(
        val_data, val_transform, batch_size=16, shuffle=False
    )
    
    # Train model with AdamW optimizer and learning rate 1e-4
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir="outputs/swin_training",
        experiment_name="swin_sipakmed"
    )
    
    # Train with specific parameters for Swin Transformer
    history = trainer.train(
        num_epochs=20,
        learning_rate=1e-4,  # AdamW with 1e-4 for stable convergence
        weight_decay=1e-4,
        patience=5,
        use_class_weights=True,
        use_amp=True,
        unfreeze_epoch=10
    )
    
    # Save model
    model_path = "outputs/models/best_swin_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # The trainer will save the best model, so we don't need to save here
    logger.info(f"Swin model will be saved to {model_path} by trainer")
    
    return model, device

def load_best_swin_model():
    """Load the best trained Swin model."""
    model_path = "outputs/models/best_swin_model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best Swin model not found at {model_path}")
    
    logger.info(f"Loading best Swin model from {model_path}")
    
    # Create model
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        pretrained=False,  # Don't need pretrained weights
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded Swin model from epoch {checkpoint['epoch']} with F1: {checkpoint['val_f1']:.4f}")
    return model, device

def load_herlev_test_dataset():
    """Load full Herlev dataset for test evaluation only."""
    logger.info("Loading full Herlev dataset for cross-dataset evaluation")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Create dataset manager for Herlev only
    dataset_manager = DatasetManager(
        root_dir=config.get('dataset', {}).get('root_dir', 'datasets'),
        classification_mode="binary",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Load datasets
    dataset_manager.load_datasets()
    dataset_manager.create_splits()
    
    # Get full Herlev dataset (use test split as entire dataset)
    herlev_data = dataset_manager.test_datasets['herlev']
    
    return herlev_data

def evaluate_model(model, device, dataset_name, split_name):
    """Evaluate model on specified dataset."""
    logger.info(f"Evaluating on {dataset_name} {split_name}")
    
    # Get transforms
    val_transform = get_val_transforms()
    
    if dataset_name == 'sipakmed':
        # Load SIPaKMeD test split
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        dataset_manager = DatasetManager(
            root_dir=config.get('dataset', {}).get('root_dir', 'datasets'),
            classification_mode="binary",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        dataset_manager.load_datasets()
        dataset_manager.create_splits()
        test_data = dataset_manager.test_datasets['sipakmed']
    elif dataset_name == 'herlev':
        # Load full Herlev dataset for cross-dataset evaluation
        test_data = load_herlev_test_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loader
    test_loader = create_data_loader_with_transforms(
        test_data, val_transform, batch_size=16, shuffle=False
    )
    
    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch[:2]
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels)
    }

def main():
    """Main training and evaluation pipeline."""
    print("Swin Transformer Training and Evaluation")
    print("=" * 50)
    
    # Train model
    train_swin_model()
    
    # Load best model for evaluation
    model, device = load_best_swin_model()
    
    # Evaluate on both datasets
    results = {
        'model': 'swin_transformer',
        'train_dataset': 'sipakmed',
        'test_results': {}
    }
    
    # Same dataset evaluation
    sipakmed_results = evaluate_model(model, device, 'sipakmed', 'test')
    results['test_results']['sipakmed'] = sipakmed_results
    
    # Cross-dataset evaluation
    herlev_results = evaluate_model(model, device, 'herlev', 'test')
    results['test_results']['herlev'] = herlev_results
    
    # Save results
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/swin_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("SWIN TRANSFORMER RESULTS")
    print("=" * 60)
    
    sipakmed_acc = sipakmed_results['accuracy']
    herlev_acc = herlev_results['accuracy']
    generalization_gap = sipakmed_acc - herlev_acc
    
    print(f"SIPaKMeD Accuracy: {sipakmed_acc:.4f}")
    print(f"Herlev Accuracy: {herlev_acc:.4f}")
    print(f"Generalization Gap: {generalization_gap:.4f}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()
