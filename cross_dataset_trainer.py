#!/usr/bin/env python3
"""
Cross-Dataset Training with Proper Dataset Separation
Trains on SIPaKMeD only, evaluates on Herlev only
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
    
    class AlbumentationsTransform:
        def __init__(self, albumentations_transform):
            self.transform = albumentations_transform
        
        def __call__(self, image):
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            return self.transform(image=image)['image']
    
    albumentations_transform = AlbumentationsTransform(transform)
    transformed_dataset = TransformDataset(dataset, albumentations_transform)
    
    return create_dataloader(
        dataset=transformed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

def load_cross_dataset_datasets():
    """Load datasets with cross-dataset separation."""
    print("Loading datasets for cross-dataset evaluation")
    print("=" * 50)
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Create dataset manager with cross-dataset mode
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
    
    # SIPaKMeD: Create splits for training
    print("Training on: SIPaKMeD")
    dataset_manager.create_splits()
    sipakmed_train = dataset_manager.train_datasets['sipakmed']
    sipakmed_val = dataset_manager.val_datasets['sipakmed']
    sipakmed_test = dataset_manager.test_datasets['sipakmed']
    
    # Herlev: Load as full dataset (no splitting)
    print("Testing on: Herlev (unseen dataset)")
    herlev_dataset = dataset_manager.herlev_dataset  # Full dataset
    
    print(f"SIPaKMeD - Train: {len(sipakmed_train)}, Val: {len(sipakmed_val)}, Test: {len(sipakmed_test)}")
    print(f"Herlev - Full: {len(herlev_dataset)} samples")
    
    return sipakmed_train, sipakmed_val, herlev_dataset, config

def train_cross_dataset_model():
    """Train Swin Transformer on SIPaKMeD only."""
    print("\n" + "=" * 50)
    print("TRAINING PHASE - SIPaKMeD ONLY")
    print("=" * 50)
    
    # Load datasets with cross-dataset separation
    train_data, val_data, herlev_data, config = load_cross_dataset_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create Swin model
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        pretrained=True,
        dropout_rate=0.3,
        freeze_backbone=False
    ).to(device)
    
    # Create data loaders
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_loader = create_data_loader_with_transforms(
        train_data, train_transform, batch_size=16, shuffle=True
    )
    
    val_loader = create_data_loader_with_transforms(
        val_data, val_transform, batch_size=16, shuffle=False
    )
    
    # Train with AdamW optimizer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir="outputs/cross_dataset_training",
        experiment_name="swin_cross_dataset"
    )
    
    history = trainer.train(
        num_epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=5,
        use_class_weights=True,
        use_amp=True,
        unfreeze_epoch=10
    )
    
    print(f"\n✅ Training completed on SIPaKMeD")
    return model, device, herlev_data

def evaluate_cross_dataset(model, device, herlev_data):
    """Evaluate trained model on Herlev dataset."""
    print("\n" + "=" * 50)
    print("EVALUATION PHASE - HERLEV (UNSEEN)")
    print("=" * 50)
    
    # Create data loader for Herlev
    val_transform = get_val_transforms()
    herlev_loader = create_data_loader_with_transforms(
        herlev_data, val_transform, batch_size=16, shuffle=False
    )
    
    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in herlev_loader:
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
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels)
    }
    
    print(f"\n📊 Cross-Dataset Results on Herlev:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Samples: {len(all_labels)}")
    
    return results

def main():
    """Main cross-dataset training and evaluation."""
    print("Cross-Dataset Training: SIPaKMeD → Herlev")
    print("=" * 50)
    
    # Train on SIPaKMeD only
    model, device, herlev_data = train_cross_dataset_model()
    
    # Evaluate on Herlev only
    herlev_results = evaluate_cross_dataset(model, device, herlev_data)
    
    # Save results
    results = {
        'model': 'swin_transformer',
        'train_dataset': 'sipakmed',
        'test_dataset': 'herlev',
        'cross_dataset_results': herlev_results
    }
    
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/cross_dataset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Cross-dataset evaluation completed!")
    print(f"📁 Results saved to: outputs/metrics/cross_dataset_results.json")
    
    return results

if __name__ == "__main__":
    main()
