"""
Simplified multi-domain training script that works with the existing infrastructure.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

from utils.config_manager import ConfigManager
from utils.label_mapping import LabelMapper, ClassificationMode
from datasets.dataset import SIPaKMeDDataset, HerlevDataset, create_dataloader
from enhanced_transforms import get_domain_robust_train_transforms, get_validation_transforms
from models.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_multi_domain_dataset():
    """Create a simple combined dataset."""
    # Create label mapper
    label_mapper = LabelMapper(ClassificationMode.BINARY)
    
    # Create transforms
    train_transform = get_domain_robust_train_transforms()
    val_transform = get_validation_transforms()
    
    # Load SIPaKMeD (all for training)
    sipakmed_dataset = SIPaKMeDDataset(
        root_dir="datasets",
        label_mapper=label_mapper,
        transform=train_transform,
        mode="train",
        use_cropped_only=True
    )
    
    # Load Herlev (split 20/80)
    herlev_full = HerlevDataset(
        root_dir="datasets",
        label_mapper=label_mapper,
        transform=None,
        mode="all",
        use_cropped_only=False
    )
    
    # Simple split
    total_size = len(herlev_full)
    train_size = int(0.2 * total_size)
    test_size = total_size - train_size
    
    herlev_train, herlev_test = torch.utils.data.random_split(
        herlev_full, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms to Herlev subsets
    class TransformSubset(torch.utils.data.Subset):
        def __init__(self, dataset, transform):
            super().__init__(dataset.dataset, dataset.indices)
            self.transform = transform
        
        def __getitem__(self, idx):
            img_path, label, orig_label = self.dataset.samples[self.indices[idx]]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                if hasattr(self.transform, '__call__') and hasattr(self.transform, 'transforms'):
                    image_np = np.array(image)
                    transformed = self.transform(image=image_np)
                    image = transformed['image']
                else:
                    image = self.transform(image)
            
            return image, label, orig_label, img_path
    
    herlev_train_transformed = TransformSubset(herlev_train, train_transform)
    herlev_test_transformed = TransformSubset(herlev_test, val_transform)
    
    # Combine training datasets
    combined_train = ConcatDataset([sipakmed_dataset, herlev_train_transformed])
    
    logger.info(f"SIPaKMeD: {len(sipakmed_dataset)} samples")
    logger.info(f"Herlev train: {len(herlev_train_transformed)} samples")
    logger.info(f"Herlev test: {len(herlev_test_transformed)} samples")
    logger.info(f"Combined train: {len(combined_train)} samples")
    
    return combined_train, herlev_test_transformed


def train_model():
    """Train the model with simplified approach."""
    logger.info("Starting simplified multi-domain training...")
    
    # Create datasets
    train_dataset, test_dataset = create_simple_multi_domain_dataset()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = ModelFactory.create_model(
        model_name='efficientnet_b0',
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
        dropout_rate=0.4
    )
    
    model.to(device)
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=2.5e-5)
    
    # Training loop
    num_epochs = 5
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        # Unfreeze after epoch 2
        if epoch == 2:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-4)
            logger.info("Backbone unfrozen")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Handle different batch formats
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch[:2]
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                if len(batch) == 4:
                    images, labels, _, _ = batch
                else:
                    images, labels = batch[:2]
                
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        val_f1 = f1_score(all_labels, all_predictions, average='binary')
        val_precision = precision_score(all_labels, all_predictions, average='binary')
        val_recall = recall_score(all_labels, all_predictions, average='binary')
        val_roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': val_f1,
            }, 'outputs/multi_domain/best_model.pth')
            logger.info(f"New best model saved with F1: {val_f1:.4f}")
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss/len(test_loader):.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        logger.info(f"ROC-AUC: {val_roc_auc:.4f}")
        
        scheduler.step()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    model.eval()
    
    final_predictions = []
    final_labels = []
    final_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch[:2]
            
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            final_predictions.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
            final_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate final metrics
    final_predictions = np.array(final_predictions)
    final_labels = np.array(final_labels)
    final_probabilities = np.array(final_probabilities)
    
    final_accuracy = accuracy_score(final_labels, final_predictions)
    final_precision = precision_score(final_labels, final_predictions, average='binary')
    final_recall = recall_score(final_labels, final_predictions, average='binary')
    final_f1 = f1_score(final_labels, final_predictions, average='binary')
    final_roc_auc = roc_auc_score(final_labels, final_probabilities[:, 1])
    final_cm = confusion_matrix(final_labels, final_predictions)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'final_metrics': {
            'accuracy': final_accuracy,
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': final_f1,
            'roc_auc': final_roc_auc,
            'confusion_matrix': final_cm.tolist()
        },
        'dataset_info': {
            'test_samples': len(final_labels),
            'normal_samples': int(np.sum(final_labels == 0)),
            'abnormal_samples': int(np.sum(final_labels == 1))
        }
    }
    
    os.makedirs('outputs/multi_domain', exist_ok=True)
    with open('outputs/multi_domain/herlev_cross_dataset.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'ABNORMAL'],
                yticklabels=['NORMAL', 'ABNORMAL'])
    plt.title('Confusion Matrix - Multi-Domain Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('outputs/multi_domain/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final results
    print("\n" + "="*80)
    print("MULTI-DOMAIN TRAINING RESULTS")
    print("="*80)
    print(f"📊 Final Metrics:")
    print(f"   Accuracy: {final_accuracy:.4f}")
    print(f"   Precision: {final_precision:.4f}")
    print(f"   Recall: {final_recall:.4f}")
    print(f"   F1 Score: {final_f1:.4f}")
    print(f"   ROC-AUC: {final_roc_auc:.4f}")
    print(f"📋 Dataset Info:")
    print(f"   Test Samples: {results['dataset_info']['test_samples']}")
    print(f"   Normal: {results['dataset_info']['normal_samples']}")
    print(f"   Abnormal: {results['dataset_info']['abnormal_samples']}")
    print("="*80)
    
    logger.info("✅ Training completed successfully!")


if __name__ == "__main__":
    from PIL import Image
    train_model()
