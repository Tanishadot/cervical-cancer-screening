"""
EfficientNet-B0 Baseline Training Script
Trains on SIPaKMeD (CROPPED-only) and evaluates on Herlev for cross-dataset generalization.
"""

import os
import sys
import argparse
import logging
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_guard import run_dataset_guard
from utils.config_manager import load_config
from utils.label_mapping import LabelMapper, ClassificationMode, DatasetType
from datasets.dataset import SIPaKMeDDataset, HerlevDataset, create_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('efficientnet_baseline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EfficientNetB0Baseline:
    """EfficientNet-B0 baseline trainer with cross-dataset evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'outputs/efficientnet_baseline'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        logger.info(f"EfficientNet-B0 Baseline initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def get_train_transforms(self):
        """Get training transforms with augmentation."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_val_transforms(self):
        """Get validation/test transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def create_model(self, num_classes, dropout_rate=0.3):
        """Create EfficientNet-B0 model with custom classifier."""
        import torchvision.models as models
        
        # Load pretrained EfficientNet-B0
        model = models.efficientnet_b0(pretrained=True)
        
        # Freeze backbone initially
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        model = model.to(self.device)
        
        logger.info(f"Created EfficientNet-B0 with {num_classes} output classes")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def create_datasets(self):
        """Create training, validation, and test datasets."""
        dataset_config = self.config.get('dataset', {})
        classification_mode = dataset_config.get('classification_mode', 'binary')
        
        # Create label mapper
        if classification_mode == "binary":
            label_mapper = LabelMapper(ClassificationMode.BINARY)
            num_classes = 2
        else:
            label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
            num_classes = label_mapper.get_num_classes(DatasetType.SIPAKMED)
        
        logger.info(f"Classification mode: {classification_mode}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Training dataset (SIPaKMeD CROPPED-only)
        train_dataset = SIPaKMeDDataset(
            root_dir=dataset_config.get('root_dir', 'datasets'),
            label_mapper=label_mapper,
            transform=self.get_train_transforms(),
            mode="train",
            use_cropped_only=True
        )
        
        # Validation dataset (SIPaKMeD CROPPED-only)
        val_dataset = SIPaKMeDDataset(
            root_dir=dataset_config.get('root_dir', 'datasets'),
            label_mapper=label_mapper,
            transform=self.get_val_transforms(),
            mode="val",
            use_cropped_only=True
        )
        
        # Test dataset (Herlev)
        test_dataset = HerlevDataset(
            root_dir=dataset_config.get('root_dir', 'datasets'),
            label_mapper=label_mapper,
            transform=self.get_val_transforms(),
            mode="test",
            use_cropped_only=False  # Use all Herlev images
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples (Herlev): {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, num_classes
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """Create data loaders."""
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 32)
        num_workers = training_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def calculate_class_weights(self, dataset):
        """Calculate class weights for imbalanced datasets."""
        labels = [sample[1] for sample in dataset.samples]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # Calculate weights inversely proportional to class frequencies
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        return class_weights
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _, _) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def evaluate_model(self, model, test_loader, dataset_name="Test"):
        """Evaluate model on test dataset with comprehensive metrics."""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels, _, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # AUC for binary classification
        auc = None
        if len(np.unique(all_labels)) == 2:
            try:
                auc = roc_auc_score(all_labels, all_probabilities[:, 1])
            except:
                auc = None
        
        # Classification report
        class_names = self.get_class_names()
        report = classification_report(all_labels, all_predictions, 
                                    target_names=class_names, 
                                    output_dict=True, 
                                    zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'dataset_name': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist()
        }
        
        logger.info(f"{dataset_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        if auc is not None:
            logger.info(f"  AUC: {auc:.4f}")
        
        return results
    
    def get_class_names(self):
        """Get class names based on classification mode."""
        classification_mode = self.config.get('dataset', {}).get('classification_mode', 'binary')
        
        if classification_mode == "binary":
            return ["NORMAL", "ABNORMAL"]
        else:
            label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
            return label_mapper.get_class_names(DatasetType.SIPAKMED)
    
    def plot_training_curves(self, history):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(history['train_acc'], label='Train Acc')
        axes[0, 1].plot(history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Class distribution (if available)
        if 'class_distribution' in history:
            class_names = self.get_class_names()
            class_counts = history['class_distribution']
            axes[1, 1].bar(class_names, class_counts)
            axes[1, 1].set_title('Class Distribution')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {self.output_dir / 'plots' / 'training_curves.png'}")
    
    def plot_confusion_matrix(self, cm, dataset_name):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        class_names = self.get_class_names()
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'plots' / f'confusion_matrix_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix for {dataset_name} saved")
    
    def train(self):
        """Main training function."""
        logger.info("🚀 Starting EfficientNet-B0 Baseline Training")
        logger.info("=" * 60)
        
        # Run dataset guard first
        logger.info("🛡️ Running Dataset Guard Validation")
        guard_results = run_dataset_guard(self.config)
        
        if guard_results['action'] == 'ABORT':
            logger.error("❌ Dataset validation failed - Training aborted")
            return False
        
        if guard_results['action'] == 'PROCEED_WITH_WARNING':
            logger.warning("⚠️ Dataset validation warnings - Proceeding with caution")
        
        logger.info("✅ Dataset validation passed - Starting training")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset, num_classes = self.create_datasets()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Create model
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.3)
        model = self.create_model(num_classes, dropout_rate)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_dataset)
        
        # Setup training
        training_config = self.config.get('training', {})
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        num_epochs = training_config.get('num_epochs', 25)
        early_stopping_patience = training_config.get('early_stopping_patience', 5)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  LR: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.output_dir / 'models' / 'best_model.pth')
                epochs_without_improvement = 0
                logger.info(f'  New best model saved (Val Acc: {val_acc:.2f}%)')
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(self.output_dir / 'models' / 'best_model.pth'))
        
        # Final validation evaluation
        logger.info("📊 Final Validation Evaluation")
        val_results = self.evaluate_model(model, val_loader, "Validation")
        
        # Cross-dataset evaluation on Herlev
        logger.info("🎯 Cross-Dataset Evaluation on Herlev")
        test_results = self.evaluate_model(model, test_loader, "Herlev")
        
        # Save results
        self.results = {
            'training_history': history,
            'validation_results': val_results,
            'test_results': test_results,
            'best_val_accuracy': best_val_acc,
            'total_epochs': epoch + 1,
            'model_config': {
                'architecture': 'efficientnet_b0',
                'num_classes': num_classes,
                'dropout_rate': dropout_rate,
                'pretrained': True
            },
            'training_config': training_config,
            'dataset_config': self.config.get('dataset', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to JSON
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Plot training curves
        self.plot_training_curves(history)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(val_results['confusion_matrix'], "Validation")
        self.plot_confusion_matrix(test_results['confusion_matrix'], "Herlev")
        
        # Print final summary
        logger.info("🎉 Training Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logger.info(f"Herlev Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Herlev F1-Score: {test_results['f1_score']:.4f}")
        if test_results['auc'] is not None:
            logger.info(f"Herlev AUC: {test_results['auc']:.4f}")
        
        # Generalization analysis
        val_acc = val_results['accuracy']
        test_acc = test_results['accuracy']
        generalization_gap = val_acc - test_acc
        
        logger.info(f"Generalization Gap: {generalization_gap:.4f}")
        
        if generalization_gap < 0.1:
            logger.info("✅ Excellent generalization (gap < 10%)")
        elif generalization_gap < 0.2:
            logger.info("⚠️ Good generalization (gap < 20%)")
        else:
            logger.info("❌ Poor generalization (gap > 20%)")
        
        return True


def create_default_config():
    """Create default configuration for EfficientNet-B0 baseline."""
    return {
        'output_dir': 'outputs/efficientnet_baseline',
        'dataset': {
            'root_dir': 'datasets',
            'train_dataset': 'sipakmed',
            'classification_mode': 'binary',  # Change to 'multiclass' for 5-class
            'use_cropped_only': True,
            'run_validation': True,
            'validation_strict_mode': True
        },
        'model': {
            'architecture': 'efficientnet_b0',
            'pretrained': True,
            'dropout_rate': 0.3
        },
        'training': {
            'num_epochs': 25,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'num_workers': 4
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='EfficientNet-B0 Baseline Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], 
                       default='binary', help='Classification mode')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config_manager = load_config(args.config)
        config = config_manager.get_config()
    else:
        config = create_default_config()
    
    # Override with command line arguments
    if args.mode:
        config['dataset']['classification_mode'] = args.mode
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.dropout:
        config['model']['dropout_rate'] = args.dropout
    
    # Print configuration
    logger.info("EfficientNet-B0 Baseline Configuration")
    logger.info("=" * 50)
    logger.info(f"Classification Mode: {config['dataset']['classification_mode']}")
    logger.info(f"Training Dataset: SIPaKMeD (CROPPED-only)")
    logger.info(f"Test Dataset: Herlev")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Dropout Rate: {config['model']['dropout_rate']}")
    logger.info("=" * 50)
    
    # Create and run trainer
    trainer = EfficientNetB0Baseline(config)
    success = trainer.train()
    
    if success:
        logger.info("🎉 Baseline training completed successfully!")
        logger.info(f"Results saved to: {trainer.output_dir}")
    else:
        logger.error("❌ Baseline training failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
