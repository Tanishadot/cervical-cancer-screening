"""
Optimized EfficientNet-B0 Training Script
Target: 85-95% validation accuracy with strong generalization to Herlev.
Advanced techniques: label smoothing, progressive unfreezing, enhanced augmentation.
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
from PIL import Image, ImageFilter
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
        logging.FileHandler('efficientnet_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GaussianBlur:
    """Custom GaussianBlur transform for medical images."""
    def __init__(self, p=0.2, kernel_size=3, sigma=(0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        if np.random.random() < self.p:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class EfficientNetB0Optimized:
    """Optimized EfficientNet-B0 trainer with advanced techniques."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'outputs/efficientnet_optimized'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        logger.info(f"Optimized EfficientNet-B0 initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def get_train_transforms(self):
        """Enhanced training transforms with medical-appropriate augmentation."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            GaussianBlur(p=0.2, kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_val_transforms(self):
        """Clean validation transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def create_model(self, num_classes, dropout_rate=0.4):
        """Create EfficientNet-B0 with enhanced classifier."""
        import torchvision.models as models
        
        # Load pretrained EfficientNet-B0
        model = models.efficientnet_b0(pretrained=True)
        
        # Freeze backbone initially
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Enhanced classifier with higher dropout
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        model = model.to(self.device)
        
        logger.info(f"Created Enhanced EfficientNet-B0 with {num_classes} output classes")
        logger.info(f"Dropout rate: {dropout_rate}")
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
        batch_size = training_config.get('batch_size', 16)
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
        """Calculate class weights with smoothing for imbalanced datasets."""
        labels = [sample[1] for sample in dataset.samples]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # Calculate weights inversely proportional to class frequencies
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        return class_weights
    
    def create_label_smoothing_loss(self, class_weights, smoothing=0.1):
        """Create CrossEntropyLoss with label smoothing."""
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, num_classes, weight=None, smoothing=0.1):
                super().__init__()
                self.num_classes = num_classes
                self.weight = weight
                self.smoothing = smoothing
                self.confidence = 1.0 - smoothing
                
            def forward(self, pred, target):
                pred_log_softmax = torch.log_softmax(pred, dim=-1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(pred_log_softmax)
                    true_dist.fill_(self.smoothing / (self.num_classes - 1))
                    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
                
                return torch.mean(torch.sum(-true_dist * pred_log_softmax, dim=-1))
        
        num_classes = len(class_weights)
        return LabelSmoothingLoss(num_classes, class_weights, smoothing)
    
    def unfreeze_backbone(self, model, unfreeze_from_layer=-1):
        """Progressively unfreeze backbone layers."""
        logger.info(f"Unfreezing backbone from layer {unfreeze_from_layer}")
        
        # Unfreeze all backbone parameters
        for param in model.features.parameters():
            param.requires_grad = True
        
        # Optionally freeze early layers (keep only later layers trainable)
        if unfreeze_from_layer > 0:
            for i, layer in enumerate(model.features.children()):
                if i < unfreeze_from_layer:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters after unfreeze: {trainable_params:,} ({trainable_params/total_params:.1%})")
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch with enhanced logging."""
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        """Validate for one epoch with detailed metrics."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate additional metrics
        from sklearn.metrics import f1_score, recall_score
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return epoch_loss, epoch_acc, f1, recall, all_predictions, all_labels, all_probabilities
    
    def evaluate_model(self, model, test_loader, dataset_name="Test"):
        """Comprehensive model evaluation."""
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
        
        # Calculate comprehensive metrics
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
        """Enhanced training curves with multiple metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-score curves
        if 'val_f1' in history:
            axes[0, 2].plot(history['val_f1'], label='Val F1-Score', linewidth=2, color='green')
            axes[0, 2].set_title('Validation F1-Score')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('F1-Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate curve
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Recall curves
        if 'val_recall' in history:
            axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2, color='red')
            axes[1, 1].set_title('Validation Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Target accuracy line
        if 'val_acc' in history:
            axes[1, 2].axhline(y=85, color='g', linestyle='--', alpha=0.7, label='Target (85%)')
            axes[1, 2].axhline(y=95, color='r', linestyle='--', alpha=0.7, label='Target (95%)')
            axes[1, 2].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
            axes[1, 2].set_title('Validation Accuracy vs Targets')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Accuracy (%)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_curves_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced training curves saved")
    
    def plot_confusion_matrix(self, cm, dataset_name):
        """Enhanced confusion matrix with percentages."""
        plt.figure(figsize=(10, 8))
        class_names = self.get_class_names()
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'Confusion Matrix (Counts) - {dataset_name}')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Percentages
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title(f'Confusion Matrix (Percentages) - {dataset_name}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / f'confusion_matrix_{dataset_name.lower()}_optimized.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced confusion matrix for {dataset_name} saved")
    
    def train(self):
        """Main optimized training function."""
        logger.info("🚀 Starting Optimized EfficientNet-B0 Training")
        logger.info("Target: 85-95% validation accuracy with strong generalization")
        logger.info("=" * 80)
        
        # Run dataset guard first
        logger.info("🛡️ Running Dataset Guard Validation")
        guard_results = run_dataset_guard(self.config)
        
        if guard_results['action'] == 'ABORT':
            logger.error("❌ Dataset validation failed - Training aborted")
            return False
        
        if guard_results['action'] == 'PROCEED_WITH_WARNING':
            logger.warning("⚠️ Dataset validation warnings - Proceeding with caution")
        
        logger.info("✅ Dataset validation passed - Starting optimized training")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset, num_classes = self.create_datasets()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Create model with higher dropout
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.4)
        model = self.create_model(num_classes, dropout_rate)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_dataset)
        
        # Create label smoothing loss
        label_smoothing = self.config.get('training', {}).get('label_smoothing', 0.1)
        criterion = self.create_label_smoothing_loss(class_weights, label_smoothing)
        
        logger.info(f"Using label smoothing: {label_smoothing}")
        
        # Setup optimized training
        training_config = self.config.get('training', {})
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        num_epochs = training_config.get('num_epochs', 30)
        early_stopping_patience = training_config.get('early_stopping_patience', 5)
        unfreeze_epoch = training_config.get('unfreeze_epoch', 5)
        
        # Enhanced training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1': [], 'val_recall': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        
        logger.info(f"Starting optimized training for {num_epochs} epochs")
        logger.info(f"Backbone unfreeze scheduled for epoch {unfreeze_epoch}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Progressive unfreezing
            if epoch == unfreeze_epoch:
                logger.info(f"🔓 UNFREEZING BACKBONE at epoch {epoch+1}")
                self.unfreeze_backbone(model)
                
                # Reduce learning rate after unfreezing
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                logger.info(f"Learning rate reduced after unfreezing")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_recall, _, _, _ = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_recall'].append(val_recall)
            history['lr'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}')
            logger.info(f'  LR: {current_lr:.6f}')
            
            # Target achievement check
            if val_acc >= 85:
                logger.info(f"🎯 ACHIEVED TARGET: {val_acc:.2f}% accuracy (≥85%)")
            if val_acc >= 95:
                logger.info(f"🏆 ACHIEVED EXCELLENCE: {val_acc:.2f}% accuracy (≥95%)")
            
            # Save best model (consider both accuracy and F1)
            current_score = val_acc + val_f1  # Combined metric
            best_score = best_val_acc + best_val_f1
            
            if current_score > best_score:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                torch.save(model.state_dict(), self.output_dir / 'models' / 'best_model_optimized.pth')
                epochs_without_improvement = 0
                logger.info(f'  New best model saved (Val Acc: {val_acc:.2f}%, F1: {val_f1:.4f})')
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(self.output_dir / 'models' / 'best_model_optimized.pth'))
        
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
            'best_val_f1': best_val_f1,
            'total_epochs': epoch + 1,
            'model_config': {
                'architecture': 'efficientnet_b0',
                'num_classes': num_classes,
                'dropout_rate': dropout_rate,
                'pretrained': True,
                'label_smoothing': label_smoothing,
                'unfreeze_epoch': unfreeze_epoch
            },
            'training_config': training_config,
            'dataset_config': self.config.get('dataset', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to JSON
        with open(self.output_dir / 'results_optimized.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Plot enhanced training curves
        self.plot_training_curves(history)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(val_results['confusion_matrix'], "Validation")
        self.plot_confusion_matrix(test_results['confusion_matrix'], "Herlev")
        
        # Print final summary
        logger.info("🎉 Optimized Training Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logger.info(f"Best Validation F1-Score: {best_val_f1:.4f}")
        logger.info(f"Herlev Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Herlev F1-Score: {test_results['f1_score']:.4f}")
        if test_results['auc'] is not None:
            logger.info(f"Herlev AUC: {test_results['auc']:.4f}")
        
        # Target achievement analysis
        logger.info("🎯 TARGET ACHIEVEMENT ANALYSIS:")
        if best_val_acc >= 95:
            logger.info("✅ EXCELLENCE: Validation accuracy ≥95%")
        elif best_val_acc >= 85:
            logger.info("✅ TARGET ACHIEVED: Validation accuracy ≥85%")
        else:
            logger.info(f"⚠️ TARGET MISSED: Validation accuracy {best_val_acc:.2f}% <85%")
        
        # Generalization analysis
        val_acc = val_results['accuracy']
        test_acc = test_results['accuracy']
        generalization_gap = val_acc - test_acc
        
        logger.info(f"📊 GENERALIZATION ANALYSIS:")
        logger.info(f"Generalization Gap: {generalization_gap:.4f}")
        
        if generalization_gap < 0.1:
            logger.info("✅ EXCELLENT generalization (gap < 10%)")
        elif generalization_gap < 0.2:
            logger.info("✅ GOOD generalization (gap < 20%)")
        else:
            logger.info("⚠️ POOR generalization (gap > 20%)")
        
        return True


def create_optimized_config():
    """Create optimized configuration for enhanced performance."""
    return {
        'output_dir': 'outputs/efficientnet_optimized',
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
            'dropout_rate': 0.4  # Higher dropout for better regularization
        },
        'training': {
            'num_epochs': 30,
            'batch_size': 16,  # Smaller batch for better generalization
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'label_smoothing': 0.1,  # Label smoothing for better generalization
            'unfreeze_epoch': 5,  # Progressive unfreezing
            'num_workers': 4
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Optimized EfficientNet-B0 Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], 
                       default='binary', help='Classification mode')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--label-smoothing', type=float, help='Label smoothing factor')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config_manager = load_config(args.config)
        config = config_manager.get_config()
    else:
        config = create_optimized_config()
    
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
    if args.label_smoothing:
        config['training']['label_smoothing'] = args.label_smoothing
    
    # Print configuration
    logger.info("Optimized EfficientNet-B0 Configuration")
    logger.info("=" * 60)
    logger.info(f"Classification Mode: {config['dataset']['classification_mode']}")
    logger.info(f"Training Dataset: SIPaKMeD (CROPPED-only)")
    logger.info(f"Test Dataset: Herlev")
    logger.info(f"Target Accuracy: 85-95%")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Dropout Rate: {config['model']['dropout_rate']}")
    logger.info(f"Label Smoothing: {config['training']['label_smoothing']}")
    logger.info(f"Unfreeze Epoch: {config['training']['unfreeze_epoch']}")
    logger.info("=" * 60)
    
    # Create and run trainer
    trainer = EfficientNetB0Optimized(config)
    success = trainer.train()
    
    if success:
        logger.info("🎉 Optimized baseline training completed successfully!")
        logger.info(f"Results saved to: {trainer.output_dir}")
    else:
        logger.error("❌ Optimized baseline training failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
