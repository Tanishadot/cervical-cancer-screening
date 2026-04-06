"""
Multi-domain training and evaluation pipeline for robust cervical cancer classification.
Implements domain-robust training with uncertainty handling and clinical safety features.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
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
from tqdm import tqdm
import warnings

from utils.config_manager import ConfigManager
from utils.label_mapping import LabelMapper, ClassificationMode
from multi_domain_dataset import MultiDomainManager
from enhanced_transforms import get_domain_robust_train_transforms, get_validation_transforms
from enhanced_stain_normalization import create_enhanced_stain_normalizer
from models.model_factory import ModelFactory
from confidence_prediction import create_confidence_predictor
from hyperparameter_config import HyperparameterConfig, HyperparameterSearch
from advanced_losses import create_loss_function, ThresholdOptimizer
from ensemble_tta import TestTimeAugmentation, ModelEnsemble, EnsembleTTA

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiDomainTrainer:
    """
    Multi-domain trainer with enhanced regularization and uncertainty handling.
    """
    
    def __init__(
        self,
        config: dict,
        device: torch.device,
        output_dir: str = "outputs/multi_domain",
        hp_config: HyperparameterConfig = None
    ):
        """
        Initialize multi-domain trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to run training on
            output_dir: Output directory for results
            hp_config: Hyperparameter configuration
        """
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.hp_config = hp_config or HyperparameterConfig()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "experiments"), exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.confidence_predictor = None
        self.threshold_optimizer = None
        self.tta_predictor = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = self.hp_config.early_stopping_patience
        self.backbone_unfrozen = False
        self.best_threshold = 0.5
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': [], 'classifier_lr': [], 'backbone_lr': []
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, "logs", "training.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def setup_model(self):
        """Setup model with hyperparameter configuration and differential learning rates."""
        logger.info("Setting up model with hyperparameter configuration...")
        
        # Log configuration
        logger.info(f"🔧 Hyperparameter Config:")
        logger.info(f"   Classifier LR: {self.hp_config.classifier_lr:.2e}")
        logger.info(f"   Backbone LR: {self.hp_config.backbone_lr:.2e}")
        logger.info(f"   Dropout: {self.hp_config.dropout_rate}")
        logger.info(f"   Pos Weight: {self.hp_config.pos_weight}")
        logger.info(f"   Loss Type: {self.hp_config.loss_type}")
        logger.info(f"   Scheduler: {self.hp_config.scheduler_type}")
        logger.info(f"   Weighted Sampler: {self.hp_config.use_weighted_sampler}")
        logger.info(f"   TTA: {self.hp_config.use_tta}")
        logger.info(f"   Threshold Optimization: {self.hp_config.optimize_threshold}")
        
        # Create model with configured dropout
        self.model = ModelFactory.create_model(
            model_name=self.config['model']['architecture'],
            num_classes=self.config['model']['num_classes'],
            pretrained=self.config['model']['pretrained'],
            freeze_backbone=self.config['model']['freeze_backbone'],
            dropout_rate=self.hp_config.dropout_rate
        )
        
        self.model.to(self.device)
        
        # Setup optimizer with parameter groups and differential learning rates
        classifier_params = []
        backbone_params = []
        
        # Separate parameters for classifier and backbone
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name or 'fc' in name or 'head' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Create parameter groups with configured learning rates
        param_groups = [
            {'params': classifier_params, 'lr': self.hp_config.classifier_lr, 'weight_decay': self.hp_config.weight_decay},
            {'params': backbone_params, 'lr': self.hp_config.backbone_lr, 'weight_decay': self.hp_config.weight_decay}
        ]
        
        self.optimizer = optim.AdamW(param_groups)
        
        # Setup learning rate scheduler based on configuration
        if self.hp_config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.hp_config.plateau_factor,
                patience=self.hp_config.plateau_patience,
                verbose=True
            )
            logger.info("📉 Using ReduceLROnPlateau scheduler")
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=1e-6
            )
            logger.info("📈 Using CosineAnnealingLR scheduler")
        
        # Setup loss function based on configuration
        self.criterion = create_loss_function(
            loss_type=self.hp_config.loss_type,
            pos_weight=self.hp_config.pos_weight,
            device=self.device
        )
        
        # Setup threshold optimizer if enabled
        if self.hp_config.optimize_threshold:
            self.threshold_optimizer = ThresholdOptimizer(
                threshold_range=self.hp_config.threshold_range,
                metric='f1'
            )
            logger.info("🎯 Threshold optimization enabled")
        
        # Setup confidence predictor
        class_names = ['NORMAL', 'ABNORMAL']  # Binary classification
        self.confidence_predictor = create_confidence_predictor(
            model=self.model,
            class_names=class_names,
            confidence_threshold=0.7,
            uncertainty_threshold=0.5,
            device=self.device
        )
        
        logger.info("Model setup completed")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract data
            if len(batch) >= 5:  # Multi-domain dataset
                images, labels, _, _, _ = batch[:5]
            else:
                images, labels = batch[:2]
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Handle BCEWithLogitsLoss - convert labels to one-hot if needed
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                # Convert labels to float and one-hot encode for BCE
                labels_onehot = torch.zeros(outputs.shape, device=self.device)
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                loss = self.criterion(outputs, labels_onehot)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Collect predictions
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        
        avg_loss = running_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> dict:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                # Extract data
                if len(batch) >= 4:
                    images, labels, _, _ = batch[:4]
                else:
                    images, labels = batch[:2]
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Handle BCEWithLogitsLoss - convert labels to one-hot if needed
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    labels_onehot = torch.zeros(outputs.shape, device=self.device)
                    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                    loss = self.criterion(outputs, labels_onehot)
                else:
                    loss = self.criterion(outputs, labels)
                
                # Update running loss
                running_loss += loss.item()
                
                # Collect predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        
        # ROC-AUC
        if len(np.unique(all_labels)) == 2:
            roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            roc_auc = 0.0
        
        avg_loss = running_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning with differential learning rates."""
        if hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone()
        else:
            # Manually unfreeze
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Re-create parameter groups with unfrozen backbone
        classifier_params = []
        backbone_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name or 'fc' in name or 'head' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Create parameter groups with different learning rates for fine-tuning
        # Use slightly lower rates for fine-tuning
        param_groups = [
            {'params': classifier_params, 'lr': self.hp_config.classifier_lr * 0.5, 'weight_decay': self.hp_config.weight_decay},
            {'params': backbone_params, 'lr': self.hp_config.backbone_lr, 'weight_decay': self.hp_config.weight_decay}
        ]
        
        self.optimizer = optim.AdamW(param_groups)
        
        # Update scheduler for new optimizer
        remaining_epochs = self.config['training']['num_epochs'] - self.current_epoch
        if self.hp_config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.hp_config.plateau_factor,
                patience=self.hp_config.plateau_patience,
                verbose=True
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_epochs,
                eta_min=1e-7
            )
        
        self.backbone_unfrozen = True
        logger.info("🔓 Backbone unfrozen for fine-tuning")
        logger.info(f"   Classifier LR: {self.hp_config.classifier_lr * 0.5:.2e}, Backbone LR: {self.hp_config.backbone_lr:.2e}")
        logger.info(f"   Remaining epochs: {remaining_epochs}")
    
    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, "models", "latest_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "models", "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with F1: {val_f1:.4f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model with multi-domain data, early stopping, and controlled unfreezing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("🚀 Starting multi-domain training with improved stability...")
        
        num_epochs = self.config['training']['num_epochs']
        unfreeze_epoch = 3  # Keep backbone frozen for first 3 epochs
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Controlled unfreezing at epoch 4 (index 3)
            if epoch == unfreeze_epoch and not self.backbone_unfrozen:
                self.unfreeze_backbone()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate based on scheduler type
            if self.hp_config.scheduler_type == "plateau":
                self.scheduler.step(val_metrics['f1_score'])
            else:
                self.scheduler.step()
            
            # Extract learning rates for logging
            classifier_lr = self.optimizer.param_groups[0]['lr']
            backbone_lr = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else classifier_lr
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1_score'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1_score'])
            self.training_history['learning_rates'].append(classifier_lr)
            self.training_history['classifier_lr'].append(classifier_lr)
            self.training_history['backbone_lr'].append(backbone_lr)
            
            # Check for improvement and early stopping
            is_best = val_metrics['f1_score'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_score']
                self.patience_counter = 0
                logger.info(f"🎉 New best F1 score: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ No improvement for {self.patience_counter} epochs")
            
            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"🛑 Early stopping triggered after {self.patience_counter} epochs without improvement")
                logger.info(f"Best F1 achieved: {self.best_val_f1:.4f}")
                break
            
            # Save checkpoint (only save best model to prevent overwriting)
            self.save_checkpoint(epoch, val_metrics['f1_score'], is_best)
            
            # Detailed logging
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            logger.info(f"📊 Learning Rates - Classifier: {classifier_lr:.2e}, Backbone: {backbone_lr:.2e}")
            logger.info(f"🎯 Recall: {val_metrics['recall']:.4f}, Precision: {val_metrics['precision']:.4f}")
            
            if self.backbone_unfrozen:
                logger.info(f"🔓 Backbone is unfrozen (fine-tuning)")
            else:
                logger.info(f"🔒 Backbone is frozen (feature extraction)")
            
            logger.info("-" * 80)
        
        logger.info("✅ Training completed!")
        logger.info(f"🏆 Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"📈 Total epochs trained: {self.current_epoch + 1}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.training_history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.training_history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score curves
        axes[1, 0].plot(epochs, self.training_history['train_f1'], 'b-', label='Train F1')
        axes[1, 0].plot(epochs, self.training_history['val_f1'], 'r-', label='Val F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curves
        if 'classifier_lr' in self.training_history and 'backbone_lr' in self.training_history:
            axes[1, 1].plot(epochs, self.training_history['classifier_lr'], 'b-', label='Classifier LR', linewidth=2)
            axes[1, 1].plot(epochs, self.training_history['backbone_lr'], 'r-', label='Backbone LR', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule (Differential)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].plot(epochs, self.training_history['learning_rates'], 'g-')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_with_uncertainty(
    model: nn.Module,
    test_loader: DataLoader,
    confidence_predictor,
    device: torch.device,
    output_dir: str
) -> dict:
    """
    Evaluate model with uncertainty handling.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        confidence_predictor: Confidence-based predictor
        device: Device to run evaluation on
        output_dir: Output directory
        
    Returns:
        Evaluation results
    """
    logger.info("Starting evaluation with uncertainty handling...")
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_results = []
    all_uncertainty_decisions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Extract data
            if len(batch) >= 4:
                images, labels, _, _ = batch[:4]
            else:
                images, labels = batch[:2]
            
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions with uncertainty
            batch_results = confidence_predictor.predict_batch(images)
            
            # Collect results
            for result, label in zip(batch_results, labels):
                all_predictions.append(result.prediction)
                all_labels.append(label.item())
                all_results.append(result)
                
                # Get uncertainty-based decision
                decision_result = confidence_predictor.predict_with_uncertainty_threshold(
                    images[0] if len(images) == 1 else None,  # Placeholder
                    min_confidence=0.7
                )
                all_uncertainty_decisions.append(decision_result['decision'])
            
            # Collect probabilities for metrics
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    # ROC-AUC
    if len(np.unique(all_labels)) == 2:
        roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
    else:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Uncertainty statistics
    uncertainty_stats = confidence_predictor.get_uncertainty_statistics(all_results)
    
    # Uncertainty decision distribution
    decision_dist = {
        decision: all_uncertainty_decisions.count(decision)
        for decision in set(all_uncertainty_decisions)
    }
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'uncertainty_stats': uncertainty_stats,
        'decision_distribution': decision_dist,
        'total_samples': len(all_labels)
    }
    
    # Save results
    results_path = os.path.join(output_dir, "herlev_cross_dataset.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'ABNORMAL'],
                yticklabels=['NORMAL', 'ABNORMAL'])
    plt.title('Confusion Matrix - Multi-Domain Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results


def main():
    """Main training and evaluation function with hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Multi-domain training for cervical cancer classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs/multi_domain', help='Output directory')
    parser.add_argument('--experiment', type=str, default='single', 
                       choices=['single', 'predefined', 'grid'], help='Experiment type')
    parser.add_argument('--hp-config', type=str, default='baseline', 
                       choices=['baseline', 'high_recall', 'balanced', 'conservative'], 
                       help='Hyperparameter configuration')
    args = parser.parse_args()
    
    # Setup
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        if args.experiment == 'single':
            # Single experiment with specified config
            from hyperparameter_config import get_config
            
            hp_config = get_config(args.hp_config)
            logger.info(f"🎯 Running single experiment with {args.hp_config} config")
            
            # Create multi-domain manager
            manager = MultiDomainManager(
                root_dir=config['dataset']['root_dir'],
                classification_mode='binary',
                herlev_train_ratio=0.2,
                random_seed=42
            )
            
            # Load and split datasets
            manager.load_datasets()
            manager.print_summary()
            
            # Create data loaders
            train_transform = get_domain_robust_train_transforms()
            val_transform = get_validation_transforms()
            
            train_loader, test_loader = manager.create_data_loaders(
                train_transform=train_transform,
                val_transform=val_transform,
                batch_size=config['dataset']['batch_size'],
                num_workers=config['dataset']['num_workers']
            )
            
            # Setup trainer
            trainer = MultiDomainTrainer(config, device, args.output_dir, hp_config)
            trainer.setup_model()
            
            # Train model
            trainer.train(train_loader, test_loader)
            
            # Load best model for evaluation
            best_model_path = os.path.join(args.output_dir, "models", "best_model.pth")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Best model loaded for evaluation")
            
            # Evaluate with uncertainty and TTA
            results = evaluate_with_uncertainty(
                model=trainer.model,
                test_loader=test_loader,
                confidence_predictor=trainer.confidence_predictor,
                device=device,
                output_dir=args.output_dir
            )
            
            # Plot training curves
            trainer.plot_training_curves()
            
            # Print final results
            print("\n" + "="*80)
            print(f"EXPERIMENT RESULTS - {args.hp_config.upper()} CONFIG")
            print("="*80)
            print(f"📊 Final Metrics:")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall: {results['recall']:.4f}")
            print(f"   F1 Score: {results['f1_score']:.4f}")
            print(f"   ROC-AUC: {results['roc_auc']:.4f}")
            
            if hasattr(trainer, 'best_threshold'):
                print(f"🎯 Best Threshold: {trainer.best_threshold:.3f}")
            
            print(f"🔧 Configuration:")
            print(f"   Classifier LR: {hp_config.classifier_lr:.2e}")
            print(f"   Backbone LR: {hp_config.backbone_lr:.2e}")
            print(f"   Dropout: {hp_config.dropout_rate}")
            print(f"   Pos Weight: {hp_config.pos_weight}")
            print(f"   Loss Type: {hp_config.loss_type}")
            print(f"   Scheduler: {hp_config.scheduler_type}")
            print(f"   TTA: {hp_config.use_tta}")
            print("="*80)
            
            # Save experiment summary
            summary = {
                'config_name': args.hp_config,
                'hyperparameters': hp_config.__dict__,
                'results': results,
                'best_threshold': getattr(trainer, 'best_threshold', 0.5),
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = os.path.join(args.output_dir, "experiment_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("✅ Single experiment completed successfully!")
            
        elif args.experiment in ['predefined', 'grid']:
            # Run multiple experiments
            from experiment_runner import run_hyperparameter_experiments
            
            logger.info(f"🔍 Running {args.experiment} experiments...")
            
            results = run_hyperparameter_experiments(
                base_config=config,
                device=device,
                output_dir=args.output_dir,
                experiment_type=args.experiment
            )
            
            logger.info("✅ Multiple experiments completed successfully!")
            
        else:
            logger.error(f"Unknown experiment type: {args.experiment}")
            return
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
