"""
Training pipeline for cervical cancer classification with early stopping, class weighting, and metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import os
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Validation loss
            model: Model to save weights from
            
        Returns:
            Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best weights")
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                return True
        
        return False


class MetricsCalculator:
    """Metrics calculation utility."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # AUC metrics (if probabilities provided)
        if y_prob is not None:
            if self.num_classes == 2:
                # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                metrics['auc_roc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_roc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
        
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)


class Trainer:
    """
    Training pipeline with comprehensive metrics and early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs",
        experiment_name: str = "cervical_cancer"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_classes: Number of classes
            class_names: Class names
            device: Device to train on
            output_dir: Output directory
            experiment_name: Experiment name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(num_classes, class_names)
        
        # Training history
        self.history = defaultdict(list)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    def calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = torch.zeros(self.num_classes)
        
        # Count samples in training set
        for _, labels, _, _ in self.train_loader:
            for label in labels:
                class_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights.to(self.device)
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            use_amp: Whether to use mixed precision
            
        Returns:
            Tuple of (loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels, _, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds, all_probs)
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        return avg_loss, metrics
    
    def validate_epoch(self, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            criterion: Loss function
            
        Returns:
            Tuple of (loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels, _, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Collect predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds, all_probs)
        
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, metrics
    
    def train(
        self,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        use_class_weights: bool = True,
        use_amp: bool = True,
        unfreeze_epoch: int = 10
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            use_class_weights: Whether to use class weights
            use_amp: Whether to use mixed precision
            unfreeze_epoch: Epoch to unfreeze backbone
            
        Returns:
            Training history
        """
        # Setup optimizer and criterion
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if use_class_weights:
            class_weights = self.calculate_class_weights()
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Unfreeze backbone at specified epoch
            if epoch == unfreeze_epoch and hasattr(self.model, 'unfreeze_backbone'):
                self.model.unfreeze_backbone()
                # Update optimizer for new parameters
                optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate * 0.1, weight_decay=weight_decay)
            
            # Training
            train_loss, train_metrics = self.train_epoch(optimizer, criterion, scheduler, use_amp)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1_macro'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train F1: {train_metrics['f1_macro']:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")
            
            # Save best model based on F1-score
            current_val_f1 = val_metrics['f1_macro']
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model with F1-based naming
                model_path = os.path.join(self.output_dir, 'models', 'best_swin_model.pth' if 'swin' in self.experiment_name else 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': current_val_f1,
                    'val_loss': val_loss
                }, model_path)
                logger.info(f"New best model saved at epoch {epoch + 1} with F1: {current_val_f1:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Check early stopping based on F1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load best model based on experiment type
        if 'swin' in self.experiment_name:
            best_model_path = os.path.join(self.output_dir, 'models', 'best_swin_model.pth')
        else:
            best_model_path = os.path.join(self.output_dir, 'models', 'best_model.pth')
        
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with F1: {checkpoint['val_f1']:.4f}")
        else:
            logger.warning(f"Best model not found at {best_model_path}")
        
        logger.info(f"Training completed. Best epoch: {best_epoch + 1}, Best val F1: {best_val_f1:.4f}")
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        return dict(self.history)
    
    def test(self) -> Dict[str, float]:
        """
        Test the model on test set.
        
        Returns:
            Test metrics
        """
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for images, labels, _, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds, all_probs)
        
        # Generate confusion matrix
        cm = self.metrics_calculator.get_confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
        
        # Save test results
        self.save_test_results(metrics, cm)
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
        if 'auc_roc' in metrics:
            logger.info(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def save_history(self):
        """Save training history."""
        history_path = os.path.join(self.output_dir, 'logs', f'{self.experiment_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        axes[1, 0].plot(self.history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1')
        axes[1, 0].set_title('F1 Score Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history:
            axes[1, 1].plot(self.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'{self.experiment_name}_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {self.output_dir}/plots")
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'{self.experiment_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_test_results(self, metrics: Dict, cm: np.ndarray):
        """Save test results."""
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names
        }
        
        results_path = os.path.join(self.output_dir, 'logs', f'{self.experiment_name}_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {results_path}")


if __name__ == "__main__":
    # Example usage
    from models.model_factory import create_model
    
    # Create dummy data loaders
    # (In practice, these would be created from actual datasets)
    
    # Create model
    model = create_model('efficientnet_b0', num_classes=2)
    
    print("Trainer module ready for use")
