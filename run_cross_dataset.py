#!/usr/bin/env python3
"""
Cross-Dataset Training - Simplified Version
Trains Swin Transformer on SIPaKMeD, evaluates on Herlev
"""

import os
import sys
import json
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from datasets.dataset import DatasetManager
    from preprocessing.transforms import get_train_transforms, get_val_transforms
    from models.model_factory import CervicalCancerModel
    from training.trainer import Trainer
    from utils.config_manager import ConfigManager
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from datasets.dataset import TransformDataset, create_dataloader
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check your Python environment and dependencies")
    sys.exit(1)

class AlbumentationsTransform:
    """Wrapper for albumentations transforms."""
    def __init__(self, albumentations_transform):
        self.transform = albumentations_transform
    
    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        return self.transform(image=image)['image']

def create_data_loader(dataset, transform, batch_size=16, shuffle=False):
    """Create data loader with transforms."""
    albumentations_transform = AlbumentationsTransform(transform)
    transformed_dataset = TransformDataset(dataset, albumentations_transform)
    
    return create_dataloader(
        dataset=transformed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

def main():
    """Main cross-dataset training."""
    print("Cross-Dataset Training: SIPaKMeD → Herlev")
    print("=" * 50)
    
    try:
        # Load config
        config_manager = ConfigManager()
        config = config_manager.get_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Device: {device}")
        
        # Create dataset manager
        dataset_manager = DatasetManager(
            root_dir=config.get('dataset', {}).get('root_dir', 'datasets'),
            classification_mode="binary",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Load datasets
        print("Loading datasets...")
        dataset_manager.load_datasets()
        
        # Create splits ONLY for SIPaKMeD
        print("Creating splits for SIPaKMeD...")
        dataset_manager.create_splits()
        
        # Get SIPaKMeD splits for training
        print("Training on: SIPaKMeD")
        sipakmed_train = dataset_manager.train_datasets['sipakmed']
        sipakmed_val = dataset_manager.val_datasets['sipakmed']
        sipakmed_test = dataset_manager.test_datasets['sipakmed']
        
        print(f"SIPaKMeD - Train: {len(sipakmed_train)}, Val: {len(sipakmed_val)}, Test: {len(sipakmed_test)}")
        
        # Get FULL Herlev dataset (NO splitting)
        print("Using full Herlev dataset as test set...")
        herlev_data = dataset_manager.herlev_dataset  # Full dataset, no splitting
        print(f"Herlev - Test: {len(herlev_data)} samples (full dataset)")
        
        # Create model
        print("Creating Swin Transformer model...")
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
        
        train_loader = create_data_loader(sipakmed_train, train_transform, batch_size=16, shuffle=True)
        val_loader = create_data_loader(sipakmed_val, val_transform, batch_size=16, shuffle=False)
        
        # Train model
        print("Training Swin Transformer...")
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
        
        # Load best model
        best_model_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with F1: {checkpoint['val_f1']:.4f}")
        
        # Evaluate on Herlev
        print("Evaluating on Herlev...")
        herlev_loader = create_data_loader(herlev_data, val_transform, batch_size=16, shuffle=False)
        
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(herlev_loader):
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
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx * len(images)} samples...")
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        # Save results
        results = {
            'model': 'swin_transformer',
            'train_dataset': 'sipakmed',
            'test_dataset': 'herlev',
            'cross_dataset_results': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'num_samples': len(all_labels)
            }
        }
        
        os.makedirs('outputs/metrics', exist_ok=True)
        with open('outputs/metrics/cross_dataset_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 50)
        print("CROSS-DATASET RESULTS")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Samples: {len(all_labels)}")
        print(f"Results saved to: outputs/metrics/cross_dataset_results.json")
        print("=" * 50)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in cross-dataset training: {e}")
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
