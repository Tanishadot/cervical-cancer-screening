#!/usr/bin/env python3
"""
Cross-Dataset Evaluation: SIPaKMeD → Herlev
Loads trained model and evaluates on different dataset
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from datasets.dataset import DatasetManager
    from preprocessing.transforms import get_val_transforms
    from models.model_factory import CervicalCancerModel
    from utils.config_manager import ConfigManager
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from datasets.dataset import TransformDataset, create_dataloader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class AlbumentationsTransform:
    """Wrapper for albumentations transforms."""
    def __init__(self, albumentations_transform):
        self.transform = albumentations_transform
    
    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        return self.transform(image=image)['image']

def load_trained_model(model_path):
    """Load trained model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading trained model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model (same architecture as training)
    model = CervicalCancerModel(
        model_name="efficientnet_b0",  # Assuming this is what you trained
        num_classes=2,
        pretrained=False,  # Don't need pretrained weights
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def load_herlev_dataset():
    """Load Herlev dataset for evaluation."""
    print("Loading Herlev dataset...")
    
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
    
    # Load datasets
    dataset_manager.load_datasets()
    dataset_manager.create_splits()
    
    # Use Herlev test split (not full dataset to avoid attribute error)
    herlev_data = dataset_manager.test_datasets['herlev']
    print(f"Herlev dataset loaded: {len(herlev_data)} samples")
    
    return herlev_data

def create_data_loader(dataset, transform, batch_size=32, shuffle=False):
    """Create data loader with transforms."""
    albumentations_transform = AlbumentationsTransform(transform)
    transformed_dataset = TransformDataset(dataset, albumentations_transform)
    
    return create_dataloader(
        dataset=transformed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

def evaluate_model(model, device, dataset, dataset_name):
    """Evaluate model on dataset."""
    print(f"Evaluating on {dataset_name}...")
    
    # Get validation transforms (same as training)
    val_transform = get_val_transforms()
    
    # Create data loader
    data_loader = create_data_loader(dataset, val_transform, batch_size=32, shuffle=False)
    
    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Handle different batch formats
            if len(batch) == 4:
                images, labels, _, _ = batch
            elif len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch[:2]
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
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
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels)
    }
    
    print(f"\n📊 Results on {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Samples: {len(all_labels)}")
    
    return results

def compare_with_sipakmed(model, device):
    """Compare with SIPaKMeD test set if available."""
    print("\n" + "="*50)
    print("COMPARING WITH SIPaKMeD TEST SET")
    print("="*50)
    
    try:
        # Load SIPaKMeD test set
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
        
        sipakmed_test = dataset_manager.test_datasets['sipakmed']
        sipakmed_results = evaluate_model(model, device, sipakmed_test, "SIPaKMeD Test")
        
        return sipakmed_results
        
    except Exception as e:
        print(f"Could not evaluate on SIPaKMeD: {e}")
        return None

def generate_visualizations(sipakmed_results, herlev_results):
    """Generate comparison visualizations."""
    print("\nGenerating visualizations...")
    
    # Create output directory
    viz_dir = Path("outputs/cross_dataset/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for visualization
    datasets = []
    accuracies = []
    f1_scores = []
    
    if sipakmed_results:
        datasets.append("SIPaKMeD")
        accuracies.append(sipakmed_results['accuracy'])
        f1_scores.append(sipakmed_results['f1_score'])
    
    if herlev_results:
        datasets.append("Herlev")
        accuracies.append(herlev_results['accuracy'])
        f1_scores.append(herlev_results['f1_score'])
    
    if not datasets:
        print("No results to visualize")
        return
    
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(datasets, accuracies, color=['#2E86AB', '#A23B72'])
    plt.title('Cross-Dataset Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # F1 score comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(datasets, f1_scores, color=['#2E86AB', '#A23B72'])
    plt.title('Cross-Dataset F1 Score Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {viz_dir}")

def main():
    """Main cross-dataset evaluation."""
    print("Cross-Dataset Evaluation: Trained Model → Herlev")
    print("="*60)
    
    # Load trained model
    model_path = "outputs/models/best_model.pth"
    model, device = load_trained_model(model_path)
    
    # Load Herlev dataset
    herlev_data = load_herlev_dataset()
    
    # Evaluate on Herlev
    herlev_results = evaluate_model(model, device, herlev_data, "Herlev")
    
    # Compare with SIPaKMeD if possible
    sipakmed_results = compare_with_sipakmed(model, device)
    
    # Generate summary
    print("\n" + "="*60)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("="*60)
    
    results = {
        'model_path': model_path,
        'train_dataset': 'sipakmed',
        'test_datasets': {}
    }
    
    if sipakmed_results:
        results['test_datasets']['sipakmed'] = sipakmed_results
        print(f"SIPaKMeD Test - Accuracy: {sipakmed_results['accuracy']:.4f}, F1: {sipakmed_results['f1_score']:.4f}")
    
    if herlev_results:
        results['test_datasets']['herlev'] = herlev_results
        print(f"Herlev Test - Accuracy: {herlev_results['accuracy']:.4f}, F1: {herlev_results['f1_score']:.4f}")
    
    # Calculate generalization gap
    if sipakmed_results and herlev_results:
        accuracy_gap = sipakmed_results['accuracy'] - herlev_results['accuracy']
        f1_gap = sipakmed_results['f1_score'] - herlev_results['f1_score']
        
        print(f"\n🎯 Generalization Analysis:")
        print(f"Accuracy Gap: {accuracy_gap:.4f}")
        print(f"F1 Score Gap: {f1_gap:.4f}")
        
        if accuracy_gap > 0.1:
            print("⚠️  Significant performance drop on cross-dataset")
        elif accuracy_gap > 0.05:
            print("⚡ Moderate performance drop on cross-dataset")
        else:
            print("✅ Good generalization performance")
    
    # Save results
    os.makedirs('outputs/cross_dataset', exist_ok=True)
    with open('outputs/cross_dataset/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: outputs/cross_dataset/evaluation_results.json")
    
    # Generate visualizations
    generate_visualizations(sipakmed_results, herlev_results)
    
    print("\n✅ Cross-dataset evaluation completed!")
    return results

if __name__ == "__main__":
    main()
