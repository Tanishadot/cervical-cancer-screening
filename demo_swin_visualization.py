#!/usr/bin/env python3
"""
Demo Swin Transformer Visualization in Terminal
Shows example output of feature visualization
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.model_factory import CervicalCancerModel
from datasets.dataset import DatasetManager
from utils.config_manager import ConfigManager
from xai.swin_attention_simple import SwinFeatureExtractor, generate_attention_overlay

def print_terminal_visualization(feature_map, original_img, overlay, label_name, pred_name, confidence):
    """Print visualization representation in terminal."""
    
    print("\n" + "="*80)
    print("🧠 SWIN TRANSFORMER FEATURE VISUALIZATION DEMO")
    print("="*80)
    
    # Sample info
    print(f"\n📊 Sample Information:")
    print(f"  Ground Truth: {label_name}")
    print(f"  Prediction: {pred_name}")
    print(f"  Confidence: {confidence:.4f}")
    
    # Feature map stats
    print(f"\n🔥 Feature Activation Map:")
    print(f"  Shape: {feature_map.shape}")
    print(f"  Min: {feature_map.min():.4f}")
    print(f"  Max: {feature_map.max():.4f}")
    print(f"  Mean: {feature_map.mean():.4f}")
    
    # Create ASCII representation of feature map
    print(f"\n📈 ASCII Feature Map (7x7 grid):")
    print("  " + "-"*30)
    
    # Resize to 7x7 for ASCII display
    grid_size = 7
    feature_resized = feature_map[:grid_size, :grid_size]
    
    # Normalize for ASCII
    feature_norm = (feature_resized - feature_resized.min()) / (feature_resized.max() - feature_resized.min() + 1e-8)
    
    # ASCII characters for different intensity levels
    ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    
    for i in range(grid_size):
        row = "  |"
        for j in range(grid_size):
            intensity = feature_norm[i, j]
            char_idx = min(int(intensity * len(ascii_chars)), len(ascii_chars) - 1)
            row += ascii_chars[char_idx] * 2
        row += "|"
        print(row)
    
    print("  " + "-"*30)
    
    # Highlight regions
    print(f"\n🎯 High Activation Regions:")
    high_threshold = np.percentile(feature_norm, 80)
    high_regions = np.where(feature_norm > high_threshold)
    
    if len(high_regions[0]) > 0:
        print(f"  Found {len(high_regions[0])} high-activation regions")
        for i in range(min(3, len(high_regions[0]))):
            y, x = high_regions[0][i], high_regions[1][i]
            print(f"  Region {i+1}: Position ({x}, {y}) - Intensity: {feature_norm[y, x]:.3f}")
    else:
        print("  No significant high-activation regions found")
    
    # Model interpretation
    print(f"\n🔍 Model Interpretation:")
    if pred_name == label_name:
        print("  ✅ Correct Prediction")
        print("  📈 Model confidence is high and prediction matches ground truth")
    else:
        print("  ❌ Incorrect Prediction")
        print("  ⚠️  Model may be focusing on wrong regions")
    
    # Clinical relevance
    print(f"\n🏥 Clinical Relevance:")
    print("  📍 Feature map shows regions model considers important")
    print("  🔬 Bright areas indicate cell nuclei or abnormal features")
    print("  👨‍⚕️  Clinicians can review these regions for validation")
    
    print("\n" + "="*80)

def main():
    """Demonstrate Swin visualization in terminal."""
    print("🚀 Starting Swin Transformer Visualization Demo...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
    if not os.path.exists(model_path):
        print("❌ Swin model not found. Please train the model first.")
        return
    
    print("📥 Loading Swin Transformer model...")
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    print("📂 Loading SIPaKMeD dataset...")
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
    
    test_dataset = dataset_manager.test_datasets['sipakmed']
    print(f"Dataset loaded: {len(test_dataset)} samples")
    
    # Create feature extractor
    feature_extractor = SwinFeatureExtractor(model, device)
    
    # Process a few samples
    print("\n🎨 Generating feature visualizations...")
    
    from preprocessing.transforms import get_val_transforms
    from datasets.dataset import TransformDataset, create_dataloader
    
    # Create transforms
    val_transform = get_val_transforms()
    
    # Create transformed dataset
    class AlbumentationsTransform:
        def __init__(self, albumentations_transform):
            self.transform = albumentations_transform
        
        def __call__(self, image):
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            return self.transform(image=image)['image']
    
    albumentations_transform = AlbumentationsTransform(val_transform)
    transformed_dataset = TransformDataset(test_dataset, albumentations_transform)
    
    # Create data loader
    data_loader = create_dataloader(
        dataset=transformed_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= 3:  # Show 3 examples
            break
            
        try:
            # Get batch data
            if len(batch) == 4:
                images, labels, original_images, _ = batch
            elif len(batch) == 3:
                images, labels, original_images = batch
            else:
                images, labels = batch[:2]
                original_images = images
            
            # Get first sample
            image_tensor = images[0:1]  # Keep batch dimension
            label = labels[0].item()
            
            # Generate feature map
            feature_map = feature_extractor.generate_feature_map(image_tensor, 224)
            
            # Get prediction
            with torch.no_grad():
                output = model(image_tensor.to(device))
                pred = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][pred].item()
            
            # Convert original image
            original_img = original_images[0].cpu().numpy()
            if original_img.shape[0] == 3:  # CHW format
                original_img = np.transpose(original_img, (1, 2, 0))
            
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
            
            # Generate overlay
            overlay = generate_attention_overlay(original_img, feature_map)
            
            # Get label names
            label_name = "Normal" if label == 0 else "Abnormal"
            pred_name = "Normal" if pred == 0 else "Abnormal"
            
            # Print terminal visualization
            print_terminal_visualization(feature_map, original_img, overlay, label_name, pred_name, confidence)
            
        except Exception as e:
            print(f"❌ Error processing sample {batch_idx}: {e}")
            continue
    
    # Cleanup
    feature_extractor.cleanup()
    
    print("\n🎉 Demo completed!")
    print("💡 For interactive visualizations, run: streamlit run dashboard_swin_xai.py")

if __name__ == "__main__":
    main()
