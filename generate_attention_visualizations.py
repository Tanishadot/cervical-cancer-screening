#!/usr/bin/env python3
"""
Generate Attention Rollout Visualizations for Swin Transformer
Creates visualizations for multiple samples from SIPaKMeD dataset
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from xai.swin_attention_rollout import visualize_multiple_samples
from datasets.dataset import DatasetManager
from models.model_factory import CervicalCancerModel
from utils.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate attention visualizations for Swin Transformer."""
    print("Generating Attention Rollout Visualizations")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Load Swin model
    print("Loading Swin Transformer model...")
    model_path = "outputs/cross_dataset_training/models/best_swin_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Swin model not found at {model_path}")
        print("Please train the Swin model first using run_cross_dataset.py")
        return
    
    # Create model
    model = CervicalCancerModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded Swin model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded Swin model state dict")
    
    model.eval()
    
    # Load SIPaKMeD dataset
    print("Loading SIPaKMeD dataset...")
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
    
    # Use test set for visualization
    test_dataset = dataset_manager.test_datasets['sipakmed']
    print(f"SIPaKMeD test dataset: {len(test_dataset)} samples")
    
    # Generate visualizations
    print("Generating attention visualizations...")
    try:
        visualize_multiple_samples(
            model=model,
            dataset=test_dataset,
            num_samples=6,  # 3 normal + 3 abnormal
            device=device,
            save_dir='outputs/xai/swin_attention/',
            patch_size=16,
            image_size=224
        )
        
        print("✅ Attention visualizations generated successfully!")
        print("📁 Saved to: outputs/xai/swin_attention/")
        
        # List generated files
        output_dir = Path('outputs/xai/swin_attention/')
        if output_dir.exists():
            files = list(output_dir.glob('*.png'))
            print(f"📊 Generated {len(files)} visualization files:")
            for file in sorted(files):
                print(f"  - {file.name}")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        logger.error(f"Visualization generation failed: {e}")

if __name__ == "__main__":
    main()
