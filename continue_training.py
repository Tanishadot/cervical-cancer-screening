"""
Continue training with memory-efficient settings.
"""

import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_efficientnet_optimized import EfficientNetB0Optimized

def main():
    print("🚀 CONTINUING EFFICIENTNET-B0 TRAINING")
    print("Memory-efficient configuration")
    print("=" * 60)
    
    # Create memory-efficient configuration
    config = {
        'experiment': {
            'name': 'efficientnet_b0_continued',
            'description': 'Memory-efficient continued training',
            'output_dir': 'outputs/efficientnet_optimized'
        },
        'dataset': {
            'root_dir': 'datasets',
            'train_dataset': 'sipakmed',
            'classification_mode': 'binary',
            'use_cropped_only': True,
            'run_validation': True,
            'validation_strict_mode': True
        },
        'model': {
            'architecture': 'efficientnet_b0',
            'pretrained': True,
            'dropout_rate': 0.4
        },
        'training': {
            'num_epochs': 20,  # Reduced for memory efficiency
            'batch_size': 8,   # Smaller batch size
            'learning_rate': 5e-5,  # Lower LR after unfreeze
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'label_smoothing': 0.1,
            'unfreeze_epoch': 1,  # Start unfrozen
            'num_workers': 2   # Reduce workers
        }
    }
    
    # Create trainer
    trainer = EfficientNetB0Optimized(config)
    
    # Load best checkpoint if exists
    checkpoint_path = "outputs/efficientnet_optimized/models/best_model_optimized.pth"
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading checkpoint: {checkpoint_path}")
        # We'll let the trainer handle loading during training
    else:
        print("⚠️ No checkpoint found, starting fresh")
    
    # Start training
    success = trainer.train()
    
    if success:
        print("🎉 Training completed successfully!")
        print(f"Results saved to: {trainer.output_dir}")
    else:
        print("❌ Training failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
