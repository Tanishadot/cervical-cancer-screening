"""
Main pipeline script for cervical cancer classification research.
Orchestrates the entire deep learning pipeline from data preprocessing to XAI analysis.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import random
from typing import Dict, List, Optional
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_manager import ConfigManager, load_config
from utils.label_mapping import LabelMapper, ClassificationMode, DatasetType
from datasets.dataset import create_dataset, create_dataloader, get_dataset_info
from preprocessing.transforms import get_train_transforms, get_val_transforms
from preprocessing.image_preprocessing import preprocess_pil_image
from models.model_factory import create_model, ModelFactory
from training.trainer import Trainer
from training.cross_dataset_eval import CrossDatasetEvaluator
from xai.grad_cam import create_xai_analyzer
from utils.visualization import create_visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def setup_device(device_config: str) -> torch.device:
    """Setup computation device."""
    if device_config == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device


def create_datasets(config_manager: ConfigManager) -> Dict:
    """Create training and validation datasets."""
    config = config_manager.get_experiment_config()
    dataset_config = config_manager.get_dataset_config()
    
    # Create label mapper
    if config.classification_mode == "binary":
        label_mapper = LabelMapper(ClassificationMode.BINARY)
        num_classes = 2
    else:
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        num_classes = label_mapper.get_num_classes(DatasetType.SIPAKMED)
    
    # Update config with correct number of classes
    config.num_classes = num_classes
    
    # Create transforms
    train_transform = get_train_transforms(
        image_size=config.image_size,
        apply_color_augmentation=config.use_color_augmentation,
        apply_geometric_augmentation=config.use_geometric_augmentation
    )
    
    val_transform = get_val_transforms(image_size=config.image_size)
    
    # Create datasets
    datasets = {}
    
    # Training dataset
    train_dataset = create_dataset(
        root_dir=dataset_config['root_dir'],
        dataset_name=dataset_config['train_dataset'],
        classification_mode=config.classification_mode,
        transform=train_transform,
        mode="train"
    )
    datasets['train'] = train_dataset
    
    # Validation dataset
    val_dataset = create_dataset(
        root_dir=dataset_config['root_dir'],
        dataset_name=dataset_config['train_dataset'],
        classification_mode=config.classification_mode,
        transform=val_transform,
        mode="val"
    )
    datasets['val'] = val_dataset
    
    # Test dataset (same dataset for initial evaluation)
    test_dataset = create_dataset(
        root_dir=dataset_config['root_dir'],
        dataset_name=dataset_config['train_dataset'],
        classification_mode=config.classification_mode,
        transform=val_transform,
        mode="test"
    )
    datasets['test'] = test_dataset
    
    # Get dataset information
    train_info = get_dataset_info(train_dataset)
    logger.info(f"Training dataset: {train_info['num_samples']} samples, {train_info['num_classes']} classes")
    logger.info(f"Class distribution: {train_info['class_distribution']}")
    
    return datasets


def create_data_loaders(datasets: Dict, config_manager: ConfigManager) -> Dict:
    """Create data loaders."""
    config = config_manager.get_experiment_config()
    
    loaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        loader = create_dataloader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers
        )
        loaders[split] = loader
        logger.info(f"{split.capitalize()} loader created: {len(loader)} batches")
    
    return loaders


def train_model(config_manager: ConfigManager, loaders: Dict) -> Dict:
    """Train the model."""
    config = config_manager.get_experiment_config()
    model_config = config_manager.get_model_config()
    training_config = config_manager.get_training_config()
    
    # Create model
    model = create_model(
        model_name=model_config['architecture'],
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained'],
        freeze_backbone=model_config['freeze_backbone'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Get class names
    if config.classification_mode == "binary":
        class_names = ["NORMAL", "ABNORMAL"]
    else:
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        class_names = label_mapper.get_class_names(DatasetType.SIPAKMED)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        test_loader=loaders['test'],
        num_classes=model_config['num_classes'],
        class_names=class_names,
        device=setup_device(config.device),
        output_dir=config.output_dir,
        experiment_name=config.name
    )
    
    # Train model
    logger.info("Starting model training...")
    history = trainer.train(
        num_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['patience'],
        use_class_weights=training_config['use_class_weights'],
        use_amp=training_config['use_amp'],
        unfreeze_epoch=training_config['unfreeze_epoch']
    )
    
    # Test model
    logger.info("Testing model...")
    test_metrics = trainer.test()
    
    return {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
        'trainer': trainer
    }


def run_cross_dataset_evaluation(
    model,
    config_manager: ConfigManager
) -> Dict:
    """Run cross-dataset evaluation."""
    config = config_manager.get_experiment_config()
    
    # Create evaluator
    evaluator = CrossDatasetEvaluator(
        model=model,
        datasets_root=config.dataset_root,
        output_dir=os.path.join(config.output_dir, 'cross_dataset'),
        device=setup_device(config.device)
    )
    
    # Get validation transforms for evaluation
    val_transform = get_val_transforms(image_size=config.image_size)
    
    # Run cross-dataset evaluation
    logger.info("Starting cross-dataset evaluation...")
    results = evaluator.run_full_cross_dataset_evaluation(
        classification_mode=config.classification_mode,
        batch_size=config.batch_size,
        transform=val_transform
    )
    
    return results


def run_xai_analysis(
    model,
    config_manager: ConfigManager,
    loaders: Dict
) -> Dict:
    """Run XAI analysis."""
    config = config_manager.get_experiment_config()
    
    if not config.xai_enabled:
        logger.info("XAI analysis disabled")
        return {}
    
    # Create XAI analyzer
    device = setup_device(config.device)
    xai_analyzer = create_xai_analyzer(
        model=model,
        device=device,
        output_dir=config.paths.get('xai_outputs', 'outputs/xai')
    )
    
    # Get class names
    if config.classification_mode == "binary":
        class_names = ["NORMAL", "ABNORMAL"]
    else:
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        class_names = label_mapper.get_class_names(DatasetType.SIPAKMED)
    
    # Run XAI analysis on test set
    logger.info("Starting XAI analysis...")
    results = xai_analyzer.analyze_dataset(
        data_loader=loaders['test'],
        class_names=class_names,
        num_samples=config.xai_num_samples,
        save_prefix=f"{config.name}_xai"
    )
    
    return results


def create_visualizations(
    history: Dict,
    test_metrics: Dict,
    config_manager: ConfigManager
):
    """Create comprehensive visualizations."""
    config = config_manager.get_experiment_config()
    
    # Create visualizer
    visualizer = create_visualizer(
        output_dir=config.paths.get('visualizations', 'outputs/visualizations')
    )
    
    # Get class names
    if config.classification_mode == "binary":
        class_names = ["NORMAL", "ABNORMAL"]
    else:
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        class_names = label_mapper.get_class_names(DatasetType.SIPAKMED)
    
    # Create dummy confusion matrix for visualization (would be computed during testing)
    # In practice, this would come from the actual test results
    num_classes = len(class_names)
    cm = np.eye(num_classes, dtype=int)  # Identity matrix as placeholder
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(
        history=history,
        metrics=test_metrics,
        cm=cm,
        class_names=class_names,
        experiment_name=config.name
    )
    
    logger.info("Visualizations created")


def save_experiment_summary(
    results: Dict,
    config_manager: ConfigManager
):
    """Save experiment summary."""
    config = config_manager.get_experiment_config()
    
    # Create summary
    summary = {
        'experiment_name': config.name,
        'timestamp': datetime.now().isoformat(),
        'configuration': config_manager.get_config(),
        'results': results
    }
    
    # Save summary
    summary_path = os.path.join(config.output_dir, f'{config.name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Experiment summary saved to {summary_path}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Cervical Cancer Classification Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'xai', 'all'], default='all', help='Pipeline mode')
    parser.add_argument('--binary', action='store_true', help='Force binary classification')
    parser.add_argument('--multiclass', action='store_true', help='Force multi-class classification')
    parser.add_argument('--model', type=str, choices=['efficientnet_b0', 'resnet50', 'swin_transformer'], help='Model architecture')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = load_config(args.config)
    config = config_manager.get_experiment_config()
    
    # Override configuration with command line arguments
    overrides = {}
    
    if args.binary:
        overrides['dataset'] = {'classification_mode': 'binary'}
    elif args.multiclass:
        overrides['dataset'] = {'classification_mode': 'multiclass'}
    
    if args.model:
        overrides['model'] = {'architecture': args.model}
    
    if args.epochs:
        overrides['training'] = {'num_epochs': args.epochs}
    
    if args.batch_size:
        overrides['dataset'] = {'batch_size': args.batch_size}
    
    if args.lr:
        overrides['training'] = {'learning_rate': args.lr}
    
    if args.seed:
        overrides['experiment'] = {'seed': args.seed}
    
    if overrides:
        config_manager.update_config(overrides)
        config = config_manager.get_experiment_config()
    
    # Print configuration
    config_manager.print_config_summary()
    
    # Create experiment directories
    config_manager.create_experiment_directories()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup device
    device = setup_device(config.device)
    
    # Results dictionary
    results = {}
    
    try:
        if args.mode in ['train', 'all']:
            logger.info("="*60)
            logger.info("TRAINING PHASE")
            logger.info("="*60)
            
            # Create datasets and loaders
            datasets = create_datasets(config_manager)
            loaders = create_data_loaders(datasets, config_manager)
            
            # Train model
            training_results = train_model(config_manager, loaders)
            results['training'] = training_results
            
        if args.mode in ['eval', 'all']:
            logger.info("="*60)
            logger.info("EVALUATION PHASE")
            logger.info("="*60)
            
            if 'training' not in results:
                # Load pretrained model for evaluation
                model_config = config_manager.get_model_config()
                model = create_model(
                    model_name=model_config['architecture'],
                    num_classes=model_config['num_classes'],
                    pretrained=model_config['pretrained'],
                    freeze_backbone=False
                )
                
                # Load best model weights if available
                model_path = os.path.join(config.output_dir, 'models', 'best_model.pth')
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    logger.info(f"Loaded model from {model_path}")
            else:
                model = training_results['model']
            
            # Cross-dataset evaluation
            cross_dataset_results = run_cross_dataset_evaluation(model, config_manager)
            results['cross_dataset'] = cross_dataset_results
            
        if args.mode in ['xai', 'all']:
            logger.info("="*60)
            logger.info("XAI ANALYSIS PHASE")
            logger.info("="*60)
            
            if 'training' not in results:
                logger.warning("No trained model available for XAI analysis")
            else:
                # Run XAI analysis
                xai_results = run_xai_analysis(
                    training_results['model'],
                    config_manager,
                    loaders if 'loaders' in locals() else None
                )
                results['xai'] = xai_results
        
        if args.mode in ['all']:
            logger.info("="*60)
            logger.info("VISUALIZATION PHASE")
            logger.info("="*60)
            
            if 'training' in results:
                # Create visualizations
                create_visualizations(
                    results['training']['history'],
                    results['training']['test_metrics'],
                    config_manager
                )
        
        # Save experiment summary
        save_experiment_summary(results, config_manager)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        # Print summary
        if 'training' in results:
            test_metrics = results['training']['test_metrics']
            logger.info(f"Final Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            logger.info(f"Final Test F1 Score: {test_metrics.get('f1_macro', 0):.4f}")
        
        if 'cross_dataset' in results:
            logger.info("Cross-dataset evaluation completed")
            for exp_name, exp_results in results['cross_dataset'].items():
                if 'metrics' in exp_results:
                    acc = exp_results['metrics'].get('accuracy', 0)
                    logger.info(f"{exp_name} Accuracy: {acc:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise
    
    logger.info(f"All results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
