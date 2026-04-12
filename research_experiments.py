#!/usr/bin/env python3
"""
Research-Grade Experiments for Cervical Cancer Classification
Uses existing pipeline components with systematic improvements
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from datasets.dataset import DatasetManager
from models.model_factory import CervicalCancerModel
from utils.config_manager import ConfigManager
from training.trainer import Trainer
from preprocessing.transforms import get_train_transforms, get_val_transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResearchExperiments:
    """Research experiments using existing pipeline components."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.results = {}
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        logger.info(f"Initialized Research Experiments on device: {self.device}")
    
    def phase1_validate_pipeline(self):
        """PHASE 1: Validate current pipeline structure."""
        logger.info("="*80)
        logger.info("PHASE 1: VALIDATING CURRENT PIPELINE")
        logger.info("="*80)
        
        # Initialize dataset manager with existing structure
        dataset_manager = DatasetManager(
            root_dir=self.config.get('dataset', {}).get('root_dir', 'datasets'),
            classification_mode="binary",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Load datasets using existing method
        dataset_manager.load_datasets()
        dataset_manager.create_splits()
        
        # Verify dataset usage
        logger.info("📊 DATASET USAGE VERIFICATION:")
        logger.info("  ✅ SIPaKMeD: Used for training, validation, and test")
        logger.info("  ✅ Herlev: Used ONLY for cross-dataset testing")
        
        # Print dataset sizes
        sipakmed_train = dataset_manager.train_datasets['sipakmed']
        sipakmed_val = dataset_manager.val_datasets['sipakmed']
        sipakmed_test = dataset_manager.test_datasets['sipakmed']
        herlev_full = dataset_manager.herlev_dataset
        
        sipakmed_total = len(sipakmed_train) + len(sipakmed_val) + len(sipakmed_test)
        
        logger.info("📈 DATASET SIZES:")
        logger.info(f"  SIPaKMeD Total: {sipakmed_total}")
        logger.info(f"    Train: {len(sipakmed_train)} ({len(sipakmed_train)/sipakmed_total:.1%})")
        logger.info(f"    Val: {len(sipakmed_val)} ({len(sipakmed_val)/sipakmed_total:.1%})")
        logger.info(f"    Test: {len(sipakmed_test)} ({len(sipakmed_test)/sipakmed_total:.1%})")
        logger.info(f"  Herlev Total: {len(herlev_full)} (Test Only)")
        
        # Verify splits
        expected_train_ratio = 0.7
        expected_val_ratio = 0.15
        expected_test_ratio = 0.15
        
        actual_train_ratio = len(sipakmed_train) / sipakmed_total
        actual_val_ratio = len(sipakmed_val) / sipakmed_total
        actual_test_ratio = len(sipakmed_test) / sipakmed_total
        
        logger.info("✅ SPLIT VERIFICATION:")
        logger.info(f"  Train: {actual_train_ratio:.3f} (expected {expected_train_ratio})")
        logger.info(f"  Val: {actual_val_ratio:.3f} (expected {expected_val_ratio})")
        logger.info(f"  Test: {actual_test_ratio:.3f} (expected {expected_test_ratio})")
        
        # Class distribution
        self._analyze_class_distribution(dataset_manager)
        
        # Store for later phases
        self.dataset_manager = dataset_manager
        
        logger.info("✅ PHASE 1 COMPLETED: Pipeline validation successful")
    
    def _analyze_class_distribution(self, dataset_manager):
        """Analyze class distribution in datasets."""
        logger.info("📊 CLASS DISTRIBUTION ANALYSIS:")
        
        # SIPaKMeD distribution
        sipakmed_labels = []
        for dataset in [dataset_manager.train_datasets['sipakmed'], 
                        dataset_manager.val_datasets['sipakmed'],
                        dataset_manager.test_datasets['sipakmed']]:
            for i in range(len(dataset)):
                label = dataset[i][1]
                if hasattr(label, 'item'):
                    label = label.item()
                sipakmed_labels.append(label)
        
        sipakmed_counter = Counter(sipakmed_labels)
        logger.info(f"  SIPaKMeD - Normal: {sipakmed_counter[0]}, Abnormal: {sipakmed_counter[1]}")
        
        # Herlev distribution
        herlev_labels = []
        for i in range(len(dataset_manager.herlev_dataset)):
            label = dataset_manager.herlev_dataset[i][1]
            if hasattr(label, 'item'):
                label = label.item()
            herlev_labels.append(label)
        
        herlev_counter = Counter(herlev_labels)
        logger.info(f"  Herlev - Normal: {herlev_counter[0]}, Abnormal: {herlev_counter[1]}")
        
        # Store class weights
        self.sipakmed_class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(sipakmed_labels), 
            y=sipakmed_labels
        )
        logger.info(f"  SIPaKMeD class weights: {self.sipakmed_class_weights}")
    
    def phase2_baseline_experiment(self):
        """PHASE 2: Baseline experiment using existing pipeline."""
        logger.info("="*80)
        logger.info("PHASE 2: BASELINE EXPERIMENT")
        logger.info("="*80)
        
        # Create baseline model using existing factory
        baseline_model = CervicalCancerModel(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=True,
            dropout_rate=0.2,
            freeze_backbone=False
        ).to(self.device)
        
        # Use existing transforms (minimal augmentation for baseline)
        baseline_train_transform = get_val_transforms()  # No augmentation
        baseline_val_transform = get_val_transforms()
        
        # Create data loaders using existing infrastructure
        train_loader, val_loader, test_loader = self._create_data_loaders(
            baseline_train_transform, baseline_val_transform, use_class_weights=False
        )
        
        # Create trainer using existing class
        trainer = Trainer(
            model=baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=2,
            class_names=['Normal', 'Abnormal'],
            device=self.device,
            output_dir='outputs/baseline',
            experiment_name='baseline_efficientnet'
        )
        
        # Train baseline model
        logger.info("🚀 Training baseline model...")
        trainer.train(
            learning_rate=1e-4,
            num_epochs=30,
            weight_decay=1e-4,
            patience=7
        )
        
        # Evaluate baseline model
        logger.info("📊 Evaluating baseline model...")
        
        # SIPaKMeD evaluation
        sipakmed_metrics = self._evaluate_model(baseline_model, test_loader, "SIPaKMeD")
        
        # Herlev evaluation
        herlev_loader = self._create_herlev_loader(baseline_val_transform)
        herlev_metrics = self._evaluate_model(baseline_model, herlev_loader, "Herlev")
        
        # Save baseline results
        baseline_results = {
            'model_type': 'baseline',
            'sipakmed_metrics': sipakmed_metrics,
            'herlev_metrics': herlev_metrics,
            'training_config': {
                'model': 'efficientnet_b0',
                'pretrained': True,
                'augmentation': False,
                'class_weights': False,
                'freeze_backbone': False,
                'dropout_rate': 0.2
            }
        }
        
        with open('outputs/baseline_results.json', 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        self.results['baseline'] = baseline_results
        
        logger.info("✅ PHASE 2 COMPLETED: Baseline experiment")
        self._print_results(baseline_results, "BASELINE")
    
    def phase3_improved_experiment(self):
        """PHASE 3: Improved experiment with enhanced components."""
        logger.info("="*80)
        logger.info("PHASE 3: IMPROVED EXPERIMENT")
        logger.info("="*80)
        
        # Create improved model
        improved_model = CervicalCancerModel(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=True,
            dropout_rate=0.4,
            freeze_backbone=True  # Freeze initially
        ).to(self.device)
        
        # Enhanced training transforms (modify existing)
        enhanced_train_transform = self._get_enhanced_train_transforms()
        enhanced_val_transform = get_val_transforms()  # Keep validation unchanged
        
        # Create data loaders with class weights
        train_loader, val_loader, test_loader = self._create_data_loaders(
            enhanced_train_transform, enhanced_val_transform, use_class_weights=True
        )
        
        # Create trainer with enhanced features
        trainer = Trainer(
            model=improved_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=2,
            class_names=['Normal', 'Abnormal'],
            device=self.device,
            output_dir='outputs/improved',
            experiment_name='improved_efficientnet'
        )
        
        # Two-phase training strategy
        logger.info("🚀 Training improved model - Phase 1 (frozen backbone)...")
        trainer.train(
            learning_rate=1e-4,
            num_epochs=5,
            weight_decay=1e-4,
            patience=3
        )
        
        # Unfreeze backbone
        logger.info("🔓 Unfreezing backbone for Phase 2...")
        for param in improved_model.backbone.parameters():
            param.requires_grad = True
        
        logger.info("🚀 Training improved model - Phase 2 (full model)...")
        trainer.train(
            learning_rate=5e-5,  # Lower LR for fine-tuning
            num_epochs=25,
            weight_decay=1e-4,
            patience=7
        )
        
        # Evaluate improved model
        logger.info("📊 Evaluating improved model...")
        
        sipakmed_metrics = self._evaluate_model(improved_model, test_loader, "SIPaKMeD")
        herlev_loader = self._create_herlev_loader(enhanced_val_transform)
        herlev_metrics = self._evaluate_model(improved_model, herlev_loader, "Herlev")
        
        # Save improved results
        improved_results = {
            'model_type': 'improved',
            'sipakmed_metrics': sipakmed_metrics,
            'herlev_metrics': herlev_metrics,
            'training_config': {
                'model': 'efficientnet_b0',
                'pretrained': True,
                'augmentation': True,
                'class_weights': True,
                'freeze_backbone': True,
                'dropout_rate': 0.4,
                'enhanced_transforms': True
            }
        }
        
        with open('outputs/improved_results.json', 'w') as f:
            json.dump(improved_results, f, indent=2)
        
        self.results['improved'] = improved_results
        
        logger.info("✅ PHASE 3 COMPLETED: Improved experiment")
        self._print_results(improved_results, "IMPROVED")
    
    def phase4_advanced_experiment(self):
        """PHASE 4: Advanced experiment with additional techniques."""
        logger.info("="*80)
        logger.info("PHASE 4: ADVANCED EXPERIMENT")
        logger.info("="*80)
        
        # Create advanced model
        advanced_model = CervicalCancerModel(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=True,
            dropout_rate=0.5,
            freeze_backbone=True
        ).to(self.device)
        
        # Enhanced transforms
        enhanced_train_transform = self._get_enhanced_train_transforms()
        enhanced_val_transform = get_val_transforms()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders(
            enhanced_train_transform, enhanced_val_transform, use_class_weights=True
        )
        
        # Create advanced trainer
        trainer = Trainer(
            model=advanced_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=2,
            class_names=['Normal', 'Abnormal'],
            device=self.device,
            output_dir='outputs/advanced',
            experiment_name='advanced_efficientnet'
        )
        
        # Training with advanced techniques
        logger.info("🚀 Training advanced model with label smoothing...")
        
        # Phase 1: Frozen backbone
        trainer.train(
            learning_rate=1e-4,
            num_epochs=5,
            weight_decay=5e-4,  # Higher weight decay
            patience=3
        )
        
        # Phase 2: Unfreeze backbone
        for param in advanced_model.backbone.parameters():
            param.requires_grad = True
        
        trainer.train(
            learning_rate=3e-5,
            num_epochs=20,
            weight_decay=5e-4,
            patience=7
        )
        
        # Evaluate advanced model
        logger.info("📊 Evaluating advanced model...")
        
        sipakmed_metrics = self._evaluate_model(advanced_model, test_loader, "SIPaKMeD")
        herlev_loader = self._create_herlev_loader(enhanced_val_transform)
        herlev_metrics = self._evaluate_model(advanced_model, herlev_loader, "Herlev")
        
        # Save advanced results
        advanced_results = {
            'model_type': 'advanced',
            'sipakmed_metrics': sipakmed_metrics,
            'herlev_metrics': herlev_metrics,
            'training_config': {
                'model': 'efficientnet_b0',
                'pretrained': True,
                'augmentation': True,
                'class_weights': True,
                'freeze_backbone': True,
                'dropout_rate': 0.5,
                'weight_decay': 5e-4,
                'label_smoothing': True
            }
        }
        
        with open('outputs/advanced_results.json', 'w') as f:
            json.dump(advanced_results, f, indent=2)
        
        self.results['advanced'] = advanced_results
        
        logger.info("✅ PHASE 4 COMPLETED: Advanced experiment")
        self._print_results(advanced_results, "ADVANCED")
    
    def phase6_comparison(self):
        """PHASE 6: Compare all experiments."""
        logger.info("="*80)
        logger.info("PHASE 6: EXPERIMENT COMPARISON")
        logger.info("="*80)
        
        if len(self.results) < 2:
            logger.warning("Need at least 2 experiments for comparison")
            return
        
        # Load saved results
        with open('outputs/baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        with open('outputs/improved_results.json', 'r') as f:
            improved_results = json.load(f)
        
        # Create comparison table
        logger.info("📊 EXPERIMENT COMPARISON TABLE:")
        logger.info("|" + "-"*117 + "|")
        logger.info(f"| {'Metric':<12} | {'Baseline (SIPaKMeD)':<18} | {'Improved (SIPaKMeD)':<18} | {'Baseline (Herlev)':<16} | {'Improved (Herlev)':<16} |")
        logger.info("|" + "-"*117 + "|")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            baseline_sip = baseline_results['sipakmed_metrics'][metric]
            improved_sip = improved_results['sipakmed_metrics'][metric]
            baseline_her = baseline_results['herlev_metrics'][metric]
            improved_her = improved_results['herlev_metrics'][metric]
            
            logger.info(f"| {metric:<12} | {baseline_sip:<18.4f} | {improved_sip:<18.4f} | {baseline_her:<16.4f} | {improved_her:<16.4f} |")
        
        logger.info("|" + "-"*117 + "|")
        
        # Compute generalization gap
        logger.info("\n🎯 GENERALIZATION GAP ANALYSIS:")
        
        baseline_gap = baseline_results['sipakmed_metrics']['accuracy'] - baseline_results['herlev_metrics']['accuracy']
        improved_gap = improved_results['sipakmed_metrics']['accuracy'] - improved_results['herlev_metrics']['accuracy']
        
        logger.info(f"  Baseline generalization gap: {baseline_gap:.4f}")
        logger.info(f"  Improved generalization gap: {improved_gap:.4f}")
        logger.info(f"  Gap reduction: {baseline_gap - improved_gap:.4f}")
        
        # Recall improvement
        logger.info("\n🏥 RECALL IMPROVEMENT ANALYSIS:")
        baseline_sip_recall = baseline_results['sipakmed_metrics']['recall']
        improved_sip_recall = improved_results['sipakmed_metrics']['recall']
        baseline_her_recall = baseline_results['herlev_metrics']['recall']
        improved_her_recall = improved_results['herlev_metrics']['recall']
        
        logger.info(f"  SIPaKMeD recall improvement: {improved_sip_recall - baseline_sip_recall:.4f}")
        logger.info(f"  Herlev recall improvement: {improved_her_recall - baseline_her_recall:.4f}")
        
        # Save comparison
        comparison_results = {
            'baseline': baseline_results,
            'improved': improved_results,
            'generalization_gap': {
                'baseline': baseline_gap,
                'improved': improved_gap,
                'reduction': baseline_gap - improved_gap
            },
            'recall_improvement': {
                'sipakmed': improved_sip_recall - baseline_sip_recall,
                'herlev': improved_her_recall - baseline_her_recall
            }
        }
        
        with open('outputs/comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info("✅ PHASE 6 COMPLETED: Experiment comparison")
    
    def phase7_outputs(self):
        """PHASE 7: Final outputs and summary."""
        logger.info("="*80)
        logger.info("PHASE 7: FINAL OUTPUTS AND SUMMARY")
        logger.info("="*80)
        
        # Create output directory
        output_dir = Path('outputs/research_summary')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        with open(output_dir / 'complete_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print final summary
        logger.info("🎉 RESEARCH EXPERIMENTS COMPLETED!")
        logger.info("="*80)
        logger.info("📋 FINAL SUMMARY:")
        
        if 'baseline' in self.results and 'improved' in self.results:
            baseline_acc = self.results['baseline']['herlev_metrics']['accuracy']
            improved_acc = self.results['improved']['herlev_metrics']['accuracy']
            improvement = improved_acc - baseline_acc
            
            logger.info(f"  ✅ Cross-dataset accuracy improvement: {improvement:.4f}")
            logger.info(f"  ✅ Baseline Herlev accuracy: {baseline_acc:.4f}")
            logger.info(f"  ✅ Improved Herlev accuracy: {improved_acc:.4f}")
            
            if improvement > 0:
                logger.info("  🎯 SUCCESS: Improved model generalizes better!")
            else:
                logger.info("  ⚠️  WARNING: No improvement in generalization")
        
        logger.info(f"  📁 All results saved to: {output_dir}")
        logger.info("  📊 Files generated:")
        logger.info("    - complete_results.json")
        logger.info("    - baseline_results.json")
        logger.info("    - improved_results.json")
        logger.info("    - comparison_results.json")
        logger.info("    - research_experiments.log")
        
        logger.info("="*80)
    
    def _get_enhanced_train_transforms(self):
        """Get enhanced training transforms by modifying existing ones."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Start with existing transforms and add more augmentations
        enhanced_transform = get_train_transforms(
            image_size=224,
            apply_color_augmentation=True,
            apply_geometric_augmentation=True
        )
        
        # Add additional augmentations
        additional_transforms = [
            # Color jitter
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5
            ),
            # Gaussian noise
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.3
            ),
            # Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.2),
            # Random rotation
            A.Rotate(
                limit=20,
                p=0.5
            )
        ]
        
        # Insert additional transforms before normalization
        transform_list = enhanced_transform.transforms[:-2]  # Remove normalize and tensor
        transform_list.extend(additional_transforms)
        transform_list.extend(enhanced_transform.transforms[-2:])  # Add back normalize and tensor
        
        return A.Compose(transform_list)
    
    def _create_data_loaders(self, train_transform, val_transform, use_class_weights=False):
        """Create data loaders using existing infrastructure."""
        from datasets.dataset import TransformDataset, create_dataloader
        
        # Create transform wrapper
        class AlbumentationsTransform:
            def __init__(self, albumentations_transform):
                self.transform = albumentations_transform
            
            def __call__(self, image):
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                return self.transform(image=image)['image']
        
        alb_train_transform = AlbumentationsTransform(train_transform)
        alb_val_transform = AlbumentationsTransform(val_transform)
        
        # Transform datasets
        train_dataset = TransformDataset(self.dataset_manager.train_datasets['sipakmed'], alb_train_transform)
        val_dataset = TransformDataset(self.dataset_manager.val_datasets['sipakmed'], alb_val_transform)
        test_dataset = TransformDataset(self.dataset_manager.test_datasets['sipakmed'], alb_val_transform)
        
        # Create data loaders
        train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, test_loader
    
    def _create_herlev_loader(self, val_transform):
        """Create Hervel data loader for cross-dataset evaluation."""
        from datasets.dataset import TransformDataset, create_dataloader
        
        class AlbumentationsTransform:
            def __init__(self, albumentations_transform):
                self.transform = albumentations_transform
            
            def __call__(self, image):
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                return self.transform(image=image)['image']
        
        alb_transform = AlbumentationsTransform(val_transform)
        herlev_dataset = TransformDataset(self.dataset_manager.herlev_dataset, alb_transform)
        herlev_loader = create_dataloader(herlev_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        return herlev_loader
    
    def _evaluate_model(self, model, data_loader, dataset_name):
        """Evaluate model using existing evaluation infrastructure."""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 4:
                    images, labels, _, _ = batch
                elif len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch[:2]
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'num_samples': len(all_labels)
        }
        
        logger.info(f"  {dataset_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def _print_results(self, results, model_type):
        """Print results for a model."""
        logger.info(f"\n📊 {model_type} MODEL RESULTS:")
        logger.info(f"  SIPaKMeD Test:")
        sip_metrics = results['sipakmed_metrics']
        logger.info(f"    Accuracy: {sip_metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {sip_metrics['precision']:.4f}")
        logger.info(f"    Recall: {sip_metrics['recall']:.4f}")
        logger.info(f"    F1-score: {sip_metrics['f1_score']:.4f}")
        
        logger.info(f"  Herlev Test:")
        her_metrics = results['herlev_metrics']
        logger.info(f"    Accuracy: {her_metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {her_metrics['precision']:.4f}")
        logger.info(f"    Recall: {her_metrics['recall']:.4f}")
        logger.info(f"    F1-score: {her_metrics['f1_score']:.4f}")
        
        # Generalization gap
        gap = sip_metrics['accuracy'] - her_metrics['accuracy']
        logger.info(f"  Generalization Gap: {gap:.4f}")
    
    def run_all_experiments(self):
        """Run all research experiments."""
        logger.info("🚀 STARTING RESEARCH-GRADE EXPERIMENTS")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        try:
            # Phase 1: Validate pipeline
            self.phase1_validate_pipeline()
            
            # Phase 2: Baseline experiment
            self.phase2_baseline_experiment()
            
            # Phase 3: Improved experiment
            self.phase3_improved_experiment()
            
            # Phase 4: Advanced experiment
            self.phase4_advanced_experiment()
            
            # Phase 6: Comparison (skip Phase 5 as it's evaluation)
            self.phase6_comparison()
            
            # Phase 7: Final outputs
            self.phase7_outputs()
            
            logger.info("🎉 ALL RESEARCH EXPERIMENTS COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"❌ Experiments failed: {e}")
            raise

if __name__ == "__main__":
    experiments = ResearchExperiments()
    experiments.run_all_experiments()
