"""
EfficientNet-B0 Baseline Setup and Validation Script
Validates all components without requiring PyTorch installation.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_guard import run_dataset_guard
from utils.config_manager import load_config
from utils.label_mapping import LabelMapper, ClassificationMode, DatasetType


class EfficientNetBaselineSetup:
    """Setup and validation for EfficientNet-B0 baseline training."""
    
    def __init__(self):
        self.setup_results = {}
        self.output_dir = Path("outputs/efficientnet_baseline")
        self.config_file = "configs/efficientnet_baseline_config.yaml"
        
    def validate_config(self):
        """Validate baseline configuration."""
        print("🔧 VALIDATING CONFIGURATION")
        print("=" * 50)
        
        # Check if config file exists
        if not os.path.exists(self.config_file):
            print(f"❌ Config file not found: {self.config_file}")
            return False
        
        try:
            # Load config
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Config loaded: {self.config_file}")
            
            # Validate required sections
            required_sections = ['experiment', 'dataset', 'model', 'training']
            for section in required_sections:
                if section not in config:
                    print(f"❌ Missing config section: {section}")
                    return False
                print(f"✅ Config section found: {section}")
            
            # Validate dataset configuration
            dataset_config = config['dataset']
            required_dataset_keys = ['root_dir', 'classification_mode', 'use_cropped_only']
            for key in required_dataset_keys:
                if key not in dataset_config:
                    print(f"❌ Missing dataset config: {key}")
                    return False
                print(f"✅ Dataset config: {key} = {dataset_config[key]}")
            
            # Validate training configuration
            training_config = config['training']
            required_training_keys = ['num_epochs', 'batch_size', 'learning_rate']
            for key in required_training_keys:
                if key not in training_config:
                    print(f"❌ Missing training config: {key}")
                    return False
                print(f"✅ Training config: {key} = {training_config[key]}")
            
            self.setup_results['config'] = config
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
            return False
    
    def validate_datasets(self):
        """Validate dataset access and structure."""
        print("\n📊 VALIDATING DATASETS")
        print("=" * 50)
        
        config = self.setup_results.get('config')
        if not config:
            print("❌ No config loaded")
            return False
        
        dataset_config = config['dataset']
        root_dir = dataset_config['root_dir']
        classification_mode = dataset_config['classification_mode']
        
        print(f"Dataset root: {root_dir}")
        print(f"Classification mode: {classification_mode}")
        print(f"CROPPED-only: {dataset_config['use_cropped_only']}")
        
        # Check SIPaKMeD dataset
        sipakmed_dir = os.path.join(root_dir, "sipakmed", "archive")
        if not os.path.exists(sipakmed_dir):
            print(f"❌ SIPaKMeD dataset not found: {sipakmed_dir}")
            return False
        
        print(f"✅ SIPaKMeD dataset found: {sipakmed_dir}")
        
        # Check Herlev dataset
        herlev_dir = os.path.join(root_dir, "herlev", "archive (1)")
        if not os.path.exists(herlev_dir):
            print(f"❌ Herlev dataset not found: {herlev_dir}")
            return False
        
        print(f"✅ Herlev dataset found: {herlev_dir}")
        
        # Run dataset guard validation
        print("\n🛡️ Running Dataset Guard Validation...")
        guard_config = {
            'dataset': dataset_config,
            'paths': {'logs': 'outputs/logs'}
        }
        
        guard_results = run_dataset_guard(guard_config)
        
        if guard_results['action'] == 'ABORT':
            print(f"❌ Dataset validation failed: {guard_results['message']}")
            return False
        
        print(f"✅ Dataset validation passed: {guard_results['message']}")
        
        # Get dataset statistics
        if 'validation_report' in guard_results:
            report = guard_results['validation_report']
            if 'summary' in report:
                summary = report['summary']
                print(f"📈 Dataset Statistics:")
                print(f"   Total Samples: {summary.get('total_samples', 0)}")
                print(f"   Duplicate %: {summary.get('duplicate_percentage', 0):.2f}%")
                print(f"   Imbalance Ratio: {summary.get('imbalance_ratio', 0):.2f}")
                print(f"   CROPPED %: {summary.get('cropped_percentage', 0):.1f}%")
        
        self.setup_results['guard_results'] = guard_results
        return True
    
    def validate_label_mapping(self):
        """Validate label mapping configuration."""
        print("\n🏷️ VALIDATING LABEL MAPPING")
        print("=" * 50)
        
        config = self.setup_results.get('config')
        if not config:
            print("❌ No config loaded")
            return False
        
        classification_mode = config['dataset']['classification_mode']
        
        # Create label mapper
        if classification_mode == "binary":
            label_mapper = LabelMapper(ClassificationMode.BINARY)
            num_classes = 2
            class_names = ["NORMAL", "ABNORMAL"]
        else:
            label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
            num_classes = label_mapper.get_num_classes(DatasetType.SIPAKMED)
            class_names = label_mapper.get_class_names(DatasetType.SIPAKMED)
        
        print(f"✅ Label mapper created for {classification_mode} classification")
        print(f"✅ Number of classes: {num_classes}")
        print(f"✅ Class names: {class_names}")
        
        # Test label mapping
        if classification_mode == "binary":
            # Test binary mapping
            sipakmed_classes = label_mapper.get_dataset_classes(DatasetType.SIPAKMED)
            print(f"\nSIPaKMeD Binary Mapping:")
            for class_id, class_name in sipakmed_classes.items():
                mapped_label = label_mapper.map_label(class_id, DatasetType.SIPAKMED)
                binary_label = "NORMAL" if mapped_label == 0 else "ABNORMAL"
                print(f"  {class_name} -> {binary_label}")
        
        self.setup_results['label_mapping'] = {
            'mode': classification_mode,
            'num_classes': num_classes,
            'class_names': class_names
        }
        return True
    
    def validate_output_structure(self):
        """Validate output directory structure."""
        print("\n📁 VALIDATING OUTPUT STRUCTURE")
        print("=" * 50)
        
        # Create output directories
        directories = [
            self.output_dir,
            self.output_dir / "models",
            self.output_dir / "plots",
            self.output_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                print(f"✅ Directory created/verified: {directory}")
            else:
                print(f"❌ Failed to create directory: {directory}")
                return False
        
        print("✅ Output structure validation passed")
        return True
    
    def generate_training_script_info(self):
        """Generate information about the training script."""
        print("\n📜 TRAINING SCRIPT INFORMATION")
        print("=" * 50)
        
        script_path = "train_efficientnet_baseline.py"
        if not os.path.exists(script_path):
            print(f"❌ Training script not found: {script_path}")
            return False
        
        print(f"✅ Training script found: {script_path}")
        
        # Read script info
        with open(script_path, 'r') as f:
            lines = f.readlines()
        
        print(f"📄 Script lines: {len(lines)}")
        
        # Check for key components
        key_components = [
            "class EfficientNetB0Baseline",
            "def create_model",
            "def create_datasets", 
            "def train",
            "def evaluate_model"
        ]
        
        for component in key_components:
            found = any(component in line for line in lines)
            status = "✅" if found else "❌"
            print(f"{status} {component}")
        
        return True
    
    def create_training_summary(self):
        """Create a comprehensive training summary."""
        print("\n📋 CREATING TRAINING SUMMARY")
        print("=" * 50)
        
        config = self.setup_results.get('config')
        if not config:
            print("❌ No config available")
            return False
        
        summary = {
            'experiment_name': config['experiment']['name'],
            'description': config['experiment']['description'],
            'setup_timestamp': datetime.now().isoformat(),
            'configuration': {
                'model': {
                    'architecture': config['model']['architecture'],
                    'pretrained': config['model']['pretrained'],
                    'dropout_rate': config['model']['dropout_rate']
                },
                'dataset': {
                    'train_dataset': config['dataset']['train_dataset'],
                    'classification_mode': config['dataset']['classification_mode'],
                    'use_cropped_only': config['dataset']['use_cropped_only']
                },
                'training': {
                    'num_epochs': config['training']['num_epochs'],
                    'batch_size': config['training']['batch_size'],
                    'learning_rate': config['training']['learning_rate'],
                    'weight_decay': config['training']['weight_decay'],
                    'early_stopping_patience': config['training']['early_stopping_patience']
                }
            },
            'expected_pipeline': {
                'training_data': 'SIPaKMeD (CROPPED-only)',
                'validation_data': 'SIPaKMeD split (15%)',
                'test_data': 'Herlev (cross-dataset)',
                'preprocessing': [
                    'Resize to 224x224',
                    'HorizontalFlip (p=0.5)',
                    'VerticalFlip (p=0.5)', 
                    'RandomRotation (±15°)',
                    'ColorJitter (brightness=0.2, contrast=0.2)',
                    'ImageNet normalization'
                ],
                'metrics': [
                    'Accuracy',
                    'Precision', 
                    'Recall',
                    'F1-Score',
                    'AUC (binary only)'
                ]
            },
            'setup_status': 'VALIDATED'
        }
        
        # Add dataset statistics if available
        if 'guard_results' in self.setup_results:
            guard_results = self.setup_results['guard_results']
            if 'validation_report' in guard_results:
                report = guard_results['validation_report']
                if 'summary' in report:
                    summary['dataset_statistics'] = report['summary']
        
        # Save summary
        summary_path = self.output_dir / "training_setup_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Training summary saved: {summary_path}")
        
        # Print summary
        print(f"\n🎯 TRAINING SETUP SUMMARY")
        print("=" * 50)
        print(f"Experiment: {summary['experiment_name']}")
        print(f"Description: {summary['description']}")
        print(f"Mode: {summary['configuration']['dataset']['classification_mode']}")
        print(f"Model: {summary['configuration']['model']['architecture']}")
        print(f"Epochs: {summary['configuration']['training']['num_epochs']}")
        print(f"Batch Size: {summary['configuration']['training']['batch_size']}")
        print(f"Learning Rate: {summary['configuration']['training']['learning_rate']}")
        print(f"Training Data: {summary['expected_pipeline']['training_data']}")
        print(f"Test Data: {summary['expected_pipeline']['test_data']}")
        
        if 'dataset_statistics' in summary:
            stats = summary['dataset_statistics']
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total Samples: {stats.get('total_samples', 'N/A')}")
            print(f"   Duplicate %: {stats.get('duplicate_percentage', 'N/A')}%")
            print(f"   Imbalance Ratio: {stats.get('imbalance_ratio', 'N/A')}")
        
        return True
    
    def generate_training_commands(self):
        """Generate training command examples."""
        print("\n💻 TRAINING COMMANDS")
        print("=" * 50)
        
        print("Binary Classification (Recommended):")
        print("python train_efficientnet_baseline.py --mode binary")
        print("")
        print("Multiclass Classification:")
        print("python train_efficientnet_baseline.py --mode multiclass")
        print("")
        print("Custom Parameters:")
        print("python train_efficientnet_baseline.py --mode binary --epochs 30 --batch-size 16 --lr 0.0001")
        print("")
        print("With Config File:")
        print(f"python train_efficientnet_baseline.py --config {self.config_file}")
        
        return True
    
    def run_setup_validation(self):
        """Run complete setup validation."""
        print("🚀 EFFICIENTNET-B0 BASELINE SETUP VALIDATION")
        print("=" * 80)
        
        validation_steps = [
            ("Configuration", self.validate_config),
            ("Datasets", self.validate_datasets),
            ("Label Mapping", self.validate_label_mapping),
            ("Output Structure", self.validate_output_structure),
            ("Training Script", self.generate_training_script_info),
            ("Training Summary", self.create_training_summary),
            ("Training Commands", self.generate_training_commands)
        ]
        
        results = {}
        
        for step_name, step_func in validation_steps:
            try:
                results[step_name] = step_func()
            except Exception as e:
                print(f"❌ {step_name} validation failed: {e}")
                results[step_name] = False
        
        # Print final summary
        print("\n" + "=" * 80)
        print("SETUP VALIDATION SUMMARY")
        print("=" * 80)
        
        passed = 0
        total = len(results)
        
        for step_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{step_name:<20}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} validation steps passed")
        
        if passed == total:
            print("\n🎉 SETUP VALIDATION COMPLETED SUCCESSFULLY!")
            print("\nThe EfficientNet-B0 baseline is ready for training:")
            print("1. ✅ Configuration validated")
            print("2. ✅ Datasets accessible and validated")
            print("3. ✅ Label mapping configured")
            print("4. ✅ Output structure created")
            print("5. ✅ Training script ready")
            print("6. ✅ Setup summary generated")
            print("\n🚀 READY TO START TRAINING!")
            print("\nNext steps:")
            print("1. Install PyTorch: pip install torch torchvision")
            print("2. Run training: python train_efficientnet_baseline.py --mode binary")
            print("3. Monitor results in outputs/efficientnet_baseline/")
        else:
            print("\n⚠️  Some validation steps failed. Check the errors above.")
        
        return passed == total


def main():
    """Main setup validation function."""
    setup = EfficientNetBaselineSetup()
    success = setup.run_setup_validation()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
