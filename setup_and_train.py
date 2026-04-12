"""
Complete End-to-End Training Pipeline Setup and Execution
EfficientNet-B0 training with full environment setup and validation.
"""

import os
import sys
import subprocess
import importlib
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_guard import run_dataset_guard
from utils.config_manager import load_config


class EndToEndTrainingSetup:
    """Complete pipeline setup and training execution."""
    
    def __init__(self):
        self.setup_log = []
        self.training_results = {}
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        self.setup_log.append({"time": timestamp, "message": message, "level": level})
    
    def check_python_version(self):
        """Check Python version compatibility."""
        self.log("🐍 CHECKING PYTHON VERSION")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            self.log(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
            return True
        else:
            self.log(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires ≥3.9")
            return False
    
    def install_dependencies(self):
        """Install required packages."""
        self.log("📦 INSTALLING DEPENDENCIES")
        
        packages = [
            "torch",
            "torchvision", 
            "torchaudio",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "pillow",
            "opencv-python",
            "tqdm",
            "pyyaml"
        ]
        
        failed_packages = []
        
        for package in packages:
            try:
                self.log(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    self.log(f"✅ {package} installed successfully")
                else:
                    self.log(f"❌ Failed to install {package}: {result.stderr}")
                    failed_packages.append(package)
                    
            except subprocess.TimeoutExpired:
                self.log(f"❌ Timeout installing {package}")
                failed_packages.append(package)
            except Exception as e:
                self.log(f"❌ Error installing {package}: {e}")
                failed_packages.append(package)
        
        if failed_packages:
            self.log(f"❌ Failed to install: {', '.join(failed_packages)}")
            return False
        else:
            self.log("✅ All dependencies installed successfully")
            return True
    
    def verify_pytorch_installation(self):
        """Verify PyTorch installation and GPU availability."""
        self.log("🔥 VERIFYING PYTORCH INSTALLATION")
        
        try:
            import torch
            import torchvision
            
            self.log(f"✅ PyTorch {torch.__version__} installed")
            self.log(f"✅ TorchVision {torchvision.__version__} installed")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.log(f"✅ CUDA available - {gpu_count} GPU(s)")
                self.log(f"✅ Primary GPU: {gpu_name}")
                return True, "cuda"
            else:
                self.log("⚠️ CUDA not available - using CPU")
                return True, "cpu"
                
        except ImportError as e:
            self.log(f"❌ PyTorch import failed: {e}")
            return False, None
        except Exception as e:
            self.log(f"❌ PyTorch verification failed: {e}")
            return False, None
    
    def verify_other_dependencies(self):
        """Verify other critical dependencies."""
        self.log("🔍 VERIFYING OTHER DEPENDENCIES")
        
        dependencies = {
            "numpy": "np",
            "pandas": "pd", 
            "matplotlib": "plt",
            "seaborn": "sns",
            "sklearn": "sklearn",
            "PIL": "Image",
            "cv2": "cv2",
            "tqdm": "tqdm",
            "yaml": "yaml"
        }
        
        failed_deps = []
        
        for package, alias in dependencies.items():
            try:
                importlib.import_module(package)
                self.log(f"✅ {package} imported successfully")
            except ImportError:
                self.log(f"❌ {package} import failed")
                failed_deps.append(package)
        
        if failed_deps:
            self.log(f"❌ Failed dependencies: {', '.join(failed_deps)}")
            return False
        else:
            self.log("✅ All dependencies verified")
            return True
    
    def run_dataset_validation(self):
        """Run dataset guard validation."""
        self.log("🛡️ RUNNING DATASET GUARD VALIDATION")
        
        try:
            # Create validation config
            config = {
                'dataset': {
                    'root_dir': 'datasets',
                    'train_dataset': 'sipakmed',
                    'classification_mode': 'binary',
                    'use_cropped_only': True,
                    'run_validation': True,
                    'validation_strict_mode': True
                },
                'paths': {'logs': 'outputs/logs'}
            }
            
            # Run validation
            guard_results = run_dataset_guard(config)
            
            if guard_results['action'] == 'ABORT':
                self.log(f"❌ Dataset validation FAILED: {guard_results['message']}")
                self.log("🛑 TRAINING STOPPED - Dataset issues detected")
                return False, guard_results
            
            elif guard_results['action'] == 'PROCEED_WITH_WARNING':
                self.log(f"⚠️ Dataset validation WARNINGS: {guard_results['message']}")
                self.log("🚀 Proceeding with caution...")
                return True, guard_results
            
            else:
                self.log(f"✅ Dataset validation PASSED: {guard_results['message']}")
                return True, guard_results
                
        except Exception as e:
            self.log(f"❌ Dataset validation error: {e}")
            return False, None
    
    def setup_training_environment(self):
        """Setup training directories and configuration."""
        self.log("🏗️ SETTING UP TRAINING ENVIRONMENT")
        
        # Create output directories
        directories = [
            "outputs/efficientnet_end2end",
            "outputs/efficientnet_end2end/models",
            "outputs/efficientnet_end2end/plots", 
            "outputs/efficientnet_end2end/logs",
            "outputs/logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.log(f"✅ Directory ready: {directory}")
        
        # Create training configuration
        config = {
            'experiment': {
                'name': 'efficientnet_b0_end2end',
                'description': 'End-to-End EfficientNet-B0 training with full pipeline',
                'output_dir': 'outputs/efficientnet_end2end'
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
                'num_epochs': 30,
                'batch_size': 16,
                'learning_rate': 0.0001,
                'weight_decay': 0.00001,
                'early_stopping_patience': 5,
                'label_smoothing': 0.1,
                'unfreeze_epoch': 5,
                'num_workers': 4
            },
            'preprocessing': {
                'image_size': 224,
                'train_augmentations': {
                    'resize_size': 256,
                    'center_crop': 224,
                    'horizontal_flip_p': 0.5,
                    'vertical_flip_p': 0.5,
                    'rotation_degrees': 15,
                    'random_resized_crop': {'scale': [0.9, 1.0]},
                    'color_jitter': {
                        'brightness': 0.2,
                        'contrast': 0.2,
                        'saturation': 0.2,
                        'hue': 0.1
                    },
                    'gaussian_blur': {
                        'p': 0.2,
                        'kernel_size': 3,
                        'sigma_range': [0.1, 1.5]
                    }
                }
            }
        }
        
        # Save configuration
        config_path = "outputs/efficientnet_end2end/training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.log(f"✅ Training configuration saved: {config_path}")
        return config
    
    def execute_training(self, config):
        """Execute the actual training."""
        self.log("🚀 STARTING EFFICIENTNET-B0 TRAINING")
        
        try:
            # Import training modules
            from train_efficientnet_optimized import EfficientNetB0Optimized
            
            # Create trainer
            trainer = EfficientNetB0Optimized(config)
            
            # Start training
            success = trainer.train()
            
            if success:
                self.log("✅ Training completed successfully")
                self.training_results = trainer.results
                return True
            else:
                self.log("❌ Training failed")
                return False
                
        except Exception as e:
            self.log(f"❌ Training execution error: {e}")
            return False
    
    def generate_final_report(self, setup_success, training_success, guard_results=None):
        """Generate comprehensive final report."""
        self.log("📊 GENERATING FINAL REPORT")
        
        total_time = time.time() - self.start_time
        
        report = {
            'pipeline_execution': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_minutes': total_time / 60,
                'setup_successful': setup_success,
                'training_successful': training_success
            },
            'environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            },
            'dataset_validation': guard_results if guard_results else None,
            'training_results': self.training_results if training_success else None,
            'setup_log': self.setup_log
        }
        
        # Save report
        report_path = "outputs/efficientnet_end2end/pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"✅ Final report saved: {report_path}")
        
        # Print summary
        self.log("\n" + "="*80)
        self.log("🎉 PIPELINE EXECUTION SUMMARY")
        self.log("="*80)
        
        if setup_success and training_success:
            self.log("✅ COMPLETE SUCCESS - All steps completed successfully")
            
            if self.training_results:
                val_acc = self.training_results.get('validation_results', {}).get('accuracy', 0)
                test_acc = self.training_results.get('test_results', {}).get('accuracy', 0)
                
                self.log(f"📊 Final Validation Accuracy: {val_acc:.4f}")
                self.log(f"🎯 Final Herlev Accuracy: {test_acc:.4f}")
                
                if val_acc >= 0.85:
                    self.log("🏆 TARGET ACHIEVED: Validation accuracy ≥85%")
                else:
                    self.log("⚠️ TARGET MISSED: Validation accuracy <85%")
                
                generalization_gap = abs(val_acc - test_acc)
                if generalization_gap < 0.2:
                    self.log("✅ GOOD GENERALIZATION: Gap <20%")
                else:
                    self.log("⚠️ POOR GENERALIZATION: Gap ≥20%")
        
        else:
            self.log("❌ PIPELINE FAILED - Check errors above")
        
        self.log(f"⏱️ Total Duration: {total_time/60:.1f} minutes")
        self.log("="*80)
        
        return report
    
    def run_complete_pipeline(self):
        """Execute the complete end-to-end pipeline."""
        self.log("🚀 STARTING COMPLETE END-TO-END TRAINING PIPELINE")
        self.log("Objective: EfficientNet-B0 training with 85-95% validation accuracy")
        self.log("="*80)
        
        # Step 0: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 1: Install dependencies
        if not self.install_dependencies():
            self.log("❌ Dependency installation failed - aborting")
            return False
        
        # Step 2: Verify PyTorch
        pytorch_ok, device = self.verify_pytorch_installation()
        if not pytorch_ok:
            self.log("❌ PyTorch verification failed - aborting")
            return False
        
        # Step 3: Verify other dependencies
        if not self.verify_other_dependencies():
            self.log("❌ Dependency verification failed - aborting")
            return False
        
        # Step 4: Dataset validation
        validation_ok, guard_results = self.run_dataset_validation()
        if not validation_ok:
            self.log("❌ Dataset validation failed - training aborted")
            return self.generate_final_report(False, False, guard_results)
        
        # Step 5: Setup training environment
        config = self.setup_training_environment()
        
        # Step 6: Execute training
        training_success = self.execute_training(config)
        
        # Step 7: Generate final report
        report = self.generate_final_report(True, training_success, guard_results)
        
        return training_success


def main():
    """Main pipeline execution."""
    print("🚀 EFFICIENTNET-B0 END-TO-END TRAINING PIPELINE")
    print("="*80)
    print("This script will:")
    print("1. ✅ Install all required dependencies")
    print("2. ✅ Verify environment and PyTorch installation")
    print("3. ✅ Run dataset guard validation")
    print("4. ✅ Setup training configuration")
    print("5. ✅ Execute EfficientNet-B0 training")
    print("6. ✅ Generate comprehensive report")
    print("="*80)
    print()
    
    # Run pipeline
    pipeline = EndToEndTrainingSetup()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("Check results in: outputs/efficientnet_end2end/")
    else:
        print("\n❌ PIPELINE FAILED!")
        print("Check the log above for error details.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
