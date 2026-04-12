#!/usr/bin/env python3
"""
Pipeline Verification and Diagnostics
Comprehensive validation of dataset and classification setup
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from datasets.dataset import DatasetManager
from utils.config_manager import ConfigManager
from utils.label_mapping import LabelMapper, DatasetType, ClassificationMode

class PipelineVerifier:
    """Comprehensive pipeline verification."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.issues = []
        self.warnings = []
        self.errors = []
        
    def log_info(self, message):
        print(f"ℹ️  {message}")
    
    def log_warning(self, message):
        warning_msg = f"⚠️  {message}"
        print(warning_msg)
        self.warnings.append(warning_msg)
    
    def log_error(self, message):
        error_msg = f"❌ {message}"
        print(error_msg)
        self.errors.append(error_msg)
    
    def log_success(self, message):
        print(f"✅ {message}")
    
    def phase1_sipakmed_verification(self):
        """PHASE 1: SIPaKMeD dataset verification."""
        print("\n" + "="*80)
        print("PHASE 1: SIPaKMeD DATASET VERIFICATION")
        print("="*80)
        
        sipakmed_path = Path("datasets/sipakmed")
        
        if not sipakmed_path.exists():
            self.log_error("SIPaKMeD dataset path does not exist")
            return
        
        # Scan entire dataset
        print("📁 Scanning SIPaKMeD dataset...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        total_images = 0
        class_images = defaultdict(list)
        
        for root, dirs, files in os.walk(sipakmed_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    total_images += 1
                    # Determine class from path
                    rel_path = Path(root).relative_to(sipakmed_path)
                    class_name = str(rel_path).replace(os.sep, '_')
                    class_images[class_name].append(Path(root) / file)
        
        print(f"📊 Total images found: {total_images}")
        
        if total_images < 4000:
            self.log_warning(f"Dataset size ({total_images}) is smaller than expected (~5000+)")
        else:
            self.log_success(f"Dataset size ({total_images}) is within expected range")
        
        # Expected classes
        expected_classes = [
            'superficial_intermediate',
            'parabasal', 
            'koilocytotic',
            'metaplastic',
            'dyskeratotic'
        ]
        
        print("\n📋 Class Distribution:")
        print(f"{'Class':<25} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        
        found_classes = []
        for expected_class in expected_classes:
            # Look for this class in the found classes
            class_count = 0
            for found_class, images in class_images.items():
                if expected_class.lower() in found_class.lower():
                    class_count = len(images)
                    found_classes.append(expected_class)
                    break
            
            percentage = (class_count / total_images * 100) if total_images > 0 else 0
            print(f"{expected_class:<25} {class_count:<8} {percentage:<10.2f}%")
        
        # Check for missing classes
        missing_classes = set(expected_classes) - set(found_classes)
        if missing_classes:
            self.log_error(f"Missing classes: {missing_classes}")
        
        # Check for unexpected classes
        unexpected_classes = set(class_images.keys()) - set(expected_classes)
        if unexpected_classes:
            self.log_warning(f"Unexpected class folders found: {unexpected_classes}")
        
        # Check for empty folders
        empty_classes = [cls for cls, imgs in class_images.items() if len(imgs) == 0]
        if empty_classes:
            self.log_error(f"Empty class folders: {empty_classes}")
        
        # Store for later phases
        self.sipakmed_total = total_images
        self.sipakmed_class_distribution = {cls: len(imgs) for cls, imgs in class_images.items()}
        
        self.log_success("SIPaKMeD dataset verification completed")
    
    def phase2_herlev_verification(self):
        """PHASE 2: Herlev dataset verification."""
        print("\n" + "="*80)
        print("PHASE 2: HERLEV DATASET VERIFICATION")
        print("="*80)
        
        herlev_path = Path("datasets/herlev")
        
        if not herlev_path.exists():
            self.log_error("Herlev dataset path does not exist")
            return
        
        # Scan Herlev dataset
        print("📁 Scanning Herlev dataset...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        total_images = 0
        class_images = defaultdict(list)
        
        for root, dirs, files in os.walk(herlev_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    total_images += 1
                    rel_path = Path(root).relative_to(herlev_path)
                    class_name = str(rel_path).replace(os.sep, '_')
                    class_images[class_name].append(Path(root) / file)
        
        print(f"📊 Total images found: {total_images}")
        
        print("\n📋 Class Distribution:")
        print(f"{'Class':<25} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        
        for class_name, images in sorted(class_images.items()):
            percentage = (len(images) / total_images * 100) if total_images > 0 else 0
            print(f"{class_name:<25} {len(images):<8} {percentage:<10.2f}%")
        
        # Store for later phases
        self.herlev_total = total_images
        self.herlev_class_distribution = {cls: len(imgs) for cls, imgs in class_images.items()}
        
        self.log_success("Herlev dataset verification completed")
    
    def phase3_dataloader_verification(self):
        """PHASE 3: Data loader verification."""
        print("\n" + "="*80)
        print("PHASE 3: DATA LOADER VERIFICATION")
        print("="*80)
        
        try:
            # Initialize dataset manager
            dataset_manager = DatasetManager(
                root_dir=self.config.get('dataset', {}).get('root_dir', 'datasets'),
                classification_mode="multiclass",  # Use multiclass to see all classes
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42
            )
            
            # Load datasets
            dataset_manager.load_datasets()
            
            print("📊 Dataset Loader Analysis:")
            
            # Check SIPaKMeD loading
            if hasattr(dataset_manager, 'sipakmed_dataset'):
                sipakmed_loaded = len(dataset_manager.sipakmed_dataset)
                print(f"  SIPaKMeD loaded: {sipakmed_loaded} samples")
                
                if sipakmed_loaded < self.sipakmed_total * 0.8:  # If loading less than 80%
                    self.log_warning(f"Pipeline is not using full SIPaKMeD dataset: {sipakmed_loaded}/{self.sipakmed_total}")
                else:
                    self.log_success(f"Pipeline loads most of SIPaKMeD dataset: {sipakmed_loaded}/{self.sipakmed_total}")
            else:
                self.log_error("SIPaKMeD dataset not found in dataset manager")
            
            # Check Herlev loading
            if hasattr(dataset_manager, 'herlev_dataset'):
                herlev_loaded = len(dataset_manager.herlev_dataset)
                print(f"  Herlev loaded: {herlev_loaded} samples")
                
                if herlev_loaded < self.herlev_total * 0.8:
                    self.log_warning(f"Pipeline is not using full Herlev dataset: {herlev_loaded}/{self.herlev_total}")
                else:
                    self.log_success(f"Pipeline loads most of Herlev dataset: {herlev_loaded}/{self.herlev_total}")
            else:
                self.log_error("Herlev dataset not found in dataset manager")
            
            # Store dataset manager for later phases
            self.dataset_manager = dataset_manager
            
        except Exception as e:
            self.log_error(f"Failed to initialize dataset manager: {e}")
    
    def phase4_split_validation(self):
        """PHASE 4: Split validation."""
        print("\n" + "="*80)
        print("PHASE 4: SPLIT VALIDATION")
        print("="*80)
        
        if not hasattr(self, 'dataset_manager'):
            self.log_error("Dataset manager not available for split validation")
            return
        
        try:
            # Create splits
            self.dataset_manager.create_splits()
            
            print("📊 SIPaKMeD Split Analysis:")
            
            # Check SIPaKMeD splits
            if 'sipakmed' in self.dataset_manager.train_datasets:
                train_size = len(self.dataset_manager.train_datasets['sipakmed'])
                val_size = len(self.dataset_manager.val_datasets['sipakmed'])
                test_size = len(self.dataset_manager.test_datasets['sipakmed'])
                total_split = train_size + val_size + test_size
                
                train_pct = train_size / total_split * 100
                val_pct = val_size / total_split * 100
                test_pct = test_size / total_split * 100
                
                print(f"  Train: {train_size} samples ({train_pct:.1f}%)")
                print(f"  Val: {val_size} samples ({val_pct:.1f}%)")
                print(f"  Test: {test_size} samples ({test_pct:.1f}%)")
                print(f"  Total: {total_split} samples")
                
                # Validate split ratios
                if abs(train_pct - 70) > 5:
                    self.log_warning(f"Train split ratio ({train_pct:.1f}%) deviates from expected (70%)")
                if abs(val_pct - 15) > 5:
                    self.log_warning(f"Val split ratio ({val_pct:.1f}%) deviates from expected (15%)")
                if abs(test_pct - 15) > 5:
                    self.log_warning(f"Test split ratio ({test_pct:.1f}%) deviates from expected (15%)")
                
                # Check for data leakage
                self._check_data_leakage('sipakmed')
            else:
                self.log_error("SIPaKMeD splits not found")
            
        except Exception as e:
            self.log_error(f"Failed to validate splits: {e}")
    
    def phase5_classification_mapping_check(self):
        """PHASE 5: Classification mapping check."""
        print("\n" + "="*80)
        print("PHASE 5: CLASSIFICATION MAPPING CHECK")
        print("="*80)
        
        try:
            # Test binary classification
            print("🔍 Binary Classification Mode:")
            binary_mapper = LabelMapper(
                dataset_type=DatasetType.SIPAKMED,
                classification_mode=ClassificationMode.BINARY
            )
            
            print("  Class mapping:")
            for class_idx, class_name in binary_mapper.class_to_idx.items():
                print(f"    {class_idx} → {class_name}")
            
            # Test multiclass classification
            print("\n🔍 Multiclass Classification Mode:")
            multiclass_mapper = LabelMapper(
                dataset_type=DatasetType.SIPAKMED,
                classification_mode=ClassificationMode.MULTICLASS
            )
            
            print("  Class mapping:")
            for class_idx, class_name in multiclass_mapper.class_to_idx.items():
                print(f"    {class_idx} → {class_name}")
            
            # Sample validation
            print("\n🎯 Sample Validation (10 random samples):")
            if hasattr(self, 'dataset_manager') and 'sipakmed' in self.dataset_manager.train_datasets:
                dataset = self.dataset_manager.train_datasets['sipakmed']
                
                import random
                random.seed(42)
                sample_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
                
                for i, idx in enumerate(sample_indices):
                    try:
                        sample = dataset[idx]
                        if len(sample) >= 3:
                            image, label, path = sample[:3]
                        else:
                            image, label = sample[:2]
                            path = f"Sample {idx}"
                        
                        class_name = binary_mapper.idx_to_class[label.item()] if hasattr(label, 'item') else binary_mapper.idx_to_class[label]
                        print(f"  {i+1}. {path} → Label: {label} ({class_name})")
                    except Exception as e:
                        print(f"  {i+1}. Error processing sample {idx}: {e}")
            
            self.log_success("Classification mapping verification completed")
            
        except Exception as e:
            self.log_error(f"Failed to verify classification mapping: {e}")
    
    def phase6_imbalance_analysis(self):
        """PHASE 6: Class imbalance analysis."""
        print("\n" + "="*80)
        print("PHASE 6: CLASS IMBALANCE ANALYSIS")
        print("="*80)
        
        if hasattr(self, 'sipakmed_class_distribution'):
            print("📊 SIPaKMeD Class Imbalance Analysis:")
            
            if self.sipakmed_class_distribution:
                max_count = max(self.sipakmed_class_distribution.values())
                min_count = min(self.sipakmed_class_distribution.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"  Largest class: {max_count} samples")
                print(f"  Smallest class: {min_count} samples")
                print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
                
                if imbalance_ratio > 2.0:
                    self.log_warning(f"Class imbalance detected ({imbalance_ratio:.2f}:1) - recommend class weights or focal loss")
                else:
                    self.log_success("Class distribution is relatively balanced")
                
                # Print detailed distribution
                print("\n  Detailed Distribution:")
                for class_name, count in sorted(self.sipakmed_class_distribution.items()):
                    percentage = count / sum(self.sipakmed_class_distribution.values()) * 100
                    print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        if hasattr(self, 'herlev_class_distribution'):
            print("\n📊 Herlev Class Imbalance Analysis:")
            
            if self.herlev_class_distribution:
                max_count = max(self.herlev_class_distribution.values())
                min_count = min(self.herlev_class_distribution.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"  Largest class: {max_count} samples")
                print(f"  Smallest class: {min_count} samples")
                print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
                
                if imbalance_ratio > 2.0:
                    self.log_warning(f"Herlev class imbalance detected ({imbalance_ratio:.2f}:1)")
    
    def phase7_summary_report(self):
        """PHASE 7: Summary report."""
        print("\n" + "="*80)
        print("PHASE 7: SUMMARY REPORT")
        print("="*80)
        
        print("📋 PIPELINE VERIFICATION SUMMARY")
        print("="*50)
        
        # Dataset sizes
        if hasattr(self, 'sipakmed_total'):
            print(f"📁 SIPaKMeD Dataset:")
            print(f"   Total images: {self.sipakmed_total}")
            if self.sipakmed_total >= 4000:
                print(f"   Status: ✅ Complete dataset")
            else:
                print(f"   Status: ⚠️  Incomplete dataset")
        
        if hasattr(self, 'herlev_total'):
            print(f"\n📁 Herlev Dataset:")
            print(f"   Total images: {self.herlev_total}")
        
        # Pipeline usage
        if hasattr(self, 'dataset_manager'):
            print(f"\n🔄 Pipeline Usage:")
            if hasattr(self.dataset_manager, 'sipakmed_dataset'):
                loaded = len(self.dataset_manager.sipakmed_dataset)
                total = self.sipakmed_total
                usage_pct = loaded / total * 100
                print(f"   SIPaKMeD: {loaded}/{total} ({usage_pct:.1f}%)")
                if usage_pct >= 80:
                    print(f"   Status: ✅ Using most of dataset")
                else:
                    print(f"   Status: ⚠️  Using subset of dataset")
        
        # Issues summary
        print(f"\n🚨 Issues Summary:")
        if self.errors:
            print(f"   Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"     {error}")
        else:
            print("   ✅ No critical errors found")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"     {warning}")
        else:
            print("   ✅ No warnings")
        
        # Final assessment
        print(f"\n🎯 Final Assessment:")
        if not self.errors and not self.warnings:
            print("   ✅ Pipeline is ready for training experiments")
        elif not self.errors and self.warnings:
            print("   ⚠️  Pipeline is mostly ready but review warnings")
        else:
            print("   ❌ Pipeline has critical issues that need addressing")
        
        print("="*80)
    
    def _check_data_leakage(self, dataset_name):
        """Check for data leakage between splits."""
        try:
            if dataset_name in self.dataset_manager.train_datasets:
                train_paths = set()
                for i in range(len(self.dataset_manager.train_datasets[dataset_name])):
                    sample = self.dataset_manager.train_datasets[dataset_name][i]
                    if len(sample) >= 3:
                        path = str(sample[2])
                    else:
                        continue
                    train_paths.add(path)
                
                val_paths = set()
                for i in range(len(self.dataset_manager.val_datasets[dataset_name])):
                    sample = self.dataset_manager.val_datasets[dataset_name][i]
                    if len(sample) >= 3:
                        path = str(sample[2])
                    else:
                        continue
                    val_paths.add(path)
                
                test_paths = set()
                for i in range(len(self.dataset_manager.test_datasets[dataset_name])):
                    sample = self.dataset_manager.test_datasets[dataset_name][i]
                    if len(sample) >= 3:
                        path = str(sample[2])
                    else:
                        continue
                    test_paths.add(path)
                
                # Check for overlaps
                train_val_overlap = train_paths & val_paths
                train_test_overlap = train_paths & test_paths
                val_test_overlap = val_paths & test_paths
                
                if train_val_overlap:
                    self.log_error(f"Data leakage: {len(train_val_overlap)} samples overlap between train and validation")
                if train_test_overlap:
                    self.log_error(f"Data leakage: {len(train_test_overlap)} samples overlap between train and test")
                if val_test_overlap:
                    self.log_error(f"Data leakage: {len(val_test_overlap)} samples overlap between validation and test")
                
                if not (train_val_overlap or train_test_overlap or val_test_overlap):
                    self.log_success("No data leakage detected between splits")
        
        except Exception as e:
            self.log_warning(f"Could not check for data leakage: {e}")
    
    def run_verification(self):
        """Run complete pipeline verification."""
        print("🚀 STARTING COMPREHENSIVE PIPELINE VERIFICATION")
        print("="*80)
        
        try:
            # Run all phases
            self.phase1_sipakmed_verification()
            self.phase2_herlev_verification()
            self.phase3_dataloader_verification()
            self.phase4_split_validation()
            self.phase5_classification_mapping_check()
            self.phase6_imbalance_analysis()
            self.phase7_summary_report()
            
            print("\n🎉 PIPELINE VERIFICATION COMPLETED")
            
        except Exception as e:
            self.log_error(f"Verification failed: {e}")
            raise

if __name__ == "__main__":
    verifier = PipelineVerifier()
    verifier.run_verification()
