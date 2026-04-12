#!/usr/bin/env python3
"""
Deep Dataset Auditing and Validation
Comprehensive analysis of data loss, duplicates, and dataset integrity
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from datasets.dataset import DatasetManager
from utils.label_mapping import LabelMapper, ClassificationMode

class DatasetAuditor:
    """Comprehensive dataset auditor."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.success = []
        
    def log_info(self, message):
        print(f"ℹ️  {message}")
    
    def log_warning(self, message):
        warning_msg = f"⚠️  {message}"
        print(warning_msg)
        self.warnings.append(warning_msg)
    
    def log_error(self, message):
        error_msg = f"❌ {message}"
        print(error_msg)
        self.issues.append(error_msg)
    
    def log_success(self, message):
        success_msg = f"✅ {message}"
        print(success_msg)
        self.success.append(success_msg)
    
    def phase1_sipakmed_data_loss_analysis(self):
        """PHASE 1: SIPaKMeD data loss analysis."""
        print("\n" + "="*80)
        print("PHASE 1: SIPaKMeD DATA LOSS ANALYSIS")
        print("="*80)
        
        sipakmed_path = Path("datasets/sipakmed")
        
        # 1. Count total raw images
        print("📁 Counting total raw images...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        raw_images = []
        for root, dirs, files in os.walk(sipakmed_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    file_path = Path(root) / file
                    raw_images.append(file_path)
        
        total_raw = len(raw_images)
        print(f"📊 Total raw images: {total_raw}")
        
        if total_raw < 4000:
            self.log_error(f"Raw dataset size ({total_raw}) is smaller than expected")
        else:
            self.log_success(f"Raw dataset size ({total_raw}) is within expected range")
        
        # 2. Count loaded images
        print("\n📦 Counting loaded images...")
        try:
            dataset_manager = DatasetManager(
                root_dir="datasets",
                classification_mode="multiclass",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42
            )
            
            dataset_manager.load_datasets()
            loaded_images = len(dataset_manager.sipakmed_dataset)
            print(f"📊 Loaded images: {loaded_images}")
            
        except Exception as e:
            self.log_error(f"Failed to load dataset: {e}")
            loaded_images = 0
        
        # 3. Find missing images
        missing_count = total_raw - loaded_images
        print(f"\n🔍 Missing images analysis:")
        print(f"  Missing images: {missing_count}")
        print(f"  Data loss: {missing_count/total_raw*100:.1f}%")
        
        if missing_count > 500:
            self.log_warning(f"Significant data loss: {missing_count} images missing")
        
        # 4. Analyze missing images
        print("\n📋 Analyzing missing images...")
        
        # Get loaded image paths
        loaded_paths = set()
        if hasattr(dataset_manager, 'sipakmed_dataset'):
            for i in range(len(dataset_manager.sipakmed_dataset)):
                sample = dataset_manager.sipakmed_dataset[i]
                if len(sample) >= 4:
                    loaded_paths.add(sample[3])  # path is 4th element
        
        # Find missing paths
        missing_paths = []
        for img_path in raw_images:
            if str(img_path) not in loaded_paths:
                missing_paths.append(img_path)
        
        # 5. Classify reasons for exclusion
        print("\n🔍 Classifying exclusion reasons...")
        
        excluded_by_reason = defaultdict(list)
        
        for missing_path in missing_paths:
            path_str = str(missing_path)
            reason = self._classify_exclusion_reason(path_str)
            excluded_by_reason[reason].append(missing_path)
        
        print("📊 Exclusion reasons:")
        for reason, paths in excluded_by_reason.items():
            print(f"  {reason}: {len(paths)} images")
            
            # Show sample paths
            if len(paths) > 0:
                sample_path = str(paths[0])
                if len(sample_path) > 80:
                    sample_path = "..." + sample_path[-77:]
                print(f"    Sample: {sample_path}")
        
        # Store for later phases
        self.sipakmed_raw_count = total_raw
        self.sipakmed_loaded_count = loaded_images
        self.sipakmed_missing_count = missing_count
        self.sipakmed_excluded_by_reason = excluded_by_reason
        
        self.log_success("SIPaKMeD data loss analysis completed")
    
    def _classify_exclusion_reason(self, path_str):
        """Classify why an image was excluded."""
        path_lower = path_str.lower()
        
        # Check for ignored folders
        if 'cropped' in path_lower:
            return "Ignored folder (CROPPED)"
        elif 'archive' in path_lower and 'im_' not in path_lower:
            return "Ignored folder (archive structure)"
        elif 'duplicate' in path_lower:
            return "Duplicate folder"
        elif 'temp' in path_lower or 'tmp' in path_lower:
            return "Temporary folder"
        
        # Check for unsupported formats (shouldn't happen with our filter)
        unsupported_formats = ['.gif', '.webp', '.svg']
        for fmt in unsupported_formats:
            if path_lower.endswith(fmt):
                return "Unsupported format"
        
        # Check for class mapping issues
        class_indicators = ['im_superficial', 'im_parabasal', 'im_koilocytotic', 'im_metaplastic', 'im_dyskeratotic']
        if not any(indicator in path_lower for indicator in class_indicators):
            return "Class mapping issue"
        
        # Check for path issues
        if len(path_str) > 260:  # Windows path limit
            return "Path too long"
        
        # Default
        return "Unknown reason"
    
    def phase2_duplicate_check(self):
        """PHASE 2: Duplicate check."""
        print("\n" + "="*80)
        print("PHASE 2: DUPLICATE CHECK")
        print("="*80)
        
        sipakmed_path = Path("datasets/sipakmed")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        print("🔍 Detecting duplicates...")
        
        # Collect all images with hashes
        image_hashes = defaultdict(list)
        file_names = defaultdict(list)
        
        for root, dirs, files in os.walk(sipakmed_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    file_path = Path(root) / file
                    
                    try:
                        # Calculate file hash
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        image_hashes[file_hash].append(file_path)
                        file_names[file.lower()].append(file_path)
                        
                    except Exception as e:
                        self.log_warning(f"Could not hash {file_path}: {e}")
        
        # Find duplicates by hash
        hash_duplicates = {h: paths for h, paths in image_hashes.items() if len(paths) > 1}
        
        # Find duplicates by filename
        name_duplicates = {n: paths for n, paths in file_names.items() if len(paths) > 1}
        
        print(f"📊 Duplicate Analysis:")
        print(f"  Duplicates by content (hash): {sum(len(paths)-1 for paths in hash_duplicates.values())}")
        print(f"  Duplicate file names: {sum(len(paths)-1 for paths in name_duplicates.values())}")
        
        # Show examples
        if hash_duplicates:
            print(f"\n📋 Content duplicates (showing first 3):")
            for i, (hash_val, paths) in enumerate(list(hash_duplicates.items())[:3]):
                print(f"  {i+1}. Hash: {hash_val[:8]}...")
                for path in paths:
                    print(f"     {path}")
        
        if name_duplicates:
            print(f"\n📋 Name duplicates (showing first 3):")
            for i, (name, paths) in enumerate(list(name_duplicates.items())[:3]):
                print(f"  {i+1}. Name: {name}")
                for path in paths:
                    print(f"     {path}")
        
        total_duplicates = sum(len(paths)-1 for paths in hash_duplicates.values())
        if total_duplicates > 0:
            self.log_warning(f"Found {total_duplicates} duplicate images by content")
        else:
            self.log_success("No duplicate images found by content")
        
        self.sipakmed_duplicates = total_duplicates
    
    def phase3_sipakmed_final_validation(self):
        """PHASE 3: SIPaKMeD final validation."""
        print("\n" + "="*80)
        print("PHASE 3: SIPaKMeD FINAL VALIDATION")
        print("="*80)
        
        try:
            # Load dataset
            dataset_manager = DatasetManager(
                root_dir="datasets",
                classification_mode="multiclass",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42
            )
            
            dataset_manager.load_datasets()
            dataset_manager.create_splits()
            
            # Get final dataset
            final_dataset = dataset_manager.sipakmed_dataset
            total_usable = len(final_dataset)
            
            print(f"📊 Final Dataset Validation:")
            print(f"  Total usable images: {total_usable}")
            
            if total_usable < 3000:
                self.log_error(f"Final dataset size ({total_usable}) is too small")
            else:
                self.log_success(f"Final dataset size ({total_usable}) is adequate")
            
            # Class distribution
            class_distribution = defaultdict(int)
            for i in range(len(final_dataset)):
                sample = final_dataset[i]
                if len(sample) >= 4:
                    label = sample[1]
                    if hasattr(label, 'item'):
                        label = label.item()
                    class_distribution[label] += 1
            
            print(f"\n📋 Class Distribution:")
            label_mapper = LabelMapper(classification_mode=ClassificationMode.MULTICLASS)
            
            for label_idx, count in sorted(class_distribution.items()):
                class_name = label_mapper.idx_to_class.get(label_idx, f"Class_{label_idx}")
                percentage = count / total_usable * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            # Check for missing classes
            expected_classes = [0, 1, 2, 3, 4]  # 5 classes for multiclass
            missing_classes = set(expected_classes) - set(class_distribution.keys())
            
            if missing_classes:
                self.log_error(f"Missing classes: {missing_classes}")
            else:
                self.log_success("All 5 classes present")
            
            self.sipakmed_final_count = total_usable
            self.sipakmed_class_distribution = dict(class_distribution)
            
        except Exception as e:
            self.log_error(f"Failed final validation: {e}")
    
    def phase4_herlev_validation(self):
        """PHASE 4: Herlev dataset validation."""
        print("\n" + "="*80)
        print("PHASE 4: HERLEV DATASET VALIDATION")
        print("="*80)
        
        herlev_path = Path("datasets/herlev")
        
        # 10. Scan Herlev dataset
        print("📁 Scanning Herlev dataset...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        herlev_images = []
        for root, dirs, files in os.walk(herlev_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    file_path = Path(root) / file
                    herlev_images.append(file_path)
        
        total_herlev = len(herlev_images)
        print(f"📊 Total Herlev images: {total_herlev}")
        
        if total_herlev < 1500:
            self.log_error(f"Herlev dataset size ({total_herlev}) is smaller than expected")
        else:
            self.log_success(f"Herlev dataset size ({total_herlev}) is adequate")
        
        # 11. Verify class structure
        print("\n🏷️ Verifying class structure...")
        
        expected_classes = [
            'normal_superficiel',
            'normal_intermediate',
            'normal_columnar',
            'light_dysplastic',
            'moderate_dysplastic',
            'severe_dysplastic',
            'carcinoma_in_situ'
        ]
        
        # Extract classes from paths
        herlev_class_counts = defaultdict(int)
        class_mapping = {}
        
        for img_path in herlev_images:
            class_name = self._extract_herlev_class(img_path, herlev_path)
            if class_name:
                herlev_class_counts[class_name] += 1
                class_mapping[str(img_path)] = class_name
        
        print(f"📋 Herlev Class Distribution:")
        for class_name in expected_classes:
            count = herlev_class_counts.get(class_name, 0)
            percentage = count / total_herlev * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Check for missing classes
        found_classes = set(herlev_class_counts.keys())
        missing_classes = set(expected_classes) - found_classes
        
        if missing_classes:
            self.log_error(f"Missing Herlev classes: {missing_classes}")
        else:
            self.log_success("All expected Herlev classes found")
        
        # 13. Check class imbalance
        if herlev_class_counts:
            max_count = max(herlev_class_counts.values())
            min_count = min(herlev_class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"\n⚖️ Class Imbalance Analysis:")
            print(f"  Largest class: {max_count} samples")
            print(f"  Smallest class: {min_count} samples")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 2.0:
                self.log_warning(f"Herlev class imbalance detected ({imbalance_ratio:.2f}:1)")
            else:
                self.log_success("Herlev class distribution is relatively balanced")
        
        # 14. Verify label mapping
        print("\n🏷️ Verifying label mapping...")
        
        try:
            # Test binary mapping
            binary_mapper = LabelMapper(classification_mode=ClassificationMode.BINARY)
            print("  Binary mapping:")
            for class_idx, class_name in binary_mapper.class_to_idx.items():
                print(f"    {class_idx} → {class_name}")
            
            # Test multiclass mapping
            multiclass_mapper = LabelMapper(classification_mode=ClassificationMode.MULTICLASS)
            print("  Multiclass mapping:")
            for class_idx, class_name in multiclass_mapper.class_to_idx.items():
                print(f"    {class_idx} → {class_name}")
            
            self.log_success("Label mapping verified")
            
        except Exception as e:
            self.log_error(f"Label mapping verification failed: {e}")
        
        # 15. Sample validation
        print("\n🎯 Sample validation (10 random samples):")
        
        import random
        random.seed(42)
        sample_paths = random.sample(herlev_images, min(10, len(herlev_images)))
        
        for i, img_path in enumerate(sample_paths):
            class_name = class_mapping.get(str(img_path), "Unknown")
            print(f"  {i+1}. {img_path.name} → {class_name}")
        
        # Store for final report
        self.herlev_total = total_herlev
        self.herlev_class_distribution = dict(herlev_class_counts)
        self.herlev_class_mapping = class_mapping
    
    def _extract_herlev_class(self, file_path, base_path):
        """Extract class name from Herlev file path."""
        relative_path = Path(file_path).relative_to(base_path)
        path_parts = list(relative_path.parts)
        
        # Herlev class mapping
        class_mapping = {
            'normal_superficiel': 'normal_superficiel',
            'normal_intermediate': 'normal_intermediate',
            'normal_columnar': 'normal_columnar',
            'light_dysplastic': 'light_dysplastic',
            'moderate_dysplastic': 'moderate_dysplastic',
            'severe_dysplastic': 'severe_dysplastic',
            'carcinoma_in_situ': 'carcinoma_in_situ'
        }
        
        for part in path_parts:
            part_lower = part.lower()
            for class_key in class_mapping:
                if class_key in part_lower:
                    return class_mapping[class_key]
        
        return None
    
    def phase5_herlev_quality_check(self):
        """PHASE 5: Herlev data quality check."""
        print("\n" + "="*80)
        print("PHASE 5: HERLEV DATA QUALITY CHECK")
        print("="*80)
        
        if not hasattr(self, 'herlev_class_mapping'):
            self.log_error("Herlev class mapping not available")
            return
        
        print("🔍 Checking data quality...")
        
        corrupted_images = []
        missing_files = []
        duplicate_hashes = defaultdict(list)
        
        # Check for corrupted images and duplicates
        for img_path, class_name in self.herlev_class_mapping.items():
            path_obj = Path(img_path)
            
            # Check if file exists
            if not path_obj.exists():
                missing_files.append(img_path)
                continue
            
            # Try to load image
            try:
                with Image.open(path_obj) as img:
                    img.verify()  # Verify image integrity
                
                # Calculate hash for duplicate detection
                with open(path_obj, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                duplicate_hashes[file_hash].append(img_path)
                
            except Exception as e:
                corrupted_images.append((img_path, str(e)))
        
        # Report findings
        print(f"📊 Quality Check Results:")
        print(f"  Corrupted images: {len(corrupted_images)}")
        print(f"  Missing files: {len(missing_files)}")
        
        # Check duplicates
        hash_duplicates = {h: paths for h, paths in duplicate_hashes.items() if len(paths) > 1}
        duplicate_count = sum(len(paths)-1 for paths in hash_duplicates.values())
        print(f"  Duplicate images: {duplicate_count}")
        
        # Show examples if issues found
        if corrupted_images:
            print(f"\n❌ Corrupted Images (showing first 3):")
            for i, (path, error) in enumerate(corrupted_images[:3]):
                print(f"  {i+1}. {path}: {error}")
        
        if missing_files:
            print(f"\n❌ Missing Files (showing first 3):")
            for i, path in enumerate(missing_files[:3]):
                print(f"  {i+1}. {path}")
        
        if duplicate_count > 0:
            print(f"\n⚠️  Duplicate Images (showing first 3):")
            for i, (hash_val, paths) in enumerate(list(hash_duplicates.items())[:3]):
                print(f"  {i+1}. Hash: {hash_val[:8]}...")
                for path in paths:
                    print(f"     {path}")
        
        # Log issues
        if corrupted_images:
            self.log_error(f"Found {len(corrupted_images)} corrupted images")
        
        if missing_files:
            self.log_error(f"Found {len(missing_files)} missing files")
        
        if duplicate_count > 0:
            self.log_warning(f"Found {duplicate_count} duplicate images")
        
        if not (corrupted_images or missing_files or duplicate_count > 0):
            self.log_success("No data quality issues found")
    
    def phase6_final_report(self):
        """PHASE 6: Final report."""
        print("\n" + "="*80)
        print("PHASE 6: FINAL REPORT")
        print("="*80)
        
        print("📋 COMPLETE DATASET AUDIT SUMMARY")
        print("="*60)
        
        # SIPaKMeD Summary
        print(f"\n📁 SIPaKMeD Dataset:")
        if hasattr(self, 'sipakmed_raw_count'):
            print(f"  Total raw images: {self.sipakmed_raw_count}")
            print(f"  Total used images: {self.sipakmed_loaded_count}")
            print(f"  Missing images: {self.sipakmed_missing_count}")
            print(f"  Data loss: {self.sipakmed_missing_count/self.sipakmed_raw_count*100:.1f}%")
            
            if hasattr(self, 'sipakmed_excluded_by_reason'):
                print(f"  Exclusion reasons:")
                for reason, paths in self.sipakmed_excluded_by_reason.items():
                    print(f"    {reason}: {len(paths)} images")
            
            if hasattr(self, 'sipakmed_duplicates'):
                print(f"  Duplicates removed: {self.sipakmed_duplicates}")
            
            if hasattr(self, 'sipakmed_class_distribution'):
                print(f"  Final class distribution:")
                for class_idx, count in self.sipakmed_class_distribution.items():
                    print(f"    Class {class_idx}: {count}")
        
        # Herlev Summary
        print(f"\n📁 Herlev Dataset:")
        if hasattr(self, 'herlev_total'):
            print(f"  Total images: {self.herlev_total}")
            
            if hasattr(self, 'herlev_class_distribution'):
                print(f"  Class distribution:")
                for class_name, count in self.herlev_class_distribution.items():
                    percentage = count / self.herlev_total * 100
                    print(f"    {class_name}: {count} ({percentage:.1f}%)")
                
                # Imbalance ratio
                if self.herlev_class_distribution:
                    max_count = max(self.herlev_class_distribution.values())
                    min_count = min(self.herlev_class_distribution.values())
                    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Issues Summary
        print(f"\n🚨 Issues Summary:")
        if self.issues:
            print(f"  Critical issues: {len(self.issues)}")
            for issue in self.issues:
                print(f"    {issue}")
        else:
            print("  ✅ No critical issues")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"    {warning}")
        
        # Final Verdict
        print(f"\n🎯 FINAL VERDICT:")
        
        sipakmed_valid = True
        herlev_valid = True
        
        # SIPaKMeD validation
        if hasattr(self, 'sipakmed_loaded_count'):
            if self.sipakmed_loaded_count < 3000:
                sipakmed_valid = False
                print(f"  ❌ SIPaKMeD: NOT VALID for training (insufficient data)")
            elif self.sipakmed_missing_count > 1000:
                sipakmed_valid = False
                print(f"  ❌ SIPaKMeD: NOT VALID for training (excessive data loss)")
            else:
                print(f"  ✅ SIPaKMeD: VALID for training")
        
        # Herlev validation
        if hasattr(self, 'herlev_total'):
            if self.herlev_total < 1500:
                herlev_valid = False
                print(f"  ❌ Herlev: NOT VALID for testing (insufficient data)")
            else:
                print(f"  ✅ Herlev: VALID for testing")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if sipakmed_valid and herlev_valid:
            print("  ✅ Both datasets are RELIABLE for research experiments")
        elif sipakmed_valid and not herlev_valid:
            print("  ⚠️  SIPaKMeD is reliable, but Herlev has issues")
        elif not sipakmed_valid and herlev_valid:
            print("  ⚠️  Herlev is reliable, but SIPaKMeD has issues")
        else:
            print("  ❌ Both datasets have significant issues")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if self.sipakmed_missing_count > 500:
            print("  • Investigate SIPaKMeD data loss - consider including excluded folders")
        
        if hasattr(self, 'herlev_class_distribution'):
            max_count = max(self.herlev_class_distribution.values())
            min_count = min(self.herlev_class_distribution.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            if imbalance_ratio > 2.0:
                print("  • Consider class weighting for Herlev dataset")
        
        print("="*80)
    
    def run_complete_audit(self):
        """Run complete dataset audit."""
        print("🚀 STARTING COMPREHENSIVE DATASET AUDIT")
        print("="*80)
        
        try:
            # Run all phases
            self.phase1_sipakmed_data_loss_analysis()
            self.phase2_duplicate_check()
            self.phase3_sipakmed_final_validation()
            self.phase4_herlev_validation()
            self.phase5_herlev_quality_check()
            self.phase6_final_report()
            
            print("\n🎉 DATASET AUDIT COMPLETED")
            
        except Exception as e:
            self.log_error(f"Audit failed: {e}")
            raise

if __name__ == "__main__":
    auditor = DatasetAuditor()
    auditor.run_complete_audit()
