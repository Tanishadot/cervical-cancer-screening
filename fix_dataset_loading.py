#!/usr/bin/env python3
"""
Fix Dataset Loading Issues
Addresses: full dataset usage, class mapping, and data leakage
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.label_mapping import LabelMapper, ClassificationMode

class FixedDatasetLoader:
    """Fixed dataset loader that addresses all identified issues."""
    
    def __init__(self):
        self.issues = []
        self.success = []
        
        # Class mapping for normalization
        self.class_mapping = {
            'im_Superficial-Intermediate': 'superficial_intermediate',
            'im_Parabasal': 'parabasal',
            'im_Koilocytotic': 'koilocytotic',
            'im_Metaplastic': 'metaplastic',
            'im_Dyskeratotic': 'dyskeratotic'
        }
        
        # Expected classes
        self.expected_classes = [
            'superficial_intermediate',
            'parabasal',
            'koilocytotic',
            'metaplastic',
            'dyskeratotic'
        ]
    
    def log_info(self, message):
        print(f"ℹ️  {message}")
    
    def log_success(self, message):
        success_msg = f"✅ {message}"
        print(success_msg)
        self.success.append(success_msg)
    
    def log_issue(self, message):
        issue_msg = f"❌ {message}"
        print(issue_msg)
        self.issues.append(issue_msg)
    
    def phase1_fix_dataset_loading(self):
        """PHASE 1: Fix dataset loading to include all images."""
        print("\n" + "="*80)
        print("PHASE 1: FIXING DATASET LOADING")
        print("="*80)
        
        sipakmed_path = Path("datasets/sipakmed")
        
        # Load all images recursively
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_images = []
        seen_files = set()  # Avoid duplicates
        
        print("📁 Scanning SIPaKMeD dataset recursively...")
        
        for root, dirs, files in os.walk(sipakmed_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    file_path = Path(root) / file
                    
                    # Create unique identifier to avoid duplicates
                    file_id = f"{file_path.stem}_{file_path.stat().st_size}"
                    
                    if file_id in seen_files:
                        continue  # Skip duplicate
                    
                    seen_files.add(file_id)
                    
                    # Extract class from path
                    class_name = self._extract_class_from_path(file_path, sipakmed_path)
                    
                    if class_name:
                        all_images.append({
                            'path': str(file_path),
                            'class': class_name,
                            'original_folder': file_path.parent.name
                        })
        
        print(f"📊 Total unique images found: {len(all_images)}")
        
        if len(all_images) < 4000:
            self.log_issue(f"Dataset size ({len(all_images)}) is still smaller than expected")
        else:
            self.log_success(f"Dataset size ({len(all_images)}) is within expected range")
        
        # Class distribution
        class_counts = Counter(img['class'] for img in all_images)
        
        print("\n📋 Class Distribution:")
        print(f"{'Class':<25} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        
        for class_name in self.expected_classes:
            count = class_counts.get(class_name, 0)
            percentage = count / len(all_images) * 100
            print(f"{class_name:<25} {count:<8} {percentage:<10.2f}%")
        
        # Check for missing classes
        missing_classes = set(self.expected_classes) - set(class_counts.keys())
        if missing_classes:
            self.log_issue(f"Missing classes: {missing_classes}")
        else:
            self.log_success("All expected classes found")
        
        # Store for next phases
        self.all_images = all_images
        self.class_counts = class_counts
        
        return all_images
    
    def _extract_class_from_path(self, file_path, base_path):
        """Extract normalized class name from file path."""
        relative_path = Path(file_path).relative_to(base_path)
        path_parts = list(relative_path.parts)
        
        # Look for class indicators in path
        for part in path_parts:
            part_lower = part.lower()
            
            # Check against class mapping
            for key, value in self.class_mapping.items():
                if key.lower() in part_lower:
                    return value
        
        return None
    
    def phase2_fix_missing_classes(self):
        """PHASE 2: Ensure all classes exist."""
        print("\n" + "="*80)
        print("PHASE 2: FIXING MISSING CLASSES")
        print("="*80)
        
        if not hasattr(self, 'class_counts'):
            self.log_issue("No class counts available")
            return
        
        missing_classes = set(self.expected_classes) - set(self.class_counts.keys())
        
        if missing_classes:
            self.log_issue(f"Missing classes: {missing_classes}")
            
            # Try to find missing classes in alternative locations
            print("🔍 Searching for missing classes...")
            
            sipakmed_path = Path("datasets/sipakmed")
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            
            for missing_class in missing_classes:
                found_images = []
                
                for root, dirs, files in os.walk(sipakmed_path):
                    for file in files:
                        if Path(file).suffix.lower() in image_extensions:
                            file_path = Path(root) / file
                            
                            # Check if this file belongs to missing class
                            extracted_class = self._extract_class_from_path(file_path, sipakmed_path)
                            if extracted_class == missing_class:
                                found_images.append({
                                    'path': str(file_path),
                                    'class': missing_class,
                                    'original_folder': file_path.parent.name
                                })
                
                if found_images:
                    print(f"  Found {len(found_images)} images for {missing_class}")
                    self.all_images.extend(found_images)
                    self.class_counts[missing_class] = len(found_images)
                    self.log_success(f"Recovered {missing_class} class")
                else:
                    self.log_issue(f"Could not recover {missing_class} class")
        
        # Final class check
        final_class_counts = Counter(img['class'] for img in self.all_images)
        final_missing = set(self.expected_classes) - set(final_class_counts.keys())
        
        if not final_missing:
            self.log_success("All 5 classes are now present")
        else:
            self.log_issue(f"Still missing classes: {final_missing}")
        
        return self.all_images
    
    def phase3_fix_data_splitting(self):
        """PHASE 3: Fix data splitting with no leakage."""
        print("\n" + "="*80)
        print("PHASE 3: FIXING DATA SPLITTING")
        print("="*80)
        
        if not hasattr(self, 'all_images'):
            self.log_issue("No images available for splitting")
            return
        
        print("🔄 Creating leak-free data splits...")
        
        # Group images by class for stratified splitting
        class_images = defaultdict(list)
        for img in self.all_images:
            class_images[img['class']].append(img)
        
        # Perform stratified split
        train_images = []
        val_images = []
        test_images = []
        
        for class_name, images in class_images.items():
            n_images = len(images)
            
            # Calculate split sizes
            n_train = int(n_images * 0.7)
            n_val = int(n_images * 0.15)
            n_test = n_images - n_train - n_val  # Ensure all images are used
            
            # Shuffle images within class
            np.random.shuffle(images)
            
            # Split
            train_images.extend(images[:n_train])
            val_images.extend(images[n_train:n_train + n_val])
            test_images.extend(images[n_train + n_val:])
        
        # Convert to lists of paths for overlap checking
        train_paths = set(img['path'] for img in train_images)
        val_paths = set(img['path'] for img in val_images)
        test_paths = set(img['path'] for img in test_images)
        
        # Check for overlaps
        train_val_overlap = train_paths & val_paths
        train_test_overlap = train_paths & test_paths
        val_test_overlap = val_paths & test_paths
        
        print("🔍 Checking for data leakage...")
        if train_val_overlap:
            self.log_issue(f"Train-Val overlap: {len(train_val_overlap)} samples")
        else:
            self.log_success("No train-validation overlap")
        
        if train_test_overlap:
            self.log_issue(f"Train-Test overlap: {len(train_test_overlap)} samples")
        else:
            self.log_success("No train-test overlap")
        
        if val_test_overlap:
            self.log_issue(f"Val-Test overlap: {len(val_test_overlap)} samples")
        else:
            self.log_success("No validation-test overlap")
        
        # Print split statistics
        total_images = len(self.all_images)
        train_pct = len(train_images) / total_images * 100
        val_pct = len(val_images) / total_images * 100
        test_pct = len(test_images) / total_images * 100
        
        print(f"\n📊 Split Statistics:")
        print(f"  Train: {len(train_images)} ({train_pct:.1f}%)")
        print(f"  Val: {len(val_images)} ({val_pct:.1f}%)")
        print(f"  Test: {len(test_images)} ({test_pct:.1f}%)")
        print(f"  Total: {total_images}")
        
        # Store splits
        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images
        
        return train_images, val_images, test_images
    
    def phase4_verify_final_dataset(self):
        """PHASE 4: Verify final dataset."""
        print("\n" + "="*80)
        print("PHASE 4: FINAL DATASET VERIFICATION")
        print("="*80)
        
        print("📋 FINAL DATASET SUMMARY")
        print("="*50)
        
        # Total images
        if hasattr(self, 'all_images'):
            print(f"📁 Total Images: {len(self.all_images)}")
            if len(self.all_images) >= 4000:
                print("   Status: ✅ Complete dataset")
            else:
                print("   Status: ⚠️  Incomplete dataset")
        
        # Class distribution
        if hasattr(self, 'class_counts'):
            print(f"\n📊 Class Distribution:")
            for class_name in self.expected_classes:
                count = self.class_counts.get(class_name, 0)
                percentage = count / len(self.all_images) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # Split verification
        if hasattr(self, 'train_images'):
            print(f"\n🔄 Split Sizes:")
            print(f"   Train: {len(self.train_images)}")
            print(f"   Val: {len(self.val_images)}")
            print(f"   Test: {len(self.test_images)}")
            
            # Verify no leakage
            train_paths = set(img['path'] for img in self.train_images)
            val_paths = set(img['path'] for img in self.val_images)
            test_paths = set(img['path'] for img in self.test_images)
            
            if not (train_paths & val_paths) and not (train_paths & test_paths) and not (val_paths & test_paths):
                print("   Status: ✅ No data leakage")
            else:
                print("   Status: ❌ Data leakage detected")
        
        # Issues summary
        print(f"\n🚨 Issues Summary:")
        if self.issues:
            print(f"   Issues found: {len(self.issues)}")
            for issue in self.issues:
                print(f"     {issue}")
        else:
            print("   ✅ No critical issues")
        
        # Success summary
        print(f"\n✅ Fixes Applied:")
        if self.success:
            for success in self.success:
                print(f"   {success}")
        
        # Final assessment
        print(f"\n🎯 Final Assessment:")
        if not self.issues:
            print("   ✅ Dataset is ready for training experiments")
        else:
            print("   ⚠️  Some issues remain - review before training")
        
        print("="*80)
    
    def create_fixed_dataset_info(self):
        """Create fixed dataset information for the pipeline."""
        if not hasattr(self, 'train_images'):
            return None
        
        dataset_info = {
            'total_images': len(self.all_images),
            'class_distribution': self.class_counts,
            'splits': {
                'train': len(self.train_images),
                'val': len(self.val_images),
                'test': len(self.test_images)
            },
            'train_paths': [img['path'] for img in self.train_images],
            'val_paths': [img['path'] for img in self.val_images],
            'test_paths': [img['path'] for img in self.test_images],
            'class_mapping': self.class_mapping
        }
        
        # Save for pipeline use
        with open('outputs/fixed_dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"📁 Fixed dataset info saved to: outputs/fixed_dataset_info.json")
        
        return dataset_info
    
    def run_all_fixes(self):
        """Run all dataset fixes."""
        print("🚀 STARTING DATASET FIXING PROCESS")
        print("="*80)
        
        try:
            # Run all phases
            self.phase1_fix_dataset_loading()
            self.phase2_fix_missing_classes()
            self.phase3_fix_data_splitting()
            self.phase4_verify_final_dataset()
            
            # Create fixed dataset info
            dataset_info = self.create_fixed_dataset_info()
            
            print("\n🎉 DATASET FIXING COMPLETED")
            
            if not self.issues:
                print("✅ All issues resolved - pipeline is ready for training!")
            else:
                print("⚠️  Some issues remain - review before training")
            
            return dataset_info
            
        except Exception as e:
            self.log_issue(f"Dataset fixing failed: {e}")
            raise

if __name__ == "__main__":
    fixer = FixedDatasetLoader()
    dataset_info = fixer.run_all_fixes()
