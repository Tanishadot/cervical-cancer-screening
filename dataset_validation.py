"""
Comprehensive dataset validation and pipeline safety check for SIPaKMeD dataset.
Implements duplicate detection, data leakage checks, class distribution analysis,
and dataset loading validation.
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset import DatasetManager, SIPaKMeDDataset
from utils.label_mapping import LabelMapper, DatasetType, ClassificationMode


class DatasetValidator:
    """Comprehensive dataset validator for cervical cancer datasets."""
    
    def __init__(self, root_dir: str, use_cropped_only: bool = False):
        """
        Initialize dataset validator.
        
        Args:
            root_dir: Root directory containing datasets
            use_cropped_only: If True, only use CROPPED folders
        """
        self.root_dir = root_dir
        self.use_cropped_only = use_cropped_only
        self.validation_results = {}
        
    def compute_file_hash(self, file_path: str, hash_type: str = 'md5') -> str:
        """
        Compute hash of file content.
        
        Args:
            file_path: Path to the file
            hash_type: Type of hash ('md5' or 'sha256')
            
        Returns:
            Hexadecimal hash string
        """
        hash_func = hashlib.md5() if hash_type == 'md5' else hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            warnings.warn(f"Failed to hash {file_path}: {e}")
            return None
    
    def detect_duplicates(self, dataset_name: str = "sipakmed") -> Dict:
        """
        Detect duplicate images using content-based hashing.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Dictionary with duplicate analysis results
        """
        print(f"\n🔍 DETECTING DUPLICATES in {dataset_name.upper()}...")
        print("-" * 60)
        
        # Load dataset without transforms
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        dataset = SIPaKMeDDataset(
            root_dir=self.root_dir,
            label_mapper=label_mapper,
            transform=None,
            mode="all"
        )
        
        # Filter for CROPPED only if requested
        if self.use_cropped_only:
            filtered_samples = []
            for img_path, mapped_label, original_label in dataset.samples:
                if "CROPPED" in img_path.upper():
                    filtered_samples.append((img_path, mapped_label, original_label))
            dataset.samples = filtered_samples
            print(f"Filtered to {len(dataset.samples)} CROPPED images")
        
        # Compute hashes for all images
        hash_to_files = defaultdict(list)
        file_hashes = {}
        
        print("Computing hashes...")
        for i, (img_path, mapped_label, original_label) in enumerate(dataset.samples):
            if i % 500 == 0:
                print(f"  Processed {i}/{len(dataset.samples)} images...")
            
            file_hash = self.compute_file_hash(img_path)
            if file_hash:
                hash_to_files[file_hash].append(img_path)
                file_hashes[img_path] = file_hash
        
        # Find duplicates
        duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}
        
        # Calculate statistics
        total_images = len(dataset.samples)
        unique_images = len(hash_to_files)
        duplicate_images = sum(len(files) - 1 for files in duplicates.values())
        duplicate_percentage = (duplicate_images / total_images) * 100 if total_images > 0 else 0
        
        results = {
            'total_images': total_images,
            'unique_images': unique_images,
            'duplicate_groups': len(duplicates),
            'duplicate_images': duplicate_images,
            'duplicate_percentage': duplicate_percentage,
            'duplicates': duplicates
        }
        
        # Print results
        print(f"\n📊 DUPLICATE ANALYSIS RESULTS:")
        print(f"  Total images: {total_images}")
        print(f"  Unique images: {unique_images}")
        print(f"  Duplicate groups: {len(duplicates)}")
        print(f"  Duplicate images: {duplicate_images}")
        print(f"  Duplicate percentage: {duplicate_percentage:.2f}%")
        
        if duplicates:
            print(f"\n⚠️  DUPLICATE DETAILS (first 10 groups):")
            for i, (hash_val, files) in enumerate(list(duplicates.items())[:10]):
                print(f"  Group {i+1}: {len(files)} identical images")
                for file in files[:3]:  # Show first 3 files per group
                    print(f"    - {file}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
        else:
            print("✅ NO DUPLICATES FOUND")
        
        self.validation_results[f'{dataset_name}_duplicates'] = results
        return results
    
    def check_data_leakage(self, dataset_name: str = "sipakmed") -> Dict:
        """
        Check for data leakage between train/val/test splits.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Dictionary with leakage analysis results
        """
        print(f"\n🔍 CHECKING DATA LEAKAGE in {dataset_name.upper()}...")
        print("-" * 60)
        
        # Create dataset manager
        manager = DatasetManager(
            root_dir=self.root_dir,
            classification_mode="multiclass",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Load and split datasets
        manager.load_datasets()
        manager.create_splits()
        
        # Get datasets
        if dataset_name == "sipakmed":
            full_dataset = manager.sipakmed_dataset
        else:
            full_dataset = manager.herlev_dataset
        
        if full_dataset is None:
            print(f"❌ Dataset {dataset_name} not found")
            return {}
        
        # Filter for CROPPED only if requested
        if self.use_cropped_only:
            filtered_samples = []
            for img_path, mapped_label, original_label in full_dataset.samples:
                if "CROPPED" in img_path.upper():
                    filtered_samples.append((img_path, mapped_label, original_label))
            full_dataset.samples = filtered_samples
        
        # Get splits
        train_dataset = manager.train_datasets[dataset_name]
        val_dataset = manager.val_datasets[dataset_name]
        test_dataset = manager.test_datasets[dataset_name]
        
        # Get file paths for each split
        train_files = set()
        val_files = set()
        test_files = set()
        
        for idx in train_dataset.indices:
            img_path = full_dataset.samples[idx][0]
            train_files.add(img_path)
        
        for idx in val_dataset.indices:
            img_path = full_dataset.samples[idx][0]
            val_files.add(img_path)
        
        for idx in test_dataset.indices:
            img_path = full_dataset.samples[idx][0]
            test_files.add(img_path)
        
        # Check for overlaps
        train_val_overlap = train_files & val_files
        train_test_overlap = train_files & test_files
        val_test_overlap = val_files & test_files
        
        total_leaked = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
        
        results = {
            'train_size': len(train_files),
            'val_size': len(val_files),
            'test_size': len(test_files),
            'train_val_overlap': len(train_val_overlap),
            'train_test_overlap': len(train_test_overlap),
            'val_test_overlap': len(val_test_overlap),
            'total_leaked_samples': total_leaked,
            'train_val_files': list(train_val_overlap),
            'train_test_files': list(train_test_overlap),
            'val_test_files': list(val_test_overlap)
        }
        
        # Print results
        print(f"\n📊 DATA LEAKAGE ANALYSIS RESULTS:")
        print(f"  Train samples: {len(train_files)}")
        print(f"  Val samples: {len(val_files)}")
        print(f"  Test samples: {len(test_files)}")
        print(f"  Train-Val overlap: {len(train_val_overlap)}")
        print(f"  Train-Test overlap: {len(train_test_overlap)}")
        print(f"  Val-Test overlap: {len(val_test_overlap)}")
        print(f"  Total leaked samples: {total_leaked}")
        
        if total_leaked > 0:
            print(f"\n⚠️  DATA LEAKAGE DETECTED!")
            if train_val_overlap:
                print(f"  Train-Val overlap examples:")
                for file in list(train_val_overlap)[:3]:
                    print(f"    - {file}")
            if train_test_overlap:
                print(f"  Train-Test overlap examples:")
                for file in list(train_test_overlap)[:3]:
                    print(f"    - {file}")
            if val_test_overlap:
                print(f"  Val-Test overlap examples:")
                for file in list(val_test_overlap)[:3]:
                    print(f"    - {file}")
        else:
            print("✅ NO DATA LEAKAGE FOUND")
        
        self.validation_results[f'{dataset_name}_leakage'] = results
        return results
    
    def analyze_class_distribution(self, dataset_name: str = "sipakmed") -> Dict:
        """
        Analyze class distribution and imbalance.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Dictionary with class distribution analysis
        """
        print(f"\n🔍 ANALYZING CLASS DISTRIBUTION in {dataset_name.upper()}...")
        print("-" * 60)
        
        # Load dataset
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        dataset = SIPaKMeDDataset(
            root_dir=self.root_dir,
            label_mapper=label_mapper,
            transform=None,
            mode="all"
        )
        
        # Filter for CROPPED only if requested
        if self.use_cropped_only:
            filtered_samples = []
            for img_path, mapped_label, original_label in dataset.samples:
                if "CROPPED" in img_path.upper():
                    filtered_samples.append((img_path, mapped_label, original_label))
            dataset.samples = filtered_samples
        
        # Count samples per class
        class_counts = defaultdict(int)
        for img_path, mapped_label, original_label in dataset.samples:
            class_counts[mapped_label] += 1
        
        # Get class names
        class_names = label_mapper.get_class_names(DatasetType.SIPAKMED)
        
        # Calculate statistics
        total_samples = sum(class_counts.values())
        class_percentages = {class_id: (count / total_samples) * 100 
                            for class_id, count in class_counts.items()}
        
        max_count = max(class_counts.values()) if class_counts else 0
        min_count = min(class_counts.values()) if class_counts else 0
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Create detailed table
        distribution_table = []
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = class_percentages[class_id]
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            distribution_table.append({
                'class_id': class_id,
                'class_name': class_name,
                'count': count,
                'percentage': percentage
            })
        
        results = {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_counts': dict(class_counts),
            'class_percentages': class_percentages,
            'max_count': max_count,
            'min_count': min_count,
            'imbalance_ratio': imbalance_ratio,
            'distribution_table': distribution_table,
            'class_names': class_names
        }
        
        # Print results
        print(f"\n📊 CLASS DISTRIBUTION RESULTS:")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of classes: {len(class_counts)}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        
        print(f"\n  CLASS DISTRIBUTION TABLE:")
        print(f"  {'Class ID':<10} {'Class Name':<25} {'Count':<8} {'Percentage':<10}")
        print(f"  {'-'*10} {'-'*25} {'-'*8} {'-'*10}")
        for row in distribution_table:
            print(f"  {row['class_id']:<10} {row['class_name']:<25} {row['count']:<8} {row['percentage']:<10.1f}%")
        
        if imbalance_ratio > 2.0:
            print(f"\n⚠️  HIGH CLASS IMBALANCE DETECTED (ratio: {imbalance_ratio:.2f})")
            print("    Consider using class weights or data augmentation")
        else:
            print(f"\n✅ CLASS DISTRIBUTION IS REASONABLY BALANCED")
        
        self.validation_results[f'{dataset_name}_class_distribution'] = results
        return results
    
    def analyze_cropped_vs_full(self, dataset_name: str = "sipakmed") -> Dict:
        """
        Analyze composition of CROPPED vs full dataset.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Dictionary with CROPPED vs full analysis
        """
        print(f"\n🔍 ANALYZING CROPPED vs FULL DATA in {dataset_name.upper()}...")
        print("-" * 60)
        
        # Load dataset
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        dataset = SIPaKMeDDataset(
            root_dir=self.root_dir,
            label_mapper=label_mapper,
            transform=None,
            mode="all"
        )
        
        # Count CROPPED vs non-CROPPED
        cropped_count = 0
        full_count = 0
        cropped_by_class = defaultdict(int)
        full_by_class = defaultdict(int)
        
        for img_path, mapped_label, original_label in dataset.samples:
            if "CROPPED" in img_path.upper():
                cropped_count += 1
                cropped_by_class[mapped_label] += 1
            else:
                full_count += 1
                full_by_class[mapped_label] += 1
        
        total_count = cropped_count + full_count
        cropped_percentage = (cropped_count / total_count) * 100 if total_count > 0 else 0
        full_percentage = (full_count / total_count) * 100 if total_count > 0 else 0
        
        results = {
            'total_images': total_count,
            'cropped_count': cropped_count,
            'full_count': full_count,
            'cropped_percentage': cropped_percentage,
            'full_percentage': full_percentage,
            'cropped_by_class': dict(cropped_by_class),
            'full_by_class': dict(full_by_class)
        }
        
        # Print results
        print(f"\n📊 CROPPED vs FULL DATA ANALYSIS:")
        print(f"  Total images: {total_count}")
        print(f"  CROPPED images: {cropped_count} ({cropped_percentage:.1f}%)")
        print(f"  Full images: {full_count} ({full_percentage:.1f}%)")
        
        print(f"\n  CROPPED IMAGES BY CLASS:")
        class_names = label_mapper.get_class_names(DatasetType.SIPAKMED)
        for class_id in sorted(cropped_by_class.keys()):
            count = cropped_by_class[class_id]
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"    {class_name}: {count}")
        
        print(f"\n  FULL IMAGES BY CLASS:")
        for class_id in sorted(full_by_class.keys()):
            count = full_by_class[class_id]
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"    {class_name}: {count}")
        
        # Recommendation
        if cropped_percentage > 50:
            recommendation = "Use CROPPED images only (higher quality, focused)"
        elif cropped_percentage < 20:
            recommendation = "Use all images (limited CROPPED data)"
        else:
            recommendation = "Consider both options and compare performance"
        
        print(f"\n💡 RECOMMENDATION: {recommendation}")
        
        self.validation_results[f'{dataset_name}_cropped_analysis'] = results
        return results
    
    def validate_dataset_loading(self, dataset_name: str = "sipakmed") -> Dict:
        """
        Validate dataset loading pipeline stability.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Dictionary with loading validation results
        """
        print(f"\n🔍 VALIDATING DATASET LOADING PIPELINE for {dataset_name.upper()}...")
        print("-" * 60)
        
        results = {
            'loading_success': False,
            'expected_samples': 0,
            'actual_samples': 0,
            'split_ratios': {},
            'errors': []
        }
        
        try:
            # Test dataset loading
            manager = DatasetManager(
                root_dir=self.root_dir,
                classification_mode="multiclass",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42
            )
            
            # Load datasets
            manager.load_datasets()
            manager.create_splits()
            
            # Get dataset info
            if dataset_name == "sipakmed":
                dataset = manager.sipakmed_dataset
            else:
                dataset = manager.herlev_dataset
            
            if dataset is None:
                results['errors'].append(f"Dataset {dataset_name} not found")
                return results
            
            # Filter for CROPPED only if requested
            if self.use_cropped_only:
                filtered_samples = []
                for img_path, mapped_label, original_label in dataset.samples:
                    if "CROPPED" in img_path.upper():
                        filtered_samples.append((img_path, mapped_label, original_label))
                dataset.samples = filtered_samples
            
            actual_samples = len(dataset)
            results['actual_samples'] = actual_samples
            
            # Expected samples (approximate)
            if dataset_name == "sipakmed":
                expected = 5015 if not self.use_cropped_only else 2500  # Approximate
            else:
                expected = 917  # Herlev approximate
            
            results['expected_samples'] = expected
            
            # Check split ratios
            if dataset_name in manager.train_datasets:
                train_size = len(manager.train_datasets[dataset_name])
                val_size = len(manager.val_datasets[dataset_name])
                test_size = len(manager.test_datasets[dataset_name])
                
                train_ratio = train_size / actual_samples
                val_ratio = val_size / actual_samples
                test_ratio = test_size / actual_samples
                
                results['split_ratios'] = {
                    'train': train_ratio,
                    'val': val_ratio,
                    'test': test_ratio,
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size
                }
            
            results['loading_success'] = True
            
            # Print results
            print(f"\n📊 DATASET LOADING VALIDATION RESULTS:")
            print(f"  Loading success: ✅")
            print(f"  Expected samples: ~{expected}")
            print(f"  Actual samples: {actual_samples}")
            print(f"  Sample match: {'✅' if abs(actual_samples - expected) < 100 else '⚠️'}")
            
            if 'split_ratios' in results:
                ratios = results['split_ratios']
                print(f"  Split ratios:")
                print(f"    Train: {ratios['train']:.1%} ({ratios['train_size']} samples)")
                print(f"    Val: {ratios['val']:.1%} ({ratios['val_size']} samples)")
                print(f"    Test: {ratios['test']:.1%} ({ratios['test_size']} samples)")
                
                # Check if ratios are close to expected
                expected_train, expected_val, expected_test = 0.7, 0.15, 0.15
                train_ok = abs(ratios['train'] - expected_train) < 0.05
                val_ok = abs(ratios['val'] - expected_val) < 0.05
                test_ok = abs(ratios['test'] - expected_test) < 0.05
                
                print(f"  Ratio accuracy: {'✅' if train_ok and val_ok and test_ok else '⚠️'}")
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"❌ Dataset loading failed: {e}")
        
        self.validation_results[f'{dataset_name}_loading_validation'] = results
        return results
    
    def run_comprehensive_validation(self, dataset_name: str = "sipakmed") -> Dict:
        """
        Run all validation checks for a dataset.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Dictionary with all validation results
        """
        print(f"\n🚀 COMPREHENSIVE DATASET VALIDATION for {dataset_name.upper()}")
        print("=" * 80)
        
        if self.use_cropped_only:
            print("🔬 VALIDATION MODE: CROPPED IMAGES ONLY")
        else:
            print("🔬 VALIDATION MODE: ALL IMAGES")
        
        # Run all validations
        duplicate_results = self.detect_duplicates(dataset_name)
        leakage_results = self.check_data_leakage(dataset_name)
        class_results = self.analyze_class_distribution(dataset_name)
        cropped_results = self.analyze_cropped_vs_full(dataset_name)
        loading_results = self.validate_dataset_loading(dataset_name)
        
        # Compile summary
        summary = {
            'dataset_name': dataset_name,
            'validation_mode': 'cropped_only' if self.use_cropped_only else 'all_images',
            'duplicate_detection': {
                'passed': duplicate_results.get('duplicate_percentage', 0) < 5,
                'duplicate_percentage': duplicate_results.get('duplicate_percentage', 0),
                'total_duplicates': duplicate_results.get('duplicate_images', 0)
            },
            'data_leakage': {
                'passed': leakage_results.get('total_leaked_samples', 0) == 0,
                'leaked_samples': leakage_results.get('total_leaked_samples', 0)
            },
            'class_distribution': {
                'passed': class_results.get('imbalance_ratio', float('inf')) <= 2.0,
                'imbalance_ratio': class_results.get('imbalance_ratio', float('inf')),
                'total_samples': class_results.get('total_samples', 0)
            },
            'dataset_loading': {
                'passed': loading_results.get('loading_success', False),
                'sample_match': abs(loading_results.get('actual_samples', 0) - 
                                  loading_results.get('expected_samples', 0)) < 100
            },
            'overall_passed': True
        }
        
        # Check overall status
        for check in ['duplicate_detection', 'data_leakage', 'class_distribution', 'dataset_loading']:
            if not summary[check]['passed']:
                summary['overall_passed'] = False
        
        # Print final summary
        print(f"\n🎯 VALIDATION SUMMARY for {dataset_name.upper()}")
        print("=" * 60)
        print(f"Overall Status: {'✅ PASSED' if summary['overall_passed'] else '❌ FAILED'}")
        print(f"Validation Mode: {summary['validation_mode']}")
        print()
        
        print("Individual Checks:")
        checks = [
            ('Duplicate Detection', summary['duplicate_detection']),
            ('Data Leakage', summary['data_leakage']),
            ('Class Distribution', summary['class_distribution']),
            ('Dataset Loading', summary['dataset_loading'])
        ]
        
        for check_name, check_result in checks:
            status = '✅ PASSED' if check_result['passed'] else '❌ FAILED'
            print(f"  {check_name:<20}: {status}")
        
        print(f"\nKey Metrics:")
        print(f"  Total samples: {class_results.get('total_samples', 0)}")
        print(f"  Duplicate percentage: {duplicate_results.get('duplicate_percentage', 0):.2f}%")
        print(f"  Class imbalance ratio: {class_results.get('imbalance_ratio', 0):.2f}")
        print(f"  Data leakage samples: {leakage_results.get('total_leaked_samples', 0)}")
        
        return summary


def main():
    """Main function to run dataset validation."""
    parser = argparse.ArgumentParser(description='Comprehensive dataset validation')
    parser.add_argument('--dataset', choices=['sipakmed', 'herlev', 'all'], 
                       default='sipakmed', help='Dataset to validate')
    parser.add_argument('--use-cropped-only', action='store_true',
                       help='Only use CROPPED images')
    parser.add_argument('--root-dir', default='datasets', 
                       help='Root directory containing datasets')
    
    args = parser.parse_args()
    
    # Create validator
    validator = DatasetValidator(
        root_dir=args.root_dir,
        use_cropped_only=args.use_cropped_only
    )
    
    # Run validation
    datasets_to_validate = ['sipakmed'] if args.dataset == 'sipakmed' else \
                         ['herlev'] if args.dataset == 'herlev' else \
                         ['sipakmed', 'herlev']
    
    all_results = {}
    
    for dataset_name in datasets_to_validate:
        try:
            results = validator.run_comprehensive_validation(dataset_name)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"❌ Validation failed for {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Print final summary
    print(f"\n🏁 FINAL VALIDATION SUMMARY")
    print("=" * 80)
    
    for dataset_name, results in all_results.items():
        if 'error' in results:
            print(f"{dataset_name.upper()}: ❌ ERROR - {results['error']}")
        else:
            status = '✅ PASSED' if results['overall_passed'] else '❌ FAILED'
            print(f"{dataset_name.upper()}: {status}")
    
    return all_results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if all(r.get('overall_passed', False) for r in results.values()) else 1)
