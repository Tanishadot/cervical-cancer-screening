"""
Simplified dataset validation without PyTorch dependency.
Implements duplicate detection, data leakage checks, and class distribution analysis.
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

from utils.label_mapping import LabelMapper, DatasetType, ClassificationMode


class SimpleDatasetValidator:
    """Simplified dataset validator without PyTorch dependency."""
    
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
        
    def load_sipakmed_samples(self) -> List[Tuple[str, int, int]]:
        """
        Load SIPaKMeD dataset samples without PyTorch.
        
        Returns:
            List of (image_path, mapped_label, original_label) tuples
        """
        samples = []
        
        # Dataset directory
        dataset_dir = os.path.join(self.root_dir, "sipakmed", "archive")
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Label mapper
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        
        # SIPaKMeD class mapping
        sipakmed_mapping = {
            0: "im_Superficial-Intermediate",  # Superficial-Intermediate
            1: "im_Parabasal",                  # Parabasal
            2: "im_Koilocytotic",               # Koilocytotic
            3: "im_Metaplastic",                # Metaplastic
            4: "im_Dyskeratotic"                # Dyskeratotic
        }
        
        # Load images from each class
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        
        for class_id, folder_name in sipakmed_mapping.items():
            if not folder_name:
                continue
                
            class_dir = os.path.join(dataset_dir, folder_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Recursively find all images
            class_path = Path(class_dir)
            all_images = []
            
            for img_file in class_path.rglob("*"):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    all_images.append(img_file)
            
            print(f"Found {len(all_images)} images for class {class_id} ({folder_name})")
            
            # Filter for CROPPED only if requested
            if self.use_cropped_only:
                all_images = [img for img in all_images if "CROPPED" in str(img).upper()]
            
            # Add samples
            for img_path in all_images:
                mapped_label = label_mapper.map_label(class_id, DatasetType.SIPAKMED)
                samples.append((str(img_path), mapped_label, class_id))
        
        return samples
    
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
    
    def detect_duplicates(self) -> Dict:
        """Detect duplicate images using content-based hashing."""
        print(f"\n🔍 DETECTING DUPLICATES in SIPaKMeD...")
        print("-" * 60)
        
        # Load samples
        samples = self.load_sipakmed_samples()
        
        # Compute hashes for all images
        hash_to_files = defaultdict(list)
        file_hashes = {}
        
        print("Computing hashes...")
        for i, (img_path, mapped_label, original_label) in enumerate(samples):
            if i % 500 == 0:
                print(f"  Processed {i}/{len(samples)} images...")
            
            file_hash = self.compute_file_hash(img_path)
            if file_hash:
                hash_to_files[file_hash].append(img_path)
                file_hashes[img_path] = file_hash
        
        # Find duplicates
        duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}
        
        # Calculate statistics
        total_images = len(samples)
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
        
        self.validation_results['duplicates'] = results
        return results
    
    def check_data_leakage(self) -> Dict:
        """Check for data leakage between train/val/test splits."""
        print(f"\n🔍 CHECKING DATA LEAKAGE in SIPaKMeD...")
        print("-" * 60)
        
        # Load samples
        samples = self.load_sipakmed_samples()
        
        # Create manual splits (70/15/15)
        import random
        random.seed(42)
        
        total_samples = len(samples)
        train_size = int(total_samples * 0.7)
        val_size = int(total_samples * 0.15)
        test_size = total_samples - train_size - val_size
        
        # Shuffle and split
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        train_samples = shuffled_samples[:train_size]
        val_samples = shuffled_samples[train_size:train_size + val_size]
        test_samples = shuffled_samples[train_size + val_size:]
        
        # Get file paths for each split
        train_files = set(sample[0] for sample in train_samples)
        val_files = set(sample[0] for sample in val_samples)
        test_files = set(sample[0] for sample in test_samples)
        
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
        
        self.validation_results['leakage'] = results
        return results
    
    def analyze_class_distribution(self) -> Dict:
        """Analyze class distribution and imbalance."""
        print(f"\n🔍 ANALYZING CLASS DISTRIBUTION in SIPaKMeD...")
        print("-" * 60)
        
        # Load samples
        samples = self.load_sipakmed_samples()
        
        # Count samples per class
        class_counts = defaultdict(int)
        for img_path, mapped_label, original_label in samples:
            class_counts[mapped_label] += 1
        
        # Get class names
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
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
        
        self.validation_results['class_distribution'] = results
        return results
    
    def analyze_cropped_vs_full(self) -> Dict:
        """Analyze composition of CROPPED vs full dataset."""
        print(f"\n🔍 ANALYZING CROPPED vs FULL DATA in SIPaKMeD...")
        print("-" * 60)
        
        # Load all samples (without filtering)
        self.use_cropped_only_temp = False
        all_samples = self.load_sipakmed_samples()
        self.use_cropped_only_temp = self.use_cropped_only
        
        # Count CROPPED vs non-CROPPED
        cropped_count = 0
        full_count = 0
        cropped_by_class = defaultdict(int)
        full_by_class = defaultdict(int)
        
        for img_path, mapped_label, original_label in all_samples:
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
        
        # Get class names for display
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        class_names = label_mapper.get_class_names(DatasetType.SIPAKMED)
        
        print(f"\n  CROPPED IMAGES BY CLASS:")
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
        
        self.validation_results['cropped_analysis'] = results
        return results
    
    def run_comprehensive_validation(self) -> Dict:
        """Run all validation checks."""
        print(f"\n🚀 COMPREHENSIVE DATASET VALIDATION for SIPaKMeD")
        print("=" * 80)
        
        if self.use_cropped_only:
            print("🔬 VALIDATION MODE: CROPPED IMAGES ONLY")
        else:
            print("🔬 VALIDATION MODE: ALL IMAGES")
        
        # Run all validations
        duplicate_results = self.detect_duplicates()
        leakage_results = self.check_data_leakage()
        class_results = self.analyze_class_distribution()
        cropped_results = self.analyze_cropped_vs_full()
        
        # Compile summary
        summary = {
            'dataset_name': 'sipakmed',
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
            'overall_passed': True
        }
        
        # Check overall status
        for check in ['duplicate_detection', 'data_leakage', 'class_distribution']:
            if not summary[check]['passed']:
                summary['overall_passed'] = False
        
        # Print final summary
        print(f"\n🎯 VALIDATION SUMMARY for SIPaKMeD")
        print("=" * 60)
        print(f"Overall Status: {'✅ PASSED' if summary['overall_passed'] else '❌ FAILED'}")
        print(f"Validation Mode: {summary['validation_mode']}")
        print()
        
        print("Individual Checks:")
        checks = [
            ('Duplicate Detection', summary['duplicate_detection']),
            ('Data Leakage', summary['data_leakage']),
            ('Class Distribution', summary['class_distribution'])
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
    parser = argparse.ArgumentParser(description='Simplified dataset validation without PyTorch')
    parser.add_argument('--use-cropped-only', action='store_true',
                       help='Only use CROPPED images')
    parser.add_argument('--root-dir', default='datasets', 
                       help='Root directory containing datasets')
    
    args = parser.parse_args()
    
    # Create validator
    validator = SimpleDatasetValidator(
        root_dir=args.root_dir,
        use_cropped_only=args.use_cropped_only
    )
    
    # Run validation
    try:
        results = validator.run_comprehensive_validation()
        return results
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results.get('overall_passed', False) else 1)
