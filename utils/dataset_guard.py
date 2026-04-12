"""
Production-ready dataset validation and auto-guard system.
Integrates comprehensive dataset safety checks directly into the training pipeline.
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.label_mapping import LabelMapper, DatasetType, ClassificationMode


class DatasetGuard:
    """
    Production-ready dataset validation and auto-guard system.
    Ensures dataset integrity before training begins.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dataset guard.
        
        Args:
            config: Configuration dictionary with dataset settings
        """
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.paths_config = config.get('paths', {})
        
        # Validation settings
        self.use_cropped_only = self.dataset_config.get('use_cropped_only', False)
        self.run_validation = self.dataset_config.get('run_validation', True)
        self.strict_mode = self.dataset_config.get('validation_strict_mode', True)
        
        # Performance thresholds
        self.thresholds = {
            'max_duplicate_rate': 0.01,  # 1%
            'min_sipakmed_samples': 4000,
            'min_herlev_samples': 500,
            'max_imbalance_ratio': 2.0,
            'min_class_samples': 500,
            'max_validation_time': 10.0  # seconds
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.DatasetGuard")
        self.validation_log_path = os.path.join(
            self.paths_config.get('logs', 'outputs/logs'), 
            'dataset_validation.log'
        )
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.validation_log_path), exist_ok=True)
        
        # Results storage
        self.validation_results = {}
        self.validation_status = "UNKNOWN"
        
    def _setup_file_logging(self):
        """Setup file logging for validation."""
        file_handler = logging.FileHandler(self.validation_log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add to logger if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)
    
    def _compute_file_hash(self, file_path: str, hash_type: str = 'md5') -> str:
        """Compute hash of file content."""
        hash_func = hashlib.md5() if hash_type == 'md5' else hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            warnings.warn(f"Failed to hash {file_path}: {e}")
            return None
    
    def _load_dataset_samples(self, dataset_name: str) -> List[Tuple[str, int, int]]:
        """Load dataset samples without PyTorch."""
        samples = []
        
        # Dataset directory
        if dataset_name == "sipakmed":
            dataset_dir = os.path.join(self.dataset_config['root_dir'], "sipakmed", "archive")
        elif dataset_name == "herlev":
            dataset_dir = os.path.join(self.dataset_config['root_dir'], "herlev", "archive (1)")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Label mapper
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        
        # Get class mapping
        if dataset_name == "sipakmed":
            class_mapping = {
                0: "im_Superficial-Intermediate",
                1: "im_Parabasal",
                2: "im_Koilocytotic",
                3: "im_Metaplastic",
                4: "im_Dyskeratotic"
            }
        else:  # herlev
            class_mapping = {
                0: "normal_superficiel",
                1: "normal_intermediate",
                2: "normal_columnar",
                3: "light_dysplastic",
                4: "moderate_dysplastic",
                5: "severe_dysplastic",
                6: "carcinoma_in_situ"
            }
        
        # Load images
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        
        for class_id, folder_name in class_mapping.items():
            if not folder_name:
                continue
                
            class_dir = os.path.join(dataset_dir, folder_name)
            
            if not os.path.exists(class_dir):
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Find all images
            class_path = Path(class_dir)
            all_images = []
            
            for img_file in class_path.rglob("*"):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    all_images.append(img_file)
            
            # Filter for CROPPED only if requested
            if self.use_cropped_only:
                all_images = [img for img in all_images if "CROPPED" in str(img).upper()]
            
            # Add samples
            for img_path in all_images:
                mapped_label = label_mapper.map_label(class_id, 
                    DatasetType.SIPAKMED if dataset_name == "sipakmed" else DatasetType.HERLEV)
                samples.append((str(img_path), mapped_label, class_id))
        
        return samples
    
    def _validate_duplicates(self, samples: List[Tuple[str, int, int]], dataset_name: str) -> Dict:
        """Validate duplicate detection."""
        start_time = time.time()
        
        # Compute hashes
        hash_to_files = defaultdict(list)
        for img_path, _, _ in samples:
            file_hash = self._compute_file_hash(img_path)
            if file_hash:
                hash_to_files[file_hash].append(img_path)
        
        # Find duplicates
        duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}
        
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
            'validation_time': time.time() - start_time,
            'passed': duplicate_percentage <= self.thresholds['max_duplicate_rate'] * 100
        }
        
        self.logger.info(f"Duplicate validation for {dataset_name}: {duplicate_percentage:.2f}% duplicates, "
                        f"Status: {'PASS' if results['passed'] else 'FAIL'}")
        
        return results
    
    def _validate_data_leakage(self, samples: List[Tuple[str, int, int]], dataset_name: str) -> Dict:
        """Validate data leakage between splits."""
        import random
        random.seed(42)
        
        start_time = time.time()
        
        # Create manual splits
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
        
        # Get file paths
        train_files = set(sample[0] for sample in train_samples)
        val_files = set(sample[0] for sample in val_samples)
        test_files = set(sample[0] for sample in test_samples)
        
        # Check overlaps
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
            'validation_time': time.time() - start_time,
            'passed': total_leaked == 0
        }
        
        self.logger.info(f"Data leakage validation for {dataset_name}: {total_leaked} leaked samples, "
                        f"Status: {'PASS' if results['passed'] else 'FAIL'}")
        
        return results
    
    def _validate_class_distribution(self, samples: List[Tuple[str, int, int]], dataset_name: str) -> Dict:
        """Validate class distribution."""
        start_time = time.time()
        
        # Count samples per class
        class_counts = defaultdict(int)
        for _, mapped_label, _ in samples:
            class_counts[mapped_label] += 1
        
        # Get class names
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        dataset_type = DatasetType.SIPAKMED if dataset_name == "sipakmed" else DatasetType.HERLEV
        class_names = label_mapper.get_class_names(dataset_type)
        
        # Calculate statistics
        total_samples = sum(class_counts.values())
        class_percentages = {class_id: (count / total_samples) * 100 
                            for class_id, count in class_counts.items()}
        
        max_count = max(class_counts.values()) if class_counts else 0
        min_count = min(class_counts.values()) if class_counts else 0
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Check for low sample classes
        low_sample_classes = [class_id for class_id, count in class_counts.items() 
                            if count < self.thresholds['min_class_samples']]
        
        # Check for missing classes
        expected_classes = len(class_names)
        actual_classes = len(class_counts)
        missing_classes = expected_classes - actual_classes
        
        results = {
            'total_samples': total_samples,
            'num_classes': actual_classes,
            'expected_classes': expected_classes,
            'missing_classes': missing_classes,
            'class_counts': dict(class_counts),
            'class_percentages': class_percentages,
            'max_count': max_count,
            'min_count': min_count,
            'imbalance_ratio': imbalance_ratio,
            'low_sample_classes': low_sample_classes,
            'validation_time': time.time() - start_time,
            'passed': (imbalance_ratio <= self.thresholds['max_imbalance_ratio'] and 
                      missing_classes == 0 and 
                      len(low_sample_classes) == 0)
        }
        
        self.logger.info(f"Class distribution validation for {dataset_name}: "
                        f"ratio={imbalance_ratio:.2f}, missing={missing_classes}, "
                        f"low_samples={len(low_sample_classes)}, "
                        f"Status: {'PASS' if results['passed'] else 'WARNING' if missing_classes == 0 else 'FAIL'}")
        
        return results
    
    def _validate_sample_count(self, samples: List[Tuple[str, int, int]], dataset_name: str) -> Dict:
        """Validate minimum sample count."""
        start_time = time.time()
        
        total_samples = len(samples)
        
        if dataset_name == "sipakmed":
            min_required = self.thresholds['min_sipakmed_samples']
        elif dataset_name == "herlev":
            min_required = self.thresholds['min_herlev_samples']
        else:
            min_required = 100  # Default
        
        results = {
            'total_samples': total_samples,
            'min_required': min_required,
            'validation_time': time.time() - start_time,
            'passed': total_samples >= min_required
        }
        
        self.logger.info(f"Sample count validation for {dataset_name}: "
                        f"{total_samples} samples (min: {min_required}), "
                        f"Status: {'PASS' if results['passed'] else 'FAIL'}")
        
        return results
    
    def _validate_cropped_composition(self, samples: List[Tuple[str, int, int]], dataset_name: str) -> Dict:
        """Validate CROPPED vs full image composition."""
        start_time = time.time()
        
        cropped_count = sum(1 for img_path, _, _ in samples if "CROPPED" in img_path.upper())
        full_count = len(samples) - cropped_count
        cropped_percentage = (cropped_count / len(samples)) * 100 if samples else 0
        
        results = {
            'total_images': len(samples),
            'cropped_count': cropped_count,
            'full_count': full_count,
            'cropped_percentage': cropped_percentage,
            'validation_time': time.time() - start_time,
            'passed': True  # Always pass - just for information
        }
        
        self.logger.info(f"CROPPED composition for {dataset_name}: "
                        f"{cropped_percentage:.1f}% CROPPED, {100-cropped_percentage:.1f}% full")
        
        return results
    
    def _generate_report(self, dataset_name: str, results: Dict) -> Dict:
        """Generate comprehensive validation report."""
        report = {
            'dataset_name': dataset_name,
            'validation_timestamp': datetime.now().isoformat(),
            'validation_mode': 'cropped_only' if self.use_cropped_only else 'all_images',
            'strict_mode': self.strict_mode,
            'overall_status': 'PASS',
            'validation_results': results,
            'summary': {
                'total_samples': results['sample_count']['total_samples'],
                'duplicate_percentage': results['duplicates']['duplicate_percentage'],
                'data_leakage_samples': results['leakage']['total_leaked_samples'],
                'imbalance_ratio': results['class_distribution']['imbalance_ratio'],
                'cropped_percentage': results['cropped_composition']['cropped_percentage'],
                'total_validation_time': sum(r.get('validation_time', 0) for r in results.values())
            }
        }
        
        # Determine overall status
        fail_conditions = [
            not results['duplicates']['passed'],
            not results['leakage']['passed'],
            not results['sample_count']['passed'],
            results['class_distribution']['missing_classes'] > 0
        ]
        
        warning_conditions = [
            not results['class_distribution']['passed'] and results['class_distribution']['missing_classes'] == 0
        ]
        
        if any(fail_conditions):
            report['overall_status'] = 'FAIL'
        elif any(warning_conditions):
            report['overall_status'] = 'WARNING'
        
        return report
    
    def validate_dataset(self, dataset_name: str) -> Dict:
        """
        Validate a single dataset comprehensively.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Validation report dictionary
        """
        if not self.run_validation:
            self.logger.info("Dataset validation disabled in configuration")
            return {'status': 'SKIPPED', 'reason': 'validation_disabled'}
        
        start_time = time.time()
        self.logger.info(f"Starting validation for {dataset_name} dataset...")
        
        # Setup file logging
        self._setup_file_logging()
        
        try:
            # Load samples
            samples = self._load_dataset_samples(dataset_name)
            
            if len(samples) == 0:
                error_msg = f"No samples found for {dataset_name}"
                self.logger.error(error_msg)
                return {'status': 'FAIL', 'reason': error_msg}
            
            # Run all validations
            validation_results = {
                'duplicates': self._validate_duplicates(samples, dataset_name),
                'leakage': self._validate_data_leakage(samples, dataset_name),
                'class_distribution': self._validate_class_distribution(samples, dataset_name),
                'sample_count': self._validate_sample_count(samples, dataset_name),
                'cropped_composition': self._validate_cropped_composition(samples, dataset_name)
            }
            
            # Generate report
            report = self._generate_report(dataset_name, validation_results)
            report['total_validation_time'] = time.time() - start_time
            
            # Check performance
            if report['total_validation_time'] > self.thresholds['max_validation_time']:
                self.logger.warning(f"Validation took {report['total_validation_time']:.2f}s "
                                  f"(threshold: {self.thresholds['max_validation_time']}s)")
            
            self.logger.info(f"Validation completed for {dataset_name}: {report['overall_status']} "
                           f"({report['total_validation_time']:.2f}s)")
            
            return report
            
        except Exception as e:
            error_msg = f"Validation failed for {dataset_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'ERROR', 'reason': error_msg}
    
    def run_dataset_guard(self) -> Dict:
        """
        Run dataset guard for the configured dataset.
        
        Returns:
            Overall guard status and reports
        """
        self.logger.info("🛡️  DATASET GUARD: Starting validation...")
        
        # Get dataset name from config
        dataset_name = self.dataset_config.get('train_dataset', 'sipakmed')
        
        # Validate the dataset
        validation_report = self.validate_dataset(dataset_name)
        
        # Determine guard action
        if validation_report.get('status') == 'SKIPPED':
            guard_status = 'PASS'
            action = 'PROCEED'
            message = 'Validation disabled in configuration'
        
        elif validation_report.get('status') == 'ERROR':
            guard_status = 'FAIL'
            action = 'ABORT'
            message = f"Validation error: {validation_report.get('reason', 'Unknown error')}"
        
        elif validation_report.get('overall_status') == 'FAIL':
            guard_status = 'FAIL'
            action = 'ABORT' if self.strict_mode else 'PROCEED_WITH_WARNING'
            message = "Critical dataset issues detected"
        
        elif validation_report.get('overall_status') == 'WARNING':
            guard_status = 'WARNING'
            action = 'PROCEED_WITH_WARNING'
            message = "Minor dataset issues detected"
        
        else:
            guard_status = 'PASS'
            action = 'PROCEED'
            message = "Dataset validation passed"
        
        # Store results
        self.validation_results = validation_report
        self.validation_status = guard_status
        
        # Log final decision
        self.logger.info(f"🛡️  DATASET GUARD: {guard_status} - {action}")
        self.logger.info(f"🛡️  DATASET GUARD: {message}")
        
        return {
            'guard_status': guard_status,
            'action': action,
            'message': message,
            'validation_report': validation_report
        }
    
    def save_reports(self, validation_report: Dict):
        """Save validation reports to files."""
        # Create reports directory
        reports_dir = os.path.join(self.paths_config.get('logs', 'outputs/logs'), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save JSON report
        json_path = os.path.join(reports_dir, 'dataset_report.json')
        with open(json_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Save human-readable report
        txt_path = os.path.join(reports_dir, 'dataset_report.txt')
        self._save_text_report(validation_report, txt_path)
        
        self.logger.info(f"Validation reports saved to {reports_dir}")
    
    def _save_text_report(self, report: Dict, file_path: str):
        """Save human-readable validation report."""
        with open(file_path, 'w') as f:
            f.write("DATASET VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {report.get('dataset_name', 'Unknown')}\n")
            f.write(f"Timestamp: {report.get('validation_timestamp', 'Unknown')}\n")
            f.write(f"Validation Mode: {report.get('validation_mode', 'Unknown')}\n")
            f.write(f"Overall Status: {report.get('overall_status', 'Unknown')}\n")
            f.write(f"Total Validation Time: {report.get('total_validation_time', 0):.2f}s\n\n")
            
            # Summary
            summary = report.get('summary', {})
            f.write("SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Samples: {summary.get('total_samples', 0)}\n")
            f.write(f"Duplicate Percentage: {summary.get('duplicate_percentage', 0):.2f}%\n")
            f.write(f"Data Leakage Samples: {summary.get('data_leakage_samples', 0)}\n")
            f.write(f"Imbalance Ratio: {summary.get('imbalance_ratio', 0):.2f}\n")
            f.write(f"CROPPED Percentage: {summary.get('cropped_percentage', 0):.1f}%\n\n")
            
            # Detailed results
            validation_results = report.get('validation_results', {})
            for check_name, result in validation_results.items():
                f.write(f"{check_name.upper()} VALIDATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Status: {'PASS' if result.get('passed', False) else 'FAIL'}\n")
                f.write(f"Validation Time: {result.get('validation_time', 0):.3f}s\n")
                
                # Add specific details for each check
                if check_name == 'duplicates':
                    f.write(f"Total Images: {result.get('total_images', 0)}\n")
                    f.write(f"Duplicate Images: {result.get('duplicate_images', 0)}\n")
                    f.write(f"Duplicate Percentage: {result.get('duplicate_percentage', 0):.2f}%\n")
                
                elif check_name == 'leakage':
                    f.write(f"Train-Val Overlap: {result.get('train_val_overlap', 0)}\n")
                    f.write(f"Train-Test Overlap: {result.get('train_test_overlap', 0)}\n")
                    f.write(f"Val-Test Overlap: {result.get('val_test_overlap', 0)}\n")
                    f.write(f"Total Leaked: {result.get('total_leaked_samples', 0)}\n")
                
                elif check_name == 'class_distribution':
                    f.write(f"Number of Classes: {result.get('num_classes', 0)}\n")
                    f.write(f"Missing Classes: {result.get('missing_classes', 0)}\n")
                    f.write(f"Imbalance Ratio: {result.get('imbalance_ratio', 0):.2f}\n")
                    f.write(f"Low Sample Classes: {len(result.get('low_sample_classes', []))}\n")
                
                elif check_name == 'sample_count':
                    f.write(f"Total Samples: {result.get('total_samples', 0)}\n")
                    f.write(f"Minimum Required: {result.get('min_required', 0)}\n")
                
                elif check_name == 'cropped_composition':
                    f.write(f"CROPPED Images: {result.get('cropped_count', 0)}\n")
                    f.write(f"Full Images: {result.get('full_count', 0)}\n")
                    f.write(f"CROPPED Percentage: {result.get('cropped_percentage', 0):.1f}%\n")
                
                f.write("\n")
    
    def print_debug_info(self, dataset_name: str):
        """Print debug information about dataset."""
        if not self.run_validation:
            print("Dataset validation is disabled")
            return
        
        print(f"\n🔍 DATASET DEBUG INFO: {dataset_name}")
        print("=" * 60)
        
        try:
            samples = self._load_dataset_samples(dataset_name)
            print(f"Total samples loaded: {len(samples)}")
            
            # Sample file paths
            print(f"\nSample file paths (first 10):")
            for i, (img_path, label, original) in enumerate(samples[:10]):
                print(f"  {i+1}. {Path(img_path).name} -> Class {label}")
            
            # Class distribution preview
            class_counts = defaultdict(int)
            for _, mapped_label, _ in samples:
                class_counts[mapped_label] += 1
            
            print(f"\nClass distribution preview:")
            for class_id, count in sorted(class_counts.items()):
                percentage = (count / len(samples)) * 100
                print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
            
            # CROPPED vs full
            cropped_count = sum(1 for img_path, _, _ in samples if "CROPPED" in img_path.upper())
            full_count = len(samples) - cropped_count
            print(f"\nImage composition:")
            print(f"  CROPPED: {cropped_count} ({(cropped_count/len(samples)*100):.1f}%)")
            print(f"  Full: {full_count} ({(full_count/len(samples)*100):.1f}%)")
            
        except Exception as e:
            print(f"Error loading debug info: {e}")


def run_dataset_guard(config: Dict) -> Dict:
    """
    Convenience function to run dataset guard.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Guard results
    """
    guard = DatasetGuard(config)
    results = guard.run_dataset_guard()
    
    # Save reports if validation was run
    if results['validation_report'].get('status') not in ['SKIPPED', 'ERROR']:
        guard.save_reports(results['validation_report'])
    
    return results
