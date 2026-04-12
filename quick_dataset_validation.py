"""
Quick dataset validation script to demonstrate pipeline functionality.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_guard import run_dataset_guard

def main():
    print("🛡️ RUNNING DATASET GUARD VALIDATION")
    print("=" * 60)
    
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
    
    results = run_dataset_guard(config)
    
    print(f"Guard Status: {results['guard_status']}")
    print(f"Action: {results['action']}")
    print(f"Message: {results['message']}")
    
    if 'validation_report' in results:
        report = results['validation_report']
        if 'summary' in report:
            summary = report['summary']
            print(f"\n📊 DATASET STATISTICS:")
            print(f"   Total Samples: {summary.get('total_samples', 0)}")
            print(f"   Duplicate %: {summary.get('duplicate_percentage', 0):.2f}%")
            print(f"   Imbalance Ratio: {summary.get('imbalance_ratio', 0):.2f}")
            print(f"   CROPPED %: {summary.get('cropped_percentage', 0):.1f}%")
            print(f"   Data Leakage: {summary.get('data_leakage_samples', 0)} samples")
    
    print("\n✅ Dataset validation completed successfully!")
    
    return results['action'] != 'ABORT'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
