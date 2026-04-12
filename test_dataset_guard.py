"""
Test script for the dataset guard system.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_guard import run_dataset_guard
from utils.config_manager import load_config


def test_dataset_guard():
    """Test the dataset guard system."""
    print("🛡️ Testing Dataset Guard System...")
    print("=" * 60)
    
    try:
        # Load config
        config_manager = load_config('configs/config.yaml')
        config = config_manager.get_config()
        
        # Run dataset guard
        results = run_dataset_guard(config)
        
        print(f"Guard Status: {results['guard_status']}")
        print(f"Action: {results['action']}")
        print(f"Message: {results['message']}")
        
        if 'validation_report' in results:
            report = results['validation_report']
            if 'summary' in report:
                summary = report['summary']
                print(f"Total Samples: {summary.get('total_samples', 0)}")
                print(f"Duplicate %: {summary.get('duplicate_percentage', 0):.2f}%")
                print(f"Imbalance Ratio: {summary.get('imbalance_ratio', 0):.2f}")
                print(f"CROPPED %: {summary.get('cropped_percentage', 0):.1f}%")
        
        print("\n✅ Dataset guard test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Dataset guard test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_dataset_guard()
    sys.exit(0 if success else 1)
