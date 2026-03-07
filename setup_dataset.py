"""
Dataset setup script for cervical cancer classification pipeline.
Automatically downloads and prepares medical imaging datasets from Kaggle.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.kaggle_downloader import create_kaggle_downloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_setup.log')
    ]
)
logger = logging.getLogger(__name__)


class DatasetSetupManager:
    """
    Manager for dataset setup operations.
    """
    
    def __init__(self, project_root: str = None):
        """
        Initialize dataset setup manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.downloader = create_kaggle_downloader(str(self.project_root))
        
        logger.info(f"Dataset setup manager initialized for: {self.project_root}")
    
    def setup_sipakmed_dataset(self, force_download: bool = False) -> bool:
        """
        Setup SIPaKMeD dataset for cervical cancer classification.
        
        Args:
            force_download: Force re-download even if dataset exists
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("="*80)
        logger.info("SIPaKMeD CERVICAL CANCER DATASET SETUP")
        logger.info("="*80)
        
        sipakmed_dir = self.project_root / "datasets" / "sipakmed"
        
        # Check if dataset already exists
        if sipakmed_dir.exists() and not force_download:
            # Verify dataset structure
            expected_classes = [
                'superficial_intermediate',
                'parabasal',
                'koilocytotic', 
                'metaplastic',
                'dyskeratotic'
            ]
            
            existing_classes = [d.name for d in sipakmed_dir.iterdir() if d.is_dir()]
            
            if all(cls in existing_classes for cls in expected_classes):
                # Count images in each class
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                
                logger.info("SIPaKMeD dataset already exists with correct structure:")
                for class_name in expected_classes:
                    class_dir = sipakmed_dir / class_name
                    if class_dir.exists():
                        image_count = len([f for f in class_dir.iterdir() 
                                         if f.is_file() and f.suffix.lower() in image_extensions])
                        logger.info(f"  {class_name}: {image_count} images")
                
                logger.info("Skipping download. Use --force to re-download.")
                return True
        
        # Download and setup dataset
        logger.info("Setting up SIPaKMeD dataset...")
        
        # Common SIPaKMeD dataset names on Kaggle
        dataset_options = [
            "marinaeplissiti/sipakmed",
            "andrewmvd/cervical-cancer-dataset",
            "birdy654/cervical-cancer-dataset",
            "aryashah2k/cervical-cancer-classification-dataset"
        ]
        
        # Try each dataset option until one works
        for dataset_name in dataset_options:
            logger.info(f"Attempting to download: {dataset_name}")
            
            try:
                success = self.downloader.setup_sipakmed_dataset(dataset_name)
                if success:
                    logger.info(f"Successfully downloaded and organized: {dataset_name}")
                    return True
                else:
                    logger.warning(f"Failed to download: {dataset_name}")
                    continue
            except Exception as e:
                logger.warning(f"Error with {dataset_name}: {e}")
                continue
        
        logger.error("Failed to download SIPaKMeD dataset from all available sources")
        logger.info("Please check Kaggle for available cervical cancer datasets")
        return False
    
    def setup_herlev_dataset(self, force_download: bool = False) -> bool:
        """
        Setup Herlev dataset for cervical cancer classification.
        
        Args:
            force_download: Force re-download even if dataset exists
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("="*80)
        logger.info("HERLEV CERVICAL CANCER DATASET SETUP")
        logger.info("="*80)
        
        herlev_dir = self.project_root / "datasets" / "herlev"
        
        # Check if dataset already exists
        if herlev_dir.exists() and not force_download:
            expected_classes = [
                'normal_superficial',
                'normal_intermediate', 
                'normal_columnar',
                'mild_dysplasia',
                'moderate_dysplasia',
                'severe_dysplasia',
                'carcinoma_in_situ'
            ]
            
            existing_classes = [d.name for d in herlev_dir.iterdir() if d.is_dir()]
            
            if any(cls in existing_classes for cls in expected_classes):
                logger.info("Herlev dataset already exists")
                logger.info("Skipping download. Use --force to re-download.")
                return True
        
        logger.info("Herlev dataset setup not yet implemented")
        logger.info("Please manually download and organize the Herlev dataset")
        logger.info("Expected structure:")
        for class_name in expected_classes:
            logger.info(f"  datasets/herlev/{class_name}/")
        
        return False
    
    def verify_dataset_structure(self) -> bool:
        """
        Verify that datasets are properly structured.
        
        Returns:
            True if structure is correct, False otherwise
        """
        logger.info("="*80)
        logger.info("VERIFYING DATASET STRUCTURE")
        logger.info("="*80)
        
        datasets_dir = self.project_root / "datasets"
        
        if not datasets_dir.exists():
            logger.error("Datasets directory not found")
            return False
        
        # Check SIPaKMeD structure
        sipakmed_dir = datasets_dir / "sipakmed"
        if sipakmed_dir.exists():
            logger.info("Checking SIPaKMeD dataset structure...")
            
            expected_sipakmed = [
                'superficial_intermediate',
                'parabasal',
                'koilocytotic',
                'metaplastic',
                'dyskeratotic'
            ]
            
            sipakmed_ok = True
            for class_name in expected_sipakmed:
                class_dir = sipakmed_dir / class_name
                if class_dir.exists():
                    image_count = len([f for f in class_dir.iterdir() if f.is_file()])
                    logger.info(f"  ✓ {class_name}: {image_count} images")
                else:
                    logger.warning(f"  ✗ {class_name}: Missing")
                    sipakmed_ok = False
            
            if sipakmed_ok:
                logger.info("SIPaKMeD dataset structure: OK")
            else:
                logger.warning("SIPaKMeD dataset structure: INCOMPLETE")
        
        # Check Herlev structure
        herlev_dir = datasets_dir / "herlev"
        if herlev_dir.exists():
            logger.info("Checking Herlev dataset structure...")
            
            expected_herlev = [
                'normal_superficial',
                'normal_intermediate',
                'normal_columnar', 
                'mild_dysplasia',
                'moderate_dysplasia',
                'severe_dysplasia',
                'carcinoma_in_situ'
            ]
            
            herlev_ok = True
            for class_name in expected_herlev:
                class_dir = herlev_dir / class_name
                if class_dir.exists():
                    image_count = len([f for f in class_dir.iterdir() if f.is_file()])
                    logger.info(f"  ✓ {class_name}: {image_count} images")
                else:
                    logger.warning(f"  ✗ {class_name}: Missing")
                    herlev_ok = False
            
            if herlev_ok:
                logger.info("Herlev dataset structure: OK")
            else:
                logger.warning("Herlev dataset structure: INCOMPLETE")
        
        logger.info("="*80)
        logger.info("DATASET VERIFICATION COMPLETED")
        logger.info("="*80)
        
        return True
    
    def show_dataset_info(self):
        """Show information about available datasets."""
        logger.info("="*80)
        logger.info("DATASET INFORMATION")
        logger.info("="*80)
        
        datasets_dir = self.project_root / "datasets"
        
        if not datasets_dir.exists():
            logger.info("No datasets directory found")
            return
        
        # SIPaKMeD info
        sipakmed_dir = datasets_dir / "sipakmed"
        if sipakmed_dir.exists():
            logger.info("SIPaKMeD Dataset:")
            logger.info(f"  Location: {sipakmed_dir}")
            
            total_images = 0
            for class_dir in sipakmed_dir.iterdir():
                if class_dir.is_dir():
                    image_count = len([f for f in class_dir.iterdir() if f.is_file()])
                    logger.info(f"  {class_dir.name}: {image_count} images")
                    total_images += image_count
            
            logger.info(f"  Total: {total_images} images")
            logger.info("")
        
        # Herlev info
        herlev_dir = datasets_dir / "herlev"
        if herlev_dir.exists():
            logger.info("Herlev Dataset:")
            logger.info(f"  Location: {herlev_dir}")
            
            total_images = 0
            for class_dir in herlev_dir.iterdir():
                if class_dir.is_dir():
                    image_count = len([f for f in class_dir.iterdir() if f.is_file()])
                    logger.info(f"  {class_dir.name}: {image_count} images")
                    total_images += image_count
            
            logger.info(f"  Total: {total_images} images")
        
        logger.info("="*80)


def main():
    """Main function for dataset setup."""
    parser = argparse.ArgumentParser(
        description="Setup medical imaging datasets for cervical cancer classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_dataset.py                          # Setup SIPaKMeD dataset
  python setup_dataset.py --dataset sipakmed       # Setup SIPaKMeD dataset
  python setup_dataset.py --dataset herlev         # Setup Herlev dataset
  python setup_dataset.py --force                 # Force re-download
  python setup_dataset.py --verify                 # Verify dataset structure
  python setup_dataset.py --info                   # Show dataset information
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['sipakmed', 'herlev', 'all'],
        default='sipakmed',
        help='Dataset to setup (default: sipakmed)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset structure only'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset information'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        help='Project root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Create setup manager
    manager = DatasetSetupManager(args.project_root)
    
    # Handle different modes
    if args.info:
        manager.show_dataset_info()
        return
    
    if args.verify:
        manager.verify_dataset_structure()
        return
    
    # Setup datasets
    success = True
    
    if args.dataset in ['sipakmed', 'all']:
        logger.info("Setting up SIPaKMeD dataset...")
        sipakmed_success = manager.setup_sipakmed_dataset(args.force)
        success = success and sipakmed_success
    
    if args.dataset in ['herlev', 'all']:
        logger.info("Setting up Herlev dataset...")
        herlev_success = manager.setup_herlev_dataset(args.force)
        success = success and herlev_success
    
    # Verify final structure
    if success:
        logger.info("\nVerifying final dataset structure...")
        manager.verify_dataset_structure()
        
        logger.info("\n" + "="*80)
        logger.info("DATASET SETUP COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("You can now run the cervical cancer classification pipeline:")
        logger.info("  python main.py")
        logger.info("\nFor binary classification:")
        logger.info("  python main.py --binary")
        logger.info("\nFor multi-class classification:")
        logger.info("  python main.py --multiclass")
        logger.info("="*80)
    else:
        logger.error("\n" + "="*80)
        logger.error("DATASET SETUP FAILED!")
        logger.error("Please check the error messages above and try again.")
        logger.error("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
