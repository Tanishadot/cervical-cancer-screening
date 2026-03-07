"""
Kaggle dataset downloader utility for medical imaging datasets.
Supports automatic CLI installation, API key verification, and dataset preparation.
"""

import os
import subprocess
import sys
import json
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KaggleDownloader:
    """
    Comprehensive Kaggle dataset downloader with CLI management and dataset preparation.
    """
    
    def __init__(self, project_root: str = None):
        """
        Initialize Kaggle downloader.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.datasets_dir = self.project_root / "datasets"
        self.kaggle_dir = Path.home() / ".kaggle"
        self.kaggle_config = self.kaggle_dir / "kaggle.json"
        self.kaggle_command = None
        
        # Ensure datasets directory exists
        self.datasets_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Kaggle downloader for project: {self.project_root}")
    
    def check_kaggle_cli(self) -> bool:
        """
        Check if Kaggle CLI is installed.
        
        Returns:
            True if Kaggle CLI is available, False otherwise
        """
        try:
            # Try multiple methods to find kaggle command
            kaggle_commands = [
                "kaggle",
                "kaggle.exe"
            ]
            
            # Add common Python installation paths
            python_paths = [
                sys.prefix,
                os.path.join(sys.prefix, "Scripts"),
                os.path.join(sys.prefix, "bin"),
                # Windows Store Python paths
                os.path.join(os.path.expanduser("~"), "AppData", "Local", "Packages", "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0", "LocalCache", "local-packages", "Python311", "Scripts"),
                os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python311", "Scripts"),
                os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python311", "Scripts"),
                # Additional Windows paths
                os.path.join(os.path.expanduser("~"), "AppData", "Local", "Packages", "PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0", "LocalCache", "local-packages", "Python311", "Scripts")
            ]
            
            # Try PATH and specific paths
            for base_cmd in kaggle_commands:
                # First try PATH
                if shutil.which(base_cmd):
                    try:
                        result = subprocess.run(
                            [base_cmd, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            version = result.stdout.strip()
                            logger.info(f"Kaggle CLI found: {version}")
                            self.kaggle_command = base_cmd
                            return True
                    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                        continue
                
                # Then try specific paths
                for path in python_paths:
                    kaggle_cmd = os.path.join(path, base_cmd)
                    if os.path.exists(kaggle_cmd):
                        try:
                            result = subprocess.run(
                                [kaggle_cmd, "--version"],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if result.returncode == 0:
                                version = result.stdout.strip()
                                logger.info(f"Kaggle CLI found: {version}")
                                self.kaggle_command = kaggle_cmd
                                return True
                        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                            continue
            
            logger.warning("Kaggle CLI not found")
            return False
        except Exception as e:
            logger.warning(f"Error checking Kaggle CLI: {e}")
            return False
    
    def install_kaggle_cli(self) -> bool:
        """
        Install Kaggle CLI using pip.
        
        Returns:
            True if installation successful, False otherwise
        """
        logger.info("Installing Kaggle CLI...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "kaggle"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Kaggle CLI installed successfully")
                # Refresh the command detection after installation
                return self.check_kaggle_cli()
            else:
                logger.error(f"Failed to install Kaggle CLI: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Kaggle CLI installation timed out")
            return False
        except Exception as e:
            logger.error(f"Error installing Kaggle CLI: {e}")
            return False
    
    def verify_api_key(self) -> bool:
        """
        Verify that Kaggle API key exists and is valid.
        
        Returns:
            True if API key exists, False otherwise
        """
        if not self.kaggle_config.exists():
            logger.error(f"Kaggle API key not found at: {self.kaggle_config}")
            self._print_api_key_instructions()
            return False
        
        try:
            with open(self.kaggle_config, 'r') as f:
                api_config = json.load(f)
            
            username = api_config.get('username')
            key = api_config.get('key')
            
            if not username or not key:
                logger.error("Invalid Kaggle API key format")
                self._print_api_key_instructions()
                return False
            
            logger.info(f"Kaggle API key found for user: {username}")
            return True
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in kaggle.json")
            self._print_api_key_instructions()
            return False
        except Exception as e:
            logger.error(f"Error reading Kaggle API key: {e}")
            self._print_api_key_instructions()
            return False
    
    def _print_api_key_instructions(self):
        """Print instructions for obtaining Kaggle API key."""
        logger.info("="*60)
        logger.info("KAGGLE API KEY SETUP INSTRUCTIONS")
        logger.info("="*60)
        logger.info("1. Go to https://www.kaggle.com/")
        logger.info("2. Sign in to your Kaggle account")
        logger.info("3. Click on your profile picture → Account")
        logger.info("4. Scroll down to 'API' section")
        logger.info("5. Click 'Create New API Token'")
        logger.info("6. Download the kaggle.json file")
        logger.info("")
        logger.info(f"7. Place the file in: {self.kaggle_dir}")
        logger.info("   - On Windows: C:/Users/<YourUsername>/.kaggle/kaggle.json")
        logger.info("   - On Mac/Linux: ~/.kaggle/kaggle.json")
        logger.info("")
        logger.info("8. The file should contain:")
        logger.info('   {"username":"your_username","key":"your_api_key"}')
        logger.info("="*60)
    
    def download_dataset(
        self,
        dataset_name: str,
        target_dir: str,
        unzip: bool = True,
        cleanup_zip: bool = True
    ) -> bool:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset name (e.g., "andrewmvd/cervical-cancer-dataset")
            target_dir: Target directory name within datasets folder
            unzip: Whether to unzip the downloaded file
            cleanup_zip: Whether to remove zip file after extraction
            
        Returns:
            True if download successful, False otherwise
        """
        target_path = self.datasets_dir / target_dir
        target_path.mkdir(exist_ok=True)
        
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info(f"Target directory: {target_path}")
        
        try:
            # Check if dataset already exists
            if self._dataset_exists(target_path):
                logger.info("Dataset already exists. Skipping download.")
                return True
            
            # Use the stored kaggle command
            if not self.kaggle_command:
                logger.error("Kaggle command not available")
                return False
            
            # Download dataset
            cmd = [self.kaggle_command, "datasets", "download", "-d", dataset_name, "--path", str(target_path)]
            
            logger.info(f"Using command: {' '.join(cmd)}")
            logger.info("Starting download...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                return False
            
            logger.info("Download completed successfully")
            
            # Unzip if requested
            if unzip:
                success = self._unzip_dataset(target_path, cleanup_zip)
                if not success:
                    return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Download timed out")
            return False
        except Exception as e:
            logger.error(f"Error during download: {e}")
            return False
    
    def _dataset_exists(self, target_dir: Path) -> bool:
        """
        Check if dataset already exists in target directory.
        
        Args:
            target_dir: Target directory path
            
        Returns:
            True if dataset exists, False otherwise
        """
        # Check for common indicators of existing dataset
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for file_path in target_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Count images (if more than 10, assume dataset exists)
                image_count = sum(1 for f in target_dir.rglob('*') 
                                if f.is_file() and f.suffix.lower() in image_extensions)
                return image_count > 10
        
        return False
    
    def _unzip_dataset(self, target_dir: Path, cleanup_zip: bool = True) -> bool:
        """
        Unzip downloaded dataset files.
        
        Args:
            target_dir: Directory containing zip files
            cleanup_zip: Whether to remove zip files after extraction
            
        Returns:
            True if unzipping successful, False otherwise
        """
        logger.info("Extracting dataset files...")
        
        zip_files = list(target_dir.glob('*.zip'))
        
        if not zip_files:
            logger.warning("No zip files found in target directory")
            return True
        
        for zip_file in zip_files:
            try:
                logger.info(f"Extracting: {zip_file.name}")
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Get total number of files for progress bar
                    file_list = zip_ref.namelist()
                    
                    with tqdm(total=len(file_list), desc="Extracting") as pbar:
                        for file in file_list:
                            zip_ref.extract(file, target_dir)
                            pbar.update(1)
                
                if cleanup_zip:
                    zip_file.unlink()
                    logger.info(f"Removed zip file: {zip_file.name}")
                
            except Exception as e:
                logger.error(f"Error extracting {zip_file}: {e}")
                return False
        
        logger.info("Dataset extraction completed")
        return True
    
    def organize_sipakmed_dataset(self) -> bool:
        """
        Organize SIPaKMeD dataset into correct class structure.
        
        Expected final structure:
        datasets/sipakmed/
            superficial_intermediate/
            parabasal/
            koilocytotic/
            metaplastic/
            dyskeratotic/
        
        Returns:
            True if organization successful, False otherwise
        """
        sipakmed_dir = self.datasets_dir / "sipakmed"
        
        if not sipakmed_dir.exists():
            logger.error("SIPaKMeD dataset directory not found")
            return False
        
        logger.info("Organizing SIPaKMeD dataset structure...")
        
        # Define class mappings (common SIPaKMeD structure)
        class_mappings = {
            # Common SIPaKMeD class folder names to standardized names
            'superficial-intermediate': 'superficial_intermediate',
            'superficial_intermediate': 'superficial_intermediate',
            'parabasal': 'parabasal',
            'koilocytotic': 'koilocytotic',
            'metaplastic': 'metaplastic',
            'dyskeratotic': 'dyskeratotic',
            # Alternative names
            'superficial': 'superficial_intermediate',
            'intermediate': 'superficial_intermediate',
            'dysplastic': 'koilocytotic',
            'carcinoma': 'dyskeratotic'
        }
        
        target_classes = [
            'superficial_intermediate',
            'parabasal', 
            'koilocytotic',
            'metaplastic',
            'dyskeratotic'
        ]
        
        # Create target class directories
        for class_name in target_classes:
            (sipakmed_dir / class_name).mkdir(exist_ok=True)
        
        # Find and organize image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        organized_count = 0
        
        # Walk through all subdirectories
        for item in sipakmed_dir.iterdir():
            if item.is_dir() and item.name not in target_classes:
                # This might be a class directory with alternative name
                class_name_lower = item.name.lower()
                
                # Try to map to standard class name
                mapped_class = None
                for alt_name, standard_name in class_mappings.items():
                    if alt_name in class_name_lower or class_name_lower in alt_name:
                        mapped_class = standard_name
                        break
                
                if mapped_class:
                    # Move images to correct class directory
                    target_dir = sipakmed_dir / mapped_class
                    logger.info(f"Moving {item.name} -> {mapped_class}")
                    
                    for img_file in item.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                            target_path = target_dir / img_file.name
                            shutil.move(str(img_file), str(target_path))
                            organized_count += 1
                    
                    # Remove empty directory
                    try:
                        item.rmdir()
                    except OSError:
                        logger.warning(f"Could not remove directory: {item}")
                else:
                    logger.warning(f"Unknown class directory: {item.name}")
        
        logger.info(f"Dataset organization completed. Organized {organized_count} images.")
        
        # Verify final structure
        final_structure = {}
        for class_name in target_classes:
            class_dir = sipakmed_dir / class_name
            if class_dir.exists():
                image_count = len([f for f in class_dir.iterdir() 
                                 if f.is_file() and f.suffix.lower() in image_extensions])
                final_structure[class_name] = image_count
        
        logger.info("Final dataset structure:")
        for class_name, count in final_structure.items():
            logger.info(f"  {class_name}: {count} images")
        
        return True
    
    def setup_sipakmed_dataset(self, dataset_name: str = "marinaeplissiti/sipakmed") -> bool:
        """
        Complete setup of SIPaKMeD dataset.
        
        Args:
            dataset_name: Kaggle dataset name
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("="*60)
        logger.info("SIPaKMeD DATASET SETUP")
        logger.info("="*60)
        
        # Step 1: Check Kaggle CLI
        logger.info("Step 1: Checking Kaggle CLI...")
        if not self.check_kaggle_cli():
            logger.info("Installing Kaggle CLI...")
            if not self.install_kaggle_cli():
                logger.error("Failed to install Kaggle CLI")
                return False
        
        # Step 2: Verify API key
        logger.info("Step 2: Verifying Kaggle API key...")
        if not self.verify_api_key():
            logger.error("Kaggle API key not configured")
            return False
        
        # Step 3: Download dataset
        logger.info("Step 3: Downloading SIPaKMeD dataset...")
        if not self.download_dataset(dataset_name, "sipakmed"):
            logger.error("Failed to download dataset")
            return False
        
        # Step 4: Organize dataset
        logger.info("Step 4: Organizing dataset structure...")
        if not self.organize_sipakmed_dataset():
            logger.error("Failed to organize dataset")
            return False
        
        logger.info("="*60)
        logger.info("SIPaKMeD DATASET SETUP COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return True


def create_kaggle_downloader(project_root: str = None) -> KaggleDownloader:
    """
    Factory function to create Kaggle downloader instance.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        KaggleDownloader instance
    """
    return KaggleDownloader(project_root)


if __name__ == "__main__":
    # Test the downloader
    downloader = create_kaggle_downloader()
    print("Kaggle downloader utility ready for use")
