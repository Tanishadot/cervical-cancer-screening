"""
Enhanced stain normalization with multi-image fitting for domain robustness.
Improves stain normalization by fitting on multiple representative images.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from sklearn.decomposition import PCA
import logging
from pathlib import Path
import random

from preprocessing.stain_normalization import MacenkoNormalizer, ReinhardNormalizer, VahadaneNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiImageMacenkoNormalizer:
    """
    Enhanced Macenko normalizer that fits on multiple images for better robustness.
    """
    
    def __init__(self):
        """Initialize multi-image Macenko normalizer."""
        self.stain_matrix = None
        self.target_concentrations = None
        self.target_mean = None
        self.target_std = None
        self.fitted_images = []
    
    def _get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        """
        Extract stain matrix using Macenko method.
        
        Args:
            img: Input RGB image
            
        Returns:
            Stain matrix
        """
        # Convert to optical density
        OD = -np.log((img.astype(np.float32) + 1) / 256)
        
        # Remove background
        ODhat = OD[~np.all(OD < 0.15, axis=1)]
        
        if len(ODhat) == 0:
            # Fallback: use all pixels if no background pixels found
            ODhat = OD.reshape(-1, 3)
        
        # Compute eigenvectors
        try:
            _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
            # Get two largest eigenvectors
            top_eigvecs = eigvecs[:, -2:]
        except:
            # Fallback: use PCA if eigendecomposition fails
            pca = PCA(n_components=2)
            pca.fit(ODhat)
            top_eigvecs = pca.components_.T
        
        # Compute stain matrix
        stains = ODhat @ top_eigvecs
        
        # Normalize stains
        stain_norm = stains / (np.linalg.norm(stains, axis=0, keepdims=True) + 1e-6)
        
        return stain_norm
    
    def _get_concentrations(self, img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """
        Get stain concentrations.
        
        Args:
            img: Input image
            stain_matrix: Stain matrix
            
        Returns:
            Stain concentrations
        """
        # Convert to optical density
        OD = -np.log((img.astype(np.float32) + 1) / 256)
        
        # Reshape for matrix multiplication
        OD_reshaped = OD.reshape(-1, 3)
        
        # Compute concentrations
        concentrations = OD_reshaped @ stain_matrix
        
        return concentrations.reshape(OD.shape[0], OD.shape[1], -1)
    
    def fit_single_image(self, img: np.ndarray):
        """Fit normalizer to a single image."""
        stain_matrix = self._get_stain_matrix(img)
        concentrations = self._get_concentrations(img, stain_matrix)
        
        # Store statistics
        mean_conc = np.mean(concentrations, axis=(0, 1))
        std_conc = np.std(concentrations, axis=(0, 1))
        
        return stain_matrix, mean_conc, std_conc, concentrations
    
    def fit(self, images: List[np.ndarray], sample_size: Optional[int] = None):
        """
        Fit normalizer to multiple images for robust statistics.
        
        Args:
            images: List of target images for normalization
            sample_size: Number of images to sample (if None, use all)
        """
        if not images:
            raise ValueError("No images provided for fitting")
        
        # Sample images if specified
        if sample_size is not None and len(images) > sample_size:
            images = random.sample(images, sample_size)
        
        logger.info(f"Fitting multi-image Macenko normalizer on {len(images)} images")
        
        # Collect statistics from all images
        all_stain_matrices = []
        all_concentrations = []
        all_means = []
        all_stds = []
        
        for i, img in enumerate(images):
            try:
                stain_matrix, mean_conc, std_conc, concentrations = self.fit_single_image(img)
                all_stain_matrices.append(stain_matrix)
                all_concentrations.append(concentrations)
                all_means.append(mean_conc)
                all_stds.append(std_conc)
                self.fitted_images.append(i)
            except Exception as e:
                logger.warning(f"Failed to process image {i}: {e}")
                continue
        
        if not all_stain_matrices:
            raise ValueError("Failed to process any images for fitting")
        
        # Average stain matrix
        self.stain_matrix = np.mean(all_stain_matrices, axis=0)
        
        # Combine all concentrations for robust statistics
        combined_concentrations = np.concatenate([c.reshape(-1, c.shape[-1]) for c in all_concentrations], axis=0)
        
        # Compute robust statistics
        self.target_mean = np.mean(combined_concentrations, axis=0)
        self.target_std = np.std(combined_concentrations, axis=0)
        
        # Store some representative concentrations for reconstruction
        self.target_concentrations = combined_concentrations[:1000].reshape(-1, 2)
        
        logger.info(f"Multi-image Macenko normalizer fitted on {len(self.fitted_images)} images")
        logger.info(f"Target mean concentrations: {self.target_mean}")
        logger.info(f"Target std concentrations: {self.target_std}")
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using multi-image Macenko method.
        
        Args:
            img: Input image to normalize
            
        Returns:
            Normalized image
        """
        if self.stain_matrix is None or self.target_mean is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Get source concentrations
        source_concentrations = self._get_concentrations(img, self.stain_matrix)
        
        # Reshape for normalization
        source_conc_reshaped = source_concentrations.reshape(-1, source_concentrations.shape[-1])
        
        # Get source statistics
        source_mean = np.mean(source_conc_reshaped, axis=0)
        source_std = np.std(source_conc_reshaped, axis=0)
        
        # Normalize concentrations
        normalized_conc = (source_conc_reshaped - source_mean) / (source_std + 1e-6)
        normalized_conc = normalized_conc * self.target_std + self.target_mean
        
        # Reconstruct image
        reconstructed_OD = normalized_conc @ self.stain_matrix.T
        reconstructed_img = np.exp(-reconstructed_OD) * 255.0
        
        # Reshape back to image format
        reconstructed_img = reconstructed_img.reshape(img.shape)
        
        return np.clip(reconstructed_img, 0, 255).astype(np.uint8)


class MultiImageReinhardNormalizer:
    """
    Enhanced Reinhard normalizer that fits on multiple images.
    """
    
    def __init__(self):
        """Initialize multi-image Reinhard normalizer."""
        self.target_mean = None
        self.target_std = None
        self.fitted_images = []
    
    def _convert_to_lab(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB image to LAB color space."""
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    def _convert_from_lab(self, lab_img: np.ndarray) -> np.ndarray:
        """Convert LAB image back to RGB."""
        return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    
    def fit(self, images: List[np.ndarray], sample_size: Optional[int] = None):
        """
        Fit normalizer to multiple images.
        
        Args:
            images: List of target images for normalization
            sample_size: Number of images to sample (if None, use all)
        """
        if not images:
            raise ValueError("No images provided for fitting")
        
        # Sample images if specified
        if sample_size is not None and len(images) > sample_size:
            images = random.sample(images, sample_size)
        
        logger.info(f"Fitting multi-image Reinhard normalizer on {len(images)} images")
        
        # Collect LAB statistics from all images
        all_lab_values = []
        
        for i, img in enumerate(images):
            try:
                lab_img = self._convert_to_lab(img)
                all_lab_values.append(lab_img.reshape(-1, 3))
                self.fitted_images.append(i)
            except Exception as e:
                logger.warning(f"Failed to process image {i}: {e}")
                continue
        
        if not all_lab_values:
            raise ValueError("Failed to process any images for fitting")
        
        # Combine all LAB values
        combined_lab = np.concatenate(all_lab_values, axis=0)
        
        # Compute robust statistics
        self.target_mean = np.mean(combined_lab, axis=0)
        self.target_std = np.std(combined_lab, axis=0)
        
        logger.info(f"Multi-image Reinhard normalizer fitted on {len(self.fitted_images)} images")
        logger.info(f"Target LAB mean: {self.target_mean}")
        logger.info(f"Target LAB std: {self.target_std}")
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using multi-image Reinhard method.
        
        Args:
            img: Input image to normalize
            
        Returns:
            Normalized image
        """
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Convert to LAB
        lab_img = self._convert_to_lab(img)
        
        # Calculate source statistics
        source_mean = np.mean(lab_img, axis=(0, 1))
        source_std = np.std(lab_img, axis=(0, 1))
        
        # Normalize
        normalized_lab = (lab_img - source_mean) / (source_std + 1e-6)
        normalized_lab = normalized_lab * self.target_std + self.target_mean
        
        # Convert back to RGB
        normalized_img = self._convert_from_lab(normalized_lab)
        
        return np.clip(normalized_img, 0, 255).astype(np.uint8)


class EnhancedStainNormalizer:
    """
    Enhanced stain normalizer with multi-image fitting and method selection.
    """
    
    def __init__(self, method: str = "macenko", num_images: int = 50):
        """
        Initialize enhanced stain normalizer.
        
        Args:
            method: Normalization method ("macenko", "reinhard")
            num_images: Number of images to use for fitting
        """
        self.method = method
        self.num_images = num_images
        self.normalizer = None
        
        if method == "macenko":
            self.normalizer = MultiImageMacenkoNormalizer()
        elif method == "reinhard":
            self.normalizer = MultiImageReinhardNormalizer()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'macenko' or 'reinhard'")
    
    def fit_from_dataset(self, dataset, sample_size: Optional[int] = None):
        """
        Fit normalizer from dataset samples.
        
        Args:
            dataset: Dataset object with samples
            sample_size: Number of samples to use (if None, use num_images)
        """
        # Sample images from dataset
        indices = list(range(len(dataset)))
        if sample_size is None:
            sample_size = min(self.num_images, len(dataset))
        
        sampled_indices = random.sample(indices, sample_size)
        
        # Load images
        images = []
        from PIL import Image
        
        for idx in sampled_indices:
            try:
                img_path, _, _ = dataset.samples[idx]
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                images.append(img_array)
            except Exception as e:
                logger.warning(f"Failed to load image {idx}: {e}")
                continue
        
        if not images:
            raise ValueError("Failed to load any images for fitting")
        
        # Fit normalizer
        self.normalizer.fit(images)
        
        logger.info(f"Enhanced {self.method} normalizer fitted on {len(images)} images")
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image.
        
        Args:
            img: Input image to normalize
            
        Returns:
            Normalized image
        """
        if self.normalizer is None:
            raise ValueError("Normalizer not fitted. Call fit_from_dataset() first.")
        
        return self.normalizer.normalize(img)


def create_enhanced_stain_normalizer(
    method: str = "macenko",
    num_images: int = 50
) -> EnhancedStainNormalizer:
    """
    Create enhanced stain normalizer.
    
    Args:
        method: Normalization method
        num_images: Number of images for fitting
        
    Returns:
        Enhanced stain normalizer instance
    """
    return EnhancedStainNormalizer(method=method, num_images=num_images)


if __name__ == "__main__":
    # Test enhanced stain normalization
    from multi_domain_dataset import MultiDomainManager
    
    # Create dummy images for testing
    test_images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(10)]
    
    # Test multi-image Macenko
    macenko_normalizer = MultiImageMacenkoNormalizer()
    macenko_normalizer.fit(test_images, sample_size=5)
    
    # Test normalization
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    normalized_img = macenko_normalizer.normalize(test_img)
    
    print(f"Original image shape: {test_img.shape}")
    print(f"Normalized image shape: {normalized_img.shape}")
    print(f"Original dtype: {test_img.dtype}")
    print(f"Normalized dtype: {normalized_img.dtype}")
    
    # Test enhanced normalizer
    enhanced_normalizer = create_enhanced_stain_normalizer("macenko", num_images=5)
    enhanced_normalizer.normalizer = macenko_normalizer
    
    normalized_img2 = enhanced_normalizer.normalize(test_img)
    print(f"Enhanced normalization result shape: {normalized_img2.shape}")
    
    print("Enhanced stain normalization ready for use")
