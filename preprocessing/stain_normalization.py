"""
Advanced stain normalization techniques for histopathology images.
Implements Macenko and Reinhard stain normalization methods.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacenkoNormalizer:
    """
    Macenko stain normalization for histopathology images.
    """
    
    def __init__(self):
        """Initialize Macenko normalizer."""
        self.stain_matrix = None
        self.target_concentrations = None
    
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
        
        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Get two largest eigenvectors
        top_eigvecs = eigvecs[:, -2:]
        
        # Compute stain matrix
        stains = ODhat @ top_eigvecs
        
        # Normalize stains
        stain_norm = stains / np.linalg.norm(stains, axis=0, keepdims=True)
        
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
    
    def fit(self, target_img: np.ndarray):
        """
        Fit normalizer to target image.
        
        Args:
            target_img: Target image for normalization
        """
        self.stain_matrix = self._get_stain_matrix(target_img)
        self.target_concentrations = self._get_concentrations(target_img, self.stain_matrix)
        
        logger.info("Macenko normalizer fitted to target image")
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using Macenko method.
        
        Args:
            img: Input image to normalize
            
        Returns:
            Normalized image
        """
        if self.stain_matrix is None or self.target_concentrations is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Get source concentrations
        source_concentrations = self._get_concentrations(img, self.stain_matrix)
        
        # Get target concentrations statistics
        target_mean = np.mean(self.target_concentrations, axis=(0, 1))
        target_std = np.std(self.target_concentrations, axis=(0, 1))
        
        # Get source concentrations statistics
        source_mean = np.mean(source_concentrations, axis=(0, 1))
        source_std = np.std(source_concentrations, axis=(0, 1))
        
        # Normalize concentrations
        normalized_concentrations = (source_concentrations - source_mean) / (source_std + 1e-6)
        normalized_concentrations = normalized_concentrations * target_std + target_mean
        
        # Reconstruct image
        reconstructed_OD = normalized_concentrations @ self.stain_matrix.T
        reconstructed_img = np.exp(-reconstructed_OD) * 255.0
        
        return np.clip(reconstructed_img, 0, 255).astype(np.uint8)


class ReinhardNormalizer:
    """
    Reinhard stain normalization for histopathology images.
    """
    
    def __init__(self):
        """Initialize Reinhard normalizer."""
        self.target_mean = None
        self.target_std = None
    
    def _convert_to_lab(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB image to LAB color space."""
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    def _convert_from_lab(self, lab_img: np.ndarray) -> np.ndarray:
        """Convert LAB image back to RGB."""
        return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    
    def fit(self, target_img: np.ndarray):
        """
        Fit normalizer to target image.
        
        Args:
            target_img: Target image for normalization
        """
        # Convert to LAB
        lab_img = self._convert_to_lab(target_img)
        
        # Calculate mean and std for each channel
        self.target_mean = np.mean(lab_img, axis=(0, 1))
        self.target_std = np.std(lab_img, axis=(0, 1))
        
        logger.info("Reinhard normalizer fitted to target image")
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using Reinhard method.
        
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


class VahadaneNormalizer:
    """
    Vahadane stain normalization for histopathology images.
    """
    
    def __init__(self):
        """Initialize Vahadane normalizer."""
        self.stain_matrix = None
        self.target_concentrations = None
    
    def _dictionary_learning(self, OD: np.ndarray, n_stains: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform dictionary learning for stain separation.
        
        Args:
            OD: Optical density image
            n_stains: Number of stains to extract
            
        Returns:
            Tuple of (stain matrix, concentrations)
        """
        # Reshape for dictionary learning
        OD_reshaped = OD.reshape(-1, 3)
        
        # Perform dictionary learning using PCA
        pca = PCA(n_components=n_stains)
        concentrations = pca.fit_transform(OD_reshaped)
        stain_matrix = pca.components_.T
        
        return stain_matrix, concentrations
    
    def fit(self, target_img: np.ndarray):
        """
        Fit normalizer to target image.
        
        Args:
            target_img: Target image for normalization
        """
        # Convert to optical density
        OD = -np.log((target_img.astype(np.float32) + 1) / 256)
        
        # Perform dictionary learning
        self.stain_matrix, self.target_concentrations = self._dictionary_learning(OD)
        
        logger.info("Vahadane normalizer fitted to target image")
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using Vahadane method.
        
        Args:
            img: Input image to normalize
            
        Returns:
            Normalized image
        """
        if self.stain_matrix is None or self.target_concentrations is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Convert to optical density
        OD = -np.log((img.astype(np.float32) + 1) / 256)
        
        # Get source concentrations
        _, source_concentrations = self._dictionary_learning(OD)
        
        # Get target concentrations statistics
        target_mean = np.mean(self.target_concentrations, axis=0)
        target_std = np.std(self.target_concentrations, axis=0)
        
        # Get source concentrations statistics
        source_mean = np.mean(source_concentrations, axis=0)
        source_std = np.std(source_concentrations, axis=0)
        
        # Normalize concentrations
        normalized_concentrations = (source_concentrations - source_mean) / (source_std + 1e-6)
        normalized_concentrations = normalized_concentrations * target_std + target_mean
        
        # Reconstruct image
        reconstructed_OD = normalized_concentrations @ self.stain_matrix.T
        reconstructed_img = np.exp(-reconstructed_OD) * 255.0
        
        return np.clip(reconstructed_img, 0, 255).astype(np.uint8)


def stain_normalization_factory(method: str = "macenko") -> object:
    """
    Factory function to create stain normalizer.
    
    Args:
        method: Normalization method ("macenko", "reinhard", "vahadane")
        
    Returns:
        Stain normalizer instance
    """
    methods = {
        "macenko": MacenkoNormalizer,
        "reinhard": ReinhardNormalizer,
        "vahadane": VahadaneNormalizer
    }
    
    if method not in methods:
        raise ValueError(f"Unknown normalization method: {method}. Available: {list(methods.keys())}")
    
    return methods[method]()


def normalize_image_stains(
    img: np.ndarray,
    target_img: np.ndarray,
    method: str = "macenko"
) -> np.ndarray:
    """
    Normalize image stains using specified method.
    
    Args:
        img: Input image to normalize
        target_img: Target image for normalization
        method: Normalization method
        
    Returns:
        Normalized image
    """
    normalizer = stain_normalization_factory(method)
    normalizer.fit(target_img)
    return normalizer.normalize(img)


def batch_normalize_images(
    images: list,
    target_img: np.ndarray,
    method: str = "macenko"
) -> list:
    """
    Normalize a batch of images.
    
    Args:
        images: List of input images
        target_img: Target image for normalization
        method: Normalization method
        
    Returns:
        List of normalized images
    """
    normalizer = stain_normalization_factory(method)
    normalizer.fit(target_img)
    
    normalized_images = []
    for img in images:
        normalized_img = normalizer.normalize(img)
        normalized_images.append(normalized_img)
    
    return normalized_images


if __name__ == "__main__":
    # Test stain normalization
    # Create dummy images
    target_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    source_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test Macenko normalization
    macenko_normalizer = MacenkoNormalizer()
    macenko_normalizer.fit(target_img)
    normalized_img = macenko_normalizer.normalize(source_img)
    
    print(f"Original image shape: {source_img.shape}")
    print(f"Normalized image shape: {normalized_img.shape}")
    print(f"Original dtype: {source_img.dtype}")
    print(f"Normalized dtype: {normalized_img.dtype}")
    
    # Test factory function
    normalized_img2 = normalize_image_stains(source_img, target_img, "macenko")
    print(f"Factory normalization result shape: {normalized_img2.shape}")
    
    print("Stain normalization methods ready for use")
