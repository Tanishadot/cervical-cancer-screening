"""
Image preprocessing utilities for cervical cancer cell classification.
Includes CLAHE, color normalization, and stain normalization techniques.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch
from PIL import Image


class ImagePreprocessor:
    """
    Image preprocessing pipeline for cervical cell images.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        apply_color_normalization: bool = True,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            apply_clahe: Whether to apply CLAHE
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile grid size
            apply_color_normalization: Whether to apply color normalization
            normalize_mean: Normalization mean values
            normalize_std: Normalization std values
        """
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.apply_color_normalization = apply_color_normalization
        self.normalize_mean = np.array(normalize_mean)
        self.normalize_std = np.array(normalize_std)
        
        # Initialize CLAHE
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input RGB image
            
        Returns:
            CLAHE enhanced image
        """
        if not self.apply_clahe:
            return image
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge channels back
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Convert back to RGB
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    def normalize_color(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image colors.
        
        Args:
            image: Input image
            
        Returns:
            Color normalized image
        """
        if not self.apply_color_normalization:
            return image
        
        # Convert to float and normalize to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Apply normalization
        normalized = (image_float - self.normalize_mean) / self.normalize_std
        
        # Convert back to uint8
        normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        
        return normalized
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using histogram equalization.
        
        Args:
            image: Input image
            
        Returns:
            Contrast enhanced image
        """
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # Apply histogram equalization to Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back to RGB
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    
    def remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small artifacts using morphological operations.
        
        Args:
            image: Input image
            
        Returns:
            Cleaned image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological opening
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Apply median filter
        denoised = cv2.medianBlur(opened, 3)
        
        # Convert back to RGB
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image: Input RGB image
            
        Returns:
            Preprocessed image
        """
        # Resize image
        processed = self.resize_image(image)
        
        # Remove artifacts
        processed = self.remove_artifacts(processed)
        
        # Apply CLAHE
        processed = self.apply_clahe_enhancement(processed)
        
        # Enhance contrast
        processed = self.enhance_contrast(processed)
        
        # Normalize colors
        processed = self.normalize_color(processed)
        
        return processed


class StainNormalizer:
    """
    Stain normalization using Macenko method.
    """
    
    def __init__(self):
        """Initialize stain normalizer."""
        self.stain_matrix = None
    
    def _normalize_stains(self, img: np.ndarray) -> np.ndarray:
        """Normalize image stains."""
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
    
    def macenko_normalization(self, img: np.ndarray, target_img: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Macenko stain normalization.
        
        Args:
            img: Source image to normalize
            target_img: Target image for normalization (optional)
            
        Returns:
            Normalized image
        """
        # Convert to float and normalize
        img_float = img.astype(np.float32) / 255.0
        
        # Convert to optical density
        OD = -np.log(img_float + 1e-6)
        
        # Remove background
        ODhat = OD[~np.all(OD < 0.15, axis=1)]
        
        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Get two largest eigenvectors
        top_eigvecs = eigvecs[:, -2:]
        
        # Compute stain matrix
        stains = ODhat @ top_eigvecs
        
        # Find extreme values
        min_vals = np.percentile(stains, 1, axis=0)
        max_vals = np.percentile(stains, 99, axis=0)
        
        # Normalize stains
        stain_norm = (stains - min_vals) / (max_vals - min_vals + 1e-6)
        
        # Reconstruct image
        if target_img is not None:
            # Use target image for normalization
            target_OD = -np.log(target_img.astype(np.float32) / 255.0 + 1e-6)
            target_stains = target_OD @ top_eigvecs
            target_min = np.percentile(target_stains, 1, axis=0)
            target_max = np.percentile(target_stains, 99, axis=0)
            
            # Map to target stain distribution
            mapped_stains = stain_norm * (target_max - target_min) + target_min
        else:
            mapped_stains = stain_norm
        
        # Reconstruct normalized image
        reconstructed_OD = mapped_stains @ top_eigvecs.T
        reconstructed_img = np.exp(-reconstructed_OD) * 255.0
        
        return np.clip(reconstructed_img, 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe: bool = True,
    apply_stain_normalization: bool = False,
    target_image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Preprocess a single image.
    
    Args:
        image: Input RGB image
        target_size: Target size (height, width)
        apply_clahe: Whether to apply CLAHE
        apply_stain_normalization: Whether to apply stain normalization
        target_image: Target image for stain normalization
        
    Returns:
        Preprocessed image
    """
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        apply_clahe=apply_clahe
    )
    
    # Apply basic preprocessing
    processed = preprocessor.preprocess(image)
    
    # Apply stain normalization if requested
    if apply_stain_normalization:
        normalizer = StainNormalizer()
        processed = normalizer.macenko_normalization(processed, target_image)
    
    return processed


def preprocess_pil_image(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe: bool = True,
    apply_stain_normalization: bool = False
) -> Image.Image:
    """
    Preprocess PIL image.
    
    Args:
        pil_image: Input PIL image
        target_size: Target size
        apply_clahe: Whether to apply CLAHE
        apply_stain_normalization: Whether to apply stain normalization
        
    Returns:
        Preprocessed PIL image
    """
    # Convert to numpy array
    image_array = np.array(pil_image.convert('RGB'))
    
    # Preprocess
    processed = preprocess_image(
        image_array,
        target_size=target_size,
        apply_clahe=apply_clahe,
        apply_stain_normalization=apply_stain_normalization
    )
    
    # Convert back to PIL
    return Image.fromarray(processed)


if __name__ == "__main__":
    # Test preprocessing
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess(test_image)
    
    print(f"Original shape: {test_image.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Original dtype: {test_image.dtype}")
    print(f"Processed dtype: {processed.dtype}")
