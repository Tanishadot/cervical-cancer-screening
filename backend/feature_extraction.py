#!/usr/bin/env python3
"""
Real Feature Extraction for Cervical Cytology
Uses OpenCV for actual image-based feature extraction.

Author: Cervical Cancer Classification Pipeline
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CytologyFeatureExtractor:
    """Extract real cytological features from images using OpenCV."""
    
    def __init__(self):
        self.feature_weights = {
            'nuclear': 0.3,
            'cytoplasmic': 0.4,
            'background': 0.3
        }
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract comprehensive cytological features from input image.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Segment regions
            nucleus_mask, cytoplasm_mask, background_mask = self._segment_regions(gray)
            
            # Extract features for each region
            nuclear_features = self._extract_nuclear_features(image, nucleus_mask, gray, hsv)
            cytoplasmic_features = self._extract_cytoplasmic_features(image, cytoplasm_mask, gray, hsv)
            background_features = self._extract_background_features(image, background_mask, gray, hsv)
            
            # Combine features
            features = {
                'nuclear': nuclear_features,
                'cytoplasmic': cytoplasmic_features,
                'background': background_features,
                'segmentation_masks': {
                    'nucleus': nucleus_mask,
                    'cytoplasm': cytoplasm_mask,
                    'background': background_mask
                }
            }
            
            logger.info(f"Extracted features: {self._summarize_features(features)}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self._get_default_features()
    
    def _segment_regions(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image into nucleus, cytoplasm, and background regions.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Tuple of (nucleus_mask, cytoplasm_mask, background_mask)
        """
        # Threshold for nuclei (dark regions)
        _, nucleus_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nucleus_mask = cv2.morphologyEx(nucleus_binary, cv2.MORPH_CLOSE, kernel)
        nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small objects
        contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 50  # Minimum nucleus area
        nucleus_mask = np.zeros_like(nucleus_mask)
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(nucleus_mask, [contour], -1, 255, -1)
        
        # Create cytoplasm mask (region around nucleus)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        cytoplasm_dilated = cv2.dilate(nucleus_mask, kernel_large, iterations=2)
        cytoplasm_mask = cv2.subtract(cytoplasm_dilated, nucleus_mask)
        
        # Background is everything else
        background_mask = 255 - cv2.add(nucleus_mask, cytoplasm_mask)
        
        return nucleus_mask, cytoplasm_mask, background_mask
    
    def _extract_nuclear_features(self, image: np.ndarray, mask: np.ndarray, 
                                gray: np.ndarray, hsv: np.ndarray) -> Dict:
        """Extract nuclear features."""
        if np.sum(mask) == 0:
            return self._get_default_nuclear_features()
        
        # Get nucleus coordinates
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return self._get_default_nuclear_features()
        
        nucleus_pixels = gray[coords]
        nucleus_hsv = hsv[coords]
        
        # Nuclear features
        features = {
            'nuclear_enlargement': self._calculate_nuclear_size(mask, image.shape),
            'chromatin_density': self._calculate_chromatin_density(nucleus_pixels),
            'nuclear_contours': self._calculate_contour_irregularity(mask),
            'nc_ratio': self._calculate_nc_ratio(mask, image.shape),
            'hyperchromasia': self._calculate_hyperchromasia(nucleus_hsv),
            'overall_nuclear_score': 0.0  # Will be calculated
        }
        
        # Calculate overall score
        features['overall_nuclear_score'] = np.mean([
            features['nuclear_enlargement'],
            features['chromatin_density'],
            features['nuclear_contours'],
            features['nc_ratio'],
            features['hyperchromasia']
        ])
        
        return features
    
    def _extract_cytoplasmic_features(self, image: np.ndarray, mask: np.ndarray,
                                    gray: np.ndarray, hsv: np.ndarray) -> Dict:
        """Extract cytoplasmic features."""
        if np.sum(mask) == 0:
            return self._get_default_cytoplasmic_features()
        
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return self._get_default_cytoplasmic_features()
        
        cytoplasm_pixels = gray[coords]
        cytoplasm_hsv = hsv[coords]
        
        features = {
            'perinuclear_halo': self._calculate_perinuclear_halo(mask, gray),
            'keratinization': self._calculate_keratinization(cytoplasm_hsv),
            'cytoplasmic_texture': self._calculate_cytoplasmic_texture(cytoplasm_pixels),
            'cell_maturity': self._calculate_cell_maturity(cytoplasm_hsv),
            'koilocytosis_score': self._calculate_koilocytosis(mask, gray),
            'overall_cytoplasmic_score': 0.0  # Will be calculated
        }
        
        # Calculate overall score
        features['overall_cytoplasmic_score'] = np.mean([
            features['perinuclear_halo'],
            features['keratinization'],
            features['cytoplasmic_texture'],
            features['cell_maturity'],
            features['koilocytosis_score']
        ])
        
        return features
    
    def _extract_background_features(self, image: np.ndarray, mask: np.ndarray,
                                  gray: np.ndarray, hsv: np.ndarray) -> Dict:
        """Extract background features."""
        if np.sum(mask) == 0:
            return self._get_default_background_features()
        
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return self._get_default_background_features()
        
        background_pixels = gray[coords]
        background_hsv = hsv[coords]
        
        features = {
            'background_debris': self._calculate_background_debris(background_pixels),
            'inflammatory_cells': self._calculate_inflammatory_cells(mask),
            'tumor_diathesis': self._calculate_tumor_diathesis(background_hsv),
            'background_cleanliness': self._calculate_background_cleanliness(background_pixels),
            'overall_background_score': 0.0  # Will be calculated
        }
        
        # Calculate overall score
        features['overall_background_score'] = np.mean([
            features['background_debris'],
            features['inflammatory_cells'],
            features['tumor_diathesis'],
            features['background_cleanliness']
        ])
        
        return features
    
    # Feature calculation methods
    def _calculate_nuclear_size(self, mask: np.ndarray, image_shape: Tuple) -> float:
        """Calculate normalized nuclear size."""
        nucleus_area = np.sum(mask > 0)
        image_area = image_shape[0] * image_shape[1]
        normalized_size = nucleus_area / image_area
        return min(normalized_size * 100, 1.0)  # Normalize to 0-1
    
    def _calculate_chromatin_density(self, nucleus_pixels: np.ndarray) -> float:
        """Calculate chromatin density from pixel intensity variation."""
        if len(nucleus_pixels) == 0:
            return 0.0
        # Higher density = more variation in intensity
        density = np.std(nucleus_pixels) / 255.0
        return min(density * 2, 1.0)
    
    def _calculate_contour_irregularity(self, mask: np.ndarray) -> float:
        """Calculate nuclear contour irregularity."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter and area
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        if area == 0:
            return 0.0
        
        # Circularity: 1 = perfect circle, 0 = irregular
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        irregularity = 1.0 - circularity
        
        return min(irregularity, 1.0)
    
    def _calculate_nc_ratio(self, mask: np.ndarray, image_shape: Tuple) -> float:
        """Calculate nucleus-to-cytoplasm ratio."""
        nucleus_area = np.sum(mask > 0)
        image_area = image_shape[0] * image_shape[1]
        
        if image_area == 0:
            return 0.0
        
        # Estimate cytoplasm area as surrounding region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        cytoplasm_dilated = cv2.dilate(mask, kernel, iterations=2)
        cytoplasm_area = np.sum(cytoplasm_dilated > 0) - nucleus_area
        
        if cytoplasm_area == 0:
            return 1.0
        
        nc_ratio = nucleus_area / cytoplasm_area
        return min(nc_ratio, 1.0)
    
    def _calculate_hyperchromasia(self, nucleus_hsv: np.ndarray) -> float:
        """Calculate hyperchromasia from HSV color space."""
        if len(nucleus_hsv) == 0:
            return 0.0
        
        # Hyperchromasia = high saturation (dark staining)
        saturation = nucleus_hsv[:, 1]
        hyperchromasia = np.mean(saturation) / 255.0
        return min(hyperchromasia, 1.0)
    
    def _calculate_perinuclear_halo(self, cytoplasm_mask: np.ndarray, gray: np.ndarray) -> float:
        """Calculate perinuclear halo (clear zone around nucleus)."""
        # Find edges in cytoplasm region
        cytoplasm_region = cv2.bitwise_and(gray, gray, mask=cytoplasm_mask)
        edges = cv2.Canny(cytoplasm_region, 50, 150)
        
        # Count edge pixels in cytoplasm
        edge_pixels = cv2.bitwise_and(edges, edges, mask=cytoplasm_mask)
        edge_density = np.sum(edge_pixels > 0) / np.sum(cytoplasm_mask > 0)
        
        return min(edge_density * 5, 1.0)  # Scale to 0-1
    
    def _calculate_keratinization(self, cytoplasm_hsv: np.ndarray) -> float:
        """Calculate keratinization from HSV values."""
        if len(cytoplasm_hsv) == 0:
            return 0.0
        
        # Keratinization = high value (brightness) in cytoplasm
        value = cytoplasm_hsv[:, 2]
        keratinization = np.mean(value) / 255.0
        return min(keratinization, 1.0)
    
    def _calculate_cytoplasmic_texture(self, cytoplasm_pixels: np.ndarray) -> float:
        """Calculate cytoplasmic texture variation."""
        if len(cytoplasm_pixels) == 0:
            return 0.0
        
        # Texture = local variation
        texture = np.std(cytoplasm_pixels) / 255.0
        return min(texture * 2, 1.0)
    
    def _calculate_cell_maturity(self, cytoplasm_hsv: np.ndarray) -> float:
        """Calculate cell maturity from color characteristics."""
        if len(cytoplasm_hsv) == 0:
            return 0.0
        
        # Mature cells have higher saturation
        saturation = cytoplasm_hsv[:, 1]
        maturity = np.mean(saturation) / 255.0
        return min(maturity, 1.0)
    
    def _calculate_koilocytosis(self, cytoplasm_mask: np.ndarray, gray: np.ndarray) -> float:
        """Calculate koilocytosis score."""
        # Koilocytosis = perinuclear clearing + nuclear enlargement
        halo_score = self._calculate_perinuclear_halo(cytoplasm_mask, gray)
        
        # Additional texture analysis for koilocytosis
        cytoplasm_region = cv2.bitwise_and(gray, gray, mask=cytoplasm_mask)
        texture_score = self._calculate_cytoplasmic_texture(cytoplasm_region)
        
        return (halo_score + texture_score) / 2
    
    def _calculate_background_debris(self, background_pixels: np.ndarray) -> float:
        """Calculate background debris level."""
        if len(background_pixels) == 0:
            return 0.0
        
        # Debris = high variation in background
        debris = np.std(background_pixels) / 255.0
        return min(debris * 3, 1.0)
    
    def _calculate_inflammatory_cells(self, background_mask: np.ndarray) -> float:
        """Calculate inflammatory cell presence."""
        # Look for small circular structures in background
        contours, _ = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cell_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 200:  # Typical inflammatory cell size
                cell_count += 1
        
        # Normalize by image size
        image_area = background_mask.shape[0] * background_mask.shape[1]
        cell_density = cell_count / (image_area / 10000)  # Normalize
        
        return min(cell_density, 1.0)
    
    def _calculate_tumor_diathesis(self, background_hsv: np.ndarray) -> float:
        """Calculate tumor diathesis (necrotic background)."""
        if len(background_hsv) == 0:
            return 0.0
        
        # Tumor diathesis = dark, reddish background
        hue = background_hsv[:, 0]
        saturation = background_hsv[:, 1]
        value = background_hsv[:, 2]
        
        # Look for dark, low saturation regions
        dark_mask = (value < 100) & (saturation < 100)
        diathesis_score = np.sum(dark_mask) / len(value)
        
        return min(diathesis_score * 2, 1.0)
    
    def _calculate_background_cleanliness(self, background_pixels: np.ndarray) -> float:
        """Calculate background cleanliness (inverse of debris)."""
        debris = self._calculate_background_debris(background_pixels)
        cleanliness = 1.0 - debris
        return max(cleanliness, 0.0)
    
    def _summarize_features(self, features: Dict) -> str:
        """Summarize extracted features."""
        nuclear_score = features['nuclear'].get('overall_nuclear_score', 0)
        cytoplasmic_score = features['cytoplasmic'].get('overall_cytoplasmic_score', 0)
        background_score = features['background'].get('overall_background_score', 0)
        
        return f"Nuclear: {nuclear_score:.3f}, Cytoplasmic: {cytoplasmic_score:.3f}, Background: {background_score:.3f}"
    
    def _get_default_features(self) -> Dict:
        """Return default features when extraction fails."""
        return {
            'nuclear': self._get_default_nuclear_features(),
            'cytoplasmic': self._get_default_cytoplasmic_features(),
            'background': self._get_default_background_features(),
            'segmentation_masks': {
                'nucleus': np.zeros((224, 224), dtype=np.uint8),
                'cytoplasm': np.zeros((224, 224), dtype=np.uint8),
                'background': np.ones((224, 224), dtype=np.uint8) * 255
            }
        }
    
    def _get_default_nuclear_features(self) -> Dict:
        """Return default nuclear features."""
        return {
            'nuclear_enlargement': 0.0,
            'chromatin_density': 0.0,
            'nuclear_contours': 0.0,
            'nc_ratio': 0.0,
            'hyperchromasia': 0.0,
            'overall_nuclear_score': 0.0
        }
    
    def _get_default_cytoplasmic_features(self) -> Dict:
        """Return default cytoplasmic features."""
        return {
            'perinuclear_halo': 0.0,
            'keratinization': 0.0,
            'cytoplasmic_texture': 0.0,
            'cell_maturity': 0.0,
            'koilocytosis_score': 0.0,
            'overall_cytoplasmic_score': 0.0
        }
    
    def _get_default_background_features(self) -> Dict:
        """Return default background features."""
        return {
            'background_debris': 0.0,
            'inflammatory_cells': 0.0,
            'tumor_diathesis': 0.0,
            'background_cleanliness': 1.0,
            'overall_background_score': 0.0
        }


def extract_features_from_image(image: np.ndarray) -> Dict:
    """
    Convenience function to extract features from image.
    
    Args:
        image: RGB image array
        
    Returns:
        Dictionary of extracted features
    """
    extractor = CytologyFeatureExtractor()
    return extractor.extract_features(image)


if __name__ == "__main__":
    # Test feature extraction
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = extract_features_from_image(test_image)
    print("Feature extraction test completed")
    print(f"Nuclear score: {features['nuclear']['overall_nuclear_score']:.3f}")
    print(f"Cytoplasmic score: {features['cytoplasmic']['overall_cytoplasmic_score']:.3f}")
    print(f"Background score: {features['background']['overall_background_score']:.3f}")
