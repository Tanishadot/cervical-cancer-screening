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
            
            # Segment regions using improved pipeline
            nucleus_mask, cytoplasm_mask, background_mask = self._segment_regions(image)
            
            # Extract features for each region
            nuclear_features = self._extract_nuclear_features(image, nucleus_mask, gray, hsv)
            cytoplasmic_features = self._extract_cytoplasmic_features(
                image, cytoplasm_mask, gray, hsv, 
                nuclear_features['nucleus_area']
            )
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
    
    def _segment_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image into nucleus, cytoplasm, and background regions using improved pipeline.
        
        Args:
            image: RGB image array
            
        Returns:
            Tuple of (nucleus_mask, cytoplasm_mask, background_mask)
        """
        # 1. Preprocessing Improvements
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE on L-channel to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Apply Gaussian blur to reduce noise
        l_blurred = cv2.GaussianBlur(l_enhanced, (5, 5), 0)
        
        # 2. Color-based Filtering
        # Convert to HSV for stain masking
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract purple/blue stain regions (hematoxylin)
        # Adjusted range to better capture hematoxylin-stained nuclei
        lower_hsv = np.array([100, 30, 30])
        upper_hsv = np.array([140, 255, 255])
        stain_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Combine enhanced L-channel with stain mask
        combined = cv2.bitwise_and(l_blurred, l_blurred, mask=stain_mask)
        
        # 3. Initial Thresholding
        # Use adaptive thresholding instead of Otsu for better nucleus detection
        nucleus_binary = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
        
        # Invert to get nuclei as white objects
        nucleus_binary = cv2.bitwise_not(nucleus_binary)
        
        # 4. Morphological Filtering
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove noise with opening
        nucleus_clean = cv2.morphologyEx(nucleus_binary, cv2.MORPH_OPEN, kernel_small)
        
        # Fill holes with closing
        nucleus_filled = cv2.morphologyEx(nucleus_clean, cv2.MORPH_CLOSE, kernel_medium)
        
        # 5. Shape Constraints and Connected Component Filtering
        nucleus_mask = self._filter_valid_nuclei(nucleus_filled)
        
        # 6. Create cytoplasm mask (region around nucleus)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        cytoplasm_dilated = cv2.dilate(nucleus_mask, kernel_large, iterations=2)
        cytoplasm_mask = cv2.subtract(cytoplasm_dilated, nucleus_mask)
        
        # 7. Background is everything else
        background_mask = 255 - cv2.add(nucleus_mask, cytoplasm_mask)
        
        return nucleus_mask, cytoplasm_mask, background_mask
    
    def _filter_valid_nuclei(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Filter binary mask to keep only biologically valid nuclei.
        
        Args:
            binary_mask: Initial binary mask of potential nuclei
            
        Returns:
            Clean binary mask with only valid nuclei
        """
        # Find connected components
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean mask
        clean_mask = np.zeros_like(binary_mask)
        
        # Biological constraints for nuclei
        min_area = 30  # Minimum nucleus area in pixels (reduced for testing)
        max_area = 5000  # Maximum nucleus area in pixels
        min_circularity = 0.4  # Minimum circularity (reduced for testing)
        min_solidity = 0.7  # Minimum solidity (reduced for testing)
        max_aspect_ratio = 3.0  # Maximum aspect ratio (increased for testing)
        
        for contour in contours:
            # Area filtering
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # Shape constraints
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Circularity: 4π * Area / Perimeter²
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            
            # Solidity: Area / Convex Hull Area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            
            solidity = area / hull_area
            if solidity < min_solidity:
                continue
            
            # Aspect ratio filtering
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > max_aspect_ratio:
                continue
            
            # Additional validation: enclosed structure
            if not self._is_enclosed_structure(contour, binary_mask):
                continue
            
            # If all constraints passed, add to clean mask
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)
        
        return clean_mask
    
    def _is_enclosed_structure(self, contour: np.ndarray, binary_mask: np.ndarray) -> bool:
        """
        Check if the contour represents an enclosed structure with consistent boundary.
        
        Args:
            contour: Contour to validate
            binary_mask: Original binary mask
            
        Returns:
            True if valid enclosed structure, False otherwise
        """
        # Create a mask for this specific contour
        contour_mask = np.zeros_like(binary_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Check if the contour has a reasonable area
        area = cv2.contourArea(contour)
        if area < 30:  # Too small to be a valid nucleus
            return False
        
        # Check if the contour is closed (contour should have at least 3 points)
        if len(contour) < 3:
            return False
        
        # Check if the contour forms a closed loop
        # For a closed contour, the first and last points should be close
        if len(contour) > 0:
            first_point = contour[0][0]
            last_point = contour[-1][0]
            distance = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)
            if distance > 5:  # Points too far apart, not closed
                return False
        
        # Check if the contour has a reasonable perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or perimeter > 1000:  # Unreasonable perimeter
            return False
        
        # Check if the contour is well-defined in the original mask
        # The contour should correspond to a filled region in the binary mask
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ensure bounding box is within image bounds
        if x < 0 or y < 0 or x + w >= binary_mask.shape[1] or y + h >= binary_mask.shape[0]:
            return False
        
        # Extract the region from the original mask
        roi = binary_mask[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        # Check if the region has sufficient white pixels (should be mostly filled)
        white_pixels = np.sum(roi > 0)
        total_pixels = roi.size
        fill_ratio = white_pixels / total_pixels
        
        # Should have reasonable fill ratio (not too empty)
        if fill_ratio < 0.1:  # Less than 10% filled
            return False
        
        return True
    
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
        
        # Calculate nuclear area for maturity assessment
        nucleus_area = np.sum(mask > 0)
        
        # Nuclear features
        features = {
            'nuclear_enlargement': self._calculate_nuclear_size(mask, image.shape),
            'chromatin_density': self._calculate_chromatin_density(nucleus_pixels),
            'nuclear_contours': self._calculate_contour_irregularity(mask),
            'nc_ratio': self._calculate_nc_ratio(mask, image.shape),
            'hyperchromasia': self._calculate_hyperchromasia(nucleus_hsv),
            'nucleus_area': nucleus_area,
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
                                    gray: np.ndarray, hsv: np.ndarray, nucleus_area: int = 0) -> Dict:
        """Extract cytoplasmic features."""
        if np.sum(mask) == 0:
            return self._get_default_cytoplasmic_features()
        
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return self._get_default_cytoplasmic_features()
        
        cytoplasm_pixels = gray[coords]
        cytoplasm_hsv = hsv[coords]
        
        # Calculate cytoplasm area for maturity assessment
        cytoplasm_area = np.sum(mask > 0)
        
        # Calculate cell maturity ratio
        cell_maturity_ratio = cytoplasm_area / nucleus_area if nucleus_area > 0 else 1.0
        
        features = {
            'perinuclear_halo': self._calculate_perinuclear_halo(mask, gray),
            'keratinization': self._calculate_keratinization(cytoplasm_hsv),
            'cytoplasmic_texture': self._calculate_cytoplasmic_texture(cytoplasm_pixels),
            'cell_maturity': self._calculate_cell_maturity(cytoplasm_hsv),
            'koilocytosis_score': self._calculate_koilocytosis(mask, gray),
            'cytoplasm_area': cytoplasm_area,
            'cell_maturity_ratio': cell_maturity_ratio,
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
            'nucleus_area': 0,
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
            'cytoplasm_area': 0,
            'cell_maturity_ratio': 1.0,
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
