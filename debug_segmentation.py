#!/usr/bin/env python3
"""
Debug script for nucleus segmentation pipeline
"""

import numpy as np
import cv2
from backend.feature_extraction import CytologyFeatureExtractor
import matplotlib.pyplot as plt

def debug_segmentation_pipeline():
    """Debug the nucleus segmentation pipeline step by step."""
    
    # Create test image with realistic nuclei
    test_image = np.full((224, 224, 3), 240, dtype=np.uint8)
    
    # Add hematoxylin-stained nuclei
    cv2.circle(test_image, (60, 60), 18, (80, 60, 120), -1)
    cv2.circle(test_image, (150, 80), 15, (90, 70, 130), -1)
    cv2.ellipse(test_image, (100, 150), (20, 25), 30, 0, 360, (85, 65, 125), -1)
    
    # Initialize extractor
    extractor = CytologyFeatureExtractor()
    
    print("=== DEBUGGING SEGMENTATION PIPELINE ===")
    
    # Step 1: LAB preprocessing
    print("1. LAB preprocessing...")
    lab = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    l_blurred = cv2.GaussianBlur(l_enhanced, (5, 5), 0)
    
    print(f"   L-channel range: [{l_channel.min():.1f}, {l_channel.max():.1f}]")
    print(f"   Enhanced L range: [{l_enhanced.min():.1f}, {l_enhanced.max():.1f}]")
    
    # Step 2: HSV masking
    print("2. HSV masking...")
    hsv = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)
    lower_hsv = np.array([100, 30, 30])
    upper_hsv = np.array([140, 255, 255])
    stain_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    print(f"   HSV ranges - H: [{hsv[:,:,0].min():.1f}, {hsv[:,:,0].max():.1f}]")
    print(f"               S: [{hsv[:,:,1].min():.1f}, {hsv[:,:,1].max():.1f}]")
    print(f"               V: [{hsv[:,:,2].min():.1f}, {hsv[:,:,2].max():.1f}]")
    print(f"   Stain mask pixels: {np.sum(stain_mask > 0)}")
    
    # Step 3: Combine and threshold
    print("3. Combine and threshold...")
    combined = cv2.bitwise_and(l_blurred, l_blurred, mask=stain_mask)
    
    # Use adaptive thresholding
    nucleus_binary = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    nucleus_binary = cv2.bitwise_not(nucleus_binary)
    
    print(f"   Combined range: [{combined.min():.1f}, {combined.max():.1f}]")
    print(f"   Binary mask pixels: {np.sum(nucleus_binary > 0)}")
    
    # Step 4: Morphological filtering
    print("4. Morphological filtering...")
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    nucleus_clean = cv2.morphologyEx(nucleus_binary, cv2.MORPH_OPEN, kernel_small)
    nucleus_filled = cv2.morphologyEx(nucleus_clean, cv2.MORPH_CLOSE, kernel_medium)
    
    print(f"   After opening pixels: {np.sum(nucleus_clean > 0)}")
    print(f"   After closing pixels: {np.sum(nucleus_filled > 0)}")
    
    # Step 5: Shape filtering
    print("5. Shape filtering...")
    contours, _ = cv2.findContours(nucleus_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Initial contours found: {len(contours)}")
    
    valid_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"   Contour {i+1}: area = {area:.1f}")
        
        if area < 30 or area > 5000:
            print(f"     -> Rejected: area out of range")
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            print(f"     -> Rejected: zero perimeter")
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        print(f"     -> Circularity: {circularity:.3f}")
        
        if circularity < 0.4:
            print(f"     -> Rejected: circularity too low")
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            print(f"     -> Rejected: zero hull area")
            continue
        
        solidity = area / hull_area
        print(f"     -> Solidity: {solidity:.3f}")
        
        if solidity < 0.7:
            print(f"     -> Rejected: solidity too low")
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        print(f"     -> Aspect ratio: {aspect_ratio:.2f}")
        
        if aspect_ratio > 3.0:
            print(f"     -> Rejected: aspect ratio too high")
            continue
        
        enclosed = extractor._is_enclosed_structure(contour, nucleus_filled)
        print(f"     -> Enclosed structure: {enclosed}")
        if not enclosed:
            print(f"     -> Rejected: not enclosed structure")
            continue
        
        print(f"     -> ACCEPTED!")
        valid_count += 1
    
    print(f"   Valid nuclei after filtering: {valid_count}")
    
    # Save debug images
    debug_images = {
        'original': test_image,
        'l_enhanced': l_enhanced,
        'stain_mask': stain_mask,
        'combined': combined,
        'binary': nucleus_binary,
        'filled': nucleus_filled
    }
    
    return debug_images, valid_count

if __name__ == "__main__":
    images, count = debug_segmentation_pipeline()
    print(f"\n=== FINAL RESULT: {count} valid nuclei detected ===")
