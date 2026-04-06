"""
Enhanced data augmentation transforms for domain-robust cervical cancer classification.
Includes strong color augmentations and domain-specific transformations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import numpy as np
from typing import Optional, Tuple


def get_domain_robust_train_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Get domain-robust training transforms with strong augmentations.
    
    Args:
        image_size: Target image size
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        
    Returns:
        Albumentations compose transform
    """
    transforms_list = [
        # Geometric augmentations
        A.Resize(height=image_size, width=image_size),
        
        # Random crop and resize
        A.RandomCrop(height=image_size, width=image_size, p=0.5),
        
        # Horizontal flip
        A.HorizontalFlip(p=0.5),
        
        # Shift, scale, rotate
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        
        # Color augmentations (critical for domain robustness)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Noise and blur (simulate imaging variations)
        A.GaussNoise(
            var_limit=(10.0, 30.0),
            mean=0,
            p=0.3
        ),
        
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.3
        ),
        
        # Additional robust augmentations
        A.OneOf([
            A.ElasticTransform(
                alpha=100,
                sigma=100 * 0.05,
                alpha_affine=100 * 0.03,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.2,
                p=1.0
            ),
            A.OpticalDistortion(
                distort_limit=0.2,
                shift_limit=0.2,
                p=1.0
            ),
        ], p=0.3),
        
        # Color jitter for stain variations
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        
        # CLAHE for contrast enhancement
        A.CLAHE(
            clip_limit=3.0,
            tile_grid_size=(8, 8),
            p=0.5
        ),
        
        # Coarse dropout (simulate artifacts)
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.2
        ),
        
        # Normalization
        A.Normalize(
            mean=normalize_mean,
            std=normalize_std
        ),
        
        # Convert to tensor
        ToTensorV2()
    ]
    
    return A.Compose(transforms_list, p=1.0)


def get_validation_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        
    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=normalize_mean,
            std=normalize_std
        ),
        ToTensorV2()
    ])


def get_strain_robust_augmentation() -> A.Compose:
    """
    Get stain-robust augmentation pipeline specifically for histopathology.
    
    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        A.Resize(height=224, width=224),
        
        # Stain-specific augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
        ], p=0.8),
        
        # Geometric variations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        
        # Contrast enhancement
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        
        # Normalization
        A.Normalize(),
        ToTensorV2()
    ])


def get_light_augmentation() -> A.Compose:
    """
    Get light augmentation for fine-tuning.
    
    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(),
        ToTensorV2()
    ])


def get_test_time_augmentation_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    tta_steps: int = 5
) -> list:
    """
    Get test-time augmentation transforms for robust inference.
    
    Args:
        image_size: Target image size
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        tta_steps: Number of TTA steps
        
    Returns:
        List of augmentation pipelines
    """
    transforms_list = []
    
    # Original
    transforms_list.append(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    )
    
    # Horizontal flip
    transforms_list.append(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    )
    
    # Brightness adjustment
    transforms_list.append(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    )
    
    # Hue adjustment
    transforms_list.append(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1.0),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    )


def get_test_time_augmentation_transforms(tta_steps: int = 5) -> list:
    """
    Get test-time augmentation transforms for robust inference.
    
    Args:
        tta_steps: Number of augmentation steps
        
    Returns:
        List of augmentation transforms
    """
    transforms_list = []
    
    # Original (no augmentation)
    transforms_list.append(A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]))
    
    # Horizontal flip
    transforms_list.append(A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]))
    
    # Small rotation
    transforms_list.append(A.Compose([
        A.Resize(height=224, width=224),
        A.Rotate(limit=10, p=1.0, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]))
    
    # Brightness adjustment
    transforms_list.append(A.Compose([
        A.Resize(height=224, width=224),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]))
    
    # Gaussian blur
    transforms_list.append(A.Compose([
        A.Resize(height=224, width=224),
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]))
    
    return transforms_list[:tta_steps]


if __name__ == "__main__":
    # Test transforms
    train_transform = get_domain_robust_train_transforms()
    val_transform = get_validation_transforms()
    tta_transforms = get_test_time_augmentation_transforms()
    
    print(f"Train transform: {len(train_transform.transforms)} steps")
    print(f"Val transform: {len(val_transform.transforms)} steps")
    print(f"TTA transforms: {len(tta_transforms)} steps")
    
    # Test on dummy image
    import numpy as np
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    result = train_transform(image=dummy_img)
    print(f"Train result type: {type(result['image'])}, shape: {result['image'].shape}")
    
    result = val_transform(image=dummy_img)
    print(f"Val result type: {type(result['image'])}, shape: {result['image'].shape}")
    
    result = tta_transforms[0](image=dummy_img)
    print(f"TTA result type: {type(result['image'])}, shape: {result['image'].shape}")
