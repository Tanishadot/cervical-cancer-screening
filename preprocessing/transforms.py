"""
Data augmentation transforms using Albumentations for cervical cancer classification.
Includes strong color augmentations to simulate staining variations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import numpy as np
from typing import Optional, Tuple


def get_train_transforms(
    image_size: int = 224,
    apply_color_augmentation: bool = True,
    apply_geometric_augmentation: bool = True,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Get training transforms with strong augmentations.
    
    Args:
        image_size: Target image size
        apply_color_augmentation: Whether to apply color augmentations
        apply_geometric_augmentation: Whether to apply geometric augmentations
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        
    Returns:
        Albumentations compose transform
    """
    transforms_list = []
    
    # Geometric augmentations
    if apply_geometric_augmentation:
        transforms_list.extend([
            # Random resize and crop
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0  # Always apply
            ),
            
            # Horizontal and vertical flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            
            # Rotation
            A.Rotate(
                limit=30,
                interpolation=1,
                border_mode=0,
                p=0.7
            ),
            
            # Affine transforms
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-8, 8),
                p=0.5
            ),
            
            # Elastic transform
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=0.3
            ),
            
            # Grid distortion
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3
            ),
        ])
    else:
        # Simple resize for no geometric augmentation
        transforms_list.append(
            A.Resize(height=image_size, width=image_size)
        )
    
    # Color augmentations (important for staining variations)
    if apply_color_augmentation:
        transforms_list.extend([
            # Random brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            
            # Hue, saturation, value changes
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.8
            ),
            
            # RGB shift
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.5
            ),
            
            # Color jitter
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.7
            ),
            
            # Gamma correction
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.5
            ),
            
            # Channel shuffle
            A.ChannelShuffle(p=0.1),
            
            # CLAHE
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.5
            ),
        ])
    
    # Noise and blur
    transforms_list.extend([
        # Gaussian noise
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0,
            p=0.3
        ),
        
        # Blur and sharpen
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
            A.MotionBlur(blur_limit=7),
        ], p=0.3),
        
        # Coarse dropout (simulates tissue artifacts)
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
    ])
    
    # Normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(
            mean=normalize_mean,
            std=normalize_std
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list, p=1.0)


def get_val_transforms(
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


def get_test_time_augmentation_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    tta_steps: int = 5
) -> list:
    """
    Get test-time augmentation transforms.
    
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
    
    # Vertical flip
    transforms_list.append(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    )
    
    # Rotation 90 degrees
    transforms_list.append(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=90, p=1.0),
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
    
    return transforms_list[:tta_steps]


class StrongColorAugmentation:
    """
    Strong color augmentation pipeline specifically for histopathology images.
    """
    
    def __init__(
        self,
        stain_variation_p: float = 0.8,
        color_shift_p: float = 0.7,
        intensity_p: float = 0.6
    ):
        """
        Initialize strong color augmentation.
        
        Args:
            stain_variation_p: Probability of stain variation
            color_shift_p: Probability of color shift
            intensity_p: Probability of intensity variation
        """
        self.stain_variation_p = stain_variation_p
        self.color_shift_p = color_shift_p
        self.intensity_p = intensity_p
    
    def get_transform(self, image_size: int = 224) -> A.Compose:
        """
        Get strong color augmentation transform.
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations compose transform
        """
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            
            # Stain-like variations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.4,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=30,
                    sat_shift_limit=40,
                    val_shift_limit=30,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=30,
                    g_shift_limit=30,
                    b_shift_limit=30,
                    p=1.0
                ),
            ], p=self.stain_variation_p),
            
            # Color channel variations
            A.OneOf([
                A.ChannelShuffle(p=1.0),
                A.ToGray(p=1.0),
                A.ToSepia(p=1.0),
            ], p=self.color_shift_p),
            
            # Intensity variations
            A.OneOf([
                A.RandomGamma(gamma_limit=(60, 140), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=1.0),
            ], p=self.intensity_p),
            
            # Final normalization
            A.Normalize(),
            ToTensorV2()
        ])


def apply_test_time_augmentation(
    model: torch.nn.Module,
    image: torch.Tensor,
    tta_transforms: list,
    device: torch.device
) -> torch.Tensor:
    """
    Apply test-time augmentation to improve predictions.
    
    Args:
        model: Trained model
        image: Input image tensor
        tta_transforms: List of TTA transforms
        device: Device to run inference on
        
    Returns:
        Averaged predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # Convert tensor back to numpy for augmentation
            if isinstance(image, torch.Tensor):
                # Denormalize and convert to numpy
                img_np = image.cpu().numpy()
                if img_np.shape[0] == 3:  # CHW format
                    img_np = img_np.transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = image
            
            # Apply augmentation
            augmented = transform(image=img_np)['image']
            augmented = augmented.unsqueeze(0).to(device)
            
            # Get prediction
            pred = model(augmented)
            predictions.append(pred)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)


def create_custom_augmentation_pipeline(
    config: dict
) -> A.Compose:
    """
    Create custom augmentation pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Albumentations compose transform
    """
    transforms_list = []
    
    # Add transforms based on config
    if config.get('horizontal_flip', False):
        transforms_list.append(A.HorizontalFlip(p=config.get('horizontal_flip_p', 0.5)))
    
    if config.get('vertical_flip', False):
        transforms_list.append(A.VerticalFlip(p=config.get('vertical_flip_p', 0.3)))
    
    if config.get('rotation', False):
        transforms_list.append(A.Rotate(limit=config.get('rotation_limit', 30), p=0.7))
    
    if config.get('brightness_contrast', False):
        transforms_list.append(A.RandomBrightnessContrast(
            brightness_limit=config.get('brightness_limit', 0.3),
            contrast_limit=config.get('contrast_limit', 0.3),
            p=config.get('brightness_contrast_p', 0.8)
        ))
    
    if config.get('hue_saturation', False):
        transforms_list.append(A.HueSaturationValue(
            hue_shift_limit=config.get('hue_limit', 20),
            sat_shift_limit=config.get('sat_limit', 30),
            val_shift_limit=config.get('val_limit', 20),
            p=config.get('hue_saturation_p', 0.8)
        ))
    
    # Always add normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


if __name__ == "__main__":
    # Test transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    print("Train transforms:", train_transform)
    print("Validation transforms:", val_transform)
    
    # Test with sample image
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Apply training transform
    transformed = train_transform(image=sample_image)
    print(f"Transformed image shape: {transformed['image'].shape}")
    print(f"Transformed image type: {type(transformed['image'])}")
