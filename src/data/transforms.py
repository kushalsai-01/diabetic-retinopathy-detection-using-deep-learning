"""
Image transforms and augmentations for diabetic retinopathy detection.
Uses albumentations for efficient augmentation pipeline.
"""

from typing import Dict, Any, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


def get_train_transforms(
    image_size: int = 512,
    crop_size: int = 448,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    config: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Includes geometric transforms, color augmentation, and regularization
    techniques suitable for retinal fundus images.
    """
    config = config or {}
    aug_config = config.get("augmentation", {}).get("train", {})
    
    transforms = [
        A.Resize(image_size, image_size),
        A.RandomCrop(crop_size, crop_size),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5 if aug_config.get("horizontal_flip", True) else 0),
        A.VerticalFlip(p=0.5 if aug_config.get("vertical_flip", True) else 0),
        A.Rotate(
            limit=aug_config.get("rotation_limit", 180),
            border_mode=0,
            p=0.7
        ),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=aug_config.get("brightness_limit", 0.2),
            contrast_limit=aug_config.get("contrast_limit", 0.2),
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=int(aug_config.get("hue_saturation_limit", 0.1) * 20),
            sat_shift_limit=int(aug_config.get("hue_saturation_limit", 0.1) * 30),
            val_shift_limit=int(aug_config.get("hue_saturation_limit", 0.1) * 20),
            p=0.3
        ),
        
        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=aug_config.get("blur_limit", 3)),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ]
    
    # Coarse dropout (cutout-style regularization)
    dropout_config = aug_config.get("coarse_dropout", {})
    if dropout_config.get("enabled", True):
        transforms.append(
            A.CoarseDropout(
                max_holes=dropout_config.get("max_holes", 8),
                max_height=dropout_config.get("max_height", 32),
                max_width=dropout_config.get("max_width", 32),
                min_holes=1,
                fill_value=0,
                p=0.3
            )
        )
    
    # Normalize and convert to tensor
    transforms.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


def get_val_transforms(
    image_size: int = 512,
    crop_size: int = 448,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get validation/test transform pipeline (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CenterCrop(crop_size, crop_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_inference_transforms(
    image_size: int = 512,
    crop_size: int = 448,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Alias for validation transforms, used during inference."""
    return get_val_transforms(image_size, crop_size, mean, std)


def get_tta_transforms(
    image_size: int = 512,
    crop_size: int = 448,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> list:
    """
    Get test-time augmentation transforms.
    
    Returns list of transforms for TTA ensemble predictions.
    """
    base = [
        A.Resize(image_size, image_size),
        A.CenterCrop(crop_size, crop_size),
    ]
    
    post = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    
    tta_list = [
        A.Compose(base + post),  # Original
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + post),
        A.Compose(base + [A.VerticalFlip(p=1.0)] + post),
        A.Compose(base + [A.Rotate(limit=(90, 90), p=1.0)] + post),
        A.Compose(base + [A.Rotate(limit=(180, 180), p=1.0)] + post),
    ]
    
    return tta_list


class AlbumentationsWrapper:
    """Wrapper to make albumentations compatible with torchvision-style calls."""
    
    def __init__(self, transform: A.Compose):
        self.transform = transform
        
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        augmented = self.transform(image=image)
        return augmented["image"]


def build_transforms(config: Dict[str, Any], split: str) -> AlbumentationsWrapper:
    """
    Build transforms from config for specified split.
    
    Args:
        config: Dataset configuration dictionary.
        split: One of 'train', 'val', 'test'.
        
    Returns:
        Wrapped transform callable.
    """
    preprocess = config.get("preprocessing", {})
    image_size = preprocess.get("image_size", 512)
    crop_size = preprocess.get("crop_size", 448)
    normalize = preprocess.get("normalize", {})
    mean = tuple(normalize.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(normalize.get("std", [0.229, 0.224, 0.225]))
    
    if split == "train":
        transform = get_train_transforms(image_size, crop_size, mean, std, config)
    else:
        transform = get_val_transforms(image_size, crop_size, mean, std)
        
    return AlbumentationsWrapper(transform)
