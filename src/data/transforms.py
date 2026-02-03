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
    config = config or {}
    aug_config = config.get("augmentation", {}).get("train", {})
    
    transforms = [
        A.Resize(image_size, image_size),
        A.RandomCrop(crop_size, crop_size),
        A.HorizontalFlip(p=0.5 if aug_config.get("horizontal_flip", True) else 0),
        A.VerticalFlip(p=0.5 if aug_config.get("vertical_flip", True) else 0),
        A.Rotate(limit=aug_config.get("rotation_limit", 180), border_mode=0, p=0.7),
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
        A.OneOf([
            A.GaussianBlur(blur_limit=aug_config.get("blur_limit", 3)),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ]
    
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
    
    transforms.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(transforms)


def get_val_transforms(
    image_size: int = 512,
    crop_size: int = 448,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
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
    return get_val_transforms(image_size, crop_size, mean, std)


def get_tta_transforms(
    image_size: int = 512,
    crop_size: int = 448,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> list:
    base = [A.Resize(image_size, image_size), A.CenterCrop(crop_size, crop_size)]
    post = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    
    return [
        A.Compose(base + post),
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + post),
        A.Compose(base + [A.VerticalFlip(p=1.0)] + post),
        A.Compose(base + [A.Rotate(limit=(90, 90), p=1.0)] + post),
        A.Compose(base + [A.Rotate(limit=(180, 180), p=1.0)] + post),
    ]


class AlbumentationsWrapper:
    def __init__(self, transform: A.Compose):
        self.transform = transform
        
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        augmented = self.transform(image=image)
        return augmented["image"]


def build_transforms(config: Dict[str, Any], split: str) -> AlbumentationsWrapper:
    preprocess_cfg = config.get("preprocessing", {})
    image_size = preprocess_cfg.get("resize", 512)
    crop_size = preprocess_cfg.get("crop_size", 448)
    mean = tuple(preprocess_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(preprocess_cfg.get("std", [0.229, 0.224, 0.225]))
    
    if split == "train":
        transform = get_train_transforms(image_size, crop_size, mean, std, config)
    elif split in ["val", "test"]:
        transform = get_val_transforms(image_size, crop_size, mean, std)
    else:
        transform = get_inference_transforms(image_size, crop_size, mean, std)
        
    return AlbumentationsWrapper(transform)
