"""Model architectures for diabetic retinopathy classification."""

from typing import Dict, Any
import torch.nn as nn

from .efficientnet import EfficientNetClassifier, EfficientNetMultimodal, build_efficientnet
from .resnet import ResNetClassifier, ResNetWithAttention, build_resnet
from .vit import ViTClassifier, ViTHybrid, build_vit


MODEL_REGISTRY = {
    "efficientnet_b3": build_efficientnet,
    "efficientnet": build_efficientnet,
    "resnet50": build_resnet,
    "resnet": build_resnet,
    "vit_base": build_vit,
    "vit": build_vit,
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build model from configuration.
    
    Args:
        config: Full configuration dictionary with model settings.
        
    Returns:
        Initialized model.
    """
    model_name = config.get("model", {}).get("name", "efficientnet_b3")
    
    # Normalize model name
    model_key = model_name.lower().replace("-", "_")
    
    if model_key not in MODEL_REGISTRY:
        # Try partial match
        for key in MODEL_REGISTRY:
            if key in model_key or model_key in key:
                model_key = key
                break
        else:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
    
    return MODEL_REGISTRY[model_key](config)


__all__ = [
    "EfficientNetClassifier",
    "EfficientNetMultimodal",
    "ResNetClassifier", 
    "ResNetWithAttention",
    "ViTClassifier",
    "ViTHybrid",
    "build_model",
    "build_efficientnet",
    "build_resnet",
    "build_vit",
]
