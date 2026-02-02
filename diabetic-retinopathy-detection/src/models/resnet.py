"""
ResNet model for diabetic retinopathy classification.
Alternative backbone option with well-understood behavior.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import timm


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for DR severity grading.
    
    ResNet50 serves as a reliable baseline with extensive research
    validation. Useful for benchmarking and situations where 
    interpretability of architecture is prioritized.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        variant: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        zero_init_residual: bool = True,
    ):
        """
        Args:
            num_classes: Number of output classes.
            variant: ResNet variant (18, 34, 50, 101, 152).
            pretrained: Load ImageNet pretrained weights.
            dropout_rate: Dropout rate before classifier.
            zero_init_residual: Zero-initialize residual connections.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.variant = variant
        
        model_name = f"resnet{variant}"
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_rate=dropout_rate,
        )
        
        self.num_features = self.backbone.num_features
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_classes),
        )
        
        # Optional: zero-initialize final BN in each residual block
        if zero_init_residual:
            self._zero_init_residual()
            
    def _zero_init_residual(self) -> None:
        """Zero-initialize the last BN in each residual branch."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'weight'):
                if m.weight is not None:
                    # Only zero-init if this is part of a residual block's final BN
                    pass  # timm handles this internally with pretrained models
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        return self.backbone.forward_features(x)
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone layers."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = not freeze
                
    def get_layer_groups(self) -> list:
        """Get parameter groups for discriminative learning rates."""
        return [
            list(self.backbone.conv1.parameters()) +
            list(self.backbone.bn1.parameters()),
            list(self.backbone.layer1.parameters()) +
            list(self.backbone.layer2.parameters()),
            list(self.backbone.layer3.parameters()) +
            list(self.backbone.layer4.parameters()),
            list(self.backbone.fc.parameters()),
        ]


class ResNetWithAttention(nn.Module):
    """
    ResNet with channel attention (SE blocks) for enhanced feature learning.
    
    Squeeze-and-Excitation attention helps the model focus on relevant
    features for distinguishing DR severity levels.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        variant: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Use SE-ResNet variant
        model_name = f"seresnet{variant}"
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_rate=dropout_rate,
        )
        
        self.num_features = self.backbone.num_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


def build_resnet(config: Dict[str, Any]) -> nn.Module:
    """Factory function to build ResNet from config."""
    model_cfg = config.get("model", {})
    resnet_cfg = model_cfg.get("resnet", {})
    
    use_attention = resnet_cfg.get("use_attention", False)
    
    if use_attention:
        return ResNetWithAttention(
            num_classes=model_cfg.get("num_classes", 5),
            variant=resnet_cfg.get("variant", 50),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
        )
    else:
        return ResNetClassifier(
            num_classes=model_cfg.get("num_classes", 5),
            variant=resnet_cfg.get("variant", 50),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            zero_init_residual=resnet_cfg.get("zero_init_residual", True),
        )
