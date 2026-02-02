from typing import Dict, Any

import torch
import torch.nn as nn
import timm


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        variant: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        zero_init_residual: bool = True,
    ):
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
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_classes),
        )
        
        if zero_init_residual:
            self._zero_init_residual()
            
    def _zero_init_residual(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'weight'):
                if m.weight is not None:
                    pass
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = not freeze
                
    def get_layer_groups(self) -> list:
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
    def __init__(
        self,
        num_classes: int = 5,
        variant: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        
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
