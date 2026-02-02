from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import timm


class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        variant: str = "b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        
        model_name = f"efficientnet_{variant}"
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_rate=dropout_rate,
            drop_path_rate=drop_connect_rate,
        )
        
        self.num_features = self.backbone.num_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
    def get_layer_groups(self) -> list:
        return [
            list(self.backbone.conv_stem.parameters()) + list(self.backbone.bn1.parameters()),
            list(self.backbone.blocks[:3].parameters()),
            list(self.backbone.blocks[3:].parameters()),
            list(self.backbone.conv_head.parameters()) +
            list(self.backbone.bn2.parameters()) +
            list(self.backbone.classifier.parameters()),
        ]


class EfficientNetMultimodal(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        variant: str = "b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        tabular_dim: int = 0,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.tabular_dim = tabular_dim
        
        model_name = f"efficientnet_{variant}"
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_rate=dropout_rate,
            num_classes=0,
        )
        self.num_features = self.backbone.num_features
        
        if tabular_dim > 0:
            combined_dim = self.num_features + tabular_dim
            self.fusion = nn.Sequential(
                nn.Linear(combined_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            )
            self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
        else:
            self.fusion = None
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.num_features, num_classes),
            )
            
    def forward(
        self,
        image: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_features = self.backbone(image)
        
        if self.tabular_dim > 0 and tabular is not None:
            combined = torch.cat([image_features, tabular], dim=1)
            fused = self.fusion(combined)
            return self.classifier(fused)
        else:
            return self.classifier(image_features)


def build_efficientnet(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    efficientnet_cfg = model_cfg.get("efficientnet", {})
    multimodal_cfg = config.get("dataset", {}).get("multimodal", {})
    
    tabular_features = multimodal_cfg.get("tabular_features", [])
    
    if multimodal_cfg.get("enabled", False) and tabular_features:
        return EfficientNetMultimodal(
            num_classes=model_cfg.get("num_classes", 5),
            variant=efficientnet_cfg.get("variant", "b3"),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            tabular_dim=len(tabular_features),
        )
    else:
        return EfficientNetClassifier(
            num_classes=model_cfg.get("num_classes", 5),
            variant=efficientnet_cfg.get("variant", "b3"),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            drop_connect_rate=efficientnet_cfg.get("drop_connect_rate", 0.2),
        )
