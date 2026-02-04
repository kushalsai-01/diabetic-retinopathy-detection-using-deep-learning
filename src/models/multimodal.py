"""
Multimodal Model for Diabetic Retinopathy Classification

Combines fundus images with patient clinical data (tabular features).
Supports flexible inference: works with OR without clinical data.
"""

from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TabularFeatureEncoder(nn.Module):
    """Encodes clinical/patient tabular features."""
    
    def __init__(
        self,
        num_features: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        
        layers = []
        in_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        self.output_dim = in_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) - tabular features
        Returns:
            (batch_size, output_dim) - encoded features
        """
        return self.encoder(x)


class FusionModule(nn.Module):
    """Fuses image and tabular features."""
    
    def __init__(
        self,
        image_dim: int,
        tabular_dim: int,
        fusion_type: str = "concat",
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            image_dim: Dimension of image features
            tabular_dim: Dimension of tabular features
            fusion_type: 'concat', 'addition', 'attention'
            hidden_dim: For attention fusion
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.output_dim = image_dim + tabular_dim
        elif fusion_type == "addition":
            assert image_dim == tabular_dim, "Dims must match for addition fusion"
            self.output_dim = image_dim
        elif fusion_type == "attention":
            hidden_dim = hidden_dim or (image_dim // 4)
            self.attention = nn.Sequential(
                nn.Linear(image_dim + tabular_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, image_dim),
                nn.Sigmoid()
            )
            self.output_dim = image_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        image_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch_size, image_dim)
            tabular_features: (batch_size, tabular_dim)
        Returns:
            (batch_size, output_dim) - fused features
        """
        if self.fusion_type == "concat":
            return torch.cat([image_features, tabular_features], dim=1)
        elif self.fusion_type == "addition":
            return image_features + tabular_features
        elif self.fusion_type == "attention":
            # Attention-weighted fusion
            concat = torch.cat([image_features, tabular_features], dim=1)
            attention_weights = self.attention(concat)
            return image_features * attention_weights
        

class MultimodalDRClassifier(nn.Module):
    """
    Multimodal Diabetic Retinopathy Classifier.
    
    Combines:
    1. Fundus images (via CNN/ViT backbone)
    2. Clinical/patient data (via MLP encoder)
    
    Key Feature: Works with OR without tabular data during inference.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_tabular_features: int = 9,
        tabular_hidden_dims: Tuple[int, ...] = (128, 64),
        fusion_type: str = "concat",
        dropout_rate: float = 0.3,
        tabular_dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of DR severity classes (default: 5)
            backbone: Image backbone ('resnet50', 'efficientnet_b3', 'vit_base_patch16_224')
            pretrained: Use ImageNet pretrained weights
            num_tabular_features: Number of clinical features
            tabular_hidden_dims: Hidden layer dimensions for tabular encoder
            fusion_type: How to combine modalities ('concat', 'addition', 'attention')
            dropout_rate: Dropout for classifier head
            tabular_dropout_rate: Dropout for tabular features during training (makes model robust to missing data)
            freeze_backbone: Freeze image backbone weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_tabular_features = num_tabular_features
        self.tabular_dropout_rate = tabular_dropout_rate
        
        # Image backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        self.image_feature_dim = self.backbone.num_features
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Tabular encoder
        self.tabular_encoder = TabularFeatureEncoder(
            num_features=num_tabular_features,
            hidden_dims=tabular_hidden_dims,
            dropout_rate=dropout_rate,
        )
        
        # Fusion module
        self.fusion = FusionModule(
            image_dim=self.image_feature_dim,
            tabular_dim=self.tabular_encoder.output_dim,
            fusion_type=fusion_type,
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Image-only classifier (fallback when no tabular data)
        self.image_only_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.image_feature_dim, num_classes)
        )
        
        # Tabular feature dropout during training
        self.tabular_dropout = nn.Dropout(p=tabular_dropout_rate)
    
    def forward(
        self,
        image: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        use_tabular: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with flexible tabular input.
        
        Args:
            image: (batch_size, 3, H, W) - fundus images
            tabular: (batch_size, num_features) - clinical data (optional)
            use_tabular: Whether to use tabular features if provided
            
        Returns:
            (batch_size, num_classes) - logits
        """
        # Extract image features
        image_features = self.backbone(image)  # (B, image_feature_dim)
        
        # Check if we have tabular data and should use it
        has_tabular = (tabular is not None) and use_tabular
        
        if has_tabular:
            # Apply dropout to tabular features during training
            # This makes the model robust to missing clinical data
            if self.training:
                # Randomly zero out entire tabular input for some samples
                # This forces the model to learn from images alone
                mask = torch.rand(tabular.size(0), 1, device=tabular.device) > self.tabular_dropout_rate
                tabular_dropped = tabular * mask
            else:
                tabular_dropped = tabular
            
            # Encode tabular features
            tabular_features = self.tabular_encoder(tabular_dropped)
            
            # Fuse modalities
            fused_features = self.fusion(image_features, tabular_features)
            
            # Classify
            logits = self.classifier(fused_features)
        else:
            # Image-only path (no clinical data available)
            logits = self.image_only_classifier(image_features)
        
        return logits
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze the image backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for debugging."""
        return {
            "image_features": self.image_feature_dim,
            "tabular_features": self.tabular_encoder.output_dim,
            "fused_features": self.fusion.output_dim,
        }


def create_multimodal_model(config: Dict[str, Any]) -> MultimodalDRClassifier:
    """Factory function to create multimodal model from config."""
    model_config = config.get("model", {})
    
    return MultimodalDRClassifier(
        num_classes=model_config.get("num_classes", 5),
        backbone=model_config.get("backbone", "resnet50"),
        pretrained=model_config.get("pretrained", True),
        num_tabular_features=model_config.get("num_tabular_features", 9),
        tabular_hidden_dims=tuple(model_config.get("tabular_hidden_dims", [128, 64])),
        fusion_type=model_config.get("fusion_type", "concat"),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        tabular_dropout_rate=model_config.get("tabular_dropout_rate", 0.5),
        freeze_backbone=model_config.get("freeze_backbone", False),
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    image = torch.randn(batch_size, 3, 448, 448)
    tabular = torch.randn(batch_size, 9)
    
    model = MultimodalDRClassifier(
        num_classes=5,
        backbone="resnet50",
        num_tabular_features=9,
    )
    
    print("Model created successfully!")
    print(f"Feature dimensions: {model.get_feature_dims()}")
    
    # Test with both modalities
    output_multimodal = model(image, tabular)
    print(f"Multimodal output shape: {output_multimodal.shape}")
    
    # Test with image only
    output_image_only = model(image, tabular=None)
    print(f"Image-only output shape: {output_image_only.shape}")
    
    # Test training mode (with tabular dropout)
    model.train()
    output_train = model(image, tabular)
    print(f"Training mode output shape: {output_train.shape}")
    
    print("\n✅ All tests passed!")
