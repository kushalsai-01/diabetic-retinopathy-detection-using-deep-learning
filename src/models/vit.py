"""
Vision Transformer (ViT) model for diabetic retinopathy classification.
Transformer-based alternative for capturing global image context.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    """
    Vision Transformer classifier for DR severity grading.
    
    ViT captures long-range dependencies in retinal images, potentially
    identifying relationships between distant lesions. Requires more data
    and compute but offers strong performance on large datasets.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        image_size: int = 448,
    ):
        """
        Args:
            num_classes: Number of output classes.
            patch_size: Size of image patches.
            embed_dim: Embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dim ratio.
            pretrained: Load pretrained weights.
            dropout_rate: Dropout rate.
            image_size: Input image size.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Select appropriate ViT variant based on config
        if embed_dim == 768 and depth == 12:
            model_name = "vit_base_patch16_224"
        elif embed_dim == 384 and depth == 12:
            model_name = "vit_small_patch16_224"
        elif embed_dim == 192 and depth == 12:
            model_name = "vit_tiny_patch16_224"
        else:
            model_name = "vit_base_patch16_224"
            
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=image_size,
            drop_rate=dropout_rate,
            num_classes=num_classes,
        )
        
        self.num_features = self.backbone.num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features."""
        return self.backbone.forward_features(x)[:, 0]
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all transformer blocks.
        Useful for visualization and interpretability.
        """
        attention_maps = []
        
        # Get patch embeddings
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        # Forward through blocks, collecting attention
        for block in self.backbone.blocks:
            # Get attention weights
            B, N, C = x.shape
            qkv = block.attn.qkv(block.norm1(x))
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach())
            
            # Continue forward pass
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            
        return attention_maps
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters."""
        for name, param in self.backbone.named_parameters():
            if 'head' not in name:
                param.requires_grad = not freeze


class ViTHybrid(nn.Module):
    """
    Hybrid ViT with CNN stem for better inductive bias.
    
    Combines CNN's local feature extraction with ViT's global attention,
    often performing better on smaller medical imaging datasets.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        image_size: int = 448,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # ResNet50 backbone as feature extractor
        self.backbone = timm.create_model(
            "vit_base_r50_s16_224",
            pretrained=pretrained,
            img_size=image_size,
            drop_rate=dropout_rate,
            num_classes=num_classes,
        )
        
        self.num_features = self.backbone.num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)[:, 0]


def build_vit(config: Dict[str, Any]) -> nn.Module:
    """Factory function to build ViT from config."""
    model_cfg = config.get("model", {})
    vit_cfg = model_cfg.get("vit", {})
    preprocess_cfg = config.get("preprocessing", {})
    
    use_hybrid = vit_cfg.get("use_hybrid", False)
    
    if use_hybrid:
        return ViTHybrid(
            num_classes=model_cfg.get("num_classes", 5),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.1),
            image_size=preprocess_cfg.get("crop_size", 448),
        )
    else:
        return ViTClassifier(
            num_classes=model_cfg.get("num_classes", 5),
            patch_size=vit_cfg.get("patch_size", 16),
            embed_dim=vit_cfg.get("embed_dim", 768),
            depth=vit_cfg.get("depth", 12),
            num_heads=vit_cfg.get("num_heads", 12),
            mlp_ratio=vit_cfg.get("mlp_ratio", 4.0),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.1),
            image_size=preprocess_cfg.get("crop_size", 448),
        )
