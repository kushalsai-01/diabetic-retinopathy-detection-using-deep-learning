import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 5, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DRModel(nn.Module):
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.pool = GeMPooling()
        
        in_features = backbone.classifier[1].in_features
        self.classifier = ClassifierHead(in_features, num_classes, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
