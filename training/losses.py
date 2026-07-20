import torch
import torch.nn as nn
from typing import Optional

def get_loss_fn(
    loss_type: str = "cross_entropy",
    num_classes: int = 5,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1
) -> nn.Module:
    if loss_type in ["cross_entropy", "label_smoothing"]:
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing if loss_type == "label_smoothing" else 0.0)
    raise ValueError(f"Unsupported loss type: {loss_type}")
