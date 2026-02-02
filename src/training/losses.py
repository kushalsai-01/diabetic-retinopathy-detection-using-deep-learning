"""
Loss functions for diabetic retinopathy classification.
Includes standard cross-entropy and focal loss for class imbalance.
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in DR classification.
    
    Down-weights well-classified examples, focusing training on hard negatives.
    Particularly useful for DR where class 0 (No DR) often dominates.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            gamma: Focusing parameter. Higher values down-weight easy examples more.
            alpha: Per-class weights. If None, no class weighting is applied.
            reduction: 'none', 'mean', or 'sum'.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if alpha is not None:
            self.register_buffer(
                "alpha_tensor",
                torch.tensor(alpha, dtype=torch.float32)
            )
        else:
            self.alpha_tensor = None
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C).
            targets: Ground truth labels of shape (N,).
            
        Returns:
            Focal loss value.
        """
        num_classes = inputs.size(-1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_one_hot = (
                targets_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
            ce_loss = -targets_one_hot * F.log_softmax(inputs, dim=-1)
            ce_loss = ce_loss.sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weighting
        if self.alpha_tensor is not None:
            alpha_t = self.alpha_tensor[targets]
            focal_weight = alpha_t * focal_weight
            
        loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            smoothing: Label smoothing factor (0 to 1).
            weight: Per-class weights.
            reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = (
            targets_one_hot * (1 - self.smoothing)
            + self.smoothing / num_classes
        )
        
        # Compute loss
        loss = -targets_smooth * log_probs
        
        # Apply class weights
        if self.weight is not None:
            weight_expanded = self.weight.unsqueeze(0).expand_as(loss)
            loss = loss * weight_expanded
            
        loss = loss.sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for DR severity grading.
    
    Treats the problem as ordered classification, penalizing predictions
    that are far from the true class more than adjacent misclassifications.
    """
    
    def __init__(self, num_classes: int = 5, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Implements CORN (Conditional Ordinal Regression with Neural Networks).
        """
        batch_size = inputs.size(0)
        
        # Create ordinal targets: for class k, all labels < k should be 1
        ordinal_targets = torch.zeros(batch_size, self.num_classes - 1, device=inputs.device)
        for i in range(self.num_classes - 1):
            ordinal_targets[:, i] = (targets > i).float()
            
        # Use first num_classes-1 logits for ordinal prediction
        ordinal_logits = inputs[:, :self.num_classes - 1]
        
        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(
            ordinal_logits,
            ordinal_targets,
            reduction=self.reduction
        )
        
        return loss


def build_loss(config: dict) -> nn.Module:
    """
    Build loss function from configuration.
    
    Args:
        config: Configuration dictionary with loss settings.
        
    Returns:
        Loss function module.
    """
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "cross_entropy")
    
    if loss_name == "focal":
        return FocalLoss(
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", None),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )
    elif loss_name == "cross_entropy":
        smoothing = loss_cfg.get("label_smoothing", 0.0)
        if smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            return nn.CrossEntropyLoss()
    elif loss_name == "ordinal":
        num_classes = config.get("model", {}).get("num_classes", 5)
        return OrdinalRegressionLoss(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
