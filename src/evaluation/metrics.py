"""
Evaluation metrics for diabetic retinopathy classification.
"""

from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute quadratic weighted kappa score.
    
    Primary evaluation metric for DR severity grading, as it accounts
    for the ordinal nature of the classes and penalizes predictions
    that are further from the true class.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Quadratic weighted kappa score.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_pred: Predicted class labels.
        y_true: Ground truth labels.
        y_proba: Predicted probabilities (optional, for AUC).
        num_classes: Number of classes.
        
    Returns:
        Dictionary containing all metrics.
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    metrics["precision_per_class"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["recall_per_class"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["f1_per_class"] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro-averaged metrics
    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Weighted metrics (accounts for class imbalance)
    metrics["precision_weighted"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["recall_weighted"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["f1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    
    # Quadratic weighted kappa
    metrics["kappa"] = quadratic_weighted_kappa(y_true, y_pred)
    
    # AUC (requires probability predictions)
    if y_proba is not None:
        try:
            # One-vs-rest AUC
            metrics["auc_per_class"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average=None,
            )
            metrics["auc_macro"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
            metrics["auc_weighted"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            # Handle cases where not all classes are present
            metrics["auc_per_class"] = np.zeros(num_classes)
            metrics["auc_macro"] = 0.0
            metrics["auc_weighted"] = 0.0
            
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> str:
    """
    Generate and return a formatted classification report.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional class names.
        
    Returns:
        Formatted classification report string.
    """
    if class_names is None:
        class_names = [
            "No DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative"
        ]
        
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    
    kappa = quadratic_weighted_kappa(y_true, y_pred)
    report += f"\nQuadratic Weighted Kappa: {kappa:.4f}"
    
    return report


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Compute per-class sensitivity and specificity.
    
    Important for medical applications where false negatives
    (missing disease) can be more costly than false positives.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        
    Returns:
        Dictionary with sensitivity and specificity per class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def binary_metrics_referable_dr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary metrics for referable DR detection.
    
    Referable DR is defined as moderate NPDR or worse (class >= 2).
    This is clinically relevant as these patients require specialist referral.
    
    Args:
        y_true: Ground truth severity labels (0-4).
        y_pred: Predicted severity labels (0-4).
        
    Returns:
        Dictionary of binary classification metrics.
    """
    # Convert to binary: referable (class >= 2) vs non-referable
    y_true_binary = (y_true >= 2).astype(int)
    y_pred_binary = (y_pred >= 2).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive predictive value
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative predictive value
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
    }
