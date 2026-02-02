"""
Validation and inference utilities for DR classification.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.data import InferenceDataset, get_inference_transforms, get_tta_transforms
from src.evaluation.metrics import compute_metrics, quadratic_weighted_kappa
from src.utils.config_loader import load_config


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Run validation loop and compute metrics.
    
    Args:
        model: Trained model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device to run on.
        num_classes: Number of classes.
        
    Returns:
        Dictionary of validation metrics.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
            total_loss += loss.item()
            num_batches += 1
            
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    
    metrics = compute_metrics(all_preds, all_labels, all_logits, num_classes)
    metrics["loss"] = total_loss / num_batches
    metrics["kappa"] = quadratic_weighted_kappa(all_labels, all_preds)
    
    return metrics


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    return_probabilities: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Run inference on a dataset.
    
    Args:
        model: Trained model.
        dataloader: Inference dataloader.
        device: Device to run on.
        return_probabilities: Whether to return class probabilities.
        
    Returns:
        Tuple of (predictions, probabilities, image_paths).
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_paths = []
    
    for batch in tqdm(dataloader, desc="Predicting"):
        images = batch["image"].to(device)
        paths = batch["image_path"]
        
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_paths.extend(paths)
        
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs) if return_probabilities else None
    
    return predictions, probabilities, all_paths


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    image_paths: List[str],
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with test-time augmentation.
    
    Args:
        model: Trained model.
        image_paths: List of image paths.
        config: Configuration dictionary.
        device: Device to run on.
        batch_size: Batch size.
        
    Returns:
        Tuple of (predictions, averaged_probabilities).
    """
    model.eval()
    
    preprocess_cfg = config.get("preprocessing", {})
    tta_transforms = get_tta_transforms(
        image_size=preprocess_cfg.get("image_size", 512),
        crop_size=preprocess_cfg.get("crop_size", 448),
    )
    
    all_probs = []
    
    for transform in tta_transforms:
        # Create dataset with current transform
        from src.data.transforms import AlbumentationsWrapper
        wrapped_transform = AlbumentationsWrapper(transform)
        
        dataset = InferenceDataset(image_paths, transform=wrapped_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Run inference
        batch_probs = []
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            batch_probs.append(probs.cpu().numpy())
            
        all_probs.append(np.concatenate(batch_probs))
        
    # Average predictions across augmentations
    averaged_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(averaged_probs, axis=1)
    
    return predictions, averaged_probs


def load_model_for_inference(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration dictionary.
        device: Device to load model on.
        
    Returns:
        Loaded model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = build_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix if present (Lightning format)
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
        }
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    test_csv: str,
    data_dir: str,
) -> Dict[str, float]:
    """
    Evaluate a checkpoint on a test set.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to configuration file.
        test_csv: Path to test CSV.
        data_dir: Data directory.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    from src.data import DiabeticRetinopathyDataset, build_transforms
    
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model_for_inference(checkpoint_path, config, device)
    
    # Create test dataset
    transform = build_transforms(config, "test")
    dataset = DiabeticRetinopathyDataset(
        csv_path=test_csv,
        image_dir=data_dir,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Run validation
    from src.training.losses import build_loss
    criterion = build_loss(config)
    
    metrics = validate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        num_classes=config.get("model", {}).get("num_classes", 5),
    )
    
    return metrics
