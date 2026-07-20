import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from training.config import TrainConfig, DEFAULT_CONFIG
from training.dataset import create_dataloader, create_weighted_sampler
from training.losses import get_loss_fn
from training.metrics import MetricsTracker, EpochMetrics
from model.efficientnet import DRModel
from preprocessing.transforms import get_train_transforms, get_val_transforms


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    config: TrainConfig,
) -> EpochMetrics:
    model.train()
    tracker = MetricsTracker()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        mixed_images, labels_a, labels_b, lam = mixup_batch(images, labels, config.mixup_alpha)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=config.use_mixed_precision):
            logits = model(mixed_images)
            loss = lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)

        if config.use_mixed_precision and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        preds = logits.argmax(dim=1).cpu().tolist()
        tracker.update(loss.item(), preds, labels_a.cpu().tolist())

    return tracker.compute()


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    tracker = MetricsTracker()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        preds = logits.argmax(dim=1).cpu().tolist()
        tracker.update(loss.item(), preds, labels.cpu().tolist())

    return tracker.compute()


def save_checkpoint(
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    scaler: GradScaler,
    epoch: int,
    qwk: float,
    best_qwk: float,
    config: TrainConfig,
) -> None:
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "qwk": qwk,
        "best_qwk": best_qwk,
        "config": config,
    }
    torch.save(checkpoint, model_dir / filename)
    print(f"Saved checkpoint: {model_dir / filename} at epoch {epoch}")


def detect_newest_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[Path | None, int, float]:
    best_path = None
    max_epoch = -1
    best_qwk = -1.0
    
    # Check all possible checkpoint names in directory
    for fname in ["best_model.pth", "latest_checkpoint.pth", "latest_model.pth"]:
        path = checkpoint_dir / fname
        if path.exists():
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                epoch = ckpt.get("epoch", -1)
                qwk = ckpt.get("best_qwk", ckpt.get("qwk", -1.0))
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_path = path
                    best_qwk = qwk
            except Exception as e:
                print(f"[Warning] Could not parse checkpoint {path}: {e}")
                
    return best_path, max_epoch, best_qwk


def compute_class_weights(csv_path: str, num_classes: int, device: torch.device) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    counts = df["diagnosis"].value_counts().sort_index()
    weights = len(df) / (num_classes * counts)
    return torch.tensor(weights.values, dtype=torch.float32).to(device)


def main(config: TrainConfig = DEFAULT_CONFIG) -> None:
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    print(f"Training on device: {device}")

    model = DRModel(num_classes=config.num_classes, dropout_rate=config.dropout_rate, pretrained=True).to(device)

    class_weights = compute_class_weights(config.train_csv, config.num_classes, device)
    print(f"Class weights: {class_weights.cpu().tolist()}")

    train_sampler = create_weighted_sampler(config.train_csv, config.num_classes)

    train_loader = create_dataloader(
        config.train_csv, config.train_image_dir, get_train_transforms(), config,
        shuffle=False, sampler=train_sampler,
    )
    val_loader = create_dataloader(
        config.val_csv, config.val_image_dir, get_val_transforms(), config,
        shuffle=False,
    )

    loss_fn = get_loss_fn(
        loss_type="label_smoothing",
        num_classes=config.num_classes,
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scaler = GradScaler(enabled=config.use_mixed_precision)

    best_qwk = -1.0
    start_epoch = 1

    checkpoint_dir = Path(config.model_dir)
    best_path, max_epoch, best_qwk = detect_newest_checkpoint(checkpoint_dir, device)

    if best_path is not None:
        print(f"Automatically resuming training from newest checkpoint: {best_path} (Epoch: {max_epoch}, Best QWK: {best_qwk:.4f})")
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = max_epoch + 1
        except Exception as e:
            print(f"[Warning] Failed to restore complete checkpoint: {e}. Starting from scratch.")

    patience_counter = 0
    epoch = start_epoch

    try:
        for epoch in range(start_epoch, config.num_epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, config)
            val_metrics = validate_one_epoch(model, val_loader, loss_fn, device)
            scheduler.step()

            print(
                f"Epoch {epoch}/{config.num_epochs} - "
                f"Train Loss: {train_metrics.loss:.4f}, Acc: {train_metrics.accuracy:.4f}, QWK: {train_metrics.qwk:.4f} | "
                f"Val Loss: {val_metrics.loss:.4f}, Acc: {val_metrics.accuracy:.4f}, QWK: {val_metrics.qwk:.4f}"
            )

            # Save latest checkpoint at the end of every epoch (Replaces latest_model.pth with latest_checkpoint.pth)
            save_checkpoint("latest_checkpoint.pth", model, optimizer, scheduler, scaler, epoch, val_metrics.qwk, best_qwk, config)

            if val_metrics.qwk > best_qwk:
                best_qwk = val_metrics.qwk
                save_checkpoint("best_model.pth", model, optimizer, scheduler, scaler, epoch, val_metrics.qwk, best_qwk, config)
                patience_counter = 0
                print(f"=> Best model saved | Val QWK: {best_qwk:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print("Early stopping triggered due to patience.")
                    break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint before exiting...")
        save_checkpoint("latest_checkpoint.pth", model, optimizer, scheduler, scaler, epoch, -1.0, best_qwk, config)
        print("Checkpoint saved. Exiting.")
        sys.exit(0)

    # Automatically trigger Evaluation and GradCAM scripts upon training completion
    print("\nTraining execution finished. Automatically launching evaluation on test set...")
    try:
        from training.evaluate import main as run_evaluation
        run_evaluation(config)
    except Exception as e:
        print(f"Error during evaluation execution: {e}")

    print("\nAutomatically launching new GradCAM visualizations...")
    try:
        from scripts.generate_gradcam import main as run_gradcam
        run_gradcam()
    except Exception as e:
        print(f"Error during GradCAM generation: {e}")


if __name__ == "__main__":
    main()
