"""
Training script for diabetic retinopathy classification.
Implements PyTorch Lightning training with full configurability.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.data import create_datamodule
from src.training.losses import build_loss
from src.evaluation.metrics import compute_metrics, quadratic_weighted_kappa
from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger


logger = get_logger(__name__)


class DRClassificationModule(pl.LightningModule):
    """PyTorch Lightning module for DR classification."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Build model and loss
        self.model = build_model(config)
        self.criterion = build_loss(config)
        
        # Metrics tracking
        self.num_classes = config.get("model", {}).get("num_classes", 5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        labels = batch["label"]
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        return {
            "val_loss": loss,
            "preds": preds,
            "labels": labels,
            "logits": logits,
        }
    
    def validation_epoch_end(self, outputs: list) -> None:
        # Aggregate predictions and labels
        all_preds = torch.cat([x["preds"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs])
        all_logits = torch.cat([x["logits"] for x in outputs])
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        # Compute metrics
        metrics = compute_metrics(
            all_preds.cpu().numpy(),
            all_labels.cpu().numpy(),
            all_logits.cpu().numpy(),
            num_classes=self.num_classes
        )
        
        # Compute quadratic weighted kappa (primary metric for DR)
        kappa = quadratic_weighted_kappa(
            all_labels.cpu().numpy(),
            all_preds.cpu().numpy()
        )
        
        # Log metrics
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", metrics["accuracy"], prog_bar=True)
        self.log("val_kappa", kappa, prog_bar=True)
        self.log("val_f1_macro", metrics["f1_macro"])
        self.log("val_auc_macro", metrics.get("auc_macro", 0.0))
        
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs: list) -> None:
        all_preds = torch.cat([x["preds"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs])
        all_logits = torch.cat([x["logits"] for x in outputs])
        
        metrics = compute_metrics(
            all_preds.cpu().numpy(),
            all_labels.cpu().numpy(),
            all_logits.cpu().numpy(),
            num_classes=self.num_classes
        )
        
        kappa = quadratic_weighted_kappa(
            all_labels.cpu().numpy(),
            all_preds.cpu().numpy()
        )
        
        self.log("test_acc", metrics["accuracy"])
        self.log("test_kappa", kappa)
        self.log("test_f1_macro", metrics["f1_macro"])
        self.log("test_auc_macro", metrics.get("auc_macro", 0.0))
        
        # Log per-class metrics
        for i, (p, r, f) in enumerate(zip(
            metrics["precision_per_class"],
            metrics["recall_per_class"],
            metrics["f1_per_class"]
        )):
            self.log(f"test_precision_class_{i}", p)
            self.log(f"test_recall_class_{i}", r)
            self.log(f"test_f1_class_{i}", f)
    
    def configure_optimizers(self):
        opt_cfg = self.config.get("optimizer", {})
        sched_cfg = self.config.get("scheduler", {})
        training_cfg = self.config.get("training", {})
        
        # Build optimizer
        opt_name = opt_cfg.get("name", "adamw")
        lr = opt_cfg.get("lr", 1e-4)
        weight_decay = opt_cfg.get("weight_decay", 1e-4)
        
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=opt_cfg.get("momentum", 0.9),
                nesterov=opt_cfg.get("nesterov", True),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
            
        # Build scheduler
        sched_name = sched_cfg.get("name", "cosine")
        epochs = training_cfg.get("epochs", 50)
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)
        
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=sched_cfg.get("min_lr", 1e-7),
            )
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("gamma", 0.1),
            )
        elif sched_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=sched_cfg.get("patience", 5),
                factor=sched_cfg.get("factor", 0.5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_kappa",
                },
            }
        elif sched_name == "onecycle":
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer
            
        # Add warmup
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs],
            )
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def train(config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file.
        config: Configuration dictionary (overrides config_path).
    """
    # Load configuration
    if config is None:
        config = load_config(config_path)
        
    # Set seed for reproducibility
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    
    training_cfg = config.get("training", {})
    checkpoint_cfg = config.get("checkpoint", {})
    
    # Create data module
    datamodule = create_datamodule(config)
    
    # Create model
    model = DRClassificationModule(config)
    
    # Setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="experiments/checkpoints",
        filename="{epoch:02d}-{val_kappa:.4f}",
        save_top_k=checkpoint_cfg.get("save_top_k", 3),
        monitor=checkpoint_cfg.get("monitor", "val_kappa"),
        mode=checkpoint_cfg.get("mode", "max"),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_cfg = config.get("early_stopping", {})
    if early_stop_cfg.get("enabled", True):
        early_stopping = EarlyStopping(
            monitor=early_stop_cfg.get("monitor", "val_kappa"),
            patience=early_stop_cfg.get("patience", 10),
            mode=early_stop_cfg.get("mode", "max"),
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    # Setup logger
    tb_logger = TensorBoardLogger("experiments/logs", name="dr_classification")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_cfg.get("epochs", 50),
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=training_cfg.get("deterministic", True),
        log_every_n_steps=10,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Test
    logger.info("Running evaluation on test set...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
    logger.info(f"Best model saved to: {checkpoint_callback.best_model_path}")
    
    return trainer, model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DR classification model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    args = parser.parse_args()
    
    train(config_path=args.config)
