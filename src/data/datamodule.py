from pathlib import Path
from typing import Optional, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import DiabeticRetinopathyDataset
from .transforms import build_transforms


class DRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_csv: str = "train.csv",
        val_csv: str = "val.csv",
        test_csv: Optional[str] = "test.csv",
        image_col: str = "image_path",
        label_col: str = "diagnosis",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_weighted_sampling: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.image_col = image_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling
        self.config = config or {}
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        
        if stage == "fit" or stage is None:
            train_transform = build_transforms(self.config, "train")
            val_transform = build_transforms(self.config, "val")
            
            self.train_dataset = DiabeticRetinopathyDataset(
                csv_path=self.data_dir / self.train_csv,
                image_dir=self.data_dir,
                image_col=self.image_col,
                label_col=self.label_col,
                transform=train_transform,
            )
            
                image_dir=self.data_dir,
                image_col=self.image_col,
                label_col=self.label_col,
                transform=val_transform,
            )
            
        if stage == "test" or stage is None:
            if self.test_csv and (self.data_dir / self.test_csv).exists():
                test_transform = build_transforms(self.config, "test")
                self.test_dataset = DiabeticRetinopathyDataset(
                    csv_path=self.data_dir / self.test_csv,
                    image_dir=self.data_dir,
                    image_col=self.image_col,
                    label_col=self.label_col,
                    transform=test_transform,
                )
                
    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        
        if self.use_weighted_sampling:
            sample_weights = self.train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Check test_csv path.")
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    @property
    def num_classes(self) -> int:
        return DiabeticRetinopathyDataset.NUM_CLASSES
    
    @property
    def class_names(self) -> list:
        return DiabeticRetinopathyDataset.CLASS_NAMES


def create_datamodule(config: Dict[str, Any]) -> DRDataModule:
    """Factory function to create DataModule from config."""
    dataset_cfg = config.get("dataset", {})
    
    return DRDataModule(
        data_dir=dataset_cfg.get("root_dir", "data/processed"),
        train_csv=dataset_cfg.get("train_csv", "train.csv"),
        val_csv=dataset_cfg.get("val_csv", "val.csv"),
        test_csv=dataset_cfg.get("test_csv", "test.csv"),
        image_col=dataset_cfg.get("image_col", "image_path"),
        label_col=dataset_cfg.get("label_col", "diagnosis"),
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=training_cfg.get("pin_memory", True),
        use_weighted_sampling=dataset_cfg.get("class_weights", {}).get("enabled", True),
        config=config,
    )
