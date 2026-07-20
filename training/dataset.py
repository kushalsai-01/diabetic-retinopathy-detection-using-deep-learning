import cv2
import numpy as np
import pandas as pd
import torch
import sys
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional
import albumentations as A

from training.config import TrainConfig

class APTOSDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transforms: Optional[A.Compose] = None,
        is_test: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        max_retries = 10
        current_idx = idx
        
        for attempt in range(max_retries):
            row = self.df.iloc[current_idx]
            image_id = row["id_code"]
            image_path = self.image_dir / f"{image_id}.png"

            try:
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file does not exist: {image_path}")

                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Image could not be read/decoded: {image_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transforms:
                    augmented = self.transforms(image=image)
                    image = augmented["image"]

                sample = {"image": image, "image_id": image_id}

                if not self.is_test and "diagnosis" in row:
                    sample["label"] = int(row["diagnosis"])

                return sample
            except Exception as e:
                print(
                    f"[Warning] Skipping corrupted image {image_id} at index {current_idx} "
                    f"(Path: {image_path}). Reason: {e}",
                    file=sys.stderr,
                )
                # Pick a random fallback index from the dataset
                current_idx = np.random.randint(0, len(self.df))

        raise RuntimeError(f"Failed to load a valid image after {max_retries} attempts.")

def create_weighted_sampler(csv_path: str, num_classes: int = 5) -> WeightedRandomSampler:
    df = pd.read_csv(csv_path)
    class_counts = df["diagnosis"].value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = df["diagnosis"].map(class_weights).values
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(df),
        replacement=True,
    )

def create_dataloader(
    csv_path: str,
    image_dir: str,
    transforms: A.Compose,
    config: TrainConfig,
    shuffle: bool = False,
    is_test: bool = False,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    dataset = APTOSDataset(csv_path, image_dir, transforms, is_test)
    
    num_workers = config.num_workers
    persistent_workers = False
    prefetch_factor = None

    # Safe worker settings for Windows to prevent crash
    if num_workers > 0:
        persistent_workers = True
        prefetch_factor = 2

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
