from pathlib import Path
from typing import Optional, Callable, Dict, Any

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DiabeticRetinopathyDataset(Dataset):
    NUM_CLASSES = 5
    CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_col: str = "image_path",
        label_col: str = "diagnosis",
        transform: Optional[Callable] = None,
        tabular_features: Optional[list] = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.tabular_features = tabular_features or []
        self.df = pd.read_csv(csv_path)
        self._validate_data()
        
    def _validate_data(self):
        required_cols = [self.image_col, self.label_col]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for col in self.tabular_features:
            if col not in self.df.columns:
                raise ValueError(f"Tabular feature column not found: {col}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = self.image_dir / row[self.image_col]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        sample = {"image": image, "label": label, "image_path": str(image_path)}
        
        if self.tabular_features:
            tabular = torch.tensor([row[f] for f in self.tabular_features], dtype=torch.float32)
            sample["tabular"] = tabular
            
        return sample
    
    def get_class_weights(self, method: str = "inverse_frequency") -> torch.Tensor:
        class_counts = self.df[self.label_col].value_counts().sort_index().values
        
        if method == "inverse_frequency":
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * len(class_counts)
        elif method == "effective_samples":
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, class_counts)
            weights = (1.0 - beta) / effective_num
            weights = weights / weights.sum() * len(class_counts)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
            
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> torch.Tensor:
        class_weights = self.get_class_weights()
        labels = self.df[self.label_col].values
        return class_weights[labels]
    
    @property
    def class_distribution(self) -> Dict[int, int]:
        return self.df[self.label_col].value_counts().sort_index().to_dict()


class InferenceDataset(Dataset):
    def __init__(self, image_paths: list, transform: Optional[Callable] = None):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return {"image": image, "image_path": str(image_path)}
