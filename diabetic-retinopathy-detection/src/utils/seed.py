"""
Seed utilities for reproducibility.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Enable CUDA deterministic mode (may impact performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass
    else:
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers.
    
    Use with DataLoader's worker_init_fn parameter to ensure
    reproducibility across data loading workers.
    
    Args:
        worker_id: Worker process ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Get a seeded random generator for data sampling.
    
    Args:
        seed: Random seed. If None, uses default PyTorch seed.
        
    Returns:
        Seeded torch.Generator instance.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g
