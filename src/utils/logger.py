"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        log_file: Optional file path for logging.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
    return logger


def setup_experiment_logging(
    experiment_name: str,
    log_dir: str = "experiments/logs",
) -> logging.Logger:
    """
    Setup logging for an experiment run.
    
    Args:
        experiment_name: Name of the experiment.
        log_dir: Directory for log files.
        
    Returns:
        Configured logger for the experiment.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"
    
    logger = get_logger(
        name=experiment_name,
        log_file=str(log_file),
    )
    
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


class MetricsLogger:
    """Simple metrics logger for tracking training progress."""
    
    def __init__(self, log_dir: str = "experiments/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []
        
    def log(self, epoch: int, metrics: dict) -> None:
        """Log metrics for an epoch."""
        entry = {"epoch": epoch, **metrics}
        self.metrics.append(entry)
        
    def save(self, filename: str = "metrics.json") -> None:
        """Save metrics to JSON file."""
        import json
        
        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
    def get_best(self, metric: str, mode: str = "max") -> dict:
        """Get the best epoch based on a metric."""
        if not self.metrics:
            return {}
            
        if mode == "max":
            return max(self.metrics, key=lambda x: x.get(metric, float("-inf")))
        else:
            return min(self.metrics, key=lambda x: x.get(metric, float("inf")))
