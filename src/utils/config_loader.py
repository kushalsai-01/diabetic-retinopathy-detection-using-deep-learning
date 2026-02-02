"""
Configuration loading utilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.
    Later configs override earlier ones.
    """
    result = {}
    
    for config in configs:
        for key, value in config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
                
    return result


def load_config(
    config_path: Optional[str] = None,
    config_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load configuration from YAML files.
    
    If config_path is provided, loads that single file.
    Otherwise, loads and merges model.yaml, training.yaml, and dataset.yaml
    from the config_dir (defaults to 'configs/').
    
    Args:
        config_path: Path to a single config file.
        config_dir: Directory containing config files.
        
    Returns:
        Merged configuration dictionary.
    """
    if config_path:
        return load_yaml(config_path)
        
    # Default config directory
    if config_dir is None:
        # Try to find configs relative to project root
        possible_paths = [
            Path("configs"),
            Path(__file__).parent.parent.parent / "configs",
        ]
        for p in possible_paths:
            if p.exists():
                config_dir = str(p)
                break
        else:
            config_dir = "configs"
            
    config_dir = Path(config_dir)
    
    # Load individual config files
    configs = []
    
    config_files = ["model.yaml", "training.yaml", "dataset.yaml"]
    for filename in config_files:
        filepath = config_dir / filename
        if filepath.exists():
            configs.append(load_yaml(filepath))
            
    return merge_configs(*configs)


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required fields.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        True if valid, raises ValueError otherwise.
    """
    required_fields = {
        "model": ["name", "num_classes"],
        "training": ["epochs", "batch_size"],
        "dataset": ["root_dir"],
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
            
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field: {section}.{field}")
                
    return True


class ConfigManager:
    """Configuration manager with validation and defaults."""
    
    DEFAULTS = {
        "model": {
            "name": "efficientnet_b3",
            "num_classes": 5,
            "pretrained": True,
            "dropout_rate": 0.3,
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "num_workers": 4,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 5,
        },
        "loss": {
            "name": "cross_entropy",
            "label_smoothing": 0.1,
        },
        "dataset": {
            "root_dir": "data/processed",
            "train_csv": "train.csv",
            "val_csv": "val.csv",
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        loaded = load_config(config_path)
        self.config = merge_configs(self.DEFAULTS, loaded)
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with dot notation support."""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a config value with dot notation support."""
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
