from dataclasses import dataclass

@dataclass
class TrainConfig:
    train_csv: str = "data/splits/train.csv"
    val_csv: str = "data/splits/val.csv"
    test_csv: str = "data/splits/test.csv"
    train_image_dir: str = "data/processed/train"
    val_image_dir: str = "data/processed/val"
    test_image_dir: str = "data/processed/test"
    model_dir: str = "model/checkpoints"

    num_classes: int = 5
    image_size: int = 320
    batch_size: int = 8
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    num_workers: int = 2
    mixup_alpha: float = 0.4

    device: str = "cuda"
    seed: int = 42
    early_stopping_patience: int = 7
    label_smoothing: float = 0.1
    use_mixed_precision: bool = True

DEFAULT_CONFIG = TrainConfig()
