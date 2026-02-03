# Quick Start Guide

## Installation

```bash
git clone https://github.com/kushalsai-01/diabetic-retinopathy-detection-using-deep-learning.git
cd diabetic-retinopathy-detection-using-deep-learning
pip install -r requirements.txt
```

## Usage

### 1. Configure Your Settings

Edit config files in `configs/`:
- `model.yaml` - Model architecture
- `training.yaml` - Training parameters  
- `dataset.yaml` - Data paths

### 2. Prepare Your Data

Structure your data as:
```
data/
├── train/
│   ├── images/
│   └── train.csv
├── val/
│   ├── images/
│   └── val.csv
└── test/
    ├── images/
    └── test.csv
```

### 3. Validate Installation

```bash
python validate_project.py
```

## Model Architectures

- **EfficientNet-B3** (default, recommended)
- **ResNet50** (baseline)
- **Vision Transformer** (for large datasets)

## Performance Metrics

- Quadratic Weighted Kappa (primary)
- Multi-class AUC
- Sensitivity/Specificity
- Confusion Matrix

## Support

For issues, open a ticket on GitHub!
