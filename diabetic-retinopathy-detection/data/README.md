# Data Directory

This directory contains datasets for diabetic retinopathy detection.

## Structure

```
data/
├── raw/           # Original, unprocessed data
├── processed/     # Preprocessed images and CSV files
└── README.md
```

## Supported Datasets

### EyePACS (Kaggle Diabetic Retinopathy Detection)
- Source: https://www.kaggle.com/c/diabetic-retinopathy-detection
- ~35,000 training images, ~53,000 test images
- 5 severity classes (0-4)

### APTOS 2019 Blindness Detection
- Source: https://www.kaggle.com/c/aptos2019-blindness-detection
- ~3,600 training images
- 5 severity classes (0-4)

## Expected Format

After preprocessing, the `processed/` directory should contain:

```
processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── train.csv
├── val.csv
└── test.csv
```

Each CSV should have at minimum:
- `image_path`: Path to the image file (relative to processed directory)
- `diagnosis`: Integer label (0-4)

## Severity Classes

| Level | Description |
|-------|-------------|
| 0 | No DR |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Proliferative DR |

## Preprocessing Pipeline

1. Download raw images to `raw/`
2. Run preprocessing script (resize, crop, quality filter)
3. Generate train/val/test splits
4. Save to `processed/`

Preprocessing scripts will be added based on specific dataset requirements.
