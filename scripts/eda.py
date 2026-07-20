import sys
from pathlib import Path

# Add project root to sys.path dynamically
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def run_eda() -> None:
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = Path("data/splits")
    raw_dir = Path("data/raw")
    
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    
    # Class distribution plot
    plt.figure(figsize=(10, 5))
    df_all = pd.concat([
        train_df.assign(Split="Train"),
        val_df.assign(Split="Validation"),
        test_df.assign(Split="Test")
    ])
    sns.countplot(data=df_all, x="diagnosis", hue="Split")
    plt.title("Class Distribution by Split")
    plt.xlabel("Severity Grade")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png")
    plt.close()
    
    # Read resolutions and brightness of sample training images
    resolutions = []
    brightness = []
    
    # Sample up to 100 images for speed in stats
    sample_df = train_df.sample(min(100, len(train_df)), random_state=42)
    for _, row in sample_df.iterrows():
        img_path = raw_dir / "train" / f"{row['id_code']}.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                resolutions.append((w, h))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness.append(gray.mean())

    # Resolution distribution
    if resolutions:
        widths, heights = zip(*resolutions)
        plt.figure(figsize=(6, 6))
        plt.scatter(widths, heights, alpha=0.5)
        plt.title("Image Resolution Distribution")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.savefig(figures_dir / "resolution_distribution.png")
        plt.close()
        
    # Brightness distribution
    if brightness:
        plt.figure(figsize=(8, 4))
        sns.histplot(brightness, kde=True, bins=20)
        plt.title("Image Mean Brightness Distribution (Raw)")
        plt.xlabel("Grayscale Mean Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figures_dir / "brightness_distribution.png")
        plt.close()

    # Pixel intensity histogram of a sample image
    if len(sample_df) > 0:
        sample_path = raw_dir / "train" / f"{sample_df.iloc[0]['id_code']}.png"
        if sample_path.exists():
            img = cv2.imread(str(sample_path))
            if img is not None:
                plt.figure(figsize=(8, 4))
                colors = ('b', 'g', 'r')
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                    plt.plot(hist, color=col)
                plt.title("Pixel Intensity Histogram (Sample Image)")
                plt.xlabel("Pixel Value")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(figures_dir / "pixel_intensity_histogram.png")
                plt.close()

    # Calculate class weights for imbalance recommendation
    counts = train_df["diagnosis"].value_counts().sort_index()
    total = len(train_df)
    class_weights = total / (len(counts) * counts)
    class_weights_str = ", ".join([f"Grade {i}: {w:.2f}" for i, w in class_weights.items()])
    
    # Generate Markdown report
    report_path = Path("reports/eda_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_content = f"""# Exploratory Data Analysis Report

## Dataset Summary
- **Train count:** {len(train_df)}
- **Validation count:** {len(val_df)}
- **Test count:** {len(test_df)}
- **Total images:** {len(train_df) + len(val_df) + len(test_df)}

## Class Distribution Counts
{counts.to_string()}

## Recommended Class Weights for Loss Function
`{class_weights_str}`

## Visualizations
- Class distribution: [class_distribution.png](figures/class_distribution.png)
- Resolution scatter: [resolution_distribution.png](figures/resolution_distribution.png)
- Brightness histogram: [brightness_distribution.png](figures/brightness_distribution.png)
- Sample intensity: [pixel_intensity_histogram.png](figures/pixel_intensity_histogram.png)
"""
    
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print("EDA execution and report generation complete.")

if __name__ == "__main__":
    run_eda()
