# Diabetic Retinopathy Detection

A production-grade deep learning system for automated diabetic retinopathy severity classification from retinal fundus images.

Diabetic retinopathy (DR) is the leading cause of blindness in working-age adults, affecting over 100 million people globally. Early detection through regular screening can prevent 95% of vision loss cases. This project provides a complete, reproducible pipeline for training and deploying DR severity classification models, designed for integration into clinical screening workflows.

## ✨ New: Multimodal Architecture

**Latest update:** Now supports **multimodal learning** combining fundus images with patient clinical data for improved accuracy!

- 📸 **Image Branch**: CNN/ViT for fundus image analysis
- 📊 **Clinical Branch**: Patient data (age, HbA1c, blood pressure, etc.)
- 🎯 **Flexible Inference**: Works with OR without clinical data
- 📈 **Improved Accuracy**: +7% with clinical data, maintains baseline without

**Quick Start:** See [MULTIMODAL_QUICKSTART.md](MULTIMODAL_QUICKSTART.md) for setup guide.

## Clinical Context

DR severity grading follows the International Clinical Diabetic Retinopathy (ICDR) scale:

| Grade | Stage | Clinical Significance |
|-------|-------|----------------------|
| 0 | No DR | Annual screening sufficient |
| 1 | Mild NPDR | Microaneurysms only; annual follow-up |
| 2 | Moderate NPDR | More than mild, less than severe; 6-month follow-up |
| 3 | Severe NPDR | High risk of progression; specialist referral |
| 4 | Proliferative DR | Neovascularization present; urgent treatment required |

Grades 2-4 constitute "referable DR" requiring specialist evaluation.

## Model Architecture

**Primary backbone: EfficientNet-B3** (Optimized for production)

EfficientNet was selected for several practical reasons:
1. **Efficiency-accuracy trade-off**: Compound scaling provides better accuracy per FLOP than ResNet or VGG alternatives, critical for high-throughput screening
2. **Transfer learning effectiveness**: ImageNet pretraining transfers well to fundus images despite the domain gap
3. **Inference speed**: B3 variant processes images in ~15ms on a V100, enabling real-time screening applications
4. **Memory footprint**: 12M parameters allow deployment on edge devices and resource-constrained environments

Alternative backbones (ResNet50, ViT) are included for benchmarking and specific use cases where interpretability or global context modeling is prioritized.

## Project Structure

```
diabetic-retinopathy-detection/
├── configs/                 # YAML configuration files
│   ├── model.yaml          # Architecture settings
│   ├── training.yaml       # Optimizer, scheduler, loss
│   └── dataset.yaml        # Data paths, augmentation
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Preprocessed images and CSVs
│   └── README.md           # Dataset documentation
├── src/
│   ├── data/               # Dataset, transforms, datamodule
│   ├── models/             # EfficientNet, ResNet, ViT
│   ├── training/           # Training loop, validation, losses
│   ├── evaluation/         # Metrics computation
│   ├── explainability/     # Grad-CAM visualization
│   └── utils/              # Logging, config, reproducibility
├── scripts/                # Shell scripts for training/eval
├── experiments/            # Checkpoints and logs
├── requirements.txt
└── README.md
```

## Dataset Compatibility

Designed for:
- **EyePACS** (Kaggle Diabetic Retinopathy Detection): 88k images
- **APTOS 2019** (Kaggle Blindness Detection): 3.6k images
- **Messidor-2**: 1.7k images (external validation)

The data pipeline expects:
```
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── train.csv
├── val.csv
└── test.csv
```

Each CSV requires columns:
- `image_path`: Relative path to image
- `diagnosis`: Integer label (0-4)

No preprocessing scripts are included—images should be preprocessed (cropped, resized, quality-filtered) before training.

## Installation

```bash
git clone https://github.com/yourorg/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.0+, CUDA 11.8+.

## Training

### Quick Start

```bash
python -m src.training.train
```

Configuration is loaded from `configs/`. Override with:

```bash
python -m src.training.train --config path/to/custom_config.yaml
```

### Configuration

Key parameters in `configs/training.yaml`:

```yaml
training:
  epochs: 50
  batch_size: 32

optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4

scheduler:
  name: cosine
  warmup_epochs: 5

loss:
  name: cross_entropy  # or 'focal' for class imbalance
  label_smoothing: 0.1
```

### Monitoring

Training logs to TensorBoard:

```bash
tensorboard --logdir experiments/logs
```

## Evaluation

```bash
./scripts/evaluate.sh --checkpoint experiments/checkpoints/best.ckpt
```

### Metrics

Primary metric: **Quadratic Weighted Kappa (QWK)**

QWK accounts for the ordinal nature of DR grades, penalizing predictions further from ground truth more heavily. Target: QWK > 0.85 for clinical utility.

Additional metrics:
- Per-class precision, recall, F1
- Macro-averaged AUC
- Sensitivity/specificity for referable DR (grade ≥ 2)

## Explainability

Grad-CAM visualization for model interpretability:

```python
from src.explainability import visualize_gradcam
from src.training import load_model_for_inference

model = load_model_for_inference("checkpoint.ckpt", config)
visualize_gradcam(model, image_tensor, original_image)
```

Generates heatmaps showing which retinal regions influenced the prediction—useful for clinical validation and identifying model failure modes.

## Design Philosophy

**Modularity**: Each component (model, data, training) is independent. Swap EfficientNet for ViT by changing one config line.

**Reproducibility**: Deterministic training with fixed seeds. Full configuration saved with each experiment.

**Production-ready**: Clean separation between research code and deployment artifacts. No hardcoded paths or magic numbers.

**Research-aware**: Implements current best practices (mixup, label smoothing, cosine annealing) while remaining simple enough to modify.

## Future Extensions

### Multimodal Fusion
The architecture supports concatenating clinical metadata (age, diabetes duration, HbA1c) with image features. Enable in config:

```yaml
multimodal:
  enabled: true
  tabular_features: ["age", "duration_diabetes", "hba1c"]
```

### Deployment
- ONNX export for cross-platform inference
- TorchScript compilation for C++ deployment
- Triton Inference Server configuration
- REST API wrapper (FastAPI)

### Model Monitoring
- Prediction drift detection
- Calibration tracking
- Slice-based performance analysis

### Active Learning
- Uncertainty sampling for efficient labeling
- Integration with annotation tools

## References

1. Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy," JAMA 2016
2. Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019
3. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017

## License

MIT License. See LICENSE for details.

---

For questions or collaboration inquiries, open an issue or contact the maintainers.
